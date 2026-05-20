from __future__ import annotations

import time
from contextlib import contextmanager
from typing import List, Tuple

import torch
import minisgl.core as _core
from minisgl.attention import create_attention_backend
from minisgl.core import Batch, Context, Req, SamplingParams
from minisgl.distributed import DistributedInfo, set_tp_info
from minisgl.engine.config import EngineConfig
from minisgl.engine.engine import Engine
from minisgl.kvcache import create_kvcache_pool
from minisgl.layers import set_rope_device
from minisgl.models import create_model, load_weight
from minisgl.utils import load_tokenizer, torch_dtype


@contextmanager
def _use_ctx(ctx: Context):
    # model.forward() and prepare_metadata both call get_global_ctx() internally;
    # this swaps the global so each model sees its own KV cache.
    old = _core._GLOBAL_CTX
    _core._GLOBAL_CTX = ctx
    try:
        yield
    finally:
        _core._GLOBAL_CTX = old


def _make_batch(req: Req, phase: str, device: torch.device, page_table: torch.Tensor) -> Batch:
    batch = Batch(reqs=[req], phase=phase)
    batch.padded_reqs = [req]
    batch.input_ids = req.input_ids[req.cached_len : req.device_len].to(device)
    batch.positions = torch.arange(
        req.cached_len, req.device_len, dtype=torch.int32, device=device
    )
    batch.out_loc = page_table[req.table_idx, req.cached_len : req.device_len]
    return batch


class DraftModelWrapper:
    """
    Runs the draft model without going through Engine.__init__,
    which asserts CUDA is uninitialized (target Engine already triggered it).
    """

    def __init__(
        self,
        config: EngineConfig,
        device: torch.device,
        max_seq_len: int,
        attn_backend: str = "fi",
    ):
        self.device = device

        # Sequential page table: position i → physical page i. Safe for dirty truncation.
        self.page_table = torch.zeros(1, max_seq_len, dtype=torch.int32, device=device)
        self.page_table[0] = torch.arange(max_seq_len, dtype=torch.int32)

        self.ctx = Context(config.page_size)
        self.ctx.page_table = self.page_table
        self.ctx.kv_cache = create_kvcache_pool(
            model_config=config.model_config,
            num_pages=max_seq_len,
            page_size=config.page_size,
            dtype=config.dtype,
            device=device,
        )
        # FlashInfer's __init__ calls get_global_ctx(), so create it inside _use_ctx.
        with _use_ctx(self.ctx):
            self.ctx.attn_backend = create_attention_backend(attn_backend, config.model_config)

        with torch.device("meta"), torch_dtype(config.dtype):
            self.model = create_model(config.model_config)
        self.model.load_state_dict(
            {k: v.to(config.dtype) for k, v in load_weight(config.model_path, device)}
        )

    def _forward(self, req: Req, phase: str) -> torch.Tensor:
        batch = _make_batch(req, phase, self.device, self.page_table)
        # _use_ctx must wrap prepare_metadata too: fi/torch backends read
        # get_global_ctx().page_table inside prepare_metadata.
        with _use_ctx(self.ctx):
            self.ctx.attn_backend.prepare_metadata(batch)
            with self.ctx.forward_batch(batch):
                return self.model.forward().float()

    def prefill(self, req: Req) -> torch.Tensor:
        logits = self._forward(req, "prefill")
        req.cached_len = req.device_len
        return logits[0]

    def step(self, req: Req, token: int) -> torch.Tensor:
        req.input_ids = torch.cat(
            [req.input_ids, torch.tensor([token], dtype=torch.int32)]
        )
        req.device_len += 1
        logits = self._forward(req, "decode")
        req.cached_len = req.device_len
        return logits[0]

    @staticmethod
    def truncate(req: Req, new_len: int) -> None:
        # Dirty truncation: pages past new_len will be overwritten next round.
        req.input_ids = req.input_ids[:new_len]
        req.cached_len = new_len
        req.device_len = new_len


class SpeculativeDecoder:
    """Standalone speculative decoding. Does not integrate with the Scheduler."""

    MAX_SEQ_LEN = 2048

    def __init__(
        self,
        draft_model_path: str,
        target_model_path: str,
        k: int = 5,
        dtype: torch.dtype = torch.bfloat16,
        attn_backend: str = "fi",
    ):
        self.k = k
        device = torch.device("cuda:0")

        # Target first: Engine.__init__ asserts CUDA uninitialized, so it must go first.
        set_tp_info(rank=0, size=1)
        set_rope_device(device)
        target_config = EngineConfig(
            model_path=target_model_path,
            tp_info=DistributedInfo(0, 1),
            dtype=dtype,
            attention_backend=attn_backend,
            cuda_graph_max_bs=0,
        )
        self.target_engine = Engine(target_config)
        self.device = self.target_engine.device

        for i in range(self.MAX_SEQ_LEN):
            self.target_engine.page_table[0, i] = i

        draft_config = EngineConfig(
            model_path=draft_model_path,
            tp_info=DistributedInfo(0, 1),
            dtype=dtype,
        )
        self.draft = DraftModelWrapper(draft_config, self.device, self.MAX_SEQ_LEN, attn_backend)

        self.tokenizer = load_tokenizer(target_model_path)

    def _make_req(self, input_ids: torch.Tensor, max_new_tokens: int) -> Req:
        return Req(
            input_ids=input_ids.cpu().to(torch.int32),
            table_idx=0,
            cached_len=0,
            output_len=max_new_tokens,
            uid=0,
            sampling_params=SamplingParams(),
            cache_handle=None,  # type: ignore
        )

    @staticmethod
    def _acceptance_sample(
        draft_tokens: List[int],
        draft_logits: List[torch.Tensor],   # draft_logits[i]  = p_draft(·|prefix + d_0..d_{i-1})
        target_logits: List[torch.Tensor],  # target_logits[i] = p_target(·|prefix + d_0..d_{i-1}), K+1 elements
    ) -> List[int]:
        """
        Rejection sampling loop.  accept_prob = min(1, p_target(d_i) / p_draft(d_i)).
        On rejection: sample from normalize(max(0, p_target - p_draft)).
        All K accepted → append bonus token from target_logits[K].
        Output distribution == pure target distribution.
        """
        K = len(draft_tokens)
        accepted: List[int] = []

        for i in range(K):
            d_tok = draft_tokens[i]
            p_draft = torch.softmax(draft_logits[i], dim=-1)
            p_target = torch.softmax(target_logits[i], dim=-1)

            accept_prob = min(1.0, (p_target[d_tok] / (p_draft[d_tok] + 1e-9)).item())
            if torch.rand(1).item() < accept_prob:
                accepted.append(d_tok)
            else:
                corrected = (p_target - p_draft).clamp(min=0.0)
                corrected /= corrected.sum()
                accepted.append(int(torch.multinomial(corrected, 1).item()))
                return accepted  # tokens after position i are conditioned on wrong d_i

        accepted.append(int(torch.argmax(target_logits[K]).item()))
        return accepted

    def _target_step(self, req: Req, token: int) -> torch.Tensor:
        req.input_ids = torch.cat(
            [req.input_ids, torch.tensor([token], dtype=torch.int32)]
        )
        req.device_len += 1
        pos = req.device_len - 1
        self.target_engine.page_table[req.table_idx, pos] = pos

        batch = _make_batch(req, "decode", self.device, self.target_engine.page_table)
        self.target_engine.attn_backend.prepare_metadata(batch)
        with self.target_engine.ctx.forward_batch(batch):
            logits = self.target_engine.model.forward().float()
        req.cached_len = req.device_len
        return logits[0]

    def _target_prefill(self, req: Req) -> torch.Tensor:
        batch = _make_batch(req, "prefill", self.device, self.target_engine.page_table)
        self.target_engine.attn_backend.prepare_metadata(batch)
        with self.target_engine.ctx.forward_batch(batch):
            logits = self.target_engine.model.forward().float()
        req.cached_len = req.device_len
        return logits[0]

    def _target_verify(self, req: Req, draft_tokens: List[int]) -> torch.Tensor:
        """Verify K draft tokens in one parallel prefill. Returns (K, vocab) logits."""
        new_ids = torch.tensor(draft_tokens, dtype=torch.int32)
        req.input_ids = torch.cat([req.input_ids, new_ids])
        req.device_len += len(draft_tokens)

        batch = _make_batch(req, "prefill", self.device, self.target_engine.page_table)
        batch.verify = True
        self.target_engine.attn_backend.prepare_metadata(batch)
        with self.target_engine.ctx.forward_batch(batch):
            logits = self.target_engine.model.forward().float()  # (K, vocab)
        req.cached_len = req.device_len
        return logits

    def _one_round(
        self,
        draft_req: Req,
        target_req: Req,
        draft_logit: torch.Tensor,   # p_draft(·|current prefix)
        target_logit: torch.Tensor,  # p_target(·|current prefix)
    ) -> Tuple[List[int], torch.Tensor, torch.Tensor]:
        prefix_len = draft_req.cached_len

        # Draft generates K tokens
        draft_tokens: List[int] = []
        draft_logits: List[torch.Tensor] = [draft_logit]
        for _ in range(self.k):
            p = torch.softmax(draft_logits[-1], dim=-1)
            d_i = int(torch.multinomial(p, 1).item())
            draft_tokens.append(d_i)
            draft_logits.append(self.draft.step(draft_req, d_i))

        # Target verifies all K draft tokens in one parallel prefill.
        verify_logits = self._target_verify(target_req, draft_tokens)  # (K, vocab)
        target_logits = [target_logit] + [verify_logits[i] for i in range(self.k)]

        accepted = self._acceptance_sample(draft_tokens, draft_logits[:self.k], target_logits)
        n = len(accepted)

        # Truncate both caches to the last fully-agreed position, then commit
        # the final accepted token (which may differ from the draft's token).
        DraftModelWrapper.truncate(draft_req, prefix_len + n - 1)
        DraftModelWrapper.truncate(target_req, prefix_len + n - 1)
        new_draft_logit = self.draft.step(draft_req, accepted[-1])
        new_target_logit = self._target_step(target_req, accepted[-1])

        return accepted, new_draft_logit, new_target_logit

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        eos_token_id: int | None = None,
    ) -> str:
        if eos_token_id is None:
            eos_token_id = self.tokenizer.eos_token_id

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").view(-1).to(torch.int32)
        draft_req = self._make_req(input_ids, max_new_tokens)
        target_req = self._make_req(input_ids, max_new_tokens)

        draft_logit = self.draft.prefill(draft_req)
        target_logit = self._target_prefill(target_req)

        output_ids: List[int] = []
        total_rounds = 0
        total_accepted = 0
        t0 = time.perf_counter()

        while len(output_ids) < max_new_tokens:
            if not draft_req.can_decode or not target_req.can_decode:
                break

            accepted, draft_logit, target_logit = self._one_round(
                draft_req, target_req, draft_logit, target_logit
            )
            total_rounds += 1
            total_accepted += len(accepted)

            print(f"round {total_rounds}: accepted {len(accepted)}/{self.k + 1} tokens")

            for tok in accepted:
                output_ids.append(tok)
                if tok == eos_token_id:
                    elapsed = time.perf_counter() - t0
                    print(f"\ntokens: {len(output_ids)}  time: {elapsed:.2f}s  "
                          f"tok/s: {len(output_ids) / elapsed:.1f}  "
                          f"avg accepted/round: {total_accepted / total_rounds:.2f}/{self.k + 1}")
                    return self.tokenizer.decode(output_ids)

        elapsed = time.perf_counter() - t0
        print(f"\ntokens: {len(output_ids)}  time: {elapsed:.2f}s  "
              f"tok/s: {len(output_ids) / elapsed:.1f}  "
              f"avg accepted/round: {total_accepted / max(total_rounds, 1):.2f}/{self.k + 1}")
        return self.tokenizer.decode(output_ids)
