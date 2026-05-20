"""
Speculative decoding benchmark.

Usage:
    python benchmark/test_speculative.py
    python benchmark/test_speculative.py --draft Qwen/Qwen3-0.6B --target Qwen/Qwen3-7B --k 5
"""
import argparse
import time

import torch

PROMPT = "Tell me about the history of the Roman Empire."
MAX_NEW_TOKENS = 200


def run_speculative(draft_path: str, target_path: str, k: int) -> None:
    from minisgl.engine.speculative import SpeculativeDecoder

    print(f"\n=== Speculative Decoding (draft={draft_path}, target={target_path}, k={k}) ===")
    decoder = SpeculativeDecoder(
        draft_model_path=draft_path,
        target_model_path=target_path,
        k=k,
    )
    result = decoder.generate(PROMPT, max_new_tokens=MAX_NEW_TOKENS)
    print("\n--- Output ---")
    print(result)


def run_target_only(target_path: str) -> None:
    """Baseline: target model alone, greedy decode, using Engine directly."""
    import minisgl.core as _core
    from minisgl.core import Batch, Req, SamplingParams
    from minisgl.distributed import DistributedInfo
    from minisgl.engine.config import EngineConfig
    from minisgl.engine.engine import Engine
    from minisgl.engine.speculative import _make_batch
    from minisgl.utils import load_tokenizer, torch_dtype

    print(f"\n=== Target-Only Baseline ({target_path}) ===")

    device = torch.device("cuda:0")

    config = EngineConfig(
        model_path=target_path,
        tp_info=DistributedInfo(0, 1),
        dtype=torch.bfloat16,
        attention_backend="fi",
        cuda_graph_max_bs=0,
    )
    engine = Engine(config)

    MAX_SEQ_LEN = 2048
    for i in range(MAX_SEQ_LEN):
        engine.page_table[0, i] = i

    tokenizer = load_tokenizer(target_path)
    eos = tokenizer.eos_token_id
    input_ids = tokenizer.encode(PROMPT, return_tensors="pt").view(-1).to(torch.int32)

    req = Req(
        input_ids=input_ids.cpu().to(torch.int32),
        table_idx=0,
        cached_len=0,
        output_len=MAX_NEW_TOKENS,
        uid=0,
        sampling_params=SamplingParams(),
        cache_handle=None,  # type: ignore
    )

    # prefill
    batch = _make_batch(req, "prefill", device, engine.page_table)
    engine.attn_backend.prepare_metadata(batch)
    with engine.ctx.forward_batch(batch):
        logits = engine.model.forward().float()
    req.cached_len = req.device_len

    output_ids = []
    t0 = time.perf_counter()

    for _ in range(MAX_NEW_TOKENS):
        if not req.can_decode:
            break
        tok = int(torch.argmax(logits[0]).item())
        output_ids.append(tok)
        if tok == eos:
            break

        req.input_ids = torch.cat([req.input_ids, torch.tensor([tok], dtype=torch.int32)])
        req.device_len += 1
        pos = req.device_len - 1
        engine.page_table[req.table_idx, pos] = pos

        batch = _make_batch(req, "decode", device, engine.page_table)
        engine.attn_backend.prepare_metadata(batch)
        with engine.ctx.forward_batch(batch):
            logits = engine.model.forward().float()
        req.cached_len = req.device_len

    elapsed = time.perf_counter() - t0
    n = len(output_ids)
    print(f"\ntokens: {n}  time: {elapsed:.2f}s  tok/s: {n / elapsed:.1f}")
    print("\n--- Output ---")
    print(tokenizer.decode(output_ids))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--target", default="Qwen/Qwen3-4B")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--speculative-only", action="store_true")
    args = parser.parse_args()

    if args.speculative_only:
        run_speculative(args.draft, args.target, args.k)
    elif args.baseline_only:
        run_target_only(args.target)
    else:
        run_speculative(args.draft, args.target, args.k)
        # Note: running both in same process is not possible because Engine asserts
        # CUDA is uninitialized at startup. Run each mode separately:
        #   python benchmark/test_speculative.py --speculative-only
        #   python benchmark/test_speculative.py --baseline-only
        print("\nTip: to compare, run with --speculative-only and --baseline-only separately.")


if __name__ == "__main__":
    main()
