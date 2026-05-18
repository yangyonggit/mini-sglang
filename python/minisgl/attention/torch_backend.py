from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch
import torch.nn.functional as F
from minisgl.core import Batch, get_global_ctx

from .base import BaseAttnBackend, BaseAttnMetadata

if TYPE_CHECKING:
    from minisgl.models import ModelConfig


@dataclass
class TorchMetadata(BaseAttnMetadata):
    page_table: torch.Tensor     # (num_reqs, max_pages_per_req)
    cache_seqlens: torch.Tensor  # (num_reqs,) KV lengths, on GPU
    cache_seqlens_cpu: list      # same, on CPU for Python loop use
    cu_seqlens_q: torch.Tensor   # (num_reqs+1,) cumulative Q lengths, on GPU
    cu_seqlens_q_cpu: list       # same, on CPU for Python loop use
    max_seqlen_q: int

    def get_last_indices(self, bs: int) -> torch.Tensor:
        return self.cu_seqlens_q[1 : 1 + bs] - 1


class TorchAttentionBackend(BaseAttnBackend):
    def __init__(self, config: ModelConfig) -> None:
        ctx = get_global_ctx()
        self.kvcache = ctx.kv_cache
        self.page_size = ctx.page_size
        self.scale = config.head_dim**-0.5

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: Batch
    ) -> torch.Tensor:
        metadata = batch.attn_metadata
        assert isinstance(metadata, TorchMetadata)
        self.kvcache.store_kv(k, v, batch.out_loc, layer_id)

        k_cache = self.kvcache.k_cache(layer_id)
        v_cache = self.kvcache.v_cache(layer_id)

        if batch.is_decode:
            return _decode(q, k_cache, v_cache, metadata, self.scale, self.page_size)
        else:
            return _prefill(q, k_cache, v_cache, metadata, self.scale, self.page_size)

    def prepare_metadata(self, batch: Batch) -> None:
        reqs = batch.padded_reqs
        padded_size = len(reqs)
        seqlens_q = [req.extend_len for req in reqs]
        seqlens_k = [req.device_len for req in reqs]
        cached_lens = [req.cached_len for req in reqs]
        max_seqlen_q = max(seqlens_q)
        max_seqlen_k = max(seqlens_k)
        CPU_KWARGS = {"device": "cpu", "dtype": torch.int32, "pin_memory": True}

        device = self.kvcache.device
        cache_seqlens = torch.tensor(seqlens_k, **CPU_KWARGS).to(device, non_blocking=True)

        if max_seqlen_q == 1:
            cu_seqlens_q_cpu = list(range(padded_size + 1))
        else:
            raw = seqlens_k if all(l == 0 for l in cached_lens) else seqlens_q
            acc, cu_seqlens_q_cpu = 0, [0]
            for l in raw:
                acc += l
                cu_seqlens_q_cpu.append(acc)
        cu_seqlens_q = torch.tensor(cu_seqlens_q_cpu, **CPU_KWARGS).to(device, non_blocking=True)

        page_table = get_global_ctx().page_table
        new_page_table = torch.stack(
            [page_table[req.table_idx, : max_seqlen_k : self.page_size] for req in reqs]
        )
        if self.page_size > 1:
            new_page_table.div_(self.page_size, rounding_mode="floor")

        batch.attn_metadata = TorchMetadata(
            page_table=new_page_table,
            cache_seqlens=cache_seqlens,
            cache_seqlens_cpu=seqlens_k,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_q_cpu=cu_seqlens_q_cpu,
            max_seqlen_q=max_seqlen_q,
        )

    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        raise NotImplementedError("torch backend does not support CUDA graph")

    def prepare_for_capture(self, batch: Batch) -> None:
        raise NotImplementedError("torch backend does not support CUDA graph")

    def prepare_for_replay(self, batch: Batch) -> None:
        raise NotImplementedError("torch backend does not support CUDA graph")


def _gather_kv(
    kv_cache: torch.Tensor,   # (num_pages, page_size, kv_heads, head_dim)
    page_table: torch.Tensor,  # (num_reqs, max_pages)
    req_idx: int,
    seqlen: int,
    page_size: int,
) -> torch.Tensor:
    num_pages = (seqlen + page_size - 1) // page_size
    pages = page_table[req_idx, :num_pages]
    kv = kv_cache[pages].reshape(num_pages * page_size, kv_cache.shape[2], kv_cache.shape[3])
    return kv[:seqlen]  # (seqlen, kv_heads, head_dim)


def _decode(
    q: torch.Tensor,          # (num_reqs, num_heads, head_dim)
    k_cache: torch.Tensor,    # (num_pages, page_size, kv_heads, head_dim)
    v_cache: torch.Tensor,
    metadata: TorchMetadata,
    scale: float,
    page_size: int,
) -> torch.Tensor:
    num_reqs, num_heads, head_dim = q.shape
    kv_heads = k_cache.shape[2]
    groups = num_heads // kv_heads
    out = torch.empty_like(q)

    for i in range(num_reqs):
        seqlen = metadata.cache_seqlens_cpu[i]
        k = _gather_kv(k_cache, metadata.page_table, i, seqlen, page_size)  # (seqlen, kv_heads, head_dim)
        v = _gather_kv(v_cache, metadata.page_table, i, seqlen, page_size)

        k = k.repeat_interleave(groups, dim=1)  # (seqlen, num_heads, head_dim)
        v = v.repeat_interleave(groups, dim=1)

        qi = q[i]  # (num_heads, head_dim)
        scores = torch.einsum("hd,shd->hs", qi, k) * scale  # (num_heads, seqlen)
        scores = F.softmax(scores, dim=-1)
        out[i] = torch.einsum("hs,shd->hd", scores, v)

    return out


def _prefill(
    q: torch.Tensor,          # (total_tokens, num_heads, head_dim)
    k_cache: torch.Tensor,    # (num_pages, page_size, kv_heads, head_dim)
    v_cache: torch.Tensor,
    metadata: TorchMetadata,
    scale: float,
    page_size: int,
) -> torch.Tensor:
    num_heads = q.shape[1]
    kv_heads = k_cache.shape[2]
    groups = num_heads // kv_heads
    out = torch.empty_like(q)

    cu_q = metadata.cu_seqlens_q_cpu
    num_reqs = len(metadata.cache_seqlens_cpu)

    for i in range(num_reqs):
        q_start, q_end = cu_q[i], cu_q[i + 1]
        seqlen_k = metadata.cache_seqlens_cpu[i]
        qi = q[q_start:q_end]  # (extend_len, num_heads, head_dim)

        k = _gather_kv(k_cache, metadata.page_table, i, seqlen_k, page_size)
        v = _gather_kv(v_cache, metadata.page_table, i, seqlen_k, page_size)

        k = k.repeat_interleave(groups, dim=1)  # (seqlen_k, num_heads, head_dim)
        v = v.repeat_interleave(groups, dim=1)

        scores = torch.einsum("qhd,khd->hqk", qi, k) * scale  # (num_heads, extend_len, seqlen_k)

        # causal mask based on absolute positions (accounts for prefix cache offset)
        extend_len = q_end - q_start
        cached_len = seqlen_k - extend_len
        q_pos = torch.arange(extend_len, device=q.device) + cached_len
        k_pos = torch.arange(seqlen_k, device=q.device)
        mask = q_pos[:, None] >= k_pos[None, :]  # (extend_len, seqlen_k)
        scores = scores.masked_fill(~mask[None, :, :], float("-inf"))

        scores = F.softmax(scores, dim=-1)
        out[q_start:q_end] = torch.einsum("hqk,khd->qhd", scores, v)

    return out
