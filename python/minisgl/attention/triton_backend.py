from __future__ import annotations

from typing import TYPE_CHECKING, List

import torch
import triton
import triton.language as tl
from minisgl.core import Batch, get_global_ctx

from .base import BaseAttnBackend
from .torch_backend import TorchMetadata, _prefill

if TYPE_CHECKING:
    from minisgl.models import ModelConfig


@triton.jit
def _paged_decode_kernel(
    Q,           # (num_reqs, num_heads, head_dim)
    K_cache,     # (num_pages, page_size, kv_heads, head_dim)
    V_cache,
    Out,         # (num_reqs, num_heads, head_dim)
    Page_table,  # (num_reqs, max_pages_per_req)
    Seqlens,     # (num_reqs,) int32
    scale,
    stride_q_r, stride_q_h, stride_q_d,
    stride_k_p, stride_k_s, stride_k_h, stride_k_d,
    stride_pt_r,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUPS: tl.constexpr,
):
    req_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    kv_head_idx = head_idx // GROUPS

    seqlen = tl.load(Seqlens + req_idx)

    # load Q for this (req, head): (HEAD_DIM,)
    d_offs = tl.arange(0, HEAD_DIM)
    q = tl.load(Q + req_idx * stride_q_r + head_idx * stride_q_h + d_offs * stride_q_d).to(tl.float32)

    # online softmax state
    m_i = float("-inf")
    l_i = 0.0
    acc = tl.zeros((HEAD_DIM,), dtype=tl.float32)

    for block_start in range(0, seqlen, BLOCK_N):
        n_offs = block_start + tl.arange(0, BLOCK_N)
        mask = n_offs < seqlen

        # look up physical page index for each token position
        page_indices = tl.load(
            Page_table + req_idx * stride_pt_r + n_offs // PAGE_SIZE,
            mask=mask, other=0,
        )  # (BLOCK_N,)

        # load K: (BLOCK_N, HEAD_DIM)
        kv_ptrs = (
            page_indices[:, None] * stride_k_p
            + (n_offs % PAGE_SIZE)[:, None] * stride_k_s
            + kv_head_idx * stride_k_h
            + d_offs[None, :] * stride_k_d
        )
        k = tl.load(K_cache + kv_ptrs, mask=mask[:, None], other=0.0).to(tl.float32)
        v = tl.load(V_cache + kv_ptrs, mask=mask[:, None], other=0.0).to(tl.float32)

        # scores: (BLOCK_N,)
        scores = tl.sum(q[None, :] * k, axis=1) * scale
        scores = tl.where(mask, scores, float("-inf"))

        # online softmax update
        m_new = tl.maximum(m_i, tl.max(scores, axis=0))
        p = tl.exp(scores - m_new)           # (BLOCK_N,)
        correction = tl.exp(m_i - m_new)

        l_i = l_i * correction + tl.sum(p, axis=0)
        acc = acc * correction + tl.sum(p[:, None] * v, axis=0)

        m_i = m_new

    # normalize and store
    out_offs = req_idx * stride_q_r + head_idx * stride_q_h + d_offs * stride_q_d
    tl.store(Out + out_offs, (acc / l_i).to(q.dtype))


def _triton_decode(
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

    BLOCK_N = 64
    grid = (num_reqs, num_heads)
    _paged_decode_kernel[grid](
        q, k_cache, v_cache, out,
        metadata.page_table, metadata.cache_seqlens,
        scale,
        q.stride(0), q.stride(1), q.stride(2),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
        metadata.page_table.stride(0),
        HEAD_DIM=head_dim,
        PAGE_SIZE=page_size,
        BLOCK_N=BLOCK_N,
        GROUPS=groups,
    )
    return out


class TritonAttentionBackend(BaseAttnBackend):
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
            return _triton_decode(q, k_cache, v_cache, metadata, self.scale, self.page_size)
        else:
            return _prefill(q, k_cache, v_cache, metadata, self.scale, self.page_size)

    def prepare_metadata(self, batch: Batch) -> None:
        from .torch_backend import TorchAttentionBackend
        # reuse torch backend's metadata preparation
        TorchAttentionBackend.prepare_metadata(self, batch)

    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        raise NotImplementedError("triton backend does not support CUDA graph yet")

    def prepare_for_capture(self, batch: Batch) -> None:
        raise NotImplementedError("triton backend does not support CUDA graph yet")

    def prepare_for_replay(self, batch: Batch) -> None:
        raise NotImplementedError("triton backend does not support CUDA graph yet")
