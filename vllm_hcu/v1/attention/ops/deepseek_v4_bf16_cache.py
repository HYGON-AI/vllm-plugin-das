# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Plain-BF16 paged-cache kernels for DeepSeek V4 sparse attention."""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _gather_bf16_k_cache_kernel(
    out_ptr,
    out_stride0,
    out_stride1,
    cache_ptr,
    cache_stride0,
    cache_stride1,
    seq_lens_ptr,
    gather_lens_ptr,
    block_table_ptr,
    offset,
    max_blocks_per_seq: tl.constexpr,
    cache_block_size: tl.constexpr,
    head_dim: tl.constexpr,
    padded_head_dim: tl.constexpr,
    num_workers: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    worker_idx = tl.program_id(1)
    seq_len = tl.load(seq_lens_ptr + batch_idx)
    if gather_lens_ptr is None:
        gather_len = seq_len
    else:
        gather_len = tl.load(gather_lens_ptr + batch_idx)
    start_pos = seq_len - gather_len
    dims = tl.arange(0, padded_head_dim)
    dim_mask = dims < head_dim

    for i in range(worker_idx, gather_len, num_workers):
        pos = start_pos + i
        logical_block = pos // cache_block_size
        pos_in_block = pos % cache_block_size
        physical_block = tl.load(
            block_table_ptr + batch_idx * max_blocks_per_seq + logical_block
        )
        src = (
            cache_ptr
            + physical_block.to(tl.int64) * cache_stride0
            + pos_in_block * cache_stride1
        )
        dst = out_ptr + batch_idx * out_stride0 + (offset + i) * out_stride1
        values = tl.load(src + dims, mask=dim_mask, other=0.0)
        tl.store(dst + dims, values, mask=dim_mask)


def gather_bf16_k_cache(
    out: torch.Tensor,
    k_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor | None,
    block_table: torch.Tensor,
    block_size: int,
    offset: int,
) -> None:
    """Gather the requested tail of each paged BF16 sequence into ``out``."""

    if k_cache.dtype != torch.bfloat16:
        raise TypeError(f"expected BF16 K cache, got {k_cache.dtype}")
    if k_cache.ndim != 3:
        raise ValueError(
            "DeepSeek V4 BF16 K cache must be [blocks, block_size, head_dim], "
            f"got {tuple(k_cache.shape)}"
        )
    num_workers = 128
    _gather_bf16_k_cache_kernel[(seq_lens.shape[0], num_workers)](
        out,
        out.stride(0),
        out.stride(1),
        k_cache,
        k_cache.stride(0),
        k_cache.stride(1),
        seq_lens,
        gather_lens,
        block_table,
        offset,
        max_blocks_per_seq=block_table.shape[-1],
        cache_block_size=block_size,
        head_dim=k_cache.shape[-1],
        padded_head_dim=triton.next_power_of_2(k_cache.shape[-1]),
        num_workers=num_workers,
    )


@triton.jit
def _bf16_sparse_attn_decode_kernel(
    q_ptr,
    main_cache_ptr,
    main_indices_ptr,
    main_indptr_ptr,
    extra_cache_ptr,
    extra_indices_ptr,
    extra_indptr_ptr,
    attn_sink_ptr,
    out_ptr,
    q_stride0,
    q_stride1,
    main_cache_stride0,
    main_cache_stride1,
    extra_cache_stride0,
    extra_cache_stride1,
    out_stride0,
    out_stride1,
    main_num_rows,
    extra_num_rows,
    main_block_size,
    extra_block_size,
    scale,
    num_heads,
    HAS_EXTRA: tl.constexpr,
    HAS_ATTN_SINK: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    NOPE_BLOCK: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    query_idx = tl.program_id(0)
    head_block_idx = tl.program_id(1)
    head_offsets = head_block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    head_mask = head_offsets < num_heads
    nope_offsets = tl.arange(0, NOPE_BLOCK)
    nope_mask = nope_offsets < NOPE_DIM
    rope_offsets = tl.arange(0, ROPE_DIM)

    q_row = q_ptr + query_idx * q_stride0 + head_offsets[:, None] * q_stride1
    q_nope = tl.load(
        q_row + nope_offsets[None, :],
        mask=head_mask[:, None] & nope_mask[None, :],
        other=0.0,
    )
    q_rope = tl.load(
        q_row + NOPE_DIM + rope_offsets[None, :],
        mask=head_mask[:, None],
        other=0.0,
    )

    neg_large = -3.4028234663852886e38
    m_i = tl.full((BLOCK_H,), neg_large, dtype=tl.float32)
    l_i = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc_nope = tl.zeros((BLOCK_H, NOPE_BLOCK), dtype=tl.float32)
    acc_rope = tl.zeros((BLOCK_H, ROPE_DIM), dtype=tl.float32)
    k_offsets = tl.arange(0, BLOCK_K)
    zero_nope = tl.zeros((BLOCK_K, NOPE_BLOCK), dtype=tl.bfloat16)
    zero_rope = tl.zeros((BLOCK_K, ROPE_DIM), dtype=tl.bfloat16)

    main_start = tl.load(main_indptr_ptr + query_idx)
    main_end = tl.load(main_indptr_ptr + query_idx + 1)
    main_len = main_end - main_start
    for k_start in tl.range(0, main_len, BLOCK_K):
        k_pos = k_start + k_offsets
        in_range = k_pos < main_len
        slot = tl.load(main_indices_ptr + main_start + k_pos, mask=in_range, other=-1)
        valid = in_range & (slot >= 0) & (slot < main_num_rows)
        safe_slot = tl.where(valid, slot, 0)
        block_idx = safe_slot // main_block_size
        pos_in_block = safe_slot % main_block_size
        token_ptr = (
            main_cache_ptr
            + block_idx.to(tl.int64) * main_cache_stride0
            + pos_in_block * main_cache_stride1
        )
        k_nope = tl.load(
            token_ptr[:, None] + nope_offsets[None, :],
            mask=valid[:, None] & nope_mask[None, :],
            other=0.0,
        )
        k_rope = tl.load(
            token_ptr[:, None] + NOPE_DIM + rope_offsets[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        k_nope = tl.where(k_nope == k_nope, k_nope, zero_nope)
        k_rope = tl.where(k_rope == k_rope, k_rope, zero_rope)
        scores = tl.dot(q_nope, tl.trans(k_nope)) + tl.dot(
            q_rope, tl.trans(k_rope)
        )
        scores *= scale
        scores = tl.where(head_mask[:, None] & valid[None, :], scores, neg_large)
        m_block = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_block)
        alpha = tl.exp(m_i - m_new)
        probs = tl.exp(scores - m_new[:, None])
        probs = tl.where(head_mask[:, None] & valid[None, :], probs, 0.0)
        l_i = l_i * alpha + tl.sum(probs, axis=1)
        acc_nope = acc_nope * alpha[:, None] + tl.dot(
            probs.to(k_nope.dtype), k_nope
        )
        acc_rope = acc_rope * alpha[:, None] + tl.dot(
            probs.to(k_rope.dtype), k_rope
        )
        m_i = m_new

    if HAS_EXTRA:
        extra_start = tl.load(extra_indptr_ptr + query_idx)
        extra_end = tl.load(extra_indptr_ptr + query_idx + 1)
        extra_len = extra_end - extra_start
        for k_start in tl.range(0, extra_len, BLOCK_K):
            k_pos = k_start + k_offsets
            in_range = k_pos < extra_len
            slot = tl.load(
                extra_indices_ptr + extra_start + k_pos,
                mask=in_range,
                other=-1,
            )
            valid = in_range & (slot >= 0) & (slot < extra_num_rows)
            safe_slot = tl.where(valid, slot, 0)
            block_idx = safe_slot // extra_block_size
            pos_in_block = safe_slot % extra_block_size
            token_ptr = (
                extra_cache_ptr
                + block_idx.to(tl.int64) * extra_cache_stride0
                + pos_in_block * extra_cache_stride1
            )
            k_nope = tl.load(
                token_ptr[:, None] + nope_offsets[None, :],
                mask=valid[:, None] & nope_mask[None, :],
                other=0.0,
            )
            k_rope = tl.load(
                token_ptr[:, None] + NOPE_DIM + rope_offsets[None, :],
                mask=valid[:, None],
                other=0.0,
            )
            k_nope = tl.where(k_nope == k_nope, k_nope, zero_nope)
            k_rope = tl.where(k_rope == k_rope, k_rope, zero_rope)
            scores = tl.dot(q_nope, tl.trans(k_nope)) + tl.dot(
                q_rope, tl.trans(k_rope)
            )
            scores *= scale
            scores = tl.where(
                head_mask[:, None] & valid[None, :], scores, neg_large
            )
            m_block = tl.max(scores, axis=1)
            m_new = tl.maximum(m_i, m_block)
            alpha = tl.exp(m_i - m_new)
            probs = tl.exp(scores - m_new[:, None])
            probs = tl.where(head_mask[:, None] & valid[None, :], probs, 0.0)
            l_i = l_i * alpha + tl.sum(probs, axis=1)
            acc_nope = acc_nope * alpha[:, None] + tl.dot(
                probs.to(k_nope.dtype), k_nope
            )
            acc_rope = acc_rope * alpha[:, None] + tl.dot(
                probs.to(k_rope.dtype), k_rope
            )
            m_i = m_new

    if HAS_ATTN_SINK:
        sink = tl.load(
            attn_sink_ptr + head_offsets, mask=head_mask, other=neg_large
        ).to(tl.float32)
        m_final = tl.maximum(m_i, sink)
        alpha = tl.exp(m_i - m_final)
        denom = l_i * alpha + tl.exp(sink - m_final)
        out_nope = acc_nope * alpha[:, None] / tl.maximum(denom[:, None], 1.0e-30)
        out_rope = acc_rope * alpha[:, None] / tl.maximum(denom[:, None], 1.0e-30)
    else:
        denom = tl.maximum(l_i, 1.0e-30)
        out_nope = acc_nope / denom[:, None]
        out_rope = acc_rope / denom[:, None]

    out_row = out_ptr + query_idx * out_stride0 + head_offsets[:, None] * out_stride1
    tl.store(
        out_row + nope_offsets[None, :],
        out_nope,
        mask=head_mask[:, None] & nope_mask[None, :],
    )
    tl.store(
        out_row + NOPE_DIM + rope_offsets[None, :],
        out_rope,
        mask=head_mask[:, None],
    )


def _dense_to_ragged(
    indices: torch.Tensor,
    lengths: torch.Tensor | None,
    num_rows: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        build_ragged_indices_from_dense,
    )

    dense = indices.reshape(indices.shape[0], -1)
    if lengths is None:
        lengths = (dense >= 0).sum(dim=-1, dtype=torch.int32)
    return build_ragged_indices_from_dense(
        dense,
        lengths.reshape(-1),
        num_rows=num_rows,
    )


def bf16_sparse_attn_decode(
    q: torch.Tensor,
    kv_cache: torch.Tensor | None,
    swa_k_cache: torch.Tensor,
    swa_only: bool,
    topk_indices: torch.Tensor | None,
    topk_lens: torch.Tensor | None,
    swa_indices: torch.Tensor,
    swa_lens: torch.Tensor,
    swa_ragged_indices: torch.Tensor | None,
    swa_ragged_indptr: torch.Tensor | None,
    topk_ragged_indices: torch.Tensor | None,
    topk_ragged_indptr: torch.Tensor | None,
    attn_sink: torch.Tensor | None,
    scale: float,
    head_dim: int,
    nope_head_dim: int,
    rope_head_dim: int,
    output: torch.Tensor,
    extra_cache_nan_free: bool = False,
    adaptive_splits: bool = False,
) -> None:
    """Decode against plain BF16 SWA and compressed paged caches."""

    del extra_cache_nan_free, adaptive_splits
    if head_dim != nope_head_dim + rope_head_dim:
        raise ValueError(
            f"expected head_dim={nope_head_dim + rope_head_dim}, got {head_dim}"
        )
    if swa_k_cache.dtype != torch.bfloat16 or swa_k_cache.ndim != 3:
        raise TypeError(
            "expected plain BF16 SWA cache [blocks, block_size, head_dim], "
            f"got {swa_k_cache.dtype} {tuple(swa_k_cache.shape)}"
        )
    if swa_k_cache.shape[-1] != head_dim:
        raise ValueError(
            f"expected SWA cache head_dim={head_dim}, got {swa_k_cache.shape[-1]}"
        )

    main_num_rows = swa_k_cache.shape[0] * swa_k_cache.shape[1]
    if swa_ragged_indices is None or swa_ragged_indptr is None:
        swa_ragged_indices, swa_ragged_indptr = _dense_to_ragged(
            swa_indices,
            swa_lens,
            main_num_rows,
        )

    has_extra = not swa_only
    if has_extra:
        if kv_cache is None or kv_cache.dtype != torch.bfloat16 or kv_cache.ndim != 3:
            raise TypeError("plain BF16 sparse decode requires a BF16 extra cache")
        if kv_cache.shape[-1] != head_dim:
            raise ValueError(
                f"expected extra cache head_dim={head_dim}, got {kv_cache.shape[-1]}"
            )
        extra_num_rows = kv_cache.shape[0] * kv_cache.shape[1]
        if topk_ragged_indices is None or topk_ragged_indptr is None:
            if topk_indices is None:
                raise ValueError("missing dense and ragged top-k indices")
            topk_ragged_indices, topk_ragged_indptr = _dense_to_ragged(
                topk_indices,
                topk_lens,
                extra_num_rows,
            )
        extra_cache = kv_cache
        extra_indices = topk_ragged_indices
        extra_indptr = topk_ragged_indptr
    else:
        extra_num_rows = main_num_rows
        extra_cache = swa_k_cache
        extra_indices = swa_ragged_indices
        extra_indptr = swa_ragged_indptr

    block_h = 16
    block_k = 16
    _bf16_sparse_attn_decode_kernel[
        (q.shape[0], triton.cdiv(q.shape[1], block_h))
    ](
        q,
        swa_k_cache,
        swa_ragged_indices,
        swa_ragged_indptr,
        extra_cache,
        extra_indices,
        extra_indptr,
        attn_sink,
        output,
        q.stride(0),
        q.stride(1),
        swa_k_cache.stride(0),
        swa_k_cache.stride(1),
        extra_cache.stride(0),
        extra_cache.stride(1),
        output.stride(0),
        output.stride(1),
        main_num_rows,
        extra_num_rows,
        swa_k_cache.shape[1],
        extra_cache.shape[1],
        scale,
        q.shape[1],
        HAS_EXTRA=has_extra,
        HAS_ATTN_SINK=attn_sink is not None,
        NOPE_DIM=nope_head_dim,
        NOPE_BLOCK=triton.next_power_of_2(nope_head_dim),
        ROPE_DIM=rope_head_dim,
        BLOCK_H=block_h,
        BLOCK_K=block_k,
        num_warps=8,
    )


__all__ = ["bf16_sparse_attn_decode", "gather_bf16_k_cache"]
