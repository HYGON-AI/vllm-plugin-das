# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Plain-BF16 paged-cache gather for DeepSeek V4 sparse prefill."""

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


__all__ = ["gather_bf16_k_cache"]
