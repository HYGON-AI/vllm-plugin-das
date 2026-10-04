# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Plain-BF16 cache writer for DeepSeek V4 compressor outputs."""

from __future__ import annotations

from typing import Any

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _compress_norm_rope_store_bf16_kernel(
    state_cache_ptr,
    state_cache_stride0,
    state_cache_stride1,
    token_to_req_indices_ptr,
    positions_ptr,
    slot_mapping_ptr,
    block_table_ptr,
    block_table_stride,
    block_size,
    rms_norm_weight_ptr,
    rms_norm_eps,
    cos_sin_cache_ptr,
    cos_sin_stride,
    k_cache_ptr,
    k_cache_stride0,
    k_cache_stride1,
    kv_slot_mapping_ptr,
    kv_cache_block_size,
    HEAD_SIZE: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
    STATE_WIDTH: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    OVERLAP: tl.constexpr,
    ROPE_HEAD_DIM: tl.constexpr,
):
    token_idx = tl.program_id(0)

    slot_id = tl.load(slot_mapping_ptr + token_idx)
    if slot_id < 0:
        return
    position = tl.load(positions_ptr + token_idx)
    if (position + 1) % COMPRESS_RATIO != 0:
        return

    req_idx = tl.load(token_to_req_indices_ptr + token_idx)
    start = position - (1 + OVERLAP) * COMPRESS_RATIO + 1
    tokens = tl.arange(0, (1 + OVERLAP) * COMPRESS_RATIO)
    positions = start + tokens
    valid_positions = positions >= 0
    block_numbers = tl.load(
        block_table_ptr
        + req_idx * block_table_stride
        + positions // block_size,
        mask=valid_positions,
        other=0,
    ).to(tl.int64)
    block_offsets = positions % block_size
    head_offset = (tokens >= COMPRESS_RATIO).to(tl.int32) * HEAD_SIZE

    dims = tl.arange(0, TRITON_BLOCK_SIZE)
    dim_mask = dims < HEAD_SIZE
    row_base = (
        state_cache_ptr
        + block_numbers * state_cache_stride0
        + block_offsets * state_cache_stride1
        + head_offset
    )
    load_mask = valid_positions[:, None] & dim_mask[None, :]
    scores = tl.load(
        row_base[:, None] + STATE_WIDTH + dims[None, :],
        mask=load_mask,
        other=float("-inf"),
    )
    scores = tl.softmax(scores, dim=0)
    values = tl.load(
        row_base[:, None] + dims[None, :],
        mask=load_mask,
        other=0.0,
    )
    compressed = tl.sum(values * scores, axis=0)

    norm_weight = tl.load(
        rms_norm_weight_ptr + dims, mask=dim_mask, other=0.0
    )
    variance = tl.sum(compressed * compressed, axis=0) / HEAD_SIZE
    normed = compressed * tl.rsqrt(variance + rms_norm_eps) * norm_weight

    nope_head_dim: tl.constexpr = HEAD_SIZE - ROPE_HEAD_DIM
    half_rope: tl.constexpr = ROPE_HEAD_DIM // 2
    num_pairs: tl.constexpr = TRITON_BLOCK_SIZE // 2
    nope_pairs: tl.constexpr = nope_head_dim // 2
    even, odd = tl.split(tl.reshape(normed, (num_pairs, 2)))
    pair_idx = tl.arange(0, num_pairs)
    rope_pair_idx = pair_idx - nope_pairs
    is_rope = rope_pair_idx >= 0
    cos_sin_idx = tl.maximum(rope_pair_idx, 0)
    compressed_position = (position // COMPRESS_RATIO) * COMPRESS_RATIO
    cos_sin = cos_sin_cache_ptr + compressed_position * cos_sin_stride
    cos_value = tl.load(cos_sin + cos_sin_idx, mask=is_rope, other=1.0)
    sin_value = tl.load(
        cos_sin + half_rope + cos_sin_idx, mask=is_rope, other=0.0
    )
    result = tl.interleave(
        even * cos_value - odd * sin_value,
        odd * cos_value + even * sin_value,
    )

    kv_slot = tl.load(kv_slot_mapping_ptr + token_idx)
    if kv_slot < 0:
        return
    kv_block = kv_slot // kv_cache_block_size
    kv_offset = kv_slot % kv_cache_block_size
    output = (
        k_cache_ptr
        + kv_block.to(tl.int64) * k_cache_stride0
        + kv_offset * k_cache_stride1
    )
    tl.store(output + dims, result.to(tl.bfloat16), mask=dim_mask)


def compress_norm_rope_store_bf16(
    *,
    state_cache: torch.Tensor,
    num_actual: int,
    token_to_req_indices: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    state_width: int,
    cos_sin_cache: torch.Tensor,
    kv_cache: torch.Tensor,
    k_cache_metadata: Any,
    rms_norm_weight: torch.Tensor,
    rms_norm_eps: float,
    head_dim: int,
    rope_head_dim: int,
    compress_ratio: int,
    overlap: bool,
    **_: object,
) -> None:
    """Compress, normalize, rotate, and store one plain BF16 row."""

    if kv_cache.dtype != torch.bfloat16 or kv_cache.ndim != 3:
        raise TypeError(
            "DeepSeek V4 BF16 compressor requires "
            "[blocks, block_size, head_dim] BF16 cache storage"
        )
    if kv_cache.shape[-1] != head_dim:
        raise ValueError(
            f"expected BF16 cache head_dim={head_dim}, got {kv_cache.shape[-1]}"
        )
    _compress_norm_rope_store_bf16_kernel[(num_actual,)](
        state_cache,
        state_cache.stride(0),
        state_cache.stride(1),
        token_to_req_indices,
        positions,
        slot_mapping,
        block_table,
        block_table.stride(0),
        block_size,
        rms_norm_weight,
        rms_norm_eps,
        cos_sin_cache,
        cos_sin_cache.stride(0),
        kv_cache,
        kv_cache.stride(0),
        kv_cache.stride(1),
        k_cache_metadata.slot_mapping,
        kv_cache.shape[1],
        HEAD_SIZE=head_dim,
        TRITON_BLOCK_SIZE=triton.next_power_of_2(head_dim),
        STATE_WIDTH=state_width,
        COMPRESS_RATIO=compress_ratio,
        OVERLAP=overlap,
        ROPE_HEAD_DIM=rope_head_dim,
        num_warps=4 if head_dim == 512 else 1,
    )


__all__ = ["compress_norm_rope_store_bf16"]
