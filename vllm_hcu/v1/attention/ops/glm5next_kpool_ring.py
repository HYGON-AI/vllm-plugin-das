# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Backport the GLM-5.3 speculative kpool tail-ring correction.

The target vLLM predates upstream commit 2617fe9383. A one-pool physical
ring lets speculative tokens overwrite an incomplete accepted pool before a
rejected pool-completing draft is replayed. Keep the target call ABI while
addressing the tail by an independently sized physical ring.
"""

from __future__ import annotations

import inspect
from types import ModuleType

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

INDEX_HEAD_DIM = 128
FP8_DTYPE = current_platform.fp8_dtype()
FP8_MAX = torch.finfo(FP8_DTYPE).max

_MARKER = "_vllm_hcu_speculative_kpool_ring_applied"
_ORIGINAL_SEED = "_vllm_hcu_original_kpool_seed_tail_cache"
_ORIGINAL_DECODE = (
    "_vllm_hcu_original_kpool_decode_update_and_maybe_write_cache_batched"
)


@triton.jit
def _cache_k_offset(
    token_offset,
    dim_offset,
    head_dim: tl.constexpr,
    preshuffle: tl.constexpr,
):
    if preshuffle:
        return (
            (token_offset // 16) * 16 * head_dim
            + (dim_offset // 16) * 16 * 16
            + (token_offset % 16) * 16
            + dim_offset % 16
        )
    return token_offset * head_dim + dim_offset


@triton.jit
def _hadamard128_stage(x, groups: tl.constexpr, stride: tl.constexpr):
    x3 = tl.reshape(x, (groups, 2, stride))
    x3 = tl.trans(x3, 0, 2, 1)
    a, b = tl.split(x3)
    x3 = tl.join(a + b, a - b)
    x3 = tl.trans(x3, 0, 2, 1)
    return tl.reshape(x3, (128,))


@triton.jit
def _hadamard128(x):
    x = _hadamard128_stage(x, 64, 1)
    x = _hadamard128_stage(x, 32, 2)
    x = _hadamard128_stage(x, 16, 4)
    x = _hadamard128_stage(x, 8, 8)
    x = _hadamard128_stage(x, 4, 16)
    x = _hadamard128_stage(x, 2, 32)
    x = _hadamard128_stage(x, 1, 64)
    return x * 0.08838834764831845


@triton.jit
def _kpool_tail_seed_ring_kernel(
    key_ptr,
    score_ptr,
    tslot_ptr,
    tail_ptr,
    n_tokens,
    TAIL_BLOCK_ELEMS: tl.constexpr,
    KPOOL_HEAD: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    KPOOL: tl.constexpr,
    RING: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    i = tl.program_id(0)
    t = tl.load(tslot_ptr + i).to(tl.int64)
    if t < 0:
        return
    block = t // RING
    ahead = tl.load(
        tslot_ptr + i + KPOOL,
        mask=i + KPOOL < n_tokens,
        other=-1,
    ).to(tl.int64)
    if ahead >= 0 and ahead // RING == block:
        return
    offsets = tl.arange(0, BLOCK_D)
    mask = offsets < HEAD_DIM
    block_base = block * TAIL_BLOCK_ELEMS
    base = block_base + (t % RING) * HEAD_DIM
    key = tl.load(key_ptr + i * HEAD_DIM + offsets, mask=mask)
    score = tl.load(score_ptr + i * HEAD_DIM + offsets, mask=mask)
    tl.store(tail_ptr + base + offsets, key, mask=mask)
    tl.store(
        tail_ptr + block_base + KPOOL_HEAD + (t % RING) * HEAD_DIM + offsets,
        score,
        mask=mask,
    )


def kpool_seed_tail_cache(
    tail_kv_cache: torch.Tensor,
    key: torch.Tensor,
    gate_score: torch.Tensor,
    tslot: torch.Tensor,
    kpool: int,
    head_dim: int = INDEX_HEAD_DIM,
) -> None:
    assert tail_kv_cache.dtype == torch.bfloat16
    assert key.dtype == torch.bfloat16
    ring = tail_kv_cache.shape[2]
    assert ring >= kpool and ring % kpool == 0, (ring, kpool)
    num_tokens = tslot.shape[0]
    if num_tokens == 0:
        return
    _kpool_tail_seed_ring_kernel[(num_tokens,)](
        key,
        gate_score,
        tslot,
        tail_kv_cache,
        num_tokens,
        TAIL_BLOCK_ELEMS=tail_kv_cache.stride(0),
        KPOOL_HEAD=tail_kv_cache.stride(1),
        HEAD_DIM=head_dim,
        KPOOL=kpool,
        RING=ring,
        BLOCK_D=triton.next_power_of_2(head_dim),
    )


@triton.jit
def _kpool_decode_update_batched_ring_kernel(
    buf_fp8_ptr,
    buf_fp32_ptr,
    tail_kv_ptr,
    tail_slot_mapping_ptr,
    key_ptr,
    key_stride_b,
    key_stride_t,
    slot_score_ptr,
    ss_stride_b,
    ss_stride_t,
    ape_ptr,
    ape_stride_0,
    slot_mapping_ptr,
    positions_ptr,
    NEXT_N,
    PAGE_SIZE: tl.constexpr,
    BUF_NUMEL_PER_PAGE: tl.constexpr,
    POOL_SIZE: tl.constexpr,
    RING: tl.constexpr,
    TAIL_BLOCK_ELEMS: tl.constexpr,
    KPOOL_HEAD: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    S_OFFSET_NBYTES_IN_PAGE: tl.constexpr,
    FP8_MAX_VALUE: tl.constexpr,
    PRESHUFFLE: tl.constexpr,
    ROUND_SCALE: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    req = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_D)
    dim_mask = offsets < HEAD_DIM

    for token_index in tl.range(0, NEXT_N):
        index = req * NEXT_N + token_index
        cache_loc = tl.load(slot_mapping_ptr + index)
        position = tl.load(positions_ptr + index)
        safe_position = tl.maximum(position, 0)
        position_valid = (cache_loc >= 0) & (position >= 0)

        slot = safe_position % POOL_SIZE
        physical_slot = safe_position % RING
        tail_slot = tl.load(tail_slot_mapping_ptr + index)
        block = tl.maximum(tail_slot, 0).to(tl.int64) // RING
        block_base = block * TAIL_BLOCK_ELEMS
        stash_valid = (position >= 0) & (tail_slot >= 0)

        key = tl.load(
            key_ptr
            + req * key_stride_b
            + token_index * key_stride_t
            + offsets,
            mask=dim_mask,
            other=0.0,
        ).to(tl.float32)
        score_current = tl.load(
            slot_score_ptr
            + req * ss_stride_b
            + token_index * ss_stride_t
            + offsets,
            mask=dim_mask,
            other=0.0,
        ).to(tl.float32)

        if position_valid & (slot == POOL_SIZE - 1):
            pool_logical_start = safe_position - slot
            max_score = tl.full((BLOCK_D,), -float("inf"), tl.float32)
            for pool_slot in tl.static_range(0, POOL_SIZE):
                is_current = pool_slot == slot
                physical = (pool_logical_start + pool_slot) % RING
                score_buffer = tl.load(
                    tail_kv_ptr
                    + block_base
                    + KPOOL_HEAD
                    + physical * HEAD_DIM
                    + offsets,
                    mask=dim_mask,
                    other=0.0,
                ).to(tl.float32)
                score = tl.where(is_current, score_current, score_buffer)
                score += tl.load(
                    ape_ptr + pool_slot * ape_stride_0 + offsets,
                    mask=dim_mask,
                    other=0.0,
                ).to(tl.float32)
                max_score = tl.maximum(max_score, score)

            accumulator = tl.full((BLOCK_D,), 0.0, tl.float32)
            denominator = tl.full((BLOCK_D,), 0.0, tl.float32)
            for pool_slot in tl.static_range(0, POOL_SIZE):
                is_current = pool_slot == slot
                physical = (pool_logical_start + pool_slot) % RING
                score_buffer = tl.load(
                    tail_kv_ptr
                    + block_base
                    + KPOOL_HEAD
                    + physical * HEAD_DIM
                    + offsets,
                    mask=dim_mask,
                    other=0.0,
                ).to(tl.float32)
                score = tl.where(is_current, score_current, score_buffer)
                score += tl.load(
                    ape_ptr + pool_slot * ape_stride_0 + offsets,
                    mask=dim_mask,
                    other=0.0,
                ).to(tl.float32)
                probability = tl.exp(score - max_score)
                denominator += probability
                key_buffer = tl.load(
                    tail_kv_ptr
                    + block_base
                    + physical * HEAD_DIM
                    + offsets,
                    mask=dim_mask,
                    other=0.0,
                ).to(tl.float32)
                selected_key = tl.where(is_current, key, key_buffer)
                accumulator += selected_key * probability

            compressed = (accumulator / denominator).to(tl.bfloat16).to(tl.float32)
            compressed = _hadamard128(compressed).to(tl.bfloat16).to(tl.float32)

            fp8_max_inverse = 1.0 / FP8_MAX_VALUE
            absmax = tl.maximum(tl.max(tl.abs(compressed), axis=0), 1e-4)
            if ROUND_SCALE:
                scale = tl.exp2(tl.ceil(tl.log2(absmax * fp8_max_inverse)))
            else:
                scale = absmax * fp8_max_inverse
            quantized = tl.minimum(
                tl.maximum(compressed / scale, -FP8_MAX_VALUE),
                FP8_MAX_VALUE,
            )

            location = cache_loc.to(tl.int64)
            location_page = location // PAGE_SIZE
            location_offset = location % PAGE_SIZE
            output_offsets = location_page * BUF_NUMEL_PER_PAGE + _cache_k_offset(
                location_offset,
                offsets,
                HEAD_DIM,
                PRESHUFFLE,
            )
            scale_offset = (
                location_page * BUF_NUMEL_PER_PAGE // 4
                + S_OFFSET_NBYTES_IN_PAGE // 4
                + location_offset
            )
            tl.store(buf_fp8_ptr + output_offsets, quantized, mask=dim_mask)
            tl.store(buf_fp32_ptr + scale_offset, scale)

        update_mask = dim_mask & stash_valid
        tl.store(
            tail_kv_ptr + block_base + physical_slot * HEAD_DIM + offsets,
            key,
            mask=update_mask,
        )
        tl.store(
            tail_kv_ptr
            + block_base
            + KPOOL_HEAD
            + physical_slot * HEAD_DIM
            + offsets,
            score_current,
            mask=update_mask,
        )


def kpool_decode_update_and_maybe_write_cache_batched(
    kv_cache: torch.Tensor,
    tail_kv_cache: torch.Tensor,
    tail_slot_mapping: torch.Tensor,
    key: torch.Tensor,
    slot_score: torch.Tensor,
    ape: torch.Tensor,
    slot_mapping: torch.Tensor,
    positions: torch.Tensor,
    pool_size: int,
    head_dim: int = INDEX_HEAD_DIM,
    round_scale: bool = True,
) -> None:
    num_requests, next_n = key.shape[0], key.shape[1]
    if num_requests == 0 or next_n == 0:
        return
    assert tail_kv_cache.ndim == 4
    assert tail_kv_cache.shape[1] == 2
    ring = tail_kv_cache.shape[2]
    assert ring >= pool_size and ring % pool_size == 0, (ring, pool_size)
    assert tail_kv_cache.shape[3] == head_dim
    assert tail_kv_cache.dtype == torch.bfloat16
    assert key.ndim == 3 and key.shape[2] == head_dim
    assert slot_score.shape == key.shape
    assert ape.shape == (pool_size, head_dim)
    assert tail_slot_mapping.shape == (num_requests, next_n)
    assert slot_mapping.shape == (num_requests, next_n)
    assert positions.shape == (num_requests, next_n)
    assert key.dtype == torch.bfloat16
    assert slot_score.dtype == torch.bfloat16
    assert ape.dtype == torch.float32
    assert kv_cache.dtype == torch.uint8

    page_size = kv_cache.shape[1]
    buffer_fp8 = kv_cache.view(FP8_DTYPE)
    buffer_fp32 = kv_cache.view(torch.float32)
    tail_slot_mapping = tail_slot_mapping.contiguous()
    slot_mapping = slot_mapping.contiguous()
    positions = positions.contiguous()
    if page_size > 1:
        assert page_size % 16 == 0, "ROCm preshuffle requires 16-token tiles"

    _kpool_decode_update_batched_ring_kernel[(num_requests,)](
        buffer_fp8,
        buffer_fp32,
        tail_kv_cache,
        tail_slot_mapping,
        key,
        key.stride(0),
        key.stride(1),
        slot_score,
        slot_score.stride(0),
        slot_score.stride(1),
        ape,
        ape.stride(0),
        slot_mapping,
        positions,
        next_n,
        PAGE_SIZE=page_size,
        BUF_NUMEL_PER_PAGE=kv_cache.stride(0),
        POOL_SIZE=pool_size,
        RING=ring,
        TAIL_BLOCK_ELEMS=tail_kv_cache.stride(0),
        KPOOL_HEAD=tail_kv_cache.stride(1),
        HEAD_DIM=head_dim,
        S_OFFSET_NBYTES_IN_PAGE=page_size * head_dim,
        FP8_MAX_VALUE=FP8_MAX,
        PRESHUFFLE=page_size > 1,
        ROUND_SCALE=round_scale,
        BLOCK_D=triton.next_power_of_2(head_dim),
    )


def _require_target_signature(
    module: ModuleType,
    name: str,
    expected: tuple[str, ...],
) -> object:
    target = getattr(module, name, None)
    if not callable(target):
        raise RuntimeError(f"required callable {module.__name__}.{name} is missing")
    actual = tuple(inspect.signature(target).parameters)
    if actual != expected:
        raise RuntimeError(
            f"incompatible {module.__name__}.{name} signature: "
            f"expected {expected}, got {actual}"
        )
    return target


def install_glm5next_kpool_ring(kpool_ops: ModuleType) -> bool:
    """Install the ring-safe helpers into the frozen target kpool module."""

    if getattr(kpool_ops, _MARKER, False):
        if (
            getattr(kpool_ops, "kpool_seed_tail_cache", None)
            is not kpool_seed_tail_cache
            or getattr(
                kpool_ops,
                "kpool_decode_update_and_maybe_write_cache_batched",
                None,
            )
            is not kpool_decode_update_and_maybe_write_cache_batched
        ):
            raise RuntimeError("GLM5Next kpool ring patch marker is stale")
        return False

    original_seed = _require_target_signature(
        kpool_ops,
        "kpool_seed_tail_cache",
        ("tail_kv_cache", "key", "gate_score", "tslot", "kpool", "head_dim"),
    )
    original_decode = _require_target_signature(
        kpool_ops,
        "kpool_decode_update_and_maybe_write_cache_batched",
        (
            "kv_cache",
            "tail_kv_cache",
            "tail_slot_mapping",
            "key",
            "slot_score",
            "ape",
            "slot_mapping",
            "positions",
            "pool_size",
            "head_dim",
            "round_scale",
        ),
    )
    setattr(kpool_ops, _ORIGINAL_SEED, original_seed)
    setattr(kpool_ops, _ORIGINAL_DECODE, original_decode)
    setattr(kpool_ops, "kpool_seed_tail_cache", kpool_seed_tail_cache)
    setattr(
        kpool_ops,
        "kpool_decode_update_and_maybe_write_cache_batched",
        kpool_decode_update_and_maybe_write_cache_batched,
    )
    setattr(kpool_ops, _MARKER, True)
    return True


__all__ = [
    "install_glm5next_kpool_ring",
    "kpool_decode_update_and_maybe_write_cache_batched",
    "kpool_seed_tail_cache",
]
