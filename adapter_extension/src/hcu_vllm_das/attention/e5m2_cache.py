"""HCU Triton writer for OCP E5M2 FlashAttention KV pages."""

from __future__ import annotations

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _reshape_and_cache_e5m2_kernel(
    key_ptr,
    value_ptr,
    key_cache_ptr,
    value_cache_ptr,
    slot_mapping_ptr,
    k_scale_ptr,
    v_scale_ptr,
    key_token_stride: tl.int64,
    key_head_stride: tl.int64,
    value_token_stride: tl.int64,
    value_head_stride: tl.int64,
    cache_block_stride: tl.int64,
    cache_page_stride: tl.int64,
    cache_head_stride: tl.int64,
    block_size: tl.constexpr,
    head_size: tl.constexpr,
    scale_per_head: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    slot_idx = tl.load(slot_mapping_ptr + token_idx).to(tl.int64)
    if slot_idx < 0:
        return

    block_idx = slot_idx // block_size
    block_offset = slot_idx % block_size
    offsets = tl.arange(0, triton.next_power_of_2(head_size))
    mask = offsets < head_size
    key = tl.load(
        key_ptr
        + token_idx * key_token_stride
        + head_idx * key_head_stride
        + offsets,
        mask=mask,
    )
    value = tl.load(
        value_ptr
        + token_idx * value_token_stride
        + head_idx * value_head_stride
        + offsets,
        mask=mask,
    )
    scale_offset = head_idx if scale_per_head else 0
    key = key / tl.load(k_scale_ptr + scale_offset)
    value = value / tl.load(v_scale_ptr + scale_offset)
    cache_offsets = (
        block_idx * cache_block_stride
        + block_offset * cache_page_stride
        + head_idx * cache_head_stride
        + offsets
    )
    tl.store(key_cache_ptr + cache_offsets, key, mask=mask)
    tl.store(value_cache_ptr + cache_offsets, value, mask=mask)


def reshape_and_cache_e5m2(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
) -> None:
    """Quantize K/V to OCP E5M2 and scatter them into dense pages."""
    if key.ndim != 3 or value.ndim != 3:
        raise ValueError("HCU E5M2 cache writer requires 3D K/V tensors")
    if key.shape != value.shape:
        raise ValueError("HCU E5M2 cache writer requires matching K/V shapes")
    if key_cache.ndim != 4 or value_cache.ndim != 4:
        raise ValueError("HCU E5M2 cache writer requires 4D cache views")
    if key_cache.shape != value_cache.shape:
        raise ValueError("HCU E5M2 cache writer requires matching cache views")
    if key_cache.element_size() != 1 or value_cache.element_size() != 1:
        raise ValueError("HCU E5M2 cache writer requires one-byte cache storage")

    num_heads = key.shape[1]
    head_size = key.shape[2]
    if key_cache.shape[2:] != (num_heads, head_size):
        raise ValueError("HCU E5M2 cache writer received incompatible cache shape")
    if k_scale.shape != v_scale.shape or k_scale.numel() not in (1, num_heads):
        raise ValueError("HCU E5M2 K/V scales must be scalar or per-head")

    e5m2_key_cache = key_cache.view(torch.float8_e5m2)
    e5m2_value_cache = value_cache.view(torch.float8_e5m2)
    grid = (slot_mapping.shape[0], num_heads)
    _reshape_and_cache_e5m2_kernel[grid](
        key,
        value,
        e5m2_key_cache,
        e5m2_value_cache,
        slot_mapping,
        k_scale,
        v_scale,
        key.stride(0),
        key.stride(1),
        value.stride(0),
        value.stride(1),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        key_cache.shape[1],
        head_size,
        k_scale.numel() > 1,
        num_warps=4,
        num_stages=4,
    )
