# SPDX-License-Identifier: Apache-2.0
"""Flash 656B writer composition; Core retains cache allocation.

The public LightOp writer does not mask negative locations. Quantize into
row-local scratch with valid locations, then scatter only real Core slots.
"""
import torch
import triton
import triton.language as tl


def _validate_inputs(k_nope, k_rope, cache, slots):
    n = slots.shape[0] if slots.ndim == 1 else -1
    if (k_nope.dtype != torch.bfloat16 or k_rope.dtype != torch.bfloat16
            or cache.dtype != torch.uint8 or slots.dtype != torch.int64):
        raise TypeError('Flash fp8_ds_mla needs BF16 NoPE/RoPE, UINT8 cache, INT64 slots')
    nope_shape = (k_nope.ndim == 2 and k_nope.shape[1:] == (512,)) or (
        k_nope.ndim == 3 and k_nope.shape[1:] == (1, 512))
    rope_shape = (k_rope.ndim == 2 and k_rope.shape[1:] == (64,)) or (
        k_rope.ndim == 3 and k_rope.shape[1:] == (1, 64))
    if (n < 0 or not nope_shape or not rope_shape
            or k_nope.shape[0] < n or k_rope.shape[0] < n
            or cache.ndim != 3 or cache.shape[1] <= 0 or cache.shape[2] != 656
            or cache.stride(2) != 1 or cache.stride(1) < 656
            or cache.stride(0) < cache.shape[1] * cache.stride(1)
            or not slots.is_contiguous()):
        raise ValueError('Unsupported Flash 656B KV-cache geometry/strides')
    if (not k_nope.is_cuda or any(t.device != k_nope.device
                                  for t in (k_rope, cache, slots))):
        raise ValueError('Flash KV tensors must share one HCU device')
    return n


@triton.jit
def _scatter_656(Scratch, Slots, Cache,
                 PAGE_SIZE: tl.constexpr, PAGE_STRIDE: tl.constexpr,
                 TOKEN_STRIDE: tl.constexpr, CAPACITY: tl.constexpr,
                 BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, BLOCK)
    slot = tl.load(Slots + row).to(tl.int64)
    valid = (slot >= 0) & (slot < CAPACITY)
    safe_slot = tl.maximum(slot, 0)
    offset = (safe_slot // PAGE_SIZE) * PAGE_STRIDE + (safe_slot % PAGE_SIZE) * TOKEN_STRIDE
    byte = tl.load(Scratch + row * 656 + col, col < 656, other=0)
    tl.store(Cache + offset + col, byte, valid & (col < 656))


def write_fp8_ds_mla_cache(k_nope, k_rope, cache, slots, *, eps=1e-6):
    """Write FN512 + four FP32 scales + BF16 RoPE64 to original paged KV.

    The first len(slots) input rows are consumed; a longer source prefix is valid.
    Negative slots never write. Padding rows are still quantized into safe
    private scratch so no device-to-host filter or dynamic-size allocation
    is needed on replay. Caller owns positive-slot validity and uniqueness.
    Positive slots outside allocated capacity are also masked to prevent OOB.
    Errors from the real native writer propagate; no silent fallback exists.
    """
    n = _validate_inputs(k_nope, k_rope, cache, slots)
    if n == 0:
        return
    if eps <= 0:
        raise ValueError('eps must be positive')
    from lightop.kvcache import fused_quantize_and_store_mla_kv_cache
    from hcu_adapter._base import Missing
    if (isinstance(fused_quantize_and_store_mla_kv_cache, Missing)
            or not callable(fused_quantize_and_store_mla_kv_cache)):
        raise RuntimeError('Public LightOp 656B FP8 MLA writer is unavailable')
    scratch = torch.empty((n, 1, 656), dtype=torch.uint8, device=k_nope.device)
    local_slots = torch.arange(n, dtype=torch.int64, device=k_nope.device)
    fused_quantize_and_store_mla_kv_cache(
        k_nope[:n], k_rope[:n], scratch, local_slots, 'fp8_e4m3', eps)
    _scatter_656[(n,)](
        scratch, slots, cache, cache.shape[1], cache.stride(0), cache.stride(1),
        cache.shape[0] * cache.shape[1], 1024, num_warps=4)
