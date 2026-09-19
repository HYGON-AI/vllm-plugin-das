# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dequantize HY V4's ``fp8_ds_mla`` KV cache to BF16 for the sparse kernel.

The fp8 FlashMLA sparse kernel hardcodes DeepSeek's ``fp8_ds_mla`` geometry, so
opting out of it (via ``VLLM_HCU_HYV4_FP8_KV_DEQUANT``) means the BF16 sparse
kernel must run instead, which needs a BF16 KV cache. This module dequantizes
only the ``topk`` slots the sparse kernel actually reads into a compact BF16
buffer, keeping a fixed output shape so the decode path stays CUDA-graph-safe.

The ``fp8_ds_mla`` per-token layout is 656 bytes, written by
``concat_and_cache_ds_mla_kernel`` (csrc/hcu_cache_kernel.cu):

    bytes [0,   512):  512 x fp8_e4m3   NoPE, 4 tiles of 128, one fp32 scale each
    bytes [512, 528):    4 x fp32       per-tile NoPE scales (tile_scale=amax/448)
    bytes [528, 656):   64 x bf16       RoPE, copied verbatim

Dequant is ``nope[i] = e4m3->f32(fp8[i]) * scale[i // 128]``; RoPE is copied. The
BF16 output row is ``[NoPE(512) || RoPE(64)]`` = 576, matching the layout the fp8
*prefill* path produces and what ``flash_mla_sparse_fwd`` consumes (``d_v=512``).
"""

import torch
import triton
import triton.language as tl

# fp8_ds_mla layout constants (see module docstring / concat_and_cache_ds_mla).
_ROW_BYTES = 656
_NOPE_DIM = 512
_ROPE_DIM = 64
_TILE = 128
_N_TILES = _NOPE_DIM // _TILE  # 4
_SCALE_BYTE_OFF = _NOPE_DIM  # 512
_ROPE_BYTE_OFF = _NOPE_DIM + _N_TILES * 4  # 528


@triton.jit
def _gather_dequantize_fp8_ds_mla_kernel(
    cache_ptr,  # uint8 [num_slots, ROW_BYTES]
    idx_ptr,  # int   [num_out_rows]  flattened global slot per output row
    out_ptr,  # bf16  [num_out_rows, OUT_DIM]
    cache_row_stride,
    out_row_stride,
    ROW_BYTES: tl.constexpr,
    OUT_DIM: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    TILE: tl.constexpr,
    N_TILES: tl.constexpr,
    SCALE_BYTE_OFF: tl.constexpr,
    ROPE_BYTE_OFF: tl.constexpr,
):
    row = tl.program_id(0)
    out_row_ptr = out_ptr + row.to(tl.int64) * out_row_stride

    # One output row per (token, topk-slot). A -1 index marks an unfilled topk
    # entry; clamp it to slot 0 for a safe (in-bounds) read, then zero the row.
    # The caller sets the matching sparse index to -1, so the kernel masks it
    # out and this zeroed row is never actually read.
    slot = tl.load(idx_ptr + row)
    valid = slot >= 0
    safe_slot = tl.where(valid, slot, 0)
    row_u8_ptr = cache_ptr + safe_slot.to(tl.int64) * cache_row_stride

    # NoPE: 4 tiles of 128 fp8 values, one fp32 scale per tile.
    scale_ptr = (row_u8_ptr + SCALE_BYTE_OFF).to(tl.pointer_type(tl.float32))
    for t in tl.static_range(N_TILES):
        offs = t * TILE + tl.arange(0, TILE)
        x_u8 = tl.load(row_u8_ptr + offs)
        x_f32 = x_u8.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        scale = tl.load(scale_ptr + t)
        val = (x_f32 * scale).to(tl.bfloat16)
        tl.store(out_row_ptr + offs, tl.where(valid, val, 0.0))

    # RoPE: 64 bf16 values copied verbatim.
    rope_ptr = (row_u8_ptr + ROPE_BYTE_OFF).to(tl.pointer_type(tl.bfloat16))
    ri = tl.arange(0, ROPE_DIM)
    rope_val = tl.load(rope_ptr + ri)
    tl.store(out_row_ptr + NOPE_DIM + ri, tl.where(valid, rope_val, 0.0))


def gather_dequantize_fp8_ds_mla_cache(
    kv_cache: torch.Tensor,
    topk_indices: torch.Tensor,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    tokens_per_request: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dequantize only the topk-selected ``fp8_ds_mla`` slots to a compact BF16.

    Instead of upconverting the whole pool, gather and dequantize just the
    ``num_tokens * topk`` slots the sparse kernel will actually read. Both
    ``topk`` (constant) and ``num_tokens`` (fixed per CUDA-graph batch bucket)
    are static, so the output shape is fixed and the decode path stays
    CUDA-graph-capturable.

    Args:
        kv_cache: ``(num_blocks, block_size, 656)`` cache, ``fp8_ds_mla`` layout.
        topk_indices: ``(num_tokens, topk)`` global cache slots
            (``block * block_size + pos``), with -1 marking unfilled entries.
        kv_lora_rank: NoPE dim (512 for HY V4).
        qk_rope_head_dim: RoPE dim (64 for HY V4).
        tokens_per_request: Retained for compatibility with the fused gather
            implementation. The standalone kernel already treats every query
            row independently.

    Returns:
        ``(kv_bf16, new_indices)`` where ``kv_bf16`` is BF16
        ``(num_tokens * topk, kv_lora_rank + qk_rope_head_dim)`` laid out so
        token ``t``'s ``k``-th selected slot is row ``t * topk + k``, and
        ``new_indices`` is ``(num_tokens, topk)`` addressing those rows (-1
        preserved for unfilled entries), ready for ``flash_mla_sparse_fwd``.
    """
    del tokens_per_request
    assert kv_lora_rank == _NOPE_DIM and qk_rope_head_dim == _ROPE_DIM, (
        f"gather_dequantize_fp8_ds_mla_cache hardcodes the DeepSeek fp8_ds_mla "
        f"geometry (NoPE={_NOPE_DIM}, RoPE={_ROPE_DIM}), got "
        f"kv_lora_rank={kv_lora_rank}, qk_rope_head_dim={qk_rope_head_dim}."
    )
    cache_u8 = kv_cache.view(torch.uint8).reshape(-1, _ROW_BYTES)
    assert cache_u8.shape[1] == _ROW_BYTES, (
        f"expected {_ROW_BYTES}-byte fp8_ds_mla rows, got {cache_u8.shape[1]}."
    )
    num_tokens, topk = topk_indices.shape
    num_out = num_tokens * topk
    out_dim = kv_lora_rank + qk_rope_head_dim
    idx_flat = topk_indices.reshape(-1).contiguous()
    out = torch.empty((num_out, out_dim), dtype=torch.bfloat16, device=kv_cache.device)

    _gather_dequantize_fp8_ds_mla_kernel[(num_out,)](
        cache_u8,
        idx_flat,
        out,
        cache_u8.stride(0),
        out.stride(0),
        ROW_BYTES=_ROW_BYTES,
        OUT_DIM=out_dim,
        NOPE_DIM=_NOPE_DIM,
        ROPE_DIM=_ROPE_DIM,
        TILE=_TILE,
        N_TILES=_N_TILES,
        SCALE_BYTE_OFF=_SCALE_BYTE_OFF,
        ROPE_BYTE_OFF=_ROPE_BYTE_OFF,
    )

    # Compact row ids: token t's k-th slot lives at row t*topk+k, -1 preserved.
    base = torch.arange(
        num_out, device=kv_cache.device, dtype=topk_indices.dtype
    ).view(num_tokens, topk)
    new_indices = torch.where(topk_indices >= 0, base, base.new_full((), -1))
    return out, new_indices
