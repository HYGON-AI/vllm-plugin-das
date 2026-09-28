# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gather and dequantize ``fp8_ds_mla`` KV cache for the sparse kernel.

The fp8 FlashMLA sparse kernel hardcodes DeepSeek's ``fp8_ds_mla`` geometry, so
opting out of it (via ``VLLM_HCU_HYV4_FP8_KV_DEQUANT``) means the BF16 sparse
kernel must run instead, which needs a BF16 KV cache. This module prefers
LightOp's fused gather/up-convert op and retains the Triton implementation as a
fallback when the installed LightOp does not expose that op.

The ``fp8_ds_mla`` per-token layout is 656 bytes, written by
``concat_and_cache_ds_mla_kernel`` (csrc/hcu_cache_kernel.cu):

    bytes [0,   512):  512 x fp8_e4m3   NoPE, 4 tiles of 128, one fp32 scale each
    bytes [512, 528):    4 x fp32       per-tile NoPE scales (tile_scale=amax/448)
    bytes [528, 656):   64 x bf16       RoPE, copied verbatim

Dequant is ``nope[i] = e4m3->f32(fp8[i]) * scale[i // 128]``; RoPE is copied. The
BF16 output row is ``[NoPE(512) || RoPE(64)]`` = 576, matching the layout the fp8
*prefill* path produces and what ``flash_mla_sparse_fwd`` consumes (``d_v=512``).
"""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from vllm.logger import init_logger

logger = init_logger(__name__)

# fp8_ds_mla layout constants (see module docstring / concat_and_cache_ds_mla).
_ROW_BYTES = 656
_NOPE_DIM = 512
_ROPE_DIM = 64
_TILE = 128
_N_TILES = _NOPE_DIM // _TILE  # 4
_SCALE_BYTE_OFF = _NOPE_DIM  # 512
_ROPE_BYTE_OFF = _NOPE_DIM + _N_TILES * 4  # 528

_LIGHTOP_GATHER = None
_LIGHTOP_GATHER_RESOLVED = False


@dataclass
class LightOpKVReuseState:
    """Compact-index mapping reused within sparse IndexShare groups."""

    compact_indices: torch.Tensor
    dedup_key: tuple[int, int, int] | None = None
    supports_mapping_reuse: bool | None = None

    @classmethod
    def from_topk_buffer(
        cls,
        topk_indices_buffer: torch.Tensor,
    ) -> "LightOpKVReuseState":
        if topk_indices_buffer.dtype != torch.int32 or topk_indices_buffer.ndim != 2:
            raise ValueError(
                "Sparse TopK indices must be a two-dimensional int32 tensor; "
                f"got {tuple(topk_indices_buffer.shape)}, "
                f"{topk_indices_buffer.dtype}."
            )
        return cls(compact_indices=torch.empty_like(topk_indices_buffer))


def _resolve_lightop_gather():
    """Resolve the single raw LightOp op not yet in a public category."""
    global _LIGHTOP_GATHER, _LIGHTOP_GATHER_RESOLVED
    if not _LIGHTOP_GATHER_RESOLVED:
        try:
            from lightop.op import decode_gather_and_up_convert_with_indices
        except (AttributeError, ImportError, OSError):
            _LIGHTOP_GATHER = None
        else:
            _LIGHTOP_GATHER = decode_gather_and_up_convert_with_indices
        _LIGHTOP_GATHER_RESOLVED = True
    return _LIGHTOP_GATHER


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
    reuse_state: LightOpKVReuseState | None = None,
    allow_mapping_reuse: bool = False,
    mapping_reuse_group_size: int = 1,
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
        kv_lora_rank: NoPE dim (512 for ``fp8_ds_mla``).
        qk_rope_head_dim: RoPE dim (64 for ``fp8_ds_mla``).
        tokens_per_request: Query rows contributed by each request, including
            the current decode token. MTP3 therefore passes 4; ordinary decode
            passes 1. LightOp uses this for request-local KV deduplication.
        reuse_state: Shared compact-index storage for an IndexShare group.
        allow_mapping_reuse: Reuse ``reuse_state``'s mapping when its runtime
            shape matches. Full indexer producers pass false; following shared
            layers pass true.
        mapping_reuse_group_size: Target-verify width eligible for mapping
            reuse. Ordinary gathers and DCP pass 1, which also clears any
            mapping left by a previous target-verify group.

    Returns:
        ``(kv_bf16, new_indices)`` where ``kv_bf16`` is BF16
        ``(num_tokens * topk, kv_lora_rank + qk_rope_head_dim)`` laid out so
        token ``t``'s ``k``-th selected slot is row ``t * topk + k``, and
        ``new_indices`` is ``(num_tokens, topk)`` addressing those rows (-1
        preserved for unfilled entries), ready for ``flash_mla_sparse_fwd``.
    """
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
    idx32 = topk_indices.to(torch.int32).contiguous()
    out = torch.empty(
        (num_tokens, topk, out_dim),
        dtype=torch.bfloat16,
        device=kv_cache.device,
    )
    if reuse_state is None:
        compact_indices = torch.empty(
            (num_tokens, topk), dtype=torch.int32, device=kv_cache.device
        )
    else:
        shared_indices = reuse_state.compact_indices
        if (
            shared_indices.dtype != torch.int32
            or shared_indices.device != kv_cache.device
            or shared_indices.ndim != 2
            or shared_indices.shape[0] < num_tokens
            or shared_indices.shape[1] != topk
        ):
            raise ValueError(
                "LightOp KV reuse buffer must be int32 on the KV-cache device "
                f"with shape at least ({num_tokens}, {topk}); got "
                f"{tuple(shared_indices.shape)}, {shared_indices.dtype}, "
                f"{shared_indices.device}."
            )
        compact_indices = shared_indices[:num_tokens]

    lightop_gather = _resolve_lightop_gather()
    if lightop_gather is not None:
        # LightOp caps validity on the topk axis at valid_lengths[q]. The
        # indices already use -1 for invalid slots, so a full-width cap keeps
        # that mask authoritative and matches the Triton fallback bit-for-bit.
        valid_lengths = torch.full(
            (num_tokens,), topk, dtype=torch.int32, device=kv_cache.device
        )
        dedup_key = (num_tokens, cache_u8.numel(), tokens_per_request)
        reuse_eligible = bool(
            reuse_state is not None
            and mapping_reuse_group_size > 1
            and tokens_per_request == mapping_reuse_group_size
            and reuse_state.supports_mapping_reuse is not False
        )
        reuse_compact_indices = bool(
            reuse_eligible
            and allow_mapping_reuse
            and reuse_state.dedup_key == dedup_key
        )
        lightop_args = (
            cache_u8,
            idx32,
            out,
            valid_lengths,
            compact_indices,
            tokens_per_request,
        )
        if not reuse_compact_indices:
            lightop_gather(*lightop_args)
        else:
            try:
                lightop_gather(
                    *lightop_args,
                    reuse_compact_indices=True,
                )
            except TypeError:
                # Older LightOp builds expose only the six positional
                # arguments. Rebuild the mapping for this layer and remember
                # the ABI result so later layers keep using the compatible
                # call shape.
                assert reuse_state is not None
                reuse_state.supports_mapping_reuse = False
                lightop_gather(*lightop_args)
                reuse_compact_indices = False
                logger.warning_once(
                    "Installed LightOp does not support compact-index mapping "
                    "reuse; using the compatible gather path."
                )
        if reuse_state is not None:
            if reuse_compact_indices:
                reuse_state.supports_mapping_reuse = True
            reuse_state.dedup_key = (
                dedup_key
                if reuse_eligible
                and reuse_state.supports_mapping_reuse is not False
                else None
            )
        logger.info_once(
            "Using LightOp decode_gather_and_up_convert_with_indices "
            f"(tokens_per_request={tokens_per_request}, "
            f"reuse_compact_indices={reuse_compact_indices})."
        )
        return out.reshape(num_out, out_dim), compact_indices

    if reuse_state is not None:
        reuse_state.dedup_key = None
    logger.warning_once(
        "LightOp decode_gather_and_up_convert_with_indices is unavailable; "
        "using the Triton FP8 KV gather/dequant fallback."
    )
    idx_flat = idx32.reshape(-1)
    out_flat = out.reshape(num_out, out_dim)

    _gather_dequantize_fp8_ds_mla_kernel[(num_out,)](
        cache_u8,
        idx_flat,
        out_flat,
        cache_u8.stride(0),
        out_flat.stride(0),
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
        num_out, device=kv_cache.device, dtype=torch.int32
    ).view(num_tokens, topk)
    compact_indices.copy_(
        torch.where(idx32 >= 0, base, base.new_full((), -1))
    )
    return out_flat, compact_indices
