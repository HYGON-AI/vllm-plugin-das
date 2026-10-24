# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""HCU-owned contracts for the optional DeepSeek indexer FP4 path.

This module deliberately contains no device or native-extension import at
module load time.  It freezes the cache/query ABI before a caller allocates a
cache, and keeps the BF16-Q prefill contract separate from the packed-Q decode
contract exposed by current LightOp.
"""

from dataclasses import dataclass
import os

import torch


FP4_PAGE_BLOCK_SIZE = 64
FP4_HEAD_DIM = 128
FP4_PAYLOAD_BYTES = FP4_HEAD_DIM // 2
FP4_SCALE_BYTES = FP4_HEAD_DIM // 32
FP4_RECORD_BYTES = FP4_PAYLOAD_BYTES + FP4_SCALE_BYTES
FP4_PAGE_BYTES = FP4_PAGE_BLOCK_SIZE * FP4_RECORD_BYTES
FP4_ABI = 3


@dataclass(frozen=True, slots=True)
class Fp4IndexerCapability:
    """Immutable selection made before FP4 cache allocation."""

    enabled: bool
    ratio: int
    reason: str
    cache_format: str = "fp8"


@dataclass(frozen=True, slots=True)
class Fp4IndexerQuery:
    """Both query representations needed by the two LightOp consumers.

    ``bf16`` is retained for non-paged prefill MQA.  ``packed`` and ``scales``
    are used only by paged decode.  Keeping all three in one immutable value
    prevents a caller from silently feeding packed Q to the BF16-Q ABI.
    """

    bf16: torch.Tensor
    packed: torch.Tensor
    scales: torch.Tensor
    weights: torch.Tensor


class Fp4QTuple(tuple):
    """Tuple-compatible packed Q pair retaining the BF16 prefill query."""

    def __new__(cls, packed, scales, bf16, weights=None):
        value = super().__new__(cls, (packed, scales))
        value.packed = packed
        value.scales = scales
        value.bf16 = bf16
        value.weights = weights
        return value


def fp4_indexer_requested() -> bool:
    """Return the opt-in leaf switch under the HCU custom-op master."""
    master = os.environ.get("VLLM_HCU_USE_CUSTOM_OPS", "True").lower()
    leaf = os.environ.get("VLLM_HCU_USE_FP4_INDEXER", "True").lower()
    return master in ("1", "true") and leaf in ("1", "true")


def select_fp4_indexer_capability(
    *,
    requested: bool,
    compress_ratio: int,
    native_available: bool = True,
    shared_source_complete: bool = True,
) -> Fp4IndexerCapability:
    """Freeze the supported domain before selecting an allocation format.

    Phase 1 only admits ratio-1 pages.  Unsupported ratios intentionally
    return the original FP8 route; callers must not allocate FP4 and then
    fall back to an FP8 reader.  ``native_available`` is a static/mock input;
    phase-1 code never probes a device or imports a native extension.
    """
    if not requested:
        return Fp4IndexerCapability(False, compress_ratio, "feature is OFF")
    if compress_ratio == 2:
        return Fp4IndexerCapability(
            False, compress_ratio, "ratio-2 pages require the saved ratio2 design"
        )
    if compress_ratio != 1:
        return Fp4IndexerCapability(
            False,
            compress_ratio,
            "only allocator-compatible ratio-1 pages are admitted in phase 1",
        )
    if not shared_source_complete:
        return Fp4IndexerCapability(
            False, compress_ratio, "shared indexer producer/consumer group is incomplete"
        )
    if not native_available:
        return Fp4IndexerCapability(False, compress_ratio, "FP4 native capability is missing")
    return Fp4IndexerCapability(True, compress_ratio, "ratio-1 FP4 contract selected", "fp4")


def fp4_cache_shape(num_pages: int, block_size: int = FP4_PAGE_BLOCK_SIZE) -> tuple[int, int, int, int]:
    """Return the physical LightOp page shape, without allocating it."""
    if block_size != FP4_PAGE_BLOCK_SIZE:
        raise ValueError("LightOp FP4 indexer requires a 64-record physical page")
    if num_pages < 0:
        raise ValueError("num_pages must be non-negative")
    return num_pages, block_size, 1, FP4_RECORD_BYTES


def fp4_cache_view(cache: torch.Tensor) -> torch.Tensor:
    """Validate/restore ``[pages, 64, 1, 68]`` without changing storage."""
    if cache.dtype != torch.uint8:
        raise TypeError(f"FP4 cache must be uint8, got {cache.dtype}")
    if cache.ndim == 4:
        if tuple(cache.shape[1:]) != (64, 1, 68):
            raise ValueError(f"invalid FP4 cache shape {tuple(cache.shape)}")
        return cache
    if cache.ndim == 3 and cache.shape[1] == 64 and cache.shape[2] == 68:
        return cache.unsqueeze(2)
    if cache.ndim == 2 and cache.shape[1] == FP4_PAGE_BYTES:
        return cache.view(cache.shape[0], 64, 1, 68)
    raise ValueError(f"invalid FP4 cache shape {tuple(cache.shape)}")


def fp4_chunk_slots(
    block_table: torch.Tensor,
    token_to_seq: torch.Tensor,
    cu_seq_lens: torch.Tensor,
    total: int,
    block_size: int = 64,
) -> torch.Tensor:
    """Map packed request-local positions to allocator-owned physical slots."""
    positions = torch.arange(total, device=block_table.device, dtype=torch.long)
    req = token_to_seq[:total].to(torch.long)
    local_positions = positions - cu_seq_lens[req].to(torch.long)
    logical_block = torch.div(local_positions, block_size, rounding_mode="floor")
    offset = torch.remainder(local_positions, block_size)
    return block_table[req, logical_block].to(torch.long) * block_size + offset


def fp4_flat_page_view(cache: torch.Tensor) -> torch.Tensor:
    """Expose the two physical byte planes, preserving padded page strides."""
    page_size = fp4_physical_page_size(cache)
    return torch.as_strided(
        cache, size=(cache.shape[0], page_size * FP4_RECORD_BYTES),
        stride=(cache.stride(0), 1),
    )


def fp4_physical_page_size(cache: torch.Tensor) -> int:
    """Validate allocator geometry without pretending 32/128 rows are 64."""
    if cache.dtype != torch.uint8 or cache.stride(-1) != 1:
        raise ValueError("FP4 physical pages must contain contiguous uint8 bytes")
    if cache.ndim == 4 and tuple(cache.shape[2:]) == (1, 68):
        rows = cache.shape[1]
        if cache.stride(1) != 68:
            raise ValueError("FP4 physical rows must have nominal stride 68")
    elif cache.ndim == 3 and cache.shape[2] == 68:
        rows = cache.shape[1]
        if cache.stride(1) != 68:
            raise ValueError("FP4 physical rows must have nominal stride 68")
    elif cache.ndim == 2 and cache.shape[1] % 68 == 0:
        rows = cache.shape[1] // 68
    else:
        raise ValueError(f"invalid FP4 physical cache shape {tuple(cache.shape)}")
    if rows not in (32, 64, 128) or cache.stride(0) < rows * 68:
        raise ValueError(f"unsupported FP4 physical page geometry: rows={rows}")
    return rows


def stage_fp4_decode_pages(cache, block_table, row_ends, max_context_len):
    """Byte-copy non-64-row allocator pages into a transient native ABI pool.

    No scale conversion, dequantization, or second quantization occurs. The
    allocator cache is read-only. Capacity is static (rows * rounded maximum
    context), so this does not use .item()/device-to-host length reads. This is
    a correctness-first staging route, not a zero-copy performance claim.
    """
    page_size = fp4_physical_page_size(cache)
    if page_size == 64:
        return fp4_cache_view(cache), block_table
    rows = row_ends.numel()
    pages_per_row = (max_context_len + 63) // 64
    width = pages_per_row * 64
    offsets = torch.arange(width, device=cache.device, dtype=torch.int64)
    # Avoid reading unspecified block-table tail entries for masked columns.
    valid = offsets[None, :] < row_ends.reshape(-1, 1)
    logical_pages = torch.div(offsets, page_size, rounding_mode="floor")
    logical_pages = logical_pages.clamp_max(block_table.shape[1] - 1)
    physical_pages = block_table[:, logical_pages].to(torch.int64)
    slots = physical_pages * page_size + offsets.remainder(page_size)
    # Slot 0 is just an in-bounds read for masked columns, never a reserved
    # sentinel or a write target. The native scorer masks those columns.
    slots = torch.where(valid, slots, torch.zeros_like(slots)).reshape(-1).contiguous()
    payload, scales = lightop_gather_index_k_fp4(fp4_flat_page_view(cache), slots, page_size)
    pool = torch.empty((rows * pages_per_row, FP4_PAGE_BYTES),
                       dtype=torch.uint8, device=cache.device)
    pool[:, :4096].copy_(payload.view(torch.uint8).reshape(-1, 4096))
    pool[:, 4096:].copy_(scales.contiguous().view(torch.uint8).reshape(-1, 256))
    table = torch.arange(rows * pages_per_row, device=cache.device,
                         dtype=torch.int32).reshape(rows, pages_per_row)
    return fp4_cache_view(pool), table


def _e2m1_code(value: torch.Tensor) -> torch.Tensor:
    """Quantize normalized values to the OCP E2M1 magnitude code."""
    levels = torch.tensor(
        (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0),
        device=value.device,
        dtype=torch.float32,
    )
    magnitude = value.abs().unsqueeze(-1)
    code = (magnitude - levels).abs().argmin(-1).to(torch.uint8)
    return code | (value < 0).to(torch.uint8) * 8


def prepare_fp4_query(
    q_bf16: torch.Tensor,
    weights: torch.Tensor,
    *,
    softmax_scale: float,
    head_scale: float,
) -> Fp4IndexerQuery:
    """CPU/mock-compatible packed-Q preparation for the audited ABI.

    The production HCU RoPE kernel remains the owner of rotation.  This
    helper intentionally accepts an already-rotated BF16 Q so tests can check
    packing, scales, and the dtype split without touching a device kernel.
    """
    if q_bf16.dtype != torch.bfloat16 or q_bf16.ndim != 3:
        raise TypeError("FP4 query preparation expects BF16 Q [T,H,128]")
    if q_bf16.shape[-1] != FP4_HEAD_DIM:
        raise ValueError("FP4 query preparation expects head_dim=128")
    if weights.shape != q_bf16.shape[:2]:
        raise ValueError("indexer weights must have shape [T,H]")
    values = q_bf16.float().reshape(*q_bf16.shape[:2], 4, 32)
    amax = values.abs().amax(-1).clamp_min(1e-10)
    exponent = torch.ceil(torch.log2(amax / 6.0))
    scale = torch.exp2(exponent)
    normalized = values / scale.unsqueeze(-1)
    codes = _e2m1_code(normalized).reshape(*q_bf16.shape[:2], 128)
    packed = codes[..., 0::2] | (codes[..., 1::2] << 4)
    scales = (exponent + 127).clamp(0, 255).to(torch.uint8)
    return Fp4IndexerQuery(
        bf16=q_bf16,
        packed=packed.contiguous(),
        scales=scales.contiguous(),
        weights=(weights * softmax_scale * head_scale).float().contiguous(),
    )


def rotate_indexer_query_bf16(
    q: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    rope_dim: int = 64,
) -> torch.Tensor:
    """Tensor-only BF16 RoPE view retained for LightOp's prefill ABI."""
    if q.dtype != torch.bfloat16 or q.ndim != 3 or q.shape[-1] != 128:
        raise TypeError("V4.1 FP4 Q adapter expects BF16 [T,H,128]")
    if rope_dim != 64 or cos_sin_cache.shape[-1] != 64:
        raise ValueError("V4.1 FP4 Q adapter expects rope_dim=64")
    half = rope_dim // 2
    out = q.clone()
    pairs = q[..., -rope_dim:].float().reshape(*q.shape[:2], half, 2)
    cache = cos_sin_cache.index_select(0, positions.to(torch.long))
    cos = cache[..., :half].float().unsqueeze(1)
    sin = cache[..., half:].float().unsqueeze(1)
    even, odd = pairs[..., 0], pairs[..., 1]
    rotated = torch.stack((even * cos - odd * sin, odd * cos + even * sin), -1)
    out[..., -rope_dim:] = rotated.to(torch.bfloat16).reshape(*q.shape[:2], rope_dim)
    return out


def lightop_fp4_prefill(
    q_bf16: torch.Tensor,
    k_packed: torch.Tensor,
    k_scales: torch.Tensor,
    weights: torch.Tensor,
    cu_seq_len_k_start: torch.Tensor,
    cu_seq_len_k_end: torch.Tensor,
    *,
    clean_logits: bool = False,
    max_seqlen_k: int,
    candidate_blocks: list[torch.Tensor] | None = None,
    candidate_query_lens: list[int] | None = None,
) -> torch.Tensor:
    """Call the current LightOp non-paged BF16-Q/FP4-K ABI exactly."""
    from lightop.attention import fp8_fp4_mqa_logits

    return fp8_fp4_mqa_logits(
        (q_bf16, None),
        (k_packed, k_scales),
        weights,
        cu_seq_len_k_start,
        cu_seq_len_k_end,
        clean_logits,
        max_seqlen_k,
        candidate_blocks,
        candidate_query_lens,
    )


def lightop_gather_index_k_fp4(
    cache: torch.Tensor, slots: torch.Tensor, page_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Call the current LightOp physical FP4 page gather wrapper."""
    from lightop.gemmopt import dsv41_gather_index_k_fp4

    return dsv41_gather_index_k_fp4(cache, slots, page_size)


def lightop_fp4_decode(
    packed_q: torch.Tensor,
    q_scales: torch.Tensor,
    cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_table: torch.Tensor,
    schedule_meta: torch.Tensor | None,
    max_context_len: int,
    table_block_size: int = 64,
    out: torch.Tensor | None = None,
    rows_per_request: int = 1,
) -> torch.Tensor:
    """Call the current LightOp 11-argument paged FP4 ABI."""
    from lightop.attention import paged_mqa_logits_fp4

    return paged_mqa_logits_fp4(
        packed_q,
        q_scales,
        fp4_cache_view(cache),
        weights,
        context_lens,
        block_table,
        schedule_meta,
        max_context_len,
        table_block_size,
        out,
        rows_per_request,
    )


__all__ = [
    "FP4_ABI",
    "FP4_HEAD_DIM",
    "FP4_PAGE_BLOCK_SIZE",
    "FP4_PAGE_BYTES",
    "FP4_RECORD_BYTES",
    "Fp4IndexerCapability",
    "Fp4IndexerQuery",
    "Fp4QTuple",
    "fp4_cache_shape",
    "fp4_cache_view",
    "fp4_chunk_slots",
    "fp4_flat_page_view",
    "fp4_physical_page_size",
    "fp4_indexer_requested",
    "lightop_fp4_decode",
    "lightop_gather_index_k_fp4",
    "lightop_fp4_prefill",
    "prepare_fp4_query",
    "rotate_indexer_query_bf16",
    "select_fp4_indexer_capability",
    "stage_fp4_decode_pages",
]
