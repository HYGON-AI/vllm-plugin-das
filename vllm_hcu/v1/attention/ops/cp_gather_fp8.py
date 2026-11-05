# SPDX-License-Identifier: Apache-2.0
"""Gather packed FP8 DS-MLA KV rows into the existing BF16 prefill workspace."""

import os
from functools import lru_cache

import torch
from vllm import _custom_ops as ops
from vllm.triton_utils import tl, triton


@lru_cache(maxsize=None)
def _supports_fp8_e4m3(device: torch.device) -> bool:
    # Other ROCm targets can use FNUZ bytes for the same packed KV layout.
    arch = getattr(torch.cuda.get_device_properties(device), "gcnArchName", "")
    return arch.split(":", 1)[0] == "gfx938"


@triton.jit(do_not_specialize=["TOTAL", "NUM_REQS"])
def _cp_gather_fp8_flat(
    Cache, Out, Table, Starts,
    TOTAL, NUM_REQS, BLOCK_SIZE: tl.constexpr,
    TABLE_STRIDE: tl.constexpr, CACHE_BLOCK_STRIDE: tl.constexpr,
    CACHE_ENTRY_STRIDE: tl.constexpr, OUT_STRIDE: tl.constexpr,
    ROWS: tl.constexpr,
):
    first = tl.program_id(0) * ROWS
    lo = tl.full((), 0, tl.int32)
    hi = NUM_REQS - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        start = tl.load(Starts + mid)
        lo = tl.where(start <= first, mid, lo)
        hi = tl.where(start <= first, hi, mid - 1)
    row = first + tl.arange(0, ROWS)
    next_start = tl.load(Starts + lo + 1, lo + 1 < NUM_REQS, other=TOTAL)
    req = tl.full((ROWS,), 0, tl.int32) + lo
    if tl.minimum(first + ROWS, TOTAL) > next_start:
        # Only tiles that cross a request boundary need per-row ownership.
        row_lo = tl.full((ROWS,), 0, tl.int32) + lo
        row_hi = tl.full((ROWS,), 0, tl.int32) + NUM_REQS - 1
        while tl.sum((row_lo < row_hi).to(tl.int32), 0) > 0:
            mid = (row_lo + row_hi + 1) // 2
            start = tl.load(Starts + mid)
            right = start <= row
            row_lo = tl.where(right, mid, row_lo)
            row_hi = tl.where(right, row_hi, mid - 1)
        req = row_lo
    offset = row - tl.load(Starts + req)
    valid = row < TOTAL
    physical = tl.load(Table + req * TABLE_STRIDE + offset // BLOCK_SIZE,
                       valid, other=0)
    src = (Cache + physical.to(tl.int64) * CACHE_BLOCK_STRIDE
           + (offset % BLOCK_SIZE).to(tl.int64) * CACHE_ENTRY_STRIDE)
    col = tl.arange(0, 512)
    raw = tl.load(src[:, None] + col[None, :], valid[:, None], other=0)
    scale_ptr = (src[:, None] + 512).to(tl.pointer_type(tl.float32))
    scales = tl.load(scale_ptr + col[None, :] // 128,
                     valid[:, None], other=0)
    value = raw.to(tl.float8e4nv, bitcast=True).to(tl.float32) * scales
    tl.store(Out + row[:, None].to(tl.int64) * OUT_STRIDE + col[None, :],
             value, valid[:, None])
    rope_col = tl.arange(0, 64)
    # Copy BF16 bits directly, including NaN payloads and signed zero.
    rope = tl.load((src[:, None] + 528).to(tl.pointer_type(tl.uint16))
                   + rope_col[None, :], valid[:, None], other=0)
    rope_out = (Out + row[:, None].to(tl.int64) * OUT_STRIDE + 512).to(
        tl.pointer_type(tl.uint16))
    tl.store(rope_out + rope_col[None, :], rope, valid[:, None])


def cp_gather_and_upconvert_fp8_kv_cache(
    src_cache: torch.Tensor,
    dst: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    workspace_starts: torch.Tensor,
    batch_size: int,
) -> None:
    """Opt-in plugin kernel; unsupported layouts retain the upstream operator."""
    if (
        os.environ.get("VLLM_HCU_ENABLE_TRITON_CP_GATHER", "0") != "1"
        or not src_cache.is_cuda
        or src_cache.dtype != torch.uint8
        or dst.dtype != torch.bfloat16
        or src_cache.ndim != 3
        or src_cache.shape[2] != 656
        or dst.ndim != 2
        or dst.shape[1] != 576
        or src_cache.stride(2) != 1
        or dst.stride(1) != 1
        or block_table.stride(1) != 1
        or workspace_starts.stride(0) != 1
        or not _supports_fp8_e4m3(src_cache.device)
    ):
        ops.cp_gather_and_upconvert_fp8_kv_cache(
            src_cache, dst, block_table, seq_lens, workspace_starts, batch_size)
        return
    if dst.shape[0] == 0 or batch_size == 0:
        return
    with torch.cuda.device(src_cache.device):
        _cp_gather_fp8_flat[(triton.cdiv(dst.shape[0], 8),)](
            src_cache, dst, block_table, workspace_starts,
            dst.shape[0], batch_size, src_cache.shape[1], block_table.stride(0),
            src_cache.stride(0), src_cache.stride(1), dst.stride(0),
            8, num_warps=2,
            enable_fp_fusion=False,
        )
