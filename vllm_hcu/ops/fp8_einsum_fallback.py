# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# Modified by Hygon Information Technology Co., Ltd., 2026.

import math

import torch

from vllm.triton_utils import tl, triton


def _upcast_e8m0_to_fp32(scale: torch.Tensor) -> torch.Tensor:
    """Upcast E8M0 (exponent-only) scale to float32."""
    exp_bits = scale.view(torch.uint8).to(torch.int32)
    fp32_bits = exp_bits << 23
    return fp32_bits.view(torch.float32)


def _decode_e8m0_scales(scale: torch.Tensor) -> torch.Tensor:
    if scale.dtype == torch.float8_e8m0fnu:
        return _upcast_e8m0_to_fp32(scale).contiguous()
    return scale.to(torch.float32)


@triton.jit
def _deepseek_v4_fp8_einsum_kernel(
    a_ptr,
    a_scale_ptr,
    b_ptr,
    b_scale_ptr,
    out_ptr,
    B,
    H,
    R,
    D,
    A_SCALE_LAST,
    B_ROW_BLOCKS,
    B_COL_BLOCKS,
    A_BLOCK,
    B_ROW_BLOCK,
    B_COL_BLOCK,
    stride_ab,
    stride_ah,
    stride_ar,
    stride_asb,
    stride_ash,
    stride_ass,
    stride_bh,
    stride_bd,
    stride_br,
    stride_bsh,
    stride_bsr,
    stride_bsc,
    stride_ob,
    stride_oh,
    stride_od,
    BLOCK_D: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_d = tl.program_id(1)

    b_idx = pid_bh // H
    h_idx = pid_bh % H

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < D
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

    for r0 in range(0, R, BLOCK_R):
        offs_r = r0 + tl.arange(0, BLOCK_R)
        mask_r = offs_r < R

        a_ptrs = a_ptr + b_idx * stride_ab + h_idx * stride_ah + offs_r * stride_ar
        a_vals = tl.load(a_ptrs, mask=mask_r, other=0.0).to(tl.float32)

        a_scale_idx = offs_r // A_BLOCK
        a_scale_idx = tl.minimum(a_scale_idx, A_SCALE_LAST - 1)
        a_scale_ptrs = (
            a_scale_ptr + b_idx * stride_asb + h_idx * stride_ash + a_scale_idx * stride_ass
        )
        a_scales = tl.load(a_scale_ptrs, mask=mask_r, other=0.0).to(tl.float32)
        a_deq = a_vals * a_scales

        b_ptrs = (
            b_ptr
            + h_idx * stride_bh
            + offs_d[:, None] * stride_bd
            + offs_r[None, :] * stride_br
        )
        b_vals = tl.load(b_ptrs, mask=mask_d[:, None] & mask_r[None, :], other=0.0).to(
            tl.float32
        )

        b_row_idx = offs_d // B_ROW_BLOCK
        b_row_idx = tl.minimum(b_row_idx, B_ROW_BLOCKS - 1)
        b_col_idx = offs_r // B_COL_BLOCK
        b_col_idx = tl.minimum(b_col_idx, B_COL_BLOCKS - 1)
        b_scale_ptrs = (
            b_scale_ptr
            + h_idx * stride_bsh
            + b_row_idx[:, None] * stride_bsr
            + b_col_idx[None, :] * stride_bsc
        )
        b_scales = tl.load(
            b_scale_ptrs,
            mask=mask_d[:, None] & mask_r[None, :],
            other=0.0,
        ).to(tl.float32)

        acc += tl.sum((b_vals * b_scales) * a_deq[None, :], axis=1)

    out_ptrs = out_ptr + b_idx * stride_ob + h_idx * stride_oh + offs_d * stride_od
    tl.store(out_ptrs, acc, mask=mask_d)


def deepseek_v4_fp8_einsum_fallback_triton(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    out: torch.Tensor,
    equation: str,
) -> None:
    """Triton fallback for DeepSeek-V4 FP8 einsum.

    Supports `equation == "bhr,hdr->bhd"` only.
    """
    if equation != "bhr,hdr->bhd":
        raise RuntimeError(f"Unsupported fallback equation: {equation}")

    if a.ndim != 3 or b.ndim != 2 or out.ndim != 3:
        raise RuntimeError("Invalid input ranks for DeepSeek-V4 FP8 einsum fallback.")

    B, H, R = a.shape
    if b.shape[0] % H != 0:
        raise RuntimeError(
            f"Cannot reshape weight of shape {tuple(b.shape)} with groups={H}."
        )
    D = b.shape[0] // H

    a_scale = _decode_e8m0_scales(a_scale)
    b_scale = _decode_e8m0_scales(b_scale)
    b_3d = b.view(H, D, R)
    b_scale_3d = b_scale.view(H, -1, b_scale.shape[-1])

    a_scale_last = a_scale.shape[-1]
    b_row_blocks = b_scale_3d.shape[-2]
    b_col_blocks = b_scale_3d.shape[-1]
    a_block = math.ceil(R / a_scale_last)
    b_row_block = math.ceil(D / b_row_blocks)
    b_col_block = math.ceil(R / b_col_blocks)

    grid = (B * H, triton.cdiv(D, 128))
    _deepseek_v4_fp8_einsum_kernel[grid](
        a,
        a_scale,
        b_3d,
        b_scale_3d,
        out,
        B,
        H,
        R,
        D,
        a_scale_last,
        b_row_blocks,
        b_col_blocks,
        a_block,
        b_row_block,
        b_col_block,
        a.stride(0),
        a.stride(1),
        a.stride(2),
        a_scale.stride(0),
        a_scale.stride(1),
        a_scale.stride(2),
        b_3d.stride(0),
        b_3d.stride(1),
        b_3d.stride(2),
        b_scale_3d.stride(0),
        b_scale_3d.stride(1),
        b_scale_3d.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        BLOCK_D=128,
        BLOCK_R=64,
        num_warps=4,
    )