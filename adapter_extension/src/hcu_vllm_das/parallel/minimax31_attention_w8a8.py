# SPDX-License-Identifier: Apache-2.0
"""MiniMax VERIFY QKV input fusion with vLLM symmetric INT8 semantics."""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _gemma_rmsnorm_quant_kernel(
    x_ptr, residual_ptr, weight_ptr, quant_ptr, scale_ptr, residual_out_ptr,
    n_cols, stride_xrow, stride_rrow, eps,
    HAS_RESIDUAL: tl.constexpr, BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * stride_xrow + cols, mask, 0).to(tl.float32)
    if HAS_RESIDUAL:
        residual = tl.load(residual_ptr + row * stride_rrow + cols, mask, 0).to(tl.float32)
        x = x + residual
        tl.store(residual_out_ptr + row * n_cols + cols, x.to(tl.bfloat16), mask)
    # Match the deployed Gemma RMSNorm reduction and sqrt, keeping the FP32
    # sum before the BF16 residual write. Quantize the rounded activation.
    variance = tl.sum(x * x, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(variance + eps)
    weight = tl.load(weight_ptr + cols, mask, 0).to(tl.float32)
    normalized = (x * rstd * (1.0 + weight)).to(tl.bfloat16).to(tl.float32)
    absval = tl.where(mask & (normalized == normalized), tl.abs(normalized), 0.0)
    amax = tl.max(absval, axis=0)
    inv_scale = tl.div_rn(127.0, tl.where(amax == 0.0, 1.0, amax))
    inv_scale = tl.where(amax == 0.0, 0.0, inv_scale)
    rounded = libdevice.rint(normalized * inv_scale)
    rounded = tl.where(rounded == rounded, tl.minimum(127.0, tl.maximum(-128.0, rounded)), 0.0)
    tl.store(quant_ptr + row * n_cols + cols, rounded.to(tl.int8), mask)
    tl.store(scale_ptr + row, amax / 127.0)


def supports_gemma_quant(x, residual, weight):
    return (
        x.ndim == 2 and x.shape[0] in (64, 128)
        and x.shape[1] == 6144 and x.is_cuda and x.dtype == torch.bfloat16
        and x.stride(1) == 1 and weight.shape == (6144,)
        and weight.device == x.device and weight.dtype == x.dtype and weight.is_contiguous()
        and (
            residual is None or (
                residual.shape == x.shape and residual.device == x.device
                and residual.dtype == x.dtype and residual.stride(1) == 1
            )
        )
    )


def gemma_rmsnorm_quant(x, residual, weight, eps, *, num_warps=16):
    """Return QKV's prequantized activations, scales and decoder residual."""
    m, n = x.shape
    quant = torch.empty((m, n), device=x.device, dtype=torch.int8)
    scale = torch.empty((m, 1), device=x.device, dtype=torch.float32)
    residual_out = x if residual is None else torch.empty((m, n), device=x.device, dtype=x.dtype)
    _gemma_rmsnorm_quant_kernel[(m,)](
        x, x if residual is None else residual, weight, quant, scale, residual_out,
        n, x.stride(0), 0 if residual is None else residual.stride(0), eps,
        residual is not None, triton.next_power_of_2(n), num_warps=num_warps,
    )
    return quant, scale, residual_out
