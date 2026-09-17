# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
# Adapted from the frozen Kimi source's flash-linear-attention gated norm.
# Original FLA code: Copyright (c) 2023-2025, Songlin Yang, Yu Zhang (MIT).
"""Kimi-owned gated RMSNorm with source-compatible row-strided gate loads."""

import torch

from vllm.model_executor.layers.fla.ops.kda import (
    FusedRMSNormGated as BaseFusedRMSNormGated,
    rms_norm_gated as base_rms_norm_gated,
)
from vllm.triton_utils import tl, triton


@triton.heuristics({
    "STORE_RESIDUAL_OUT": lambda args: args["residual_out"] is not None,
    "HAS_RESIDUAL": lambda args: args["residual"] is not None,
    "HAS_WEIGHT": lambda args: args["w"] is not None,
    "HAS_BIAS": lambda args: args["b"] is not None,
})
@triton.jit
def layer_norm_gated_fwd_kernel(
    x, g, y, w, b, residual, residual_out, rstd, eps, T,
    H: tl.constexpr, g_stride_n: tl.constexpr, D: tl.constexpr,
    BT: tl.constexpr, BD: tl.constexpr, ACTIVATION: tl.constexpr,
    STORE_RESIDUAL_OUT: tl.constexpr, HAS_RESIDUAL: tl.constexpr,
    HAS_WEIGHT: tl.constexpr, HAS_BIAS: tl.constexpr,
):
    i_t = tl.program_id(0)
    o_d = tl.arange(0, BD)
    m_d = o_d < D
    p_x = tl.make_block_ptr(x, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    b_x = tl.load(p_x, boundary_check=(0, 1)).to(tl.float32)
    if HAS_RESIDUAL:
        p_res = tl.make_block_ptr(
            residual, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0)
        )
        b_x += tl.load(p_res, boundary_check=(0, 1)).to(tl.float32)
    if STORE_RESIDUAL_OUT:
        p_res_out = tl.make_block_ptr(
            residual_out, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0)
        )
        tl.store(p_res_out, b_x.to(p_res_out.dtype.element_ty), boundary_check=(0, 1))
    b_xbar = tl.where(m_d[None, :], b_x, 0.0)
    b_var = tl.sum(b_xbar * b_xbar, axis=1) / D
    b_rstd = 1 / tl.sqrt(b_var + eps)
    p_rstd = tl.make_block_ptr(rstd, (T,), (1,), (i_t * BT,), (BT,), (0,))
    tl.store(p_rstd, b_rstd.to(p_rstd.dtype.element_ty), boundary_check=(0,))
    if HAS_WEIGHT:
        b_w = tl.load(w + o_d, mask=m_d).to(tl.float32)
    if HAS_BIAS:
        b_b = tl.load(b + o_d, mask=m_d).to(tl.float32)
    b_x_hat = b_x * b_rstd[:, None]
    b_y = b_x_hat * b_w[None, :] if HAS_WEIGHT else b_x_hat
    if HAS_BIAS:
        b_y = b_y + b_b[None, :]
    o_t = i_t * BT + tl.arange(0, BT)
    o_g = (o_t // H) * g_stride_n + (o_t % H) * D
    b_g = tl.load(
        g + o_g[:, None] + o_d[None, :],
        mask=(o_t[:, None] < T) & m_d[None, :], other=0.0,
    ).to(tl.float32)
    if ACTIVATION == "swish" or ACTIVATION == "silu":
        b_y = b_y * b_g * tl.sigmoid(b_g)
    elif ACTIVATION == "sigmoid":
        b_y = b_y * tl.sigmoid(b_g)
    p_y = tl.make_block_ptr(y, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    tl.store(p_y, b_y.to(p_y.dtype.element_ty), boundary_check=(0, 1))


def rms_norm_gated(
    x, g, weight, bias, activation="swish", residual=None,
    prenorm=False, residual_in_fp32=False, eps=1e-6,
):
    """Preserve the base API and in-place output/residual ownership contract."""
    if (x.ndim < 2 or g.ndim < 2 or x.shape[-1] != g.shape[-1]
            or x.numel() != g.numel()):
        raise ValueError("Kimi gated norm requires matching x/g elements and feature width")
    if activation not in ("swish", "silu", "sigmoid"):
        raise ValueError(f"Unsupported activation: {activation}")
    if residual is not None and residual.shape != x.shape:
        raise ValueError("Residual shape must match x")
    shape = x.shape
    d = shape[-1]
    # The source also retains a contiguous, one-row kernel above this threshold.
    if d > 512:
        return base_rms_norm_gated(
            x, g, weight, bias, activation, residual, prenorm, residual_in_fp32, eps
        )
    if d == 0:
        raise ValueError("Feature dimension must be positive")
    for parameter in (weight, bias):
        if parameter is not None and parameter.shape != (d,):
            raise ValueError("Weight/bias shape must match the feature dimension")
    x = x.contiguous().view(-1, d)
    h = 1 if g.ndim == 2 else g.shape[-2]
    # Kimi's projection slice has contiguous heads/features and padding between
    # token rows. Unusual external views keep the existing materialization path.
    if g.stride(-1) != 1 or (g.ndim > 2 and g.stride(-2) != d):
        g = g.contiguous()
    try:
        g = g.view(-1, h, d)
    except RuntimeError:
        g = g.contiguous().view(-1, h, d)
    t = x.shape[0]
    if t == 0:
        return (x.view(shape), x.view(shape)) if prenorm else x.view(shape)
    if residual is not None:
        residual = residual.contiguous().view(-1, d)
    residual_dtype = residual.dtype if residual is not None else (
        torch.float32 if residual_in_fp32 else None
    )
    residual_out = torch.empty_like(x, dtype=residual_dtype) if (
        residual is not None or (residual_dtype is not None and residual_dtype != x.dtype)
    ) else None
    rstd = torch.empty((t,), dtype=torch.float32, device=x.device)
    layer_norm_gated_fwd_kernel[(triton.cdiv(t, 16),)](
        x=x, g=g, y=x, w=weight, b=bias, residual=residual,
        residual_out=residual_out, rstd=rstd, eps=eps, T=t,
        H=h, g_stride_n=g.stride(0), D=d, BT=16, BD=triton.next_power_of_2(d),
        ACTIVATION=activation, num_warps=8,
    )
    y = x.view(shape)
    return y if not prenorm else (
        y, (residual_out if residual_out is not None else x).view(shape)
    )


class KimiFusedRMSNormGated(BaseFusedRMSNormGated):
    """Keep base parameters/custom-op dispatch; specialize only Kimi GPU norm."""

    def forward_cuda(self, x, g, residual=None, prenorm=False, residual_in_fp32=False):
        return rms_norm_gated(
            x, g, self.weight, self.bias, self.activation, residual,
            prenorm, residual_in_fp32, self.eps,
        )
