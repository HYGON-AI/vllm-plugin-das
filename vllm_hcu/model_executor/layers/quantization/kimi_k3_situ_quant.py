# SPDX-License-Identifier: Apache-2.0
"""Explicit temporary Triton replacement for Kimi HT SiTU INT8 quantization."""
import math
import os

import torch
from triton.language.extra import libdevice
from vllm.triton_utils import tl, triton


@triton.jit
def _situ_mul_quant_kernel(X, Q, S, STRIDE: tl.constexpr, D: tl.constexpr,
                           BETA: tl.constexpr, LINEAR_BETA: tl.constexpr,
                           BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    gate = tl.load(X + row * STRIDE + col, col < D, other=0).to(tl.float32)
    up = tl.load(X + row * STRIDE + D + col, col < D, other=0).to(tl.float32)
    value = (BETA * libdevice.tanh(gate / BETA) * tl.sigmoid(gate)
             * LINEAR_BETA * libdevice.tanh(up / LINEAR_BETA))
    amax = tl.maximum(tl.max(tl.abs(value), 0), 1e-10)
    quant = libdevice.nearbyint(value * (127.0 / amax))
    quant = tl.minimum(tl.maximum(quant, -127.0), 127.0).to(tl.int8)
    tl.store(Q + row * D + col, quant, col < D)
    tl.store(S + row, amax / 127.0)


def triton_situ_mul_quant(x, *, beta, linear_beta):
    """Return INT8 [M,D] and FP32 [M,1]; compute SiTU in FP32 before rounding.

    This is a mathematical-reference workaround, not proven bitwise parity
    with the unavailable LightOp kernel. Zero rows use scale 1e-10 / 127.
    """
    if not all(math.isfinite(v) and v > 0 for v in (beta, linear_beta)):
        raise ValueError("SiTU beta and linear_beta must be finite and positive")
    if x.ndim != 2 or x.shape[1] == 0 or x.shape[1] % 2:
        raise ValueError("SiTU requires [M,2D] with positive even width")
    if x.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise TypeError("SiTU input must be BF16, FP16 or FP32")
    if not x.is_cuda or not x.is_contiguous():
        raise ValueError("SiTU requires contiguous CUDA/HIP input")
    rows, width = x.shape
    dim = width // 2
    if dim > 16384:
        raise ValueError("Temporary SiTU kernel supports D <= 16384")
    q = torch.empty((rows, dim), dtype=torch.int8, device=x.device)
    scale = torch.empty((rows, 1), dtype=torch.float32, device=x.device)
    if rows:
        _situ_mul_quant_kernel[(rows,)](
            x, q, scale, x.stride(0), dim, float(beta), float(linear_beta),
            triton.next_power_of_2(dim), num_warps=4,
        )
    return q, scale


def lightop_situ_mul_quant(x, *, beta, linear_beta):
    """Adapt the published contiguous LightOp API to the Kimi HT contract."""
    # The native vectorized kernel does not mask partial vector loads/stores.
    # Kimi W4A8 dimensions are multiples of 32, which satisfy every vector size.
    if x.ndim != 2 or x.shape[1] == 0 or x.shape[1] % 64:
        raise ValueError("Kimi LightOp SiTU requires [M,2D] with D divisible by 32")
    from lightop.activation import fuse_situ_mul_quant_contiguous
    return fuse_situ_mul_quant_contiguous(
        x, situ_beta=beta, situ_linear_beta=linear_beta
    )


def resolve_situ_quant():
    backend = os.environ.get("VLLM_HCU_KIMI_HT_SITU_BACKEND", "lightop")
    if backend == "triton":
        return backend, triton_situ_mul_quant
    if backend == "lightop":
        from lightop.activation import fuse_situ_mul_quant_contiguous
        if not callable(fuse_situ_mul_quant_contiguous):
            raise TypeError("LightOp contiguous SiTU operation is not callable")
        return backend, lightop_situ_mul_quant
    raise ValueError("VLLM_HCU_KIMI_HT_SITU_BACKEND must be lightop or triton")
