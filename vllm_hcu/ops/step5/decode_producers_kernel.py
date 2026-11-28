# SPDX-License-Identifier: Apache-2.0
"""Fused producers for the validated 4096-wide Step5 decode profile.

Residual sums stay FP32. Activations round to BF16 before token FP8 quantization.
"""
import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _producer(X, A, W, R, Y, Q, S, EPS: tl.constexpr,
              ADD: tl.constexpr, QUANT: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, 4096)
    offset = row * 4096 + col
    x = tl.load(X + offset).to(tl.float32)
    if ADD:
        x = x + tl.load(A + offset).to(tl.float32)
    # Store the FP32 sum before normalization; no residual narrowing is allowed.
    tl.store(R + offset, x)
    w = tl.load(W + col).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, 0) * (1.0 / 4096.0) + EPS)
    y = ((x * inv) * (1.0 + w)).to(tl.bfloat16)
    tl.store(Y + offset, y)
    if QUANT:
        yf = y.to(tl.float32)
        bad = yf != yf
        has_nan = tl.sum(bad.to(tl.int32), 0) > 0
        amax = tl.max(tl.where(bad, 0.0, tl.abs(yf)), 0)
        scale = tl.maximum(amax * tl.full((), 1.0 / 448.0, tl.float32),
                           tl.full((), 1.e-12, tl.float32))
        scale = tl.where(has_nan, float('nan'), scale)
        value = tl.div_rn(yf, scale)
        clipped = tl.minimum(tl.maximum(value, -448.0), 448.0)
        clipped = tl.where(value != value, value, clipped)
        tl.store(Q + offset, clipped.to(Q.dtype.element_ty))
        tl.store(S + row, scale)


def producer(x, weight, eps, addition=None, quant=True, warps=8):
    assert x.dtype == torch.float32 and x.ndim == 2 and x.shape[1] == 4096
    assert x.is_cuda and x.is_contiguous()
    assert weight.dtype == torch.float32 and weight.shape == (4096,)
    assert weight.is_contiguous() and weight.device == x.device
    if addition is not None:
        assert addition.shape == x.shape and addition.is_contiguous()
        assert addition.dtype in (torch.bfloat16, torch.float32)
        assert addition.device == x.device
    residual = torch.empty_like(x)
    y = torch.empty_like(x, dtype=torch.bfloat16)
    q = torch.empty_like(x, dtype=torch.float8_e4m3fn) if quant else y
    scale = torch.empty((x.shape[0], 1), device=x.device, dtype=torch.float32)
    if x.shape[0]:
        _producer[(x.shape[0],)](
            x, x if addition is None else addition, weight, residual, y, q,
            scale, eps, ADD=addition is not None, QUANT=quant,
            num_warps=warps, enable_fp_fusion=False)
    return (residual, y, q, scale) if quant else (residual, y)
