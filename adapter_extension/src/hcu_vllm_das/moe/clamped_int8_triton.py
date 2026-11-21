# SPDX-License-Identifier: Apache-2.0
"""Experimental routed W4A8 activation; not installed in serving.

Reuse the earlier 2048+256 column split, with the INT8 native arithmetic and
two BF16 boundaries. Device metadata selects live rows; capacity is unchanged.
GPU byte/scale/consumer equivalence is required before admission.
"""
from __future__ import annotations

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _value(gate, up):
    gate = tl.minimum(gate.to(tl.float32), 10.0)
    up = tl.minimum(tl.maximum(up.to(tl.float32), -10.0), 10.0)
    # Match native __builtin_amdgcn_exp2f and its explicit FP32 log2(e).
    ex = tl.exp2(-gate * 1.4426950408889634)
    activated = tl.div_rn(gate, 1.0 + ex).to(tl.bfloat16).to(tl.float32)
    return (activated * up).to(tl.bfloat16).to(tl.float32)


@triton.jit
def _int8_rne(value, inverse):
    # Saturating signed RNE, matching native nearbyint -> saturate -> INT8.
    rounded = tl.inline_asm_elementwise(
        "v_rndne_f32 $0, $1", constraints="=v,v", args=[value * inverse],
        dtype=tl.float32, is_pure=True, pack=1,
    )
    return tl.minimum(tl.maximum(rounded, -128.0), 127.0).to(tl.int8)


@triton.jit
def _clamped_int8_2304(X, Q, S, META, T: tl.constexpr,
                       MASKED: tl.constexpr, ROW_STRIDE: tl.constexpr):
    expert = tl.program_id(0) if MASKED else 0
    row = tl.program_id(1) if MASKED else tl.program_id(0)
    end = tl.minimum(tl.maximum(tl.load(META + expert), 0), T) if MASKED else T
    col = tl.arange(0, 2048)
    tail = 2048 + tl.arange(0, 256)
    while row < end:
        live = True if MASKED else tl.load(META + row) >= 0
        if live:
            offset = (expert * T + row) * 4608
            value = _value(tl.load(X + offset + col),
                           tl.load(X + offset + 2304 + col))
            last = _value(tl.load(X + offset + tail),
                          tl.load(X + offset + 2304 + tail))
            amax = tl.maximum(tl.maximum(tl.max(tl.abs(value), 0),
                                        tl.max(tl.abs(last), 0)), 1.0e-10)
            scale = tl.div_rn(amax, 127.0)
            inverse = tl.div_rn(127.0, amax)
            output = (expert * T + row) * 2304
            tl.store(Q + output + col, _int8_rne(value, inverse))
            tl.store(Q + output + tail, _int8_rne(last, inverse))
            tl.store(S + expert * T + row, scale)
        row += ROW_STRIDE


def clamped_int8_2304(x, metadata, *, masked, expected_m=32, output=None):
    """Quantize live rows only; callers must mask padding at every consumer."""
    if (not isinstance(x, torch.Tensor) or not x.is_cuda
            or x.dtype != torch.bfloat16 or not x.is_contiguous()
            or x.ndim != (3 if masked else 2) or x.shape[-1] != 4608):
        raise ValueError("experimental W4A8 activation requires contiguous HIP BF16 2N=4608")
    shape = x.shape[:-1]
    expected_meta = (shape[0],) if masked else (shape[0],)
    if (metadata.device != x.device or metadata.dtype != torch.int32
            or tuple(metadata.shape) != expected_meta or not metadata.is_contiguous()):
        raise ValueError("experimental W4A8 activation requires native device row metadata")
    if output is None:
        output = torch.empty((*shape, 2304), device=x.device, dtype=torch.int8)
    if (output.device != x.device or output.dtype != torch.int8
            or tuple(output.shape) != (*shape, 2304) or not output.is_contiguous()):
        raise ValueError("experimental W4A8 activation output layout mismatch")
    scales = torch.empty((*shape, 1), device=x.device, dtype=torch.float32)
    t = shape[1] if masked else shape[0]
    if x.numel():
        # Contiguous launches already have one program per padded row. Each
        # program must visit that row once, rather than walking all later rows.
        stride = min(t, max(1, int(expected_m))) if masked else t
        grid = (shape[0], stride) if masked else (t,)
        _clamped_int8_2304[grid](x, output, scales, metadata, t, masked, stride,
                                num_warps=4, enable_fp_fusion=False)
    return output, scales
