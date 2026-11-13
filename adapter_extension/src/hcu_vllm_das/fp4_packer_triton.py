# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""029's production E2M1 lowering, with explicit legacy NaN policy.

FP32 inputs, round to nearest even, saturate magnitude to six, retain the
IEEE sign (including -0). NaN has zero magnitude and retains its sign, as in
029's CPU contract; this is not a claim of NVIDIA NaN conformance. No scale,
RoPE, page addressing or cache ownership is implemented here.
"""
import triton
import triton.language as tl


@triton.jit
def e2m1_code(x):
    ax = tl.minimum(tl.abs(x), 6.0)
    # Make the old reference's NaN policy independent of minimum's NaN mode.
    ax = tl.where(x != x, 0.0, ax)
    idx = (ax >= 0.25).to(tl.uint8)
    idx += (ax >= 0.75).to(tl.uint8)
    idx += (ax >= 1.25).to(tl.uint8)
    idx += (ax >= 1.75).to(tl.uint8)
    idx += (ax >= 2.5).to(tl.uint8)
    idx += (ax >= 3.5).to(tl.uint8)
    idx += (ax >= 5.0).to(tl.uint8)
    boundary = ((ax == 0.25) | (ax == 0.75) | (ax == 1.25)
                | (ax == 1.75) | (ax == 2.5) | (ax == 3.5) | (ax == 5.0))
    idx = tl.where(boundary & ((idx & 1) == 1), idx - 1, idx)
    sign = ((x.to(tl.int32, bitcast=True) >> 31) & 1).to(tl.uint8)
    return idx | (sign << 3)


@triton.jit
def fp32x2_to_fp4x2(x_lo, x_hi):
    """First input occupies the low nibble, second input the high nibble."""
    return e2m1_code(x_lo) | (e2m1_code(x_hi) << 4)
