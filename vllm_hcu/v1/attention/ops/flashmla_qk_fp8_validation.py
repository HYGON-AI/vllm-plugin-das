# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Fused indices/count validation before the unmasked native QK-FP8 kernel."""

import torch
import triton
import triton.language as tl


@triton.jit
def _invalid_rows(
    Indices,
    Counts,
    Invalid,
    stride_i: tl.constexpr,
    stride_c: tl.constexpr,
    kv_length,
    HAS_COUNTS: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, 2048)
    index = tl.load(Indices + row * stride_i + offsets)
    bad = tl.max(((index < 0) | (index >= kv_length)).to(tl.int32), 0)
    if HAS_COUNTS:
        count = tl.load(Counts + row * stride_c)
        bad = bad | (count != 2048).to(tl.int32)
    tl.store(Invalid + row, bad)


def invalid_rows(indices, counts, kv_length):
    flags = torch.empty((indices.shape[0],), device=indices.device, dtype=torch.int32)
    _invalid_rows[(indices.shape[0],)](
        indices,
        counts if counts is not None else indices,
        flags,
        indices.stride(0),
        counts.stride(0) if counts is not None else 0,
        kv_length,
        counts is not None,
        num_warps=4,
    )
    return flags
