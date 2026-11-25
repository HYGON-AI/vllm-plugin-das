# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""HCU-local IndexPool fusions; no global operator or cache-layout changes."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _fwht_round(x, BIT: tl.constexpr):
    lo, hi = tl.split(tl.permute(tl.reshape(x, (128 // (2 * BIT), 2, BIT)), (0, 2, 1)))
    return tl.reshape(tl.permute(tl.join(lo + hi, lo - hi), (0, 2, 1)), (128,))


@triton.jit
def _fwht_quantize(X, Q, S, ROW: tl.constexpr, COL: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, 128)
    x = tl.load(X + row * ROW + col * COL).to(tl.float32)
    # Keep every radix stage in FP32. A BF16 Hadamard intermediate changes
    # the power-of-two scale and may therefore change the selected groups.
    x = _fwht_round(x, 1)
    x = _fwht_round(x, 2)
    x = _fwht_round(x, 4)
    x = _fwht_round(x, 8)
    x = _fwht_round(x, 16)
    x = _fwht_round(x, 32)
    x = _fwht_round(x, 64)
    x = x * (128.0**-0.5)
    magnitude = tl.maximum(tl.max(tl.abs(x), 0), 1.0e-4)
    scale = tl.exp2(tl.ceil(tl.log2(tl.div_rn(magnitude, 448.0))))
    # Explicit NaN propagation matches Torch amax (including nonfinite inputs).
    scale = tl.where(tl.sum((x != x).to(tl.int32), 0) > 0, float("nan"), scale)
    scaled = tl.div_rn(x, scale)
    quant = tl.minimum(tl.maximum(scaled, -448.0), 448.0)
    quant = tl.where(scaled != scaled, float("nan"), quant)
    tl.store(Q + row * 128 + col, quant)
    tl.store(S + row, scale)


def fwht_quantize_triton(vectors):
    rows = vectors.shape[0]
    quant = torch.empty((rows, 128), device=vectors.device, dtype=torch.float8_e4m3fn)
    scales = torch.empty((rows,), device=vectors.device, dtype=torch.float32)
    if rows:
        _fwht_quantize[(rows,)](
            vectors,
            quant,
            scales,
            vectors.stride(0),
            vectors.stride(1),
            num_warps=4,
            enable_fp_fusion=False,
        )
    return quant, scales


@triton.jit
def _expand_groups(
    G,
    P,
    A,
    O,
    GS0: tl.constexpr,
    GS1: tl.constexpr,
    PS: tl.constexpr,
    AS: tl.constexpr,
    GROUPS: tl.constexpr,
    RATIO: tl.constexpr,
    HAS_ACTIVE: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    pos = tl.load(P + row * PS)
    live = pos >= 0
    if HAS_ACTIVE:
        live = live & tl.load(A + row * AS)
    upper = pos // RATIO
    is_history = offsets < GROUPS * RATIO
    group = tl.load(G + row * GS0 + (offsets // RATIO) * GS1, mask=is_history, other=-1)
    token = tl.where(
        is_history, group * RATIO + offsets % RATIO, upper * RATIO + offsets % RATIO
    )
    valid = (
        live
        & (offsets < WIDTH)
        & tl.where(is_history, (group >= 0) & (group < upper), token <= pos)
    )
    # Torch casts candidates to int32 BEFORE re-checking validity. Preserve
    # wraparound behavior even for helper callers outside model position bounds.
    token = tl.where(valid, token, -1).to(tl.int32)
    valid = (token >= 0) & (offsets < WIDTH)
    prefix = tl.cumsum(valid.to(tl.int32), 0)
    count = tl.sum(valid.to(tl.int32), 0)
    destination = tl.where(valid, prefix - 1, count + offsets - prefix)
    # Both valid and invalid destinations form one permutation: one store per
    # output, no initialize/scatter race and no dependency on padding contents.
    tl.store(O + row * WIDTH + destination, token, mask=offsets < WIDTH)


def expand_groups_triton(group_ids, positions, ratio, active=None):
    rows, groups = group_ids.shape
    width = (groups + 1) * ratio
    output = torch.empty((rows, width), dtype=torch.int32, device=positions.device)
    if rows:
        _expand_groups[(rows,)](
            group_ids,
            positions,
            positions if active is None else active,
            output,
            *group_ids.stride(),
            positions.stride(0),
            0 if active is None else active.stride(0),
            groups,
            ratio,
            active is not None,
            width,
            triton.next_power_of_2(width),
            num_warps=4
        )
    return output
