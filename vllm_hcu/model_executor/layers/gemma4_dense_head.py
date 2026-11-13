# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Full-vocabulary draft projection fused with argmax.

The stock greedy draft step materialises the whole [rows, vocab] logits tensor
and then reduces it: at 64 rows over a 262144 vocabulary that is 32 MiB written
and read back to produce 64 indices. These kernels keep the per-column-block
maximum and its index instead, so the logits tensor is never materialised.

M is a runtime argument while N and K are compile-time constants. N and K
really are fixed -- the vocabulary and the draft hidden size -- but M is the
row count, which changes every step, and it is only used to build a row mask.
Making it constexpr would compile a separate kernel per row count, which costs
211 s of install time over the gated range (measured) because every row count
has to be pre-compiled: a first compile inside CUDA graph capture is fatal.
As a runtime argument it is bit-identical and the same speed (measured at
33..256 rows), and Triton's integer specialization keeps the variants to a
handful rather than one per row.

No vocabulary pruning and no weight quantization: the GEMM runs over the full
vocabulary in BF16. The accumulator is rounded to bfloat16 and back before the
comparison, which reproduces the dtype of the stock logits, but the GEMM's
accumulation order differs from the library's, so a logit sitting on a rounding
boundary can land on either side. With only 8 mantissa bits over 262144
columns, ties for the maximum are common, so both kernels resolve them the way
torch.argmax does -- lowest index wins -- and NaN, if present, outranks
everything, again lowest index first.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# Triton kernels can only read module globals declared as constexpr.
_INDEX_SENTINEL = tl.constexpr(2147483647)


@triton.jit
def _partial(
    X, W, SUPPRESS, VALS, IDXS,
    M, N: tl.constexpr, K: tl.constexpr,
    SX: tl.constexpr, SW: tl.constexpr, NS: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):  # fmt: skip
    """Per column-block maximum and its index, for one block of rows."""
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    rows = pm * BM + tl.arange(0, BM)
    cols = pn * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)

    acc = tl.full((BM, BN), 0, tl.float32)
    for block in range(tl.cdiv(K, BK)):
        ks = block * BK + kk
        x = tl.load(
            X + rows[:, None] * SX + ks[None, :],
            (rows[:, None] < M) & (ks[None, :] < K),
            0,
        )
        w = tl.load(
            W + cols[None, :] * SW + ks[:, None],
            (cols[None, :] < N) & (ks[:, None] < K),
            0,
        )
        acc = tl.dot(x, w, acc)

    # Round through bfloat16 so the comparison sees the same values the stock
    # logits tensor would hold.
    z = acc.to(tl.bfloat16).to(tl.float32)

    valid = cols < N
    for j in tl.static_range(NS):
        valid = valid & (cols != tl.load(SUPPRESS + j))
    z = tl.where(valid[None, :], z, float("-inf"))

    isnan = z != z
    any_nan = tl.sum(isnan.to(tl.int32), 1) > 0
    mx = tl.max(tl.where(isnan, float("-inf"), z), 1)
    winners = tl.where(any_nan[:, None], isnan, z == mx[:, None])
    ids = tl.min(tl.where(winners & valid[None, :], cols[None, :], _INDEX_SENTINEL), 1)
    mx = tl.where(any_nan, float("nan"), mx)

    groups = tl.cdiv(N, BN)
    tl.store(VALS + rows * groups + pn, mx, rows < M)
    tl.store(IDXS + rows * groups + pn, ids, rows < M)


@triton.jit
def _finish(VALS, IDXS, OUT, G: tl.constexpr, B: tl.constexpr):
    """Reduce one row's column-block winners to a single index."""
    row = tl.program_id(0)
    ix = tl.arange(0, B)
    inside = ix < G
    v = tl.load(VALS + row * G + ix, inside, float("-inf"))
    ids = tl.load(IDXS + row * G + ix, inside, _INDEX_SENTINEL)

    isnan = (v != v) & inside
    any_nan = tl.sum(isnan.to(tl.int32), 0) > 0
    mx = tl.max(tl.where(isnan, float("-inf"), v), 0)
    match = tl.where(any_nan, isnan, v == mx) & inside
    tl.store(OUT + row, tl.min(tl.where(match, ids, _INDEX_SENTINEL), 0).to(tl.int64))


def dense_argmax(x, w, suppress=None, config=None, workspace=None):
    """Return argmax over the full vocabulary without materialising logits."""
    config = config or dict(BM=32, BN=128, BK=64, num_warps=4, num_stages=2)
    m, k = x.shape
    n = w.shape[0]
    groups = triton.cdiv(n, config["BN"])
    if workspace is None:
        vals = torch.empty((m, groups), dtype=torch.float32, device=x.device)
        idxs = torch.empty((m, groups), dtype=torch.int32, device=x.device)
    else:
        vals, idxs = workspace
    out = torch.empty((m,), dtype=torch.int64, device=x.device)

    _partial[(triton.cdiv(m, config["BM"]), groups)](
        x, w, suppress if suppress is not None else x, vals, idxs,
        m, n, k, x.stride(0), w.stride(0),
        0 if suppress is None else suppress.numel(),
        **config,
    )  # fmt: skip
    _finish[(m,)](
        vals, idxs, out, groups, triton.next_power_of_2(groups), num_warps=4
    )
    return out
