# SPDX-License-Identifier: Apache-2.0
"""One launch for packed-provider padding; Core still owns route selection."""

import torch
import triton
import triton.language as tl


@triton.jit
def _prepare_provider_inputs(
    X, W, I, Y, V, J,
    H: tl.constexpr, K: tl.constexpr,
    WS0: tl.constexpr, WS1: tl.constexpr,
    IS0: tl.constexpr, IS1: tl.constexpr,
    BH: tl.constexpr, BK: tl.constexpr,
):
    row = tl.program_id(0)
    slots = tl.arange(0, BK)
    ids = tl.load(I + row * IS0 + slots * IS1, slots < K, other=-1)
    valid = (slots < K) & (ids >= 0)
    # At most K-padding candidates in [0,K) are occupied. Assign the nth
    # unused candidate to the nth padding slot, keeping every active id.
    occupied = tl.sum(
        ((ids[:, None] == slots[None, :]) & valid[:, None]).to(tl.int32), 0
    ) > 0
    unused = (slots < K) & ~occupied
    unused_rank = tl.cumsum(unused.to(tl.int32)) - 1
    padding_rank = tl.cumsum(((slots < K) & ~valid).to(tl.int32)) - 1
    padding_id = tl.min(
        tl.where(
            (padding_rank[:, None] == unused_rank[None, :]) & unused[None, :],
            slots[None, :],
            BK,
        ),
        1,
    )
    weights = tl.load(W + row * WS0 + slots * WS1, slots < K, other=0)
    tl.store(J + row * K + slots, tl.where(valid, ids, padding_id), slots < K)
    tl.store(V + row * K + slots, tl.where(valid, weights, 0), slots < K)
    cols = tl.arange(0, BH)
    x = tl.load(X + row * H + cols, cols < H, other=0)
    # Multiplying an unused NaN/Inf row by zero would still poison the GEMM.
    clean_x = tl.where(tl.sum(valid.to(tl.int32), 0) > 0, x, 0)
    tl.store(Y + row * H + cols, clean_x, cols < H)


def prepare_provider_inputs(x, weights, ids):
    out = torch.empty_like(x)
    routes = torch.empty_like(weights, memory_format=torch.contiguous_format)
    indices = torch.empty_like(ids, memory_format=torch.contiguous_format)
    if x.shape[0]:
        _prepare_provider_inputs[(x.shape[0],)](
            x, weights, ids, out, routes, indices,
            x.shape[1], ids.shape[1], *weights.stride(), *ids.stride(),
            triton.next_power_of_2(x.shape[1]),
            triton.next_power_of_2(ids.shape[1]), num_warps=4,
        )
    return out, routes, indices
