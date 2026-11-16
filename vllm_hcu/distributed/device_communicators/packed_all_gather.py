# SPDX-License-Identifier: Apache-2.0
"""Byte-exact Step5 DP dispatch packing for the uniform M=64 decode path.

All peers must supply the same tensor count, dtypes, trailing dimensions and
uniform row sizes, as required by the underlying NCCL all-gather. Local strides
are normalized instead of choosing a different collective on one peer.
"""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _pack4(A, B, C, D, P, N0: tl.constexpr, N1: tl.constexpr,
           N2: tl.constexpr, N3: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    v0 = tl.load(A + i, i < N0, other=0)
    v1 = tl.load(B + i - N0, (i >= N0) & (i < N0 + N1), other=0)
    v2 = tl.load(C + i - N0 - N1,
                 (i >= N0 + N1) & (i < N0 + N1 + N2), other=0)
    v3 = tl.load(D + i - N0 - N1 - N2,
                 (i >= N0 + N1 + N2) & (i < N0 + N1 + N2 + N3), other=0)
    value = tl.where(i < N0, v0, tl.where(i < N0 + N1, v1,
                     tl.where(i < N0 + N1 + N2, v2, v3)))
    tl.store(P + i, value, i < N0 + N1 + N2 + N3)


@triton.jit
def _unpack4(P, A, B, C, D, N0: tl.constexpr, N1: tl.constexpr,
             N2: tl.constexpr, N3: tl.constexpr, WORLD: tl.constexpr,
             BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    stride: tl.constexpr = N0 + N1 + N2 + N3
    rank, offset = i // stride, i % stride
    valid = i < WORLD * stride
    value = tl.load(P + i, valid, other=0)
    tl.store(A + rank * N0 + offset, value, valid & (offset < N0))
    tl.store(B + rank * N1 + offset - N0, value,
             valid & (offset >= N0) & (offset < N0 + N1))
    tl.store(C + rank * N2 + offset - N0 - N1, value,
             valid & (offset >= N0 + N1) & (offset < N0 + N1 + N2))
    tl.store(D + rank * N3 + offset - N0 - N1 - N2, value,
             valid & (offset >= N0 + N1 + N2) & (offset < stride))


def packed_step5_all_gatherv(comm, inputs, original, dim=0, sizes=None):
    """Return four tensors after one collective, or None for other layouts."""
    if dim != 0 or not isinstance(inputs, list) or len(inputs) != 4:
        return None
    x, weights, ids, scale = inputs
    world = comm.world_size
    if not 2 <= world <= 16 or x.ndim != 2 or x.shape != (64, 4096):
        return None
    if x.element_size() != 1 or weights.dtype != torch.float32:
        return None
    if ids.dtype not in (torch.int32, torch.int64) or scale.dtype != torch.float32:
        return None
    if weights.shape != (64, 8) or ids.shape != (64, 8) or scale.shape != (64, 1):
        return None
    if sizes is not None and (len(sizes) != world or any(s != 64 for s in sizes)):
        return None
    if not x.is_cuda or any(t.device != x.device for t in inputs):
        raise ValueError("packed Step5 all-gather requires one GPU device per rank")
    inputs = [tensor.contiguous() for tensor in inputs]
    counts = tuple(t.numel() * t.element_size() for t in inputs)
    total = sum(counts)
    packed = torch.empty(total, dtype=torch.uint8, device=x.device)
    _pack4[(triton.cdiv(total, 4096),)](
        *(t.view(torch.uint8).reshape(-1) for t in inputs), packed, *counts,
        BLOCK=4096, num_warps=4,
    )
    gathered = original(comm, packed, dim=0, sizes=None)
    if gathered.dtype != torch.uint8 or gathered.numel() != world * total:
        raise RuntimeError("packed Step5 all-gather returned an invalid byte buffer")
    outputs = [
        torch.empty((world * 64, *t.shape[1:]), dtype=t.dtype, device=t.device)
        for t in inputs
    ]
    _unpack4[(triton.cdiv(world * total, 4096),)](
        gathered, *(t.view(torch.uint8).reshape(-1) for t in outputs),
        *counts, WORLD=world, BLOCK=4096, num_warps=4,
    )
    return outputs
