# SPDX-License-Identifier: Apache-2.0
"""Apply the virtual zero-value attention sink to normalized FlashMLA output."""

import torch
import triton
import triton.language as tl


@triton.jit
def _scale_sink(
    Out,
    Lse,
    Sink,
    ROWS,
    HEADS: tl.constexpr,
    OQ: tl.constexpr,
    OH: tl.constexpr,
    LQ: tl.constexpr,
    LH: tl.constexpr,
    SH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    rowhead = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, head = rowhead // HEADS, rowhead % HEADS
    valid = rowhead < ROWS * HEADS
    lse = tl.load(Lse + row * LQ + head * LH, valid, other=0)
    sink = tl.load(Sink + head * SH, valid, other=0)
    weight = 1.0 / (1.0 + tl.exp(sink - lse))
    weight = tl.where(sink == -float("inf"), 1.0, weight)
    col = tl.arange(0, 512)
    ptr = Out + row[:, None].to(tl.int64) * OQ + head[:, None] * OH + col[None, :]
    out = tl.load(ptr, valid[:, None], other=0).to(tl.float32)
    # +inf is an explicit zero-output sink, even if the source contains NaNs.
    scaled = tl.where((sink == float("inf"))[:, None], 0.0, out * weight[:, None])
    tl.store(ptr, scaled, valid[:, None])


def apply_sink(output, lse, sink):
    with torch.cuda.device(output.device):
        _scale_sink[(triton.cdiv(output.shape[0] * output.shape[1], 8),)](
            output,
            lse,
            sink,
            output.shape[0],
            output.shape[1],
            output.stride(0),
            output.stride(1),
            lse.stride(0),
            lse.stride(1),
            sink.stride(0),
            8,
            num_warps=4,
        )
