"""Small-row BF16 compressor projection with FP32 accumulation and output."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _compressor_projection_kernel(
    hidden, weight, output,
    rows: tl.constexpr, columns: tl.constexpr, inner: tl.constexpr,
    hidden_stride: tl.constexpr, weight_stride0: tl.constexpr, weight_stride1: tl.constexpr,
    block_rows: tl.constexpr, block_columns: tl.constexpr, block_inner: tl.constexpr,
):
    row = tl.program_id(0) * block_rows + tl.arange(0, block_rows)
    column = tl.program_id(1) * block_columns + tl.arange(0, block_columns)
    offsets = tl.arange(0, block_inner)
    accumulator = tl.zeros((block_rows, block_columns), tl.float32)
    for begin in range(tl.cdiv(inner, block_inner)):
        reduction = begin * block_inner + offsets
        activations = tl.load(
            hidden + row[:, None] * hidden_stride + reduction[None, :],
            (row[:, None] < rows) & (reduction[None, :] < inner), other=0,
        )
        weights = tl.load(
            weight + reduction[:, None] * weight_stride0 + column[None, :] * weight_stride1,
            (reduction[:, None] < inner) & (column[None, :] < columns), other=0,
        )
        accumulator = tl.dot(activations, weights, accumulator)
    tl.store(
        output + row[:, None] * columns + column[None, :], accumulator,
        (row[:, None] < rows) & (column[None, :] < columns),
    )


def try_compressor_projection(hidden: torch.Tensor, weight: torch.Tensor) -> torch.Tensor | None:
    if (
        hidden.ndim != 2 or weight.ndim != 2
        or not 1 <= hidden.shape[0] <= 256 or hidden.shape[1] != 5120
        or hidden.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16
        or hidden.device.type != "cuda" or weight.device != hidden.device
        or hidden.stride(1) != 1 or torch.version.hip is None
    ):
        return None
    right = weight if weight.shape[0] == hidden.shape[1] else weight.T
    if right.shape != (5120, 1024) or 1 not in right.stride():
        return None
    if torch.cuda.get_device_properties(hidden.device).gcnArchName.split(":")[0] != "gfx938":
        return None
    result = torch.empty((hidden.shape[0], right.shape[1]), device=hidden.device, dtype=torch.float32)
    _compressor_projection_kernel[(triton.cdiv(hidden.shape[0], 16), triton.cdiv(right.shape[1], 64))](
        hidden, right, result, hidden.shape[0], right.shape[1], hidden.shape[1],
        hidden.stride(0), right.stride(0), right.stride(1), 16, 64, 64,
        num_warps=4, num_stages=2,
    )
    return result
