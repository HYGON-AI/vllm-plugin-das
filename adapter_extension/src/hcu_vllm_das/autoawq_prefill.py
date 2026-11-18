# SPDX-License-Identifier: Apache-2.0
"""Convert validated LightOp packed storage to Core's ordinary KN matmul input."""

from __future__ import annotations

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _packed_to_kn(
    weight,
    scales_zeros,
    output,
    n: tl.constexpr,
    k: tl.constexpr,
    tile_k: tl.constexpr,
    block_n: tl.constexpr,
):
    # Adjacent programs traverse contiguous N tiles in the packed backing.
    # Keeping K fastest instead scatters adjacent programs across that backing.
    tile = tl.program_id(1)
    columns = tl.program_id(0) * block_n + tl.arange(0, block_n)
    words = tl.arange(0, tile_k // 8)
    offsets = (
        tile * n * (tile_k // 8)
        + columns[:, None] * (tile_k // 8)
        + words[None, :]
    )
    packed = tl.load(weight + offsets, columns[:, None] < n, 0)
    within_word = tl.arange(0, 8)
    # LightOp packs the K axis in the same nibble order as the checkpoint.
    shifts = 4 * (within_word // 2 + (within_word % 2) * 4)
    values = (packed[:, :, None] >> shifts[None, None, :]) & 15
    groups = (tile * tile_k + words * 8) // 128
    parameters = tl.load(
        scales_zeros + columns[:, None] * (k // 128) + groups[None, :],
        columns[:, None] < n,
        0,
    )
    zeros = (
        (parameters & 65535)
        .to(tl.uint16)
        .to(tl.float16, bitcast=True)
        .to(tl.float32)
        - 64
    )
    scales = (
        (parameters >> 16)
        .to(tl.uint16)
        .to(tl.float16, bitcast=True)
        .to(tl.float32)
    )
    values = (
        (values.to(tl.float32) - zeros[:, :, None]) * scales[:, :, None]
    ).to(tl.float16)
    values = values.reshape((block_n, tile_k))
    rows = tile * tile_k + tl.arange(0, tile_k)
    tl.store(
        output + rows[:, None] * n + columns[None, :],
        tl.trans(values),
        columns[None, :] < n,
    )


def dequantize_for_matmul(
    weight: torch.Tensor, scales_zeros: torch.Tensor, k: int
) -> torch.Tensor:
    """Return FP16 KN storage without retaining a second weight representation."""
    n = scales_zeros.shape[0]
    output = torch.empty((k, n), device=weight.device, dtype=torch.float16)
    tile_k = 256 if k % 256 == 0 else 128
    _packed_to_kn[(triton.cdiv(n, 64), k // tile_k)](
        weight, scales_zeros, output, n, k, tile_k, 64, num_warps=4
    )
    return output
