# SPDX-License-Identifier: Apache-2.0
"""Preserve a stable reduction order for HYV4-VL sparse attention."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _canonicalize_topk_kernel(
    indices,
    row_stride,
    col_stride,
    width: tl.constexpr,
    block: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, block)
    ptr = indices + row * row_stride
    values = tl.load(ptr + col * col_stride, col < width, other=-1)
    # Keep invalid entries at the end, as required by the native short-row
    # selector. Already ordered rows (including all short decode rows) need
    # no sort or writeback.
    keys = tl.where(values >= 0, values, 2147483647)
    previous = tl.load(
        ptr + (col - 1) * col_stride,
        (col > 0) & (col < width),
        other=-1,
    )
    previous_keys = tl.where(previous >= 0, previous, 2147483647)
    inverted = (col > 0) & (col < width) & (keys < previous_keys)
    if tl.sum(inverted.to(tl.int32), axis=0) > 0:
        ordered = tl.sort(keys, descending=False)
        ordered = tl.where(ordered == 2147483647, -1, ordered)
        tl.store(ptr + col * col_stride, ordered, col < width)


def canonicalize_topk_indices(indices: torch.Tensor, num_tokens: int) -> None:
    """Sort only active rows in place, retaining the selected token multiset.

    Native Top-K can emit the same set in different orders above index_topk.
    BF16 sparse-attention reductions then round differently, and the HYV4
    language model can amplify those differences. Logical token order avoids
    that variation without changing scores or the native selection policy.
    The backing buffer includes inactive rows even during one-token decode;
    leave those rows untouched. No tensor values are read on the host.
    """
    if indices.ndim != 2 or indices.dtype != torch.int32:
        raise ValueError("HYV4-VL sparse indices must be a 2D int32 tensor")
    if not 0 <= num_tokens <= indices.shape[0]:
        raise ValueError("active sparse-index rows exceed the backing buffer")
    width = indices.shape[1]
    if num_tokens == 0 or width == 0:
        return
    _canonicalize_topk_kernel[(num_tokens,)](
        indices,
        indices.stride(0),
        indices.stride(1),
        width,
        triton.next_power_of_2(width),
        num_warps=4,
    )
