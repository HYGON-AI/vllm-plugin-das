# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Exact BF16 softcap by table lookup.

bfloat16 has only 65536 bit patterns, so the stock expression
``tanh(x / soft_cap) * soft_cap`` can be tabulated for every one of them and
applied by lookup instead of recomputed. Because the table is built with that
same expression, the result is bit-identical by construction rather than
approximately equal.

The table costs 128 KiB and bakes in ``soft_cap``, so a caller must check that
value, and the scale applied after the softcap, before installing.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_BLOCK = 2048


def build_lut(device: torch.device, cap: float) -> torch.Tensor:
    """Tabulate the stock softcap for every bfloat16 bit pattern.

    Entry ``i`` holds the result for the bfloat16 whose bit pattern is ``i``
    read as unsigned, which is the index ``_lookup`` computes.
    """
    bits = torch.arange(65536, dtype=torch.int32, device=device).to(torch.int16)
    values = bits.view(torch.bfloat16)
    return (torch.tanh(values / cap) * cap).contiguous()


@triton.jit
def _lookup(X, LUT, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    # Index by the raw bit pattern: a negative bfloat16 has its sign bit set,
    # so reading the pattern as unsigned lands in the table's upper half,
    # which build_lut filled from the same pattern.
    index = tl.load(X + offsets, mask, 0).to(tl.uint16, bitcast=True).to(tl.int32)
    tl.store(X + offsets, tl.load(LUT + index, cache_modifier=".ca"), mask)


def apply_lut(x: torch.Tensor, lut: torch.Tensor) -> torch.Tensor:
    """Softcap ``x`` in place. Requires contiguous bfloat16."""
    assert x.dtype == torch.bfloat16 and x.is_contiguous()
    _lookup[(triton.cdiv(x.numel(), _BLOCK),)](
        x, lut, x.numel(), BLOCK=_BLOCK, num_warps=4
    )
    return x
