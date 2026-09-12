# SPDX-License-Identifier: Apache-2.0
"""Load-time integer decoding; execution belongs to current SlimQuant.

This module intentionally defines no private linear/MoE kernel or selector.
The current linear owner stores signed INT8; the current MoE owner stores
high-first signed INT4, with its existing layout and fallback lifecycle.
"""
import torch


def unpack_int4(packed: torch.Tensor) -> torch.Tensor:
    """Expand high-first two's-complement nibbles into signed INT8 values."""
    if packed.dtype not in (torch.int8, torch.uint8):
        raise ValueError("HYV4 packed INT4 must have byte storage")
    unsigned = packed.view(torch.uint8)
    pair = torch.stack((unsigned >> 4, unsigned & 15), dim=-1).to(torch.int16)
    return torch.where(pair >= 8, pair - 16, pair).to(torch.int8).flatten(-2)
