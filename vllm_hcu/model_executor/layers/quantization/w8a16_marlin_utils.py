# SPDX-License-Identifier: Apache-2.0
"""Dequantize DeepGEMM marlin-layout W8A16 weights straight to row-major W16.

``deepgemm.w8a16_marlin_weight`` stores an int8 [N, K] weight as contiguous
2 KiB tiles ``[K/128, N/16][4][16][32]`` (k-block, n-block, k-subblock, row,
k). DeepGEMM's large-M path (dequant + rocBLAS) needs a row-major W16 weight
and only knows the raw layout, so marlin weights are dequantized here: one
pass over the tiles instead of un-shuffle + dequant.

Not cached: in a full forward pass the W16 weights of all layers far exceed
any reasonable cache budget, and layers are visited cyclically, so an LRU
never hits (measured on Qwen3.5 shapes) while still pinning its budget.
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _w8a16_marlin_dequant_kernel(w_ptr, s_ptr, out_ptr, N, K):
    # One program per 16 x 128 marlin tile.
    nb = tl.program_id(0)
    kb = tl.program_id(1)
    rows = tl.arange(0, 16)
    cols = tl.arange(0, 128)
    src = ((kb * (N // 16) + nb) * 2048 + ((cols // 32) * 512)[None, :]
           + rows[:, None] * 32 + (cols % 32)[None, :])
    w = tl.load(w_ptr + src).to(tl.float32)
    s = tl.load(s_ptr + nb * 16 + rows).to(tl.float32)
    out = (w * s[:, None]).to(out_ptr.dtype.element_ty)
    dst = (nb * 16 + rows)[:, None] * K + (kb * 128 + cols)[None, :]
    tl.store(out_ptr + dst, out)


def w8a16_marlin_dequant(w: torch.Tensor, scale: torch.Tensor,
                         dtype: torch.dtype) -> torch.Tensor:
    """Marlin int8 [N, K] + per-row scale [N] -> row-major ``dtype`` [N, K].

    Bit-exact with ``deepgemm.op.w8a16_dequant_weight{,_bf16}`` on the raw
    weight (fp32 multiply, one rounding).
    """
    n, k = w.shape
    assert n % 16 == 0 and k % 128 == 0 and w.is_contiguous()
    out = torch.empty((n, k), dtype=dtype, device=w.device)
    _w8a16_marlin_dequant_kernel[(n // 16, k // 128)](
        w, scale.contiguous().view(n), out, n, k)
    return out
