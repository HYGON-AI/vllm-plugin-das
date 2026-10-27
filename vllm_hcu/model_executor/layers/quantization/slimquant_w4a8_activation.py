# SPDX-License-Identifier: Apache-2.0
"""Clamp/SiLU/INT8 quantization for the padded DeepGEMM W4A8 workspace."""

import torch
from triton.language.extra import libdevice
from vllm.triton_utils import tl, triton


@triton.jit
def _masked_clamp_quant(X, Y, S, E, n_rows, D: tl.constexpr,
                       LIMIT: tl.constexpr, BLOCK: tl.constexpr,
                       ROW_TILE: tl.constexpr):
    # Promote before workspace offsets can exceed 2**31 elements.
    row = (tl.program_id(0) * ROW_TILE + tl.arange(0, ROW_TILE)).to(tl.int64)
    valid = tl.load(E + row, row < n_rows, other=-1) >= 0
    # DeepGEMM also skips these rows: their GEMM output is uninitialized.
    if tl.sum(valid.to(tl.int32), 0) > 0:
        col = tl.arange(0, BLOCK)
        mask = valid[:, None] & (col[None, :] < D)
        gate = tl.load(X + row[:, None] * 2 * D + col[None, :],
                       mask, other=0).to(tl.float32)
        up = tl.load(X + row[:, None] * 2 * D + D + col[None, :],
                     mask, other=0).to(tl.float32)
        # Preserve LightOp's input-dtype rounding at each activation stage.
        gate = tl.minimum(gate, LIMIT).to(X.dtype.element_ty).to(tl.float32)
        up = tl.minimum(tl.maximum(up, -LIMIT), LIMIT).to(
            X.dtype.element_ty).to(tl.float32)
        activated = tl.div_rn(
            gate, 1.0 + tl.exp2(-gate * 1.4426950408889634)
        ).to(X.dtype.element_ty).to(tl.float32)
        value = (activated * up).to(X.dtype.element_ty).to(tl.float32)
        amax = tl.maximum(tl.max(tl.abs(value), 1), 1e-10)
        quant = libdevice.nearbyint(value * tl.div_rn(127.0, amax[:, None]))
        quant = tl.minimum(tl.maximum(quant, -128.0), 127.0).to(tl.int8)
        tl.store(Y + row[:, None] * D + col[None, :], quant, mask)
        tl.store(S + row, amax / 127.0, valid)


def masked_silu_mul_clamp_quant(input, *, limit, output, expert_ids):
    """Quantize valid contiguous rows; invalid rows remain unused by GEMM."""
    rows, width = input.shape
    scales = torch.empty((rows, 1), dtype=torch.float32, device=input.device)
    if rows:
        _masked_clamp_quant[(triton.cdiv(rows, 8),)](
            input, output, scales, expert_ids, rows, width // 2, limit,
            triton.next_power_of_2(width // 2), 8,
            num_warps=4, enable_fp_fusion=False,
        )
    return output, scales
