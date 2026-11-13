# SPDX-License-Identifier: Apache-2.0
# HCU gated RMSNorm with strided-z support, forked from
# vllm/model_executor/layers/fla/ops/layernorm_guard.py::layer_norm_fwd_kernel
#
# WHY: In qwen3_next.Qwen3NextGatedDeltaNet.forward, `z` comes from
# torch.split(mixed_qkvz, ...)[3] then z.reshape(L, -1, head_v).
# The final z.reshape(-1, head_v) that the norm layer sees must materialize
# a contiguous copy (~4.96µs on qwen3.5 int8wo decode) because the split
# leaves z with stride (row_stride, head_v, 1) where row_stride > num_v*head_v.
# This kernel loads z directly with (stride_z_row_outer, stride_z_row_inner)
# — the two-level "M dimension" stride of the un-cloned z view — so the
# clone can be skipped.
#
# For the [L, num_v, head_v] logical view flattened to [L*num_v, head_v]:
#   linearized row index r  -> outer = r // num_v_per_l, inner = r % num_v_per_l
#   address = z_base + outer*stride_outer + inner*stride_inner + col

import torch

from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import cdiv, next_power_of_2
from vllm.utils.platform_utils import num_compute_units


@triton.jit
def hcu_gated_rmsnorm_strided_z_kernel(
    X,           # ptr to input [M, N], contiguous last dim
    Y,           # ptr to output [M, N], contiguous last dim
    W,           # ptr to weight [N]
    Z,           # ptr to z view; addressed via (outer, inner, col) below
    stride_x_row,          # X row stride
    stride_y_row,          # Y row stride
    stride_z_outer,        # Z outer stride: rows_per_outer sub-blocks
    stride_z_inner,        # Z inner stride within one outer block
    rows_per_outer,        # M axis: rows per outer chunk  (= num_v_per_l)
    M,                     # number of rows in X/Y  (== L * rows_per_outer)
    N: tl.constexpr,       # feature dim
    eps,
    BLOCK_N: tl.constexpr,
    ROWS_PER_BLOCK: tl.constexpr,
    NORM_BEFORE_GATE: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    row_start = tl.program_id(0) * ROWS_PER_BLOCK
    rows = row_start + tl.arange(0, ROWS_PER_BLOCK)
    cols = tl.arange(0, BLOCK_N)

    row_mask = rows[:, None] < M
    col_mask = cols[None, :] < N
    mask = row_mask & col_mask

    # Load X (contiguous last dim)
    X_base = X + rows[:, None] * stride_x_row + cols[None, :]
    x = tl.load(X_base, mask=mask, other=0.0).to(tl.float32)

    # Decode Z address from (row, col) via outer/inner
    outer_idx = rows // rows_per_outer   # [ROWS_PER_BLOCK]
    inner_idx = rows % rows_per_outer    # [ROWS_PER_BLOCK]
    z_row_offset = (
        outer_idx[:, None] * stride_z_outer
        + inner_idx[:, None] * stride_z_inner
    )
    Z_base = Z + z_row_offset + cols[None, :]

    if not NORM_BEFORE_GATE:
        z = tl.load(Z_base, mask=mask, other=0.0).to(tl.float32)
        if ACTIVATION == "swish" or ACTIVATION == "silu":
            x = x * z * tl.sigmoid(z)
        elif ACTIVATION == "sigmoid":
            x = x * tl.sigmoid(z)

    # RMSNorm
    xbar = tl.where(mask, x, 0.0)
    var = tl.sum(xbar * xbar, axis=1) / N   # [ROWS_PER_BLOCK]
    rstd = tl.rsqrt(var + eps)              # [ROWS_PER_BLOCK]

    w = tl.load(W + cols, mask=cols < N, other=0.0).to(tl.float32)
    y = x * rstd[:, None] * w[None, :]

    if NORM_BEFORE_GATE:
        z = tl.load(Z_base, mask=mask, other=0.0).to(tl.float32)
        if ACTIVATION == "swish" or ACTIVATION == "silu":
            y = y * z * tl.sigmoid(z)
        elif ACTIVATION == "sigmoid":
            y = y * tl.sigmoid(z)

    Y_base = Y + rows[:, None] * stride_y_row + cols[None, :]
    tl.store(Y_base, y, mask=mask)


def _calc_rows_per_block(M: int, device: torch.device) -> int:
    sm_count = num_compute_units(device.index)
    rows_per_block = next_power_of_2(cdiv(M, 2 * sm_count))
    rows_per_block = min(rows_per_block, 4)
    return max(rows_per_block, 1)


def hcu_gated_rmsnorm_strided_z(
    x: torch.Tensor,           # [M, N], contiguous last dim
    weight: torch.Tensor,      # [N]
    z_view: torch.Tensor,      # [L, num_v, N]; may be non-contiguous
    eps: float,
    norm_before_gate: bool = True,
    activation: str = "swish",
    out: torch.Tensor = None,
) -> torch.Tensor:
    """Gated RMSNorm that accepts a strided z view directly, skipping the
    contiguous copy that the stock fla wrapper would trigger.

    z_view shape [L, num_v, N] is flattened to [M=L*num_v, N] logically,
    without materializing a contiguous tensor.
    """
    assert x.stride(-1) == 1
    assert z_view.stride(-1) == 1
    assert x.dim() == 2
    assert z_view.dim() == 3
    M, N = x.shape
    L, num_v, Nz = z_view.shape
    assert Nz == N
    assert L * num_v == M
    assert weight.shape == (N,)

    if out is None:
        out = torch.empty_like(x)
    assert out.shape == x.shape
    assert out.stride(-1) == 1

    # The kernel address for row r in the flattened view is:
    #   outer = r // num_v, inner = r % num_v
    #   base = outer * z_view.stride(0) + inner * z_view.stride(1)
    stride_z_outer = z_view.stride(0)
    stride_z_inner = z_view.stride(1)
    rows_per_outer = num_v

    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    if N > BLOCK_N:
        raise RuntimeError(
            "hcu_gated_rmsnorm_strided_z doesn't support feature dim >= 64KB."
        )
    num_warps = min(max(BLOCK_N // 256, 1), 8)
    rows_per_block = _calc_rows_per_block(M, x.device)
    grid = (cdiv(M, rows_per_block),)

    hcu_gated_rmsnorm_strided_z_kernel[grid](
        x,
        out,
        weight,
        z_view,
        x.stride(0),
        out.stride(0),
        stride_z_outer,
        stride_z_inner,
        rows_per_outer,
        M,
        N,
        eps,
        BLOCK_N=BLOCK_N,
        ROWS_PER_BLOCK=rows_per_block,
        NORM_BEFORE_GATE=norm_before_gate,
        ACTIVATION=activation,
        num_warps=num_warps,
    )
    return out
