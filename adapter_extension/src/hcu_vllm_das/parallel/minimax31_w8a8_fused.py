# SPDX-License-Identifier: Apache-2.0
"""MiniMax VERIFY W8A8 activation producers for symmetric per-token scales."""
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from .minimax31_verify_int8 import VerifyTransposeInt8Kernel, supports_w8a8_fusion


@triton.jit
def _swiglu_tile_amax(
    gate_up, activation, amax, N: tl.constexpr, BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr, alpha, beta, limit, HAS_LIMIT: tl.constexpr,
):
    tile = tl.program_id(1)
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    columns = tile * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = columns[None, :] < N
    offsets = rows[:, None] + columns[None, :] * 128
    gate = tl.load(gate_up + offsets, mask, 0.0).to(tl.float32)
    up = tl.load(gate_up + offsets + N * 128, mask, 0.0).to(tl.float32)
    if HAS_LIMIT:
        gate = tl.minimum(gate, limit)
        up = tl.minimum(tl.maximum(up, -limit), limit)
    value = (gate * tl.sigmoid(alpha * gate) * (up + beta)).to(tl.bfloat16)
    tl.store(activation + rows[:, None] * N + columns[None, :], value, mask)
    tl.store(amax + tile * 128 + rows, tl.max(tl.abs(value.to(tl.float32)), axis=1))


@triton.jit
def _quantize_with_tile_amax(
    activation, partials, output, scales, N: tl.constexpr, TILES: tl.constexpr,
    BLOCK: tl.constexpr, BLOCK_TILES: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.arange(0, BLOCK_TILES)
    amax = tl.max(tl.load(partials + tile * 128 + row, tile < TILES, 0.0), axis=0)
    inv = tl.div_rn(127.0, tl.where(amax == 0.0, 1.0, amax))
    i = tl.arange(0, BLOCK)
    value = tl.load(activation + row * N + i, i < N, 0.0).to(tl.float32)
    tl.store(output + row * N + i, libdevice.rint(value * inv).to(tl.int8), i < N)
    tl.store(scales + row, amax / 127.0)


def swiglu_oai_int8_tiled(gate_up, alpha, beta, limit, *, block_m=32, block_n=64):
    n = gate_up.shape[1] // 2
    tiles = triton.cdiv(n, block_n)
    activation = torch.empty((128, n), device=gate_up.device, dtype=torch.bfloat16)
    amax = torch.empty((tiles, 128), device=gate_up.device, dtype=torch.float32)
    q = torch.empty_like(activation, dtype=torch.int8)
    scales = torch.empty((128, 1), device=gate_up.device, dtype=torch.float32)
    _swiglu_tile_amax[(128 // block_m, tiles)](
        gate_up, activation, amax, n, block_m, block_n, float(alpha), float(beta),
        0.0 if limit is None else float(limit), limit is not None, num_warps=4,
    )
    _quantize_with_tile_amax[(128,)](
        activation, amax, q, scales, n, tiles, triton.next_power_of_2(n),
        triton.next_power_of_2(tiles), num_warps=4,
    )
    return q, scales


def _verify_kernel(proj):
    kernel = getattr(getattr(proj, "scheme", None), "kernel", None)
    return kernel if type(kernel) is VerifyTransposeInt8Kernel else None


def shared_w8a8_kernels(mlp, x):
    """Check both projection contracts before bypassing the two linear forwards."""
    up, down = mlp.gate_up_proj, mlp.down_proj
    up_kernel, down_kernel = _verify_kernel(up), _verify_kernel(down)
    if (up_kernel is None or down_kernel is None or x.ndim != 2
            or up.tp_size != 1 or up.gather_output or up.bias is not None
            or down.tp_size != 1 or not down.input_is_parallel or down.bias is not None
            or not up_kernel.supports_input(up, x) or not supports_w8a8_fusion(x.device)
            or not down_kernel.supports_shape(down, (x.shape[0], up.weight.shape[1] // 2),
                                              x.dtype, x.device)):
        return None
    return up_kernel, down_kernel


def project_shared_gate_up(mlp, x, kernel):
    return kernel.apply_weights_strided(mlp.gate_up_proj, x)


def project_shared_down(mlp, gate_up, kernel, *, row_major=True):
    down = mlp.down_proj
    if gate_up.is_contiguous():
        from lightop.activation import swiglu_oai_int8_quant_contiguous

        q, scales = swiglu_oai_int8_quant_contiguous(
            gate_up, mlp.swiglu_alpha, mlp.swiglu_beta, mlp.swiglu_limit,
            vllm_compatible=True, block_threads=512,
        )
    else:
        q, scales = swiglu_oai_int8_tiled(
            gate_up, mlp.swiglu_alpha, mlp.swiglu_beta, mlp.swiglu_limit,
        )
    if row_major:
        from lightop.gemmopt import gemm_w8a8_smooth

        # LightOP accepts the existing KN weight view and writes [M, N] directly.
        ok, output = gemm_w8a8_smooth(
            q, down.weight, scales, down.weight_scale, None, gate_up.dtype,
        )
        if ok:
            if (not isinstance(output, torch.Tensor)
                    or output.shape != (q.shape[0], down.weight.shape[1])
                    or output.dtype != gate_up.dtype or output.device != gate_up.device
                    or not output.is_contiguous()):
                raise RuntimeError("unexpected LightOP shared down GEMM return ABI")
            return output
    return kernel.apply_prequantized(down, q, scales, gate_up.dtype, row_major=row_major)
