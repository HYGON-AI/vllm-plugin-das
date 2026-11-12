# SPDX-License-Identifier: Apache-2.0
"""HCU mHC post for V4.1 PCP; Core has no Triton post and TileLang is unsupported."""
import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _mhc_post_kernel(x, residual, post, comb, out, hidden,
                     sx_t, sx_h, sr_t, sr_i, sr_h, sp_t, sp_j,
                     sc_t, sc_i, sc_j, so_t, so_j, so_h,
                     HC: tl.constexpr, HC_BLOCK: tl.constexpr,
                     BLOCK_H: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    i = tl.arange(0, HC_BLOCK)
    hmask = h < hidden
    imask = i < HC
    xv = tl.load(x + t * sx_t + h * sx_h, mask=hmask, other=0.0).to(tl.float32)
    rv = tl.load(residual + t * sr_t + i[:, None] * sr_i + h[None, :] * sr_h,
                 mask=imask[:, None] & hmask[None, :], other=0.0).to(tl.float32)
    for j in tl.static_range(HC):
        c = tl.load(comb + t * sc_t + i * sc_i + j * sc_j, mask=imask, other=0.0)
        p = tl.load(post + t * sp_t + j * sp_j)
        acc = tl.sum(c.to(tl.float32)[:, None] * rv, 0) + p.to(tl.float32) * xv
        tl.store(out + t * so_t + j * so_j + h * so_h,
                 acc.to(out.dtype.element_ty), mask=hmask)


def mhc_post(x, residual, post_layer_mix, comb_res_mix):
    """out_j = post_j * x + sum_i comb_ij * residual_i, in FP32 like Core's Torch."""
    tokens, hc, hidden = residual.shape
    post = post_layer_mix.reshape(tokens, hc)
    if (x.shape != (tokens, hidden) or comb_res_mix.shape != (tokens, hc, hc)
            or residual.dtype not in (torch.bfloat16, torch.float16)):
        raise ValueError('HCU mHC post shape or dtype mismatch')
    out = torch.empty_like(residual)
    if tokens == 0:
        return out
    # Eight BF16 lanes per thread keep 16-byte loads; other HCU splits were ~10x slower.
    block_h = 1024
    _mhc_post_kernel[(tokens, triton.cdiv(hidden, block_h))](
        x, residual, post, comb_res_mix, out, hidden,
        *x.stride(), *residual.stride(), *post.stride(),
        *comb_res_mix.stride(), *out.stride(),
        HC=hc, HC_BLOCK=triton.next_power_of_2(hc), BLOCK_H=block_h, num_warps=2)
    return out
