# SPDX-License-Identifier: Apache-2.0

"""
vllm.model_executor.layers.fused_moe.utils: Optional, F, libdevice imports; int8 triton opt; expert_num_tokens in _int8_quantize
"""

PATCHES = [
(
"""
from math import prod

import torch

from vllm import _custom_ops as ops
""",
"""
from math import prod
from typing import Optional

import torch
import torch.nn.functional as F
from triton.language.extra import libdevice

from vllm import _custom_ops as ops
""",
),
(
"""
    return A, A_scale


def _int8_quantize(
""",
"""
    return A, A_scale


@triton.jit
def _per_token_quant_int8_one_kernel_opt(
    x_ptr,
    xq_ptr,
    scale_ptr,
    stride_x,
    stride_xq,
    N, 
    T_dim, 
    has_tokens_per_expert: tl.constexpr,
    tokens_per_expert_ptr, 
    BLOCK: tl.constexpr
):
    row_id = tl.program_id(0)

    if has_tokens_per_expert:
        e = row_id // T_dim
        t = row_id % T_dim
        
        num_valid_tokens_for_e = tl.load(tokens_per_expert_ptr + e)
        
        if t >= num_valid_tokens_for_e:
            return

    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(x_ptr + row_id * stride_x + cols, mask=mask,
                other=0.0).to(tl.float32)
    absmax = tl.maximum(tl.max(tl.abs(x)), 1e-10)
    scale_x = absmax / 127
    x_q = x * (127 / absmax)
    x_q = libdevice.nearbyint(x_q).to(tl.int8)

    tl.store(xq_ptr + row_id * stride_xq + cols, x_q, mask=mask)
    tl.store(scale_ptr + row_id, scale_x)

@triton.jit
def _per_token_quant_int8_kernel_opt(
    x_ptr,
    xq_ptr,
    scale_ptr,
    stride_x,
    stride_xq,
    N, 
    E_dim,
    T_dim,
    has_tokens_per_expert: tl.constexpr,
    tokens_per_expert_ptr, 
    BLOCK: tl.constexpr
):
    token_idx_start = tl.program_id(0)
    grid_size = tl.num_programs(0) 
    num_total_tokens = E_dim * T_dim

    for token_idx in range(token_idx_start, num_total_tokens, grid_size):
        
        is_valid_token = True
        if has_tokens_per_expert:
            e = token_idx // T_dim
            t = token_idx % T_dim
            
            num_valid_tokens_for_e = tl.load(tokens_per_expert_ptr + e)
            
            if t >= num_valid_tokens_for_e:
                is_valid_token = False 

        if is_valid_token:
            cols = tl.arange(0, BLOCK)
            mask = cols < N

            x = tl.load(x_ptr + token_idx * stride_x + cols, mask=mask,
                        other=0.0).to(tl.float32)
            absmax = tl.maximum(tl.max(tl.abs(x)), 1e-10)
            scale_x = absmax / 127
            x_q = x * (127 / absmax)
            x_q = libdevice.nearbyint(x_q).to(tl.int8)

            tl.store(xq_ptr + token_idx * stride_xq + cols, x_q, mask=mask)
            tl.store(scale_ptr + token_idx, scale_x)


def per_token_quant_int8_triton_opt(x: torch.Tensor,
                                    tokens_per_expert: Optional[torch.Tensor] = None):
    if x.dim() != 3:
        raise ValueError(f"Input must be 3D [E, T, H], but got {x.shape}")
    E, T, H = x.shape
    N = H
    
    x_q = torch.empty_like(x, device=x.device, dtype=torch.int8)
    scales = torch.empty(x.shape[:-1] + (1, ),
                         device=x.device,
                         dtype=torch.float32)
    BLOCK = triton.next_power_of_2(N)
    num_warps = min(max(BLOCK // 256, 1), 8)
    if T >= 4096:
        num_warps = 1
    
    num_tokens = E * T
    grid_opt = num_tokens
    if E == 16 and T >= 1024 :
        grid_opt = max(1, num_tokens // (T // 256))
        _per_token_quant_int8_kernel_opt[(grid_opt, )](
            x,
            x_q,
            scales,
            stride_x=x.stride(-2),
            stride_xq=x_q.stride(-2),
            N=N,
            E_dim=E,
            T_dim=T,
            has_tokens_per_expert=tokens_per_expert is not None,
            tokens_per_expert_ptr=tokens_per_expert,
            BLOCK=BLOCK,
            num_warps=num_warps,
            num_stages=1,
        )
    else:
        _per_token_quant_int8_one_kernel_opt[(grid_opt, )](
            x,
            x_q,
            scales,
            stride_x=x.stride(-2),
            stride_xq=x_q.stride(-2),
            N=N,
            T_dim=T,
            has_tokens_per_expert=tokens_per_expert is not None,
            tokens_per_expert_ptr=tokens_per_expert,
            BLOCK=BLOCK,
            num_warps=num_warps,
            num_stages=1,
        )
    return x_q, scales


def _int8_quantize(
""",
),
(
"""
def _int8_quantize(
    A: torch.Tensor,
    A_scale: torch.Tensor | None,
    per_act_token: bool,
    block_shape: list[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
""",
"""
def _int8_quantize(
    A: torch.Tensor,
    A_scale: torch.Tensor | None,
    per_act_token: bool,
    block_shape: list[int] | None = None,
    expert_num_tokens: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
""",
),
(
"""
    if block_shape is None:
        assert per_act_token, "int8 quantization only supports block or channel-wise"
        A, A_scale = per_token_quant_int8(A)
""",
"""
    if block_shape is None or per_act_token:
        assert per_act_token, "int8 quantization only supports block or channel-wise"
        if expert_num_tokens is None:
            A, A_scale = per_token_quant_int8(A)
        else:
            A, A_scale = per_token_quant_int8_triton_opt(A, expert_num_tokens)
""",
),
]
