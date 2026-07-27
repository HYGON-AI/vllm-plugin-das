# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Expert-aware per-token INT8 Triton quantization for HCU."""

from __future__ import annotations

import torch
from triton.language.extra import libdevice

from vllm.triton_utils import tl, triton


@triton.jit
def _per_token_quant_int8_one_kernel(
    x_ptr,
    xq_ptr,
    scale_ptr,
    stride_x,
    stride_xq,
    n_cols,
    tokens_per_expert,
    max_tokens,
    has_tokens_per_expert: tl.constexpr,
    block: tl.constexpr,
):
    row = tl.program_id(0)
    if has_tokens_per_expert:
        expert = row // max_tokens
        token = row % max_tokens
        if token >= tl.load(tokens_per_expert + expert):
            return
    cols = tl.arange(0, block)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    absmax = tl.maximum(tl.max(tl.abs(x)), 1e-10)
    scale = absmax / 127.0
    quant = libdevice.nearbyint(x * (127.0 / absmax)).to(tl.int8)
    tl.store(xq_ptr + row * stride_xq + cols, quant, mask=mask)
    tl.store(scale_ptr + row, scale)


@triton.jit
def _per_token_quant_int8_grid_stride_kernel(
    x_ptr,
    xq_ptr,
    scale_ptr,
    stride_x,
    stride_xq,
    n_cols,
    num_experts,
    max_tokens,
    tokens_per_expert,
    has_tokens_per_expert: tl.constexpr,
    block: tl.constexpr,
):
    start = tl.program_id(0)
    grid_size = tl.num_programs(0)
    total = num_experts * max_tokens
    for row in range(start, total, grid_size):
        valid = True
        if has_tokens_per_expert:
            expert = row // max_tokens
            token = row % max_tokens
            valid = token < tl.load(tokens_per_expert + expert)
        if valid:
            cols = tl.arange(0, block)
            mask = cols < n_cols
            x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
            absmax = tl.maximum(tl.max(tl.abs(x)), 1e-10)
            scale = absmax / 127.0
            quant = libdevice.nearbyint(x * (127.0 / absmax)).to(tl.int8)
            tl.store(xq_ptr + row * stride_xq + cols, quant, mask=mask)
            tl.store(scale_ptr + row, scale)


def per_token_quant_int8(
    x: torch.Tensor,
    tokens_per_expert: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if x.dim() != 3:
        raise ValueError(f"HCU expert-aware INT8 quantization expects [E,T,H], got {x.shape}")
    num_experts, max_tokens, hidden = x.shape
    if tokens_per_expert is not None:
        if tokens_per_expert.numel() != num_experts:
            raise ValueError(
                "tokens_per_expert length must equal the number of experts, "
                f"got {tokens_per_expert.numel()} and {num_experts}"
            )
        if tokens_per_expert.device != x.device:
            raise ValueError("tokens_per_expert must be on the same device as x")
    xq = torch.empty_like(x, dtype=torch.int8)
    scales = torch.empty(x.shape[:-1] + (1,), device=x.device, dtype=torch.float32)
    block = triton.next_power_of_2(hidden)
    num_warps = min(max(block // 256, 1), 8)
    if max_tokens >= 4096:
        num_warps = 1
    total = num_experts * max_tokens
    if num_experts == 16 and max_tokens >= 1024:
        grid = max(1, total // (max_tokens // 256))
        _per_token_quant_int8_grid_stride_kernel[(grid,)](
            x,
            xq,
            scales,
            x.stride(-2),
            xq.stride(-2),
            hidden,
            num_experts,
            max_tokens,
            tokens_per_expert,
            has_tokens_per_expert=tokens_per_expert is not None,
            block=block,
            num_warps=num_warps,
            num_stages=1,
        )
    else:
        _per_token_quant_int8_one_kernel[(total,)](
            x,
            xq,
            scales,
            x.stride(-2),
            xq.stride(-2),
            hidden,
            tokens_per_expert,
            max_tokens,
            has_tokens_per_expert=tokens_per_expert is not None,
            block=block,
            num_warps=num_warps,
            num_stages=1,
        )
    return xq, scales


__all__ = ["per_token_quant_int8"]
