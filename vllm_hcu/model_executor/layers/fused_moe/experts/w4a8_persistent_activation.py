# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Persistent masked activation kernels for HCU W4A8 experts."""

from __future__ import annotations

import torch
from triton.language.extra import libdevice

from vllm.triton_utils import tl, triton


@triton.jit
def _persistent_masked_m_silu_mul_clamp_int8_quant(
    input_ptr,
    output_ptr,
    scale_ptr,
    counts_ptr,
    stride_i_e,
    stride_i_t,
    stride_i_h,
    stride_o_e,
    stride_o_t,
    stride_o_h,
    stride_s_e,
    stride_s_t,
    stride_counts_e,
    H: tl.constexpr,
    LIMIT: tl.constexpr,
    PARALLEL_TOKENS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Persistent masked SwiGLU clamp and per-token INT8 quantization."""
    pid = tl.program_id(0)
    expert = pid // PARALLEL_TOKENS
    token_slot = pid % PARALLEL_TOKENS
    n_tokens = tl.load(counts_ptr + expert * stride_counts_e).to(tl.int64)

    cols = tl.arange(0, BLOCK)
    col_mask = cols < H
    input_base = expert * stride_i_e + cols * stride_i_h
    output_base = expert * stride_o_e + cols * stride_o_h
    scale_base = expert * stride_s_e

    # A small, graph-stable grid handles the common low-token case in one
    # iteration while retaining correctness for arbitrarily skewed routing.
    for token in tl.range(token_slot, n_tokens, PARALLEL_TOKENS):
        token = token.to(tl.int64)
        gate = tl.load(
            input_ptr + input_base + token * stride_i_t,
            mask=col_mask,
            other=0.0,
        ).to(tl.float32)
        up = tl.load(
            input_ptr + input_base + H * stride_i_h + token * stride_i_t,
            mask=col_mask,
            other=0.0,
        ).to(tl.float32)
        gate = tl.minimum(gate, LIMIT)
        up = tl.maximum(tl.minimum(up, LIMIT), -LIMIT)
        activated = gate * (1.0 / (1.0 + tl.exp(-gate))) * up

        absmax = tl.maximum(tl.max(tl.abs(activated)), 1e-10)
        quantized = libdevice.nearbyint(activated * (127.0 / absmax))
        quantized = tl.maximum(tl.minimum(quantized, 127.0), -127.0).to(tl.int8)
        tl.store(
            output_ptr + output_base + token * stride_o_t,
            quantized,
            mask=col_mask,
        )
        tl.store(scale_ptr + scale_base + token * stride_s_t, absmax / 127.0)


def persistent_masked_m_silu_mul_clamp_int8_quant(
    value: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    *,
    limit: float,
    parallel_tokens: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Clamp SwiGLU and INT8-quantize only valid rows of an EP buffer."""
    if value.ndim != 3 or value.shape[-1] % 2 != 0:
        raise ValueError("value must have shape [experts, tokens, 2 * hidden]")
    if tokens_per_expert.ndim != 1 or tokens_per_expert.shape[0] != value.shape[0]:
        raise ValueError("tokens_per_expert must have one entry per expert")
    if tokens_per_expert.device != value.device:
        raise ValueError("tokens_per_expert must be on the same device as value")
    if limit <= 0:
        raise ValueError("limit must be positive")
    if parallel_tokens <= 0:
        raise ValueError("parallel_tokens must be positive")

    experts, max_tokens, doubled_hidden = value.shape
    hidden = doubled_hidden // 2
    output = torch.empty(
        (experts, max_tokens, hidden),
        dtype=torch.int8,
        device=value.device,
    )
    scales = torch.empty(
        (experts, max_tokens, 1),
        dtype=torch.float32,
        device=value.device,
    )
    block = triton.next_power_of_2(hidden)
    num_warps = min(max(block // 256, 1), 8)
    _persistent_masked_m_silu_mul_clamp_int8_quant[
        (experts * parallel_tokens,)
    ](
        value,
        output,
        scales,
        tokens_per_expert,
        *value.stride(),
        *output.stride(),
        scales.stride(0),
        scales.stride(1),
        tokens_per_expert.stride(0),
        H=hidden,
        LIMIT=float(limit),
        PARALLEL_TOKENS=parallel_tokens,
        BLOCK=block,
        num_warps=num_warps,
        num_stages=1,
    )
    return output, scales


__all__ = ["persistent_masked_m_silu_mul_clamp_int8_quant"]
