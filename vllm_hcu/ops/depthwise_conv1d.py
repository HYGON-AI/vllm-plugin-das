# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Graph-safe Triton depthwise conv1d for HCU model adapters."""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _depthwise_conv1d_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    channels: tl.constexpr,
    input_length: tl.constexpr,
    output_length: tl.constexpr,
    stride_x_batch,
    stride_x_channel,
    stride_x_token,
    stride_weight_channel,
    stride_weight_token,
    stride_output_batch,
    stride_output_channel,
    stride_output_token,
    kernel_size: tl.constexpr,
    padding: tl.constexpr,
    dilation: tl.constexpr,
    has_bias: tl.constexpr,
    block_tokens: tl.constexpr,
):
    batch_channel = tl.program_id(0)
    token_block = tl.program_id(1)
    batch = batch_channel // channels
    channel = batch_channel % channels
    output_offsets = token_block * block_tokens + tl.arange(0, block_tokens)
    output_mask = output_offsets < output_length
    accumulator = tl.zeros((block_tokens,), dtype=tl.float32)

    for kernel_offset in tl.static_range(0, kernel_size):
        input_offsets = output_offsets + kernel_offset * dilation - padding
        input_mask = output_mask & (input_offsets >= 0) & (
            input_offsets < input_length
        )
        values = tl.load(
            x_ptr
            + batch * stride_x_batch
            + channel * stride_x_channel
            + input_offsets * stride_x_token,
            mask=input_mask,
            other=0.0,
        ).to(tl.float32)
        weight = tl.load(
            weight_ptr
            + channel * stride_weight_channel
            + kernel_offset * stride_weight_token
        ).to(tl.float32)
        accumulator += values * weight

    if has_bias:
        accumulator += tl.load(bias_ptr + channel).to(tl.float32)

    tl.store(
        output_ptr
        + batch * stride_output_batch
        + channel * stride_output_channel
        + output_offsets * stride_output_token,
        accumulator,
        mask=output_mask,
    )


def depthwise_conv1d(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    padding: int = 0,
    dilation: int = 1,
) -> torch.Tensor:
    """Apply stride-1 depthwise conv1d with PyTorch cross-correlation semantics."""
    if inputs.ndim != 3:
        raise ValueError(f"inputs must be 3D, got shape={tuple(inputs.shape)}")
    if weight.ndim != 3 or weight.shape[1] != 1:
        raise ValueError(f"weight must have shape (C, 1, K), got {tuple(weight.shape)}")
    batch, channels, input_length = inputs.shape
    if weight.shape[0] != channels:
        raise ValueError(
            f"weight channels ({weight.shape[0]}) do not match inputs ({channels})"
        )
    if bias is not None and tuple(bias.shape) != (channels,):
        raise ValueError(f"bias must have shape ({channels},), got {tuple(bias.shape)}")
    if padding < 0 or dilation <= 0:
        raise ValueError(f"invalid padding={padding} or dilation={dilation}")

    kernel_size = weight.shape[2]
    output_length = input_length + 2 * padding - dilation * (kernel_size - 1)
    if output_length <= 0:
        raise ValueError(
            "calculated output length must be positive, got "
            f"{output_length} for input={input_length}, kernel={kernel_size}, "
            f"padding={padding}, dilation={dilation}"
        )

    output = torch.empty(
        (batch, channels, output_length),
        dtype=inputs.dtype,
        device=inputs.device,
    )
    block_tokens = 32
    grid = (batch * channels, triton.cdiv(output_length, block_tokens))
    _depthwise_conv1d_kernel[grid](
        inputs,
        weight,
        bias if bias is not None else weight,
        output,
        channels,
        input_length,
        output_length,
        inputs.stride(0),
        inputs.stride(1),
        inputs.stride(2),
        weight.stride(0),
        weight.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        kernel_size,
        padding,
        dilation,
        bias is not None,
        block_tokens,
    )
    return output


__all__ = ["depthwise_conv1d"]
