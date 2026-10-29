# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""HY4 value projection using HIPBLASLt FP8 strided batched GEMM."""

import os

import torch

from vllm_hcu.platforms import envs as henvs


def prepare_fp8_v_up_proj(layer, act_dtype: torch.dtype) -> bool:
    if not henvs.VLLM_HCU_USE_CUSTOM_OPS:
        return False
    if os.environ.get("VLLM_HCU_HY4_FP8_V_UP_PROJ", "0") != "1":
        return False
    if (
        torch.version.hip is None
        or act_dtype != torch.bfloat16
        or layer.is_aiter_triton_fp4_bmm_enabled
        or layer.is_aiter_triton_fp8_bmm_enabled
    ):
        raise ValueError(
            "VLLM_HCU_HY4_FP8_V_UP_PROJ requires ROCm BF16 activations "
            "and disabled AITER FP4/FP8 BMM"
        )
    from vllm import _custom_ops as ops

    import vllm_hcu.hcu_ops  # noqa: F401

    # W_UV is [heads, latent, value]. BLASLt's FP8 TN path needs
    # contiguous [heads, value, latent] weights and one tensor scale.
    weight = layer.W_UV.transpose(1, 2).contiguous()
    quant, scale = ops.scaled_fp8_quant(weight.view(-1, layer.kv_lora_rank))
    layer.W_UV_fp8 = quant.view(weight.shape)
    layer.W_UV_fp8_scale = scale
    return True


def fp8_v_up_proj(layer, x: torch.Tensor, out: torch.Tensor) -> None:
    from vllm import _custom_ops as ops

    x = x.reshape(-1, layer.num_heads, layer.kv_lora_rank)
    out = out.view(-1, layer.num_heads, layer.v_head_dim)
    if x.shape[0] == 0:
        return
    # A single activation scale applies to every head. The native GEMM reads
    # token-major x directly and writes token-major out, avoiding transposes.
    quant, scale = ops.scaled_fp8_quant(x.reshape(-1, layer.kv_lora_rank))
    torch.ops.hcu_ops.hipblaslt_fp8_bmm(
        quant.view(x.shape), layer.W_UV_fp8, scale, layer.W_UV_fp8_scale, out
    )
