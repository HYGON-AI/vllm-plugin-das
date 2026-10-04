# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""FP32-output GLM-5.3 router projection for small gfx938 decode batches."""

from __future__ import annotations

import functools
import os

import torch
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_hcu.platforms import envs as henvs

_HIDDEN_SIZE = 4096
_NUM_EXPERTS = 288
_MAX_TOKENS = 16
_BLOCK_K = 512
_ROUTER_GEMV_ENV = "VLLM_HCU_GLM53_ROUTER_GEMV"


@triton.jit
def _router_gemv_kernel(
    x_ptr,
    weight_ptr,
    output_ptr,
    hidden_size: tl.constexpr,
    num_experts: tl.constexpr,
    block_k: tl.constexpr,
):
    expert = tl.program_id(0)
    token = tl.program_id(1)
    offsets = tl.arange(0, block_k)
    accum = tl.full((block_k,), 0, tl.float32)
    for block in range(tl.cdiv(hidden_size, block_k)):
        col = block * block_k + offsets
        x = tl.load(x_ptr + token * hidden_size + col).to(tl.float32)
        weight = tl.load(weight_ptr + expert * hidden_size + col).to(tl.float32)
        accum = tl.fma(x, weight, accum)
    tl.store(output_ptr + token * num_experts + expert, tl.sum(accum, 0))


def _glm53_router_gemv_impl(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if (
        x.device.type != "cuda"
        or weight.device != x.device
        or x.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
        or x.ndim != 2
        or weight.shape != (_NUM_EXPERTS, _HIDDEN_SIZE)
        or x.shape[1] != _HIDDEN_SIZE
        or not 1 <= x.shape[0] <= _MAX_TOKENS
        or not x.is_contiguous()
        or not weight.is_contiguous()
    ):
        raise ValueError("unsupported GLM-5.3 router GEMV input")
    output = torch.empty(
        (x.shape[0], _NUM_EXPERTS), device=x.device, dtype=torch.float32
    )
    _router_gemv_kernel[(_NUM_EXPERTS, x.shape[0])](
        x,
        weight,
        output,
        _HIDDEN_SIZE,
        _NUM_EXPERTS,
        _BLOCK_K,
        num_warps=4,
    )
    return output


def _glm53_router_gemv_fake(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.empty(
        (x.shape[0], weight.shape[0]), device=x.device, dtype=torch.float32
    )


direct_register_custom_op(
    op_name="hcu_glm53_router_gemv",
    op_func=_glm53_router_gemv_impl,
    mutates_args=[],
    fake_impl=_glm53_router_gemv_fake,
)


def glm53_router_gemv(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.ops.vllm.hcu_glm53_router_gemv(x, weight)


def _bind_gate(gate: torch.nn.Module) -> bool:
    weight = getattr(gate, "weight", None)
    if (
        not henvs.VLLM_USE_NN
        or not getattr(gate, "allow_cublas_router_gemm", False)
        or not isinstance(weight, torch.Tensor)
        or weight.shape != (_HIDDEN_SIZE, _NUM_EXPERTS)
        or weight.dtype != torch.bfloat16
        or weight.device.type != "cuda"
        or not weight.is_contiguous()
        or getattr(gate, "out_dtype", None) != torch.float32
        or getattr(gate, "bias", None) is not None
        or getattr(gate, "_vllm_hcu_glm53_router_gemv_bound", False)
    ):
        return False
    gate.register_buffer(
        "_hcu_router_gemv_weight", weight.T.contiguous(), persistent=False
    )
    packed_weight = gate._hcu_router_gemv_weight
    original_forward = gate.forward

    @functools.wraps(original_forward)
    def router_forward(x: torch.Tensor):
        if (
            isinstance(x, torch.Tensor)
            and x.device == packed_weight.device
            and x.dtype == torch.bfloat16
            and x.ndim == 2
            and 1 <= x.shape[0] <= _MAX_TOKENS
            and x.shape[1] == _HIDDEN_SIZE
            and x.is_contiguous()
        ):
            return glm53_router_gemv(x, packed_weight), None
        return original_forward(x)

    gate.forward = router_forward
    gate._vllm_hcu_glm53_router_gemv_bound = True
    return True


def bind_glm53_router_gates(model: torch.nn.Module | None) -> int:
    """Bind loaded base-model gates before warmup and CUDA graph capture."""

    from vllm_hcu.platforms.hcu import on_gfx938

    if model is None or not on_gfx938() or os.environ.get(_ROUTER_GEMV_ENV) != "1":
        return 0
    bound = 0
    for layer in model.modules():
        if type(layer).__name__ != "Glm5NextDecoderLayer":
            continue
        if getattr(layer, "is_mtp_layer", False):
            continue
        gate = getattr(getattr(layer, "mlp", None), "gate", None)
        if gate is not None:
            bound += _bind_gate(gate)
    return bound
