# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Adapt official GateLinear to the HCU unquantized NN weight layout."""

from __future__ import annotations

import functools
from types import ModuleType

import torch

from vllm_hcu.platforms import envs as henvs

from ._common import (
    already_applied,
    load_exact_module,
    require_callable,
    require_class,
    require_exact_signature,
)

TARGET_MODULE = "vllm.model_executor.layers.fused_moe.router.gate_linear"
PATCH_ID = "worker.op_opt.moe.router.gate_linear_nn_layout"
TARGETS = (
    f"{TARGET_MODULE}.GateLinear.forward",
    f"{TARGET_MODULE}.GateLinear.__init__",
)
_CLASS_MARKER = "_vllm_hcu_nn_layout_applied"
_WRAPPER_MARKER = "_vllm_hcu_nn_layout_wrapper"
_INIT_WRAPPER_MARKER = "_vllm_hcu_gate_model_wrapper"


def _get_model_type():
    from vllm.config import get_current_vllm_config_or_none

    config = get_current_vllm_config_or_none()
    model_config = getattr(config, "model_config", None)
    return getattr(getattr(model_config, "hf_config", None), "model_type", None)


def apply_to_module(module: ModuleType) -> bool:
    gate_linear = load_exact_module(TARGET_MODULE, module)
    gate_class = require_class(
        gate_linear,
        "GateLinear",
        f"{TARGET_MODULE}.GateLinear",
    )
    if already_applied(
        gate_class,
        _CLASS_MARKER,
        (
            (gate_class, "forward", TARGETS[0], _WRAPPER_MARKER),
            (gate_class, "__init__", TARGETS[1], _INIT_WRAPPER_MARKER),
        ),
    ):
        return False

    original = require_callable(gate_class, "forward", TARGETS[0])
    original_init = require_callable(gate_class, "__init__", TARGETS[1])
    require_exact_signature(
        original,
        TARGETS[0],
        positional=("self", "x"),
    )

    @functools.wraps(original_init)
    def hcu_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        # Model configuration is available during construction, not necessarily
        # during forward or graph capture. Do not infer the model from shapes.
        self._vllm_hcu_deepseek_v4_gate = _get_model_type() == "deepseek_v4"

    @functools.wraps(original)
    def hcu_forward(self, x: torch.Tensor):
        uses_hcu_nn_layout = (
            henvs.VLLM_USE_NN
            and self.allow_cublas_router_gemm
            and x.dtype == torch.bfloat16
            and self.weight.ndim == 2
            and self.weight.shape[0] == x.shape[-1]
        )
        if (
            self._vllm_hcu_deepseek_v4_gate
            and self.allow_cublas_router_gemm
            and x.dtype == torch.bfloat16
            and self.weight.dtype == torch.bfloat16
        ):
            weight = self.weight if uses_hcu_nn_layout else self.weight.T
            # Match v0.25.1: round GEMM output to BF16 before converting the
            # router logits to FP32. Direct FP32 output selects a slow kernel.
            return torch.mm(x, weight).float(), None
        if uses_hcu_nn_layout:
            output = torch.mm(x, self.weight, out_dtype=torch.float32)
            return output, None
        return original(self, x)

    setattr(hcu_forward, _WRAPPER_MARKER, True)
    setattr(hcu_init, _INIT_WRAPPER_MARKER, True)
    setattr(gate_class, "_vllm_hcu_original_init", original_init)
    setattr(gate_class, "__init__", hcu_init)
    setattr(gate_class, "_vllm_hcu_original_forward", original)
    setattr(gate_class, "forward", hcu_forward)
    setattr(gate_class, _CLASS_MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = [
    "PATCH_ID",
    "TARGET_MODULE",
    "TARGETS",
    "apply",
    "apply_to_module",
]
