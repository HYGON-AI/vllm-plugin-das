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
TARGETS = (f"{TARGET_MODULE}.GateLinear.forward",)
_CLASS_MARKER = "_vllm_hcu_nn_layout_applied"
_WRAPPER_MARKER = "_vllm_hcu_nn_layout_wrapper"


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
        ((gate_class, "forward", TARGETS[0], _WRAPPER_MARKER),),
    ):
        return False

    original = require_callable(gate_class, "forward", TARGETS[0])
    require_exact_signature(
        original,
        TARGETS[0],
        positional=("self", "x"),
    )

    @functools.wraps(original)
    def hcu_forward(self, x: torch.Tensor):
        uses_hcu_nn_layout = (
            henvs.VLLM_USE_NN
            and self.allow_cublas_router_gemm
            and x.dtype == torch.bfloat16
            and self.weight.ndim == 2
            and self.weight.shape[0] == x.shape[-1]
        )
        if uses_hcu_nn_layout:
            output = torch.mm(x, self.weight, out_dtype=torch.float32)
            return output, None
        return original(self, x)

    setattr(hcu_forward, _WRAPPER_MARKER, True)
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
