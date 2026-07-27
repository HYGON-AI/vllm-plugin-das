# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Advertise the audited Triton INT8 MoE path on ROCm/HCU."""

from __future__ import annotations

import functools
from types import ModuleType

from ._common import load_exact_module, require_callable, require_class, require_parameter_names

TARGET_MODULE = "vllm.model_executor.layers.fused_moe.experts.triton_moe"
PATCH_ID = "worker.op_opt.moe.experts.triton_int8"
TARGETS = (f"{TARGET_MODULE}.TritonExperts._supports_quant_scheme",)
_MARKER = "_vllm_hcu_triton_int8_applied"


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    if getattr(target, _MARKER, False):
        return False
    cls = require_class(target, "TritonExperts", TARGETS[0].rsplit(".", 1)[0])
    original = require_callable(cls, "_supports_quant_scheme", TARGETS[0])
    require_parameter_names(original, TARGETS[0], ("weight_key", "activation_key"))

    @functools.wraps(original)
    def hcu_supports_quant_scheme(weight_key, activation_key):
        if target.current_platform.is_rocm() and (
            weight_key,
            activation_key,
        ) == (target.kInt8StaticChannelSym, target.kInt8DynamicTokenSym):
            return True
        return original(weight_key, activation_key)

    cls._vllm_hcu_original_supports_quant_scheme = original
    cls._supports_quant_scheme = staticmethod(hcu_supports_quant_scheme)
    setattr(target, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
