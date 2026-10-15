# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
from __future__ import annotations
from types import ModuleType
from vllm_hcu.runtime_compat.step5_patch_contract import (
    already_applied,
    remember_targets,
    require_signatures,
)
from ._common import load_exact_module, require_callable

TARGET_MODULE = "vllm.model_executor.layers.fused_moe.utils"
PATCH_ID = "worker.framework_opt.step5_token_fp8"
TARGETS = ("vllm.model_executor.layers.fused_moe.utils._fp8_quantize",)
_MARKER = "_vllm_hcu_step5_token_fp8_applied"

SIGNATURES = (
    (
        "vllm.model_executor.layers.fused_moe.utils._fp8_quantize",
        ("A", "A_scale", "per_act_token", "block_shape"),
    ),
)
REVIEWED_SOURCES = ()


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    if already_applied(target, _MARKER, TARGETS):
        return False
    require_signatures(target, SIGNATURES, REVIEWED_SOURCES)
    import functools

    original = require_callable(target, "_fp8_quantize", TARGETS[0])

    @functools.wraps(original)
    def quantize(A, A_scale, per_act_token, block_shape=None):
        if block_shape is None and per_act_token and A_scale is None:
            from vllm_hcu.ops import token_fp8_quant as tq

            if tq.can_use(A):
                return tq.token_fp8_quant(A)
        return original(A, A_scale, per_act_token, block_shape)

    target._fp8_quantize = quantize
    remember_targets(target, _MARKER, TARGETS)
    return True


def apply(module=None):
    return apply_to_module(load_exact_module(TARGET_MODULE, module))
