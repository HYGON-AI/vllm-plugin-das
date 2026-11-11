# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Keep graph-padding routes valid for the LightOp W16A16 pipeline."""

from __future__ import annotations

import functools
from types import ModuleType

from ._common import (
    check_module_marker,
    load_exact_module,
    require_callable,
    require_parameter_names,
)

TARGET_MODULE = "vllm.model_executor.layers.fused_moe.router.fused_topk_router"
PATCH_ID = "worker.op_opt.moe.router.fused_topk_lightop_w16a16"
TARGETS = (f"{TARGET_MODULE}._get_padding_mask",)
_MARKER = "_vllm_hcu_lightop_w16a16_padding_applied"
_WRAPPER_MARKER = "_vllm_hcu_lightop_w16a16_padding_wrapper"


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    if check_module_marker(
        target,
        _MARKER,
        ((target, "_get_padding_mask", _WRAPPER_MARKER),),
    ):
        return False
    original = require_callable(target, "_get_padding_mask", TARGETS[0])
    require_parameter_names(original, TARGETS[0], ("num_tokens",))

    @functools.wraps(original)
    def hcu_get_padding_mask(num_tokens: int):
        from vllm_hcu.model_executor.layers.fused_moe.lightop_w16a16_runtime import (
            lightop_w16a16_backend_selected,
        )

        # LightOp 0.6's W16A16 align/GEMM pipeline assumes every route is a
        # valid expert. vLLM otherwise emits -1 for graph-padding rows, which
        # makes the LightOp align kernel index before its expert counter array.
        # The affected backend is opt-in and selected only for its audited
        # shapes; keeping padding routes valid avoids a host sync and remains
        # capture-safe. Padded model outputs are discarded by vLLM.
        if lightop_w16a16_backend_selected():
            return None
        return original(num_tokens)

    setattr(hcu_get_padding_mask, _WRAPPER_MARKER, True)
    target._vllm_hcu_original_get_padding_mask = original
    target._get_padding_mask = hcu_get_padding_mask
    setattr(target, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
