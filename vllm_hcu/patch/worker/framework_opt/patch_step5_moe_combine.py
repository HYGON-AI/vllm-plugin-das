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

TARGET_MODULE = "vllm.model_executor.layers.fused_moe.prepare_finalize.naive_dp_ep"
PATCH_ID = "worker.framework_opt.step5_moe_combine"
TARGETS = (
    "vllm.model_executor.layers.fused_moe.prepare_finalize.naive_dp_ep.MoEPrepareAndFinalizeNaiveDPEPModular.finalize",
)
_MARKER = "_vllm_hcu_step5_moe_combine_applied"

SIGNATURES = (
    (
        "vllm.model_executor.layers.fused_moe.prepare_finalize.naive_dp_ep.MoEPrepareAndFinalizeNaiveDPEPModular.finalize",
        (
            "self",
            "output",
            "fused_expert_output",
            "topk_weights",
            "topk_ids",
            "apply_router_weight_on_input",
            "weight_and_reduce_impl",
        ),
    ),
)
REVIEWED_SOURCES = ()


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    if already_applied(target, _MARKER, TARGETS):
        return False
    require_signatures(target, SIGNATURES, REVIEWED_SOURCES)
    cls = getattr(target, "MoEPrepareAndFinalizeNaiveDPEPModular", None)
    require_callable(cls, "finalize", TARGETS[0])
    from vllm_hcu.model_executor.layers.fused_moe.combine_runtime import (
        finalize,
        set_finalize_combine_hook,
    )

    cls.finalize = finalize
    target.set_finalize_combine_hook = set_finalize_combine_hook
    remember_targets(target, _MARKER, TARGETS)
    return True


def apply(module=None):
    return apply_to_module(load_exact_module(TARGET_MODULE, module))
