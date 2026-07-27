# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Optional pure-Torch EPLB map-and-record implementation."""

from __future__ import annotations

import functools
from types import ModuleType

from ._common import load_exact_module, require_callable, require_parameter_names

TARGET_MODULE = "vllm.model_executor.layers.fused_moe.router.base_router"
PATCH_ID = "worker.op_opt.moe.router.base"
TARGETS = (f"{TARGET_MODULE}.eplb_map_to_physical_and_record",)
_MARKER = "_vllm_hcu_eplb_torch_applied"


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    if getattr(target, _MARKER, False):
        return False
    from vllm_hcu.model_executor.layers.fused_moe import router_runtime

    original = require_callable(target, "eplb_map_to_physical_and_record", TARGETS[0])
    require_parameter_names(
        original,
        TARGETS[0],
        (
            "topk_ids",
            "expert_load_view",
            "logical_to_physical_map",
            "logical_replica_count",
            "record_enabled",
            "num_unpadded_tokens",
        ),
    )

    @functools.wraps(original)
    def hcu_eplb_map(
        topk_ids,
        expert_load_view,
        logical_to_physical_map,
        logical_replica_count,
        record_enabled,
        num_unpadded_tokens=None,
    ):
        return router_runtime.eplb_map_to_physical_and_record(
            target, original, topk_ids, expert_load_view, logical_to_physical_map,
            logical_replica_count, record_enabled, num_unpadded_tokens,
        )

    target._vllm_hcu_original_eplb_map_to_physical_and_record = original
    target.eplb_map_to_physical_and_record = hcu_eplb_map
    setattr(target, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
