# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Install the static primary-replica router only for offline EPLB workers."""
from __future__ import annotations

from types import ModuleType

from ._common import (
    check_module_marker,
    load_exact_module,
    marked_wrapper,
    require_callable,
    require_class,
    require_parameter_names,
)

TARGET_MODULE = "vllm.model_executor.layers.fused_moe.router.base_router"
PATCH_ID = "worker.op_opt.moe.router.static_eplb"
TARGETS = (f"{TARGET_MODULE}.BaseRouter._apply_eplb_mapping",)
_MARKER = "_vllm_hcu_static_primary_mapping_applied"
_WRAPPER = "_vllm_hcu_static_primary_mapping_wrapper"


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    router = require_class(target, "BaseRouter", TARGETS[0].rsplit(".", 1)[0])
    if check_module_marker(
        target, _MARKER, ((router, "_apply_eplb_mapping", _WRAPPER),)
    ):
        return False
    original = require_callable(router, "_apply_eplb_mapping", TARGETS[0])
    require_parameter_names(original, TARGETS[0], ("self", "topk_ids"))
    ubatch_id = require_callable(
        target, "dbo_current_ubatch_id", f"{TARGET_MODULE}.dbo_current_ubatch_id"
    )
    from vllm_hcu.model_executor.layers.fused_moe.router_runtime import (
        STATIC_PRIMARY_REPLICA_COUNT,
    )

    def apply_mapping(self, topk_ids):
        state = self.eplb_state
        primary_count = getattr(state, STATIC_PRIMARY_REPLICA_COUNT, None)
        if primary_count is None:
            return original(self, topk_ids)
        return target.eplb_map_to_physical_and_record(
            topk_ids=topk_ids,
            logical_to_physical_map=state.logical_to_physical_map,
            logical_replica_count=primary_count,
            expert_load_view=state.expert_load_view,
            record_enabled=state.should_record_tensor,
            num_unpadded_tokens=state.num_unpadded_tokens_tensors[ubatch_id()],
        )

    router._apply_eplb_mapping = marked_wrapper(original, _WRAPPER, apply_mapping)
    setattr(target, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
