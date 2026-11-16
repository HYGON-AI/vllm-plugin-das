# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Optional pure-Torch EPLB map-and-record implementation."""

from __future__ import annotations

import functools
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
PATCH_ID = "worker.op_opt.moe.router.base"
TARGETS = (
    f"{TARGET_MODULE}.eplb_map_to_physical_and_record",
    f"{TARGET_MODULE}.BaseRouter._apply_eplb_mapping",
)
_MARKER = "_vllm_hcu_eplb_torch_applied"
_MAP_WRAPPER = "_vllm_hcu_eplb_map_wrapper"
_APPLY_WRAPPER = "_vllm_hcu_eplb_apply_wrapper"


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    base_router_cls = require_class(
        target,
        "BaseRouter",
        f"{TARGET_MODULE}.BaseRouter",
    )
    if check_module_marker(
        target,
        _MARKER,
        (
            (target, "eplb_map_to_physical_and_record", _MAP_WRAPPER),
            (base_router_cls, "_apply_eplb_mapping", _APPLY_WRAPPER),
        ),
    ):
        return False
    from vllm_hcu.model_executor.layers.fused_moe import router_runtime

    original = require_callable(target, "eplb_map_to_physical_and_record", TARGETS[0])
    original_apply = require_callable(
        base_router_cls,
        "_apply_eplb_mapping",
        TARGETS[1],
    )
    dbo_current_ubatch_id = require_callable(
        target,
        "dbo_current_ubatch_id",
        f"{TARGET_MODULE}.dbo_current_ubatch_id",
    )
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
    require_parameter_names(
        original_apply,
        TARGETS[1],
        ("self", "topk_ids"),
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

    def hcu_apply_eplb_mapping(self, topk_ids):
        eplb_state = self.eplb_state
        primary_count = getattr(
            eplb_state,
            router_runtime.STATIC_PRIMARY_REPLICA_COUNT,
            None,
        )
        if primary_count is None:
            return original_apply(self, topk_ids)
        return target.eplb_map_to_physical_and_record(
            topk_ids=topk_ids,
            logical_to_physical_map=eplb_state.logical_to_physical_map,
            logical_replica_count=primary_count,
            expert_load_view=eplb_state.expert_load_view,
            record_enabled=eplb_state.should_record_tensor,
            num_unpadded_tokens=eplb_state.num_unpadded_tokens_tensors[
                dbo_current_ubatch_id()
            ],
        )

    hcu_eplb_map = marked_wrapper(original, _MAP_WRAPPER, hcu_eplb_map)
    hcu_apply_eplb_mapping = marked_wrapper(
        original_apply,
        _APPLY_WRAPPER,
        hcu_apply_eplb_mapping,
    )

    target._vllm_hcu_original_eplb_map_to_physical_and_record = original
    target._vllm_hcu_original_apply_eplb_mapping = original_apply
    target.eplb_map_to_physical_and_record = hcu_eplb_map
    base_router_cls._apply_eplb_mapping = hcu_apply_eplb_mapping
    setattr(target, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
