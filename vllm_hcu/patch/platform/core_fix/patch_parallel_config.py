# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Count HCU prefill context parallelism in the upstream EPLB world check."""

from __future__ import annotations

import copy
from functools import wraps
from types import ModuleType
from typing import Any

from ._common import (
    PatchCompatibilityError,
    apply_once,
    load_exact_module,
    require_positional_signature,
)

TARGET_MODULE = "vllm.config.parallel"
PATCH_ID = "platform.core_fix.parallel_config.pcp_eplb"
TARGETS = (f"{TARGET_MODULE}.ParallelConfig._validate_parallel_config",)
_MARKER = "_vllm_hcu_pcp_eplb_validator"


def apply_to_module(module: ModuleType) -> bool:
    from pydantic.dataclasses import rebuild_dataclass

    parallel_config = getattr(module, "ParallelConfig", None)
    if not isinstance(parallel_config, type):
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGET_MODULE}.ParallelConfig is missing"
        )
    if getattr(parallel_config, _MARKER, False):
        return False
    original = getattr(parallel_config, "_validate_parallel_config", None)
    if not callable(original):
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGETS[0]} is missing"
        )
    require_positional_signature(original, TARGETS[0], ("self",))
    decorators = getattr(parallel_config, "__pydantic_decorators__", None)
    validators = getattr(decorators, "model_validators", {})
    decorator = validators.get("_validate_parallel_config")
    if (
        decorator is None
        or decorator.func is not original
        or getattr(decorator.info, "mode", None) != "after"
    ):
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGETS[0]} has incompatible Pydantic metadata"
        )

    @wraps(original)
    def validate(self: Any) -> Any:
        if not (
            self.enable_eplb
            and self.enable_expert_parallel
            and self.tensor_parallel_size * self.data_parallel_size == 1
            and self.prefill_context_parallel_size > 1
            and module.current_platform.is_cuda_alike()
        ):
            return original(self)

        # Field validation and __post_init__ have already run with EPLB enabled,
        # including nested EPLBConfig checks and communicator selection. Only
        # this after-validator needs to bypass the upstream TP/DP-only guard.
        # Keep the caller's EPLBConfig untouched while validating the rest.
        eplb_config = self.eplb_config
        self.eplb_config = copy.copy(eplb_config)
        self.eplb_config.num_redundant_experts = 0
        self.enable_eplb = False
        try:
            return original(self)
        finally:
            self.enable_eplb = True
            self.eplb_config = eplb_config

    # Updating only the method leaves Pydantic's compiled graph pointing at the
    # original function. Update its decorator metadata before rebuilding it.
    previous = {
        name: getattr(parallel_config, name)
        for name in (
            "__pydantic_core_schema__",
            "__pydantic_validator__",
            "__pydantic_serializer__",
            "__pydantic_complete__",
        )
    }
    parallel_config._validate_parallel_config = validate
    decorator.func = validate
    try:
        if rebuild_dataclass(parallel_config, force=True) is not True:
            raise PatchCompatibilityError("ParallelConfig validator rebuild failed")
    except Exception:
        parallel_config._validate_parallel_config = original
        decorator.func = original
        for name, value in previous.items():
            setattr(parallel_config, name, value)
        raise
    setattr(parallel_config, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    parallel_module = load_exact_module(TARGET_MODULE, module)
    parallel_config = getattr(parallel_module, "ParallelConfig", None)
    if not isinstance(parallel_config, type):
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGET_MODULE}.ParallelConfig is missing"
        )
    return apply_once(
        patch_id=PATCH_ID,
        targets=TARGETS,
        marker_owner=parallel_config,
        marker=_MARKER,
        callback=lambda: apply_to_module(parallel_module),
    )


__all__ = ["PATCH_ID", "TARGETS", "TARGET_MODULE", "apply", "apply_to_module"]
