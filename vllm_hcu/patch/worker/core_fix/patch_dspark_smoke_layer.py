# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Keep DSpark draft weights complete during reduced-layer smoke loading."""

from __future__ import annotations

import functools
import importlib
from types import ModuleType

from ._common import (
    PatchCompatibilityError,
    load_exact_module,
    require_callable,
    require_exact_signature,
)

TARGET_MODULE = "vllm.v1.worker.gpu.spec_decode.dspark.utils"
WEIGHT_UTILS_MODULE = "vllm.model_executor.model_loader.weight_utils"
PATCH_ID = "worker.core_fix.dspark.smoke_layer_limit"
TARGETS = (f"{TARGET_MODULE}.get_model",)
_MARKER = "_vllm_hcu_dspark_smoke_layer_limit_applied"
_WRAPPER_MARKER = "_vllm_hcu_dspark_smoke_layer_limit_wrapper"


def apply_to_module(module: ModuleType) -> bool:
    """Disable smoke layer filtering only for the DSpark draft model load."""
    dspark = load_exact_module(TARGET_MODULE, module)
    original = require_callable(dspark, "get_model", TARGETS[0])
    require_exact_signature(
        original,
        TARGETS[0],
        keyword_only=("vllm_config", "model_config", "prefix", "load_config"),
        defaults={"model_config": None, "prefix": "", "load_config": None},
    )

    weight_utils = importlib.import_module(WEIGHT_UTILS_MODULE)
    if not isinstance(weight_utils, ModuleType) or (
        weight_utils.__name__ != WEIGHT_UTILS_MODULE
    ):
        raise PatchCompatibilityError(
            f"required runtime module {WEIGHT_UTILS_MODULE} is unavailable"
        )
    current = vars(dspark).get("get_model")
    if getattr(dspark, _MARKER, False):
        if not getattr(current, _WRAPPER_MARKER, False):
            raise PatchCompatibilityError(
                f"required HCU patch marker for {TARGETS[0]} is stale"
            )
        return False

    disable_smoke_layer_limit = getattr(
        weight_utils, "disable_smoke_layer_limit", None
    )
    if disable_smoke_layer_limit is not None and not callable(
        disable_smoke_layer_limit
    ):
        raise PatchCompatibilityError(
            f"required runtime target {WEIGHT_UTILS_MODULE}."
            "disable_smoke_layer_limit must be callable"
        )
    from vllm_hcu.runtime_compat.weight_loading import disable_weight_debug_skip

    @functools.wraps(original)
    def load_dspark_model_without_smoke_limit(*, vllm_config, **kwargs):
        # ``get_model`` is imported into DSpark's module namespace and only
        # called there for the draft model. Keep target loading unaffected.
        with disable_weight_debug_skip():
            if disable_smoke_layer_limit is None:
                return original(vllm_config=vllm_config, **kwargs)
            with disable_smoke_layer_limit():
                return original(vllm_config=vllm_config, **kwargs)

    setattr(load_dspark_model_without_smoke_limit, _WRAPPER_MARKER, True)
    setattr(dspark, "_vllm_hcu_original_dspark_get_model", original)
    setattr(dspark, "get_model", load_dspark_model_without_smoke_limit)
    setattr(dspark, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    """Apply the exact DSpark loader adapter."""
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
