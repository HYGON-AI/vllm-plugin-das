# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Enable explicit HYV4 generation head dtype on vLLM v0.25.1."""

from __future__ import annotations

import functools
from types import ModuleType

from ._common import (
    PatchCompatibilityError,
    apply_once,
    load_exact_module,
    require_callable,
    require_positional_signature,
)

TARGET_MODULE = "vllm.config.model"
PATCH_ID = "platform.core_fix.hy_v4_model_head_dtype"
TARGETS = (f"{TARGET_MODULE}.ModelConfig.head_dtype",)
_MARKER = "_vllm_hcu_hy_v4_model_head_dtype_applied"
_MODEL_TYPES = frozenset({"hy_v4", "hy_v4_mtp"})


def apply_to_module(module: ModuleType) -> bool:
    model_module = load_exact_module(TARGET_MODULE, module)
    if getattr(model_module, _MARKER, False):
        return False

    model_config = getattr(model_module, "ModelConfig", None)
    if not isinstance(model_config, type):
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGET_MODULE}.ModelConfig is missing"
        )
    descriptor = vars(model_config).get("head_dtype")
    if not isinstance(descriptor, property) or descriptor.fget is None:
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGETS[0]} is not a readable property"
        )
    original = descriptor.fget
    require_positional_signature(original, TARGETS[0], ("self",))
    get_head_dtype = require_callable(
        model_module,
        "_get_head_dtype",
        f"{TARGET_MODULE}._get_head_dtype",
    )
    require_positional_signature(
        get_head_dtype,
        f"{TARGET_MODULE}._get_head_dtype",
        ("config", "dtype", "runner_type"),
    )
    current_platform = getattr(model_module, "current_platform", None)
    if current_platform is None or not hasattr(current_platform, "supported_dtypes"):
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGET_MODULE}.current_platform is missing"
        )

    @functools.wraps(original)
    def hcu_head_dtype(self):
        hf_config = getattr(self, "hf_config", None)
        if (
            getattr(hf_config, "model_type", None) in _MODEL_TYPES
            and getattr(hf_config, "head_dtype", None) is not None
        ):
            requested = get_head_dtype(hf_config, self.dtype, self.runner_type)
            if requested in current_platform.supported_dtypes:
                return requested
        return original(self)

    setattr(model_config, "_vllm_hcu_original_head_dtype", descriptor)
    setattr(model_config, "head_dtype", property(hcu_head_dtype, descriptor.fset))
    setattr(model_module, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    model_module = load_exact_module(TARGET_MODULE, module)
    return apply_once(
        patch_id=PATCH_ID,
        targets=TARGETS,
        marker_owner=model_module,
        marker=_MARKER,
        callback=lambda: apply_to_module(model_module),
    )


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
