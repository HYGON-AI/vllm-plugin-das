# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Bind static EPLB maps before any vLLM checkpoint loader can run."""

from __future__ import annotations

import functools
import sys
from types import ModuleType
from typing import Any

from vllm_hcu.model_executor.layers.fused_moe.static_eplb import (
    bind_static_eplb_plan,
)

from ._common import (
    PatchCompatibilityError,
    load_exact_module,
    require_callable,
    require_exact_signature,
)


TARGET_MODULE = "vllm.model_executor.model_loader.utils"
PATCH_ID = "worker.framework_opt.model_loader.static_eplb_preload"
TARGETS = (f"{TARGET_MODULE}.initialize_model",)

_MARKER = "_vllm_hcu_static_eplb_model_loader_applied"
_WRAPPER_MARKER = "_vllm_hcu_static_eplb_model_loader_wrapper"
_INITIALIZER_ALIASES = (
    "vllm.model_executor.model_loader.base_loader",
    "vllm.model_executor.model_loader.tensorizer_loader",
    "vllm.model_executor.models.mllama4",
)


def _validate_loaded_aliases(original: object) -> tuple[ModuleType, ...]:
    aliases: list[ModuleType] = []
    for module_name in _INITIALIZER_ALIASES:
        alias_module = sys.modules.get(module_name)
        if alias_module is None:
            continue
        alias = getattr(alias_module, "initialize_model", None)
        if alias is not original:
            raise PatchCompatibilityError(
                "already-imported alias "
                f"{module_name}.initialize_model does not match {TARGETS[0]}"
            )
        aliases.append(alias_module)
    return tuple(aliases)


def apply_to_module(module: ModuleType) -> bool:
    """Wrap the common construction boundary with static-plan binding."""

    model_loader_utils = load_exact_module(TARGET_MODULE, module)
    original = require_callable(model_loader_utils, "initialize_model", TARGETS[0])
    if getattr(model_loader_utils, _MARKER, False):
        if not getattr(original, _WRAPPER_MARKER, False):
            raise PatchCompatibilityError(
                f"required HCU patch marker for {TARGETS[0]} is stale"
            )
        return False
    require_exact_signature(
        original,
        TARGETS[0],
        positional=("vllm_config",),
        keyword_only=("prefix", "model_class", "model_config"),
        defaults={"prefix": "", "model_class": None, "model_config": None},
    )
    aliases = _validate_loaded_aliases(original)

    @functools.wraps(original)
    def hcu_initialize_model(
        vllm_config: object,
        *,
        prefix: str = "",
        model_class: type | None = None,
        model_config: object | None = None,
    ) -> Any:
        model = original(
            vllm_config,
            prefix=prefix,
            model_class=model_class,
            model_config=model_config,
        )
        bind_static_eplb_plan(vllm_config, model)
        return model

    setattr(hcu_initialize_model, _WRAPPER_MARKER, True)
    setattr(model_loader_utils, "_vllm_hcu_original_initialize_model", original)
    setattr(model_loader_utils, "initialize_model", hcu_initialize_model)
    for alias_module in aliases:
        setattr(alias_module, "initialize_model", hcu_initialize_model)
    setattr(model_loader_utils, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "apply", "apply_to_module"]
