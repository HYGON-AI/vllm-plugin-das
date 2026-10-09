# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Bind static EPLB plans after construction and before checkpoint loading."""

from __future__ import annotations

import functools
import sys
from contextvars import ContextVar
from types import ModuleType

from ._common import (
    PatchCompatibilityError,
    load_exact_module,
    require_callable,
    require_exact_signature,
)

TARGET_MODULE = "vllm.model_executor.model_loader.utils"
PATCH_ID = "worker.framework_opt.model_loader.static_eplb_preload"
TARGETS = (f"{TARGET_MODULE}.initialize_model",)
INITIALIZER_ALIASES = (
    "vllm.model_executor.model_loader.base_loader",
    "vllm.model_executor.model_loader.tensorizer_loader",
    "vllm.model_executor.model_loader.weight_cache.ipc_loader",
    "vllm.model_executor.models.mllama4",
)
_MARKER = "_vllm_hcu_static_eplb_preload_wrapper"
_OWNER_MARKER = "_vllm_hcu_static_eplb_preload_applied"
_DEPTH = ContextVar("hcu_static_initialization_depth", default=0)
_AUDITED_LOAD_FORMATS = frozenset(
    {"auto", "pt", "safetensors", "npcache", "mistral"}
)


def validate_static_loader(vllm_config: object, load_config: object) -> None:
    """Reject loaders that bypass per-logical-expert checkpoint callbacks."""

    from vllm_hcu.patch.config import bind_hcu_eplb_config

    bind_hcu_eplb_config(vllm_config)
    parallel_config = getattr(vllm_config, "parallel_config", None)
    path = getattr(parallel_config, "_vllm_hcu_expert_map_path", None)
    if not path:
        return
    load_format = getattr(load_config, "load_format", None)
    if load_format not in _AUDITED_LOAD_FORMATS:
        raise ValueError(
            "Static EPLB requires an audited default model loader format; "
            f"got {load_format!r}"
        )


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    current = require_callable(target, "initialize_model", TARGETS[0])
    if getattr(target, _OWNER_MARKER, False):
        if not getattr(current, _MARKER, False):
            raise PatchCompatibilityError(
                "Static EPLB initializer marker is stale; restart the process"
            )
        return False

    require_exact_signature(
        current,
        TARGETS[0],
        positional=("vllm_config",),
        keyword_only=("prefix", "model_class", "model_config"),
        defaults={"prefix": "", "model_class": None, "model_config": None},
    )

    aliases: list[ModuleType] = []
    for name in INITIALIZER_ALIASES:
        alias = sys.modules.get(name)
        if alias is None:
            continue
        value = getattr(alias, "initialize_model", None)
        if value is None and getattr(
            getattr(alias, "__spec__", None),
            "_initializing",
            False,
        ):
            continue
        if value is not current:
            raise PatchCompatibilityError(
                f"Static EPLB initializer alias mismatch: {name}"
            )
        aliases.append(alias)

    @functools.wraps(current)
    def initialize_model(
        vllm_config,
        *,
        prefix="",
        model_class=None,
        model_config=None,
    ):
        from vllm_hcu.model_executor.layers.fused_moe.static_eplb import (
            bind_static_eplb_plan,
        )

        validate_static_loader(
            vllm_config,
            getattr(vllm_config, "load_config", None),
        )
        depth = _DEPTH.get()
        token = _DEPTH.set(depth + 1)
        try:
            model = current(
                vllm_config,
                prefix=prefix,
                model_class=model_class,
                model_config=model_config,
            )
            if depth == 0:
                bind_static_eplb_plan(vllm_config, model)
            return model
        finally:
            _DEPTH.reset(token)

    setattr(initialize_model, _MARKER, True)
    setattr(target, "_vllm_hcu_original_initialize_model", current)
    target.initialize_model = initialize_model
    for alias in aliases:
        alias.initialize_model = initialize_model
    setattr(target, _OWNER_MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = [
    "INITIALIZER_ALIASES",
    "PATCH_ID",
    "TARGET_MODULE",
    "TARGETS",
    "apply",
    "apply_to_module",
    "validate_static_loader",
]
