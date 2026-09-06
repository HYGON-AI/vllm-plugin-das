# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Bind static EPLB maps before any vLLM checkpoint loader can run."""

from __future__ import annotations

import functools
import sys
from contextvars import ContextVar
from types import ModuleType
from typing import Any

from vllm_hcu.model_executor.layers.fused_moe.static_eplb import (
    bind_static_eplb_plan,
)

from ._common import (
    PatchCompatibilityError,
    load_exact_module,
    require_callable,
    require_class,
    require_exact_signature,
)


TARGET_MODULE = "vllm.model_executor.model_loader.utils"
PATCH_ID = "worker.framework_opt.model_loader.static_eplb_preload"
TARGETS = (f"{TARGET_MODULE}.initialize_model",)
ENTRYPOINT_TARGET_MODULE = "vllm.model_executor.model_loader"
ENTRYPOINT_TARGETS = (f"{ENTRYPOINT_TARGET_MODULE}.get_model_loader",)

_MARKER = "_vllm_hcu_static_eplb_model_loader_applied"
_WRAPPER_MARKER = "_vllm_hcu_static_eplb_model_loader_wrapper"
_ENTRYPOINT_MARKER = "_vllm_hcu_static_eplb_loader_gate_applied"
_ENTRYPOINT_WRAPPER_MARKER = "_vllm_hcu_static_eplb_loader_gate_wrapper"
_LOADER_WRAPPER_MARKER = "_vllm_hcu_static_eplb_load_model_gate_wrapper"
_INITIALIZER_DEPTH = ContextVar(
    "vllm_hcu_static_eplb_initializer_depth",
    default=0,
)
_INITIALIZER_ALIASES = (
    "vllm.model_executor.model_loader.base_loader",
    "vllm.model_executor.model_loader.tensorizer_loader",
    "vllm.model_executor.models.mllama4",
)
_LOADER_FACTORY_ALIASES = (
    "vllm.v1.worker.gpu_model_runner",
    "vllm.v1.worker.gpu.model_runner",
)


def _validate_loaded_aliases(original: object) -> tuple[ModuleType, ...]:
    aliases: list[ModuleType] = []
    for module_name in _INITIALIZER_ALIASES:
        alias_module = sys.modules.get(module_name)
        if alias_module is None:
            continue
        alias = getattr(alias_module, "initialize_model", None)
        module_spec = getattr(alias_module, "__spec__", None)
        if alias is None and bool(getattr(module_spec, "_initializing", False)):
            # A parent such as base_loader can be suspended inside
            # ``from ...utils import initialize_model`` while the utils
            # completion callback runs.  It will naturally receive this
            # wrapper when the dependency import returns.
            continue
        if alias is not original:
            raise PatchCompatibilityError(
                "already-imported alias "
                f"{module_name}.initialize_model does not match {TARGETS[0]}"
            )
        aliases.append(alias_module)
    return tuple(aliases)


def _validate_loaded_loader_factory_aliases(
    original: object,
) -> tuple[ModuleType, ...]:
    aliases: list[ModuleType] = []
    for module_name in _LOADER_FACTORY_ALIASES:
        alias_module = sys.modules.get(module_name)
        if alias_module is None:
            continue
        alias = getattr(alias_module, "get_model_loader", None)
        module_spec = getattr(alias_module, "__spec__", None)
        if alias is None and bool(getattr(module_spec, "_initializing", False)):
            continue
        if alias is not original:
            raise PatchCompatibilityError(
                "already-imported alias "
                f"{module_name}.get_model_loader does not match "
                f"{ENTRYPOINT_TARGETS[0]}"
            )
        aliases.append(alias_module)
    return tuple(aliases)


def _is_vllm_tensorized_loader(loader: object) -> bool:
    from vllm.model_executor.model_loader.tensorizer import is_vllm_tensorized

    return bool(is_vllm_tensorized(loader.tensorizer_config))


def _require_static_preload_binding() -> None:
    initializer_module = load_exact_module(TARGET_MODULE, sys.modules.get(TARGET_MODULE))
    initializer = require_callable(initializer_module, "initialize_model", TARGETS[0])
    if not getattr(initializer, _WRAPPER_MARKER, False):
        raise PatchCompatibilityError(
            "Static EPLB loader preflight requires the audited initialize_model "
            "binding patch before checkpoint loading."
        )


def apply_entrypoint_to_module(module: ModuleType) -> bool:
    """Reject loaders that bypass the audited logical-expert load contract."""

    model_loader = load_exact_module(ENTRYPOINT_TARGET_MODULE, module)
    original_get_model_loader = require_callable(
        model_loader,
        "get_model_loader",
        ENTRYPOINT_TARGETS[0],
    )
    if getattr(model_loader, _ENTRYPOINT_MARKER, False):
        if not getattr(
            original_get_model_loader,
            _ENTRYPOINT_WRAPPER_MARKER,
            False,
        ):
            raise PatchCompatibilityError(
                f"required HCU patch marker for {ENTRYPOINT_TARGETS[0]} is stale"
            )
        return False
    require_exact_signature(
        original_get_model_loader,
        ENTRYPOINT_TARGETS[0],
        positional=("load_config",),
    )
    loader_registry = getattr(model_loader, "_LOAD_FORMAT_TO_MODEL_LOADER", None)
    if not isinstance(loader_registry, dict):
        raise PatchCompatibilityError(
            f"required HCU patch target {ENTRYPOINT_TARGET_MODULE}."
            "_LOAD_FORMAT_TO_MODEL_LOADER is missing"
        )

    default_loader = require_class(
        model_loader,
        "DefaultModelLoader",
        f"{ENTRYPOINT_TARGET_MODULE}.DefaultModelLoader",
    )
    bitsandbytes_loader = require_class(
        model_loader,
        "BitsAndBytesModelLoader",
        f"{ENTRYPOINT_TARGET_MODULE}.BitsAndBytesModelLoader",
    )
    runai_loader = require_class(
        model_loader,
        "RunaiModelStreamerLoader",
        f"{ENTRYPOINT_TARGET_MODULE}.RunaiModelStreamerLoader",
    )
    tensorizer_loader = require_class(
        model_loader,
        "TensorizerLoader",
        f"{ENTRYPOINT_TARGET_MODULE}.TensorizerLoader",
    )
    audited_loader_types = (
        default_loader,
        bitsandbytes_loader,
        runai_loader,
        tensorizer_loader,
    )
    aliases = _validate_loaded_loader_factory_aliases(original_get_model_loader)

    @functools.wraps(original_get_model_loader)
    def hcu_get_model_loader(load_config: object) -> Any:
        loader = original_get_model_loader(load_config)
        original_load_model = require_callable(
            loader,
            "load_model",
            f"{type(loader).__name__}.load_model",
        )
        if getattr(original_load_model, _LOADER_WRAPPER_MARKER, False):
            return loader

        load_format = getattr(load_config, "load_format", None)
        loader_type = loader_registry.get(load_format)

        @functools.wraps(original_load_model)
        def hcu_load_model(*args: object, **kwargs: object) -> Any:
            if args:
                vllm_config = args[0]
            else:
                vllm_config = kwargs.get("vllm_config")
            parallel_config = getattr(vllm_config, "parallel_config", None)
            if not getattr(parallel_config, "_vllm_hcu_expert_map_path", None):
                return original_load_model(*args, **kwargs)

            if loader_type not in audited_loader_types or type(loader) is not loader_type:
                loader_name = type(loader).__name__
                raise PatchCompatibilityError(
                    f"Model loader {loader_name} for load format {load_format!r} "
                    "does not support static EPLB direct loading through the "
                    "audited model/RoutedExperts contract."
                )

            _require_static_preload_binding()
            if loader_type is tensorizer_loader and _is_vllm_tensorized_loader(loader):
                raise PatchCompatibilityError(
                    "The vLLM-tensorized Tensorizer load path bypasses model "
                    "weight loaders and does not support static EPLB direct loading."
                )
            return original_load_model(*args, **kwargs)

        setattr(hcu_load_model, _LOADER_WRAPPER_MARKER, True)
        try:
            setattr(loader, "load_model", hcu_load_model)
        except (AttributeError, TypeError) as exc:
            raise PatchCompatibilityError(
                f"Model loader {type(loader).__name__} does not expose a "
                "wrappable load_model boundary for static EPLB gating."
            ) from exc
        return loader

    setattr(hcu_get_model_loader, _ENTRYPOINT_WRAPPER_MARKER, True)
    setattr(
        model_loader,
        "_vllm_hcu_original_static_eplb_get_model_loader",
        original_get_model_loader,
    )
    setattr(model_loader, "get_model_loader", hcu_get_model_loader)
    for alias_module in aliases:
        setattr(alias_module, "get_model_loader", hcu_get_model_loader)
    setattr(model_loader, _ENTRYPOINT_MARKER, True)
    return True


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
        parallel_config = getattr(vllm_config, "parallel_config", None)
        if not getattr(parallel_config, "_vllm_hcu_expert_map_path", None):
            return original(
                vllm_config,
                prefix=prefix,
                model_class=model_class,
                model_config=model_config,
            )

        depth = _INITIALIZER_DEPTH.get()
        token = _INITIALIZER_DEPTH.set(depth + 1)
        try:
            model = original(
                vllm_config,
                prefix=prefix,
                model_class=model_class,
                model_config=model_config,
            )
        finally:
            _INITIALIZER_DEPTH.reset(token)
        if depth == 0:
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


__all__ = [
    "ENTRYPOINT_TARGET_MODULE",
    "ENTRYPOINT_TARGETS",
    "PATCH_ID",
    "TARGET_MODULE",
    "apply",
    "apply_entrypoint_to_module",
    "apply_to_module",
]
