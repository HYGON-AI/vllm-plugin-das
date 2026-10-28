# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Port CacheConfig.swa_bounded_replay + CLI onto pinned vLLM 0.29.

0.29.1rc0.dev283's CacheConfig / EngineArgs lack ``swa_bounded_replay``.
Upstream (#58132 / #59039) adds the field (default True) with
``--swa-bounded-replay`` / ``--no-swa-bounded-replay``. This adapter:

* Gives every ``CacheConfig`` instance the attribute (default True)
* Adds the EngineArgs attribute + BooleanOptionalAction CLI
* Copies the EngineArgs value onto ``VllmConfig.cache_config`` after construction

It intentionally does **not** live under ``HcuFeatureConfig`` /
``additional_config['hcu']`` (unknown HCU keys raise).
"""

from __future__ import annotations

import argparse
import functools
from types import ModuleType
from typing import Any

from ._common import (
    PatchCompatibilityError,
    apply_once,
    load_exact_module,
    require_callable,
)

# Primary exact-import target (CacheConfig). EngineArgs is patched from
# ``register_platform_core_callbacks`` after ``patch_engine_args``.
TARGET_MODULE = "vllm.config.cache"
ENGINE_TARGET_MODULE = "vllm.engine.arg_utils"
PATCH_ID = "platform.core_fix.cache.swa_bounded_replay"
ENGINE_PATCH_ID = "platform.core_fix.cache.swa_bounded_replay.engine"
TARGETS = (f"{TARGET_MODULE}.CacheConfig",)
ENGINE_TARGETS = (
    f"{ENGINE_TARGET_MODULE}.EngineArgs.__init__",
    f"{ENGINE_TARGET_MODULE}.EngineArgs.add_cli_args",
    f"{ENGINE_TARGET_MODULE}.EngineArgs.from_cli_args",
    f"{ENGINE_TARGET_MODULE}.EngineArgs.create_engine_config",
)
_CACHE_MARKER = "_vllm_hcu_swa_bounded_replay_cache_applied"
_ENGINE_MARKER = "_vllm_hcu_swa_bounded_replay_engine_applied"
_DEFAULT = True


def _ensure_cache_attr(cache_config: object, value: bool = _DEFAULT) -> None:
    if not hasattr(cache_config, "swa_bounded_replay"):
        object.__setattr__(cache_config, "swa_bounded_replay", value)


def _install_cache_config(module: ModuleType) -> bool:
    cache_module = load_exact_module(TARGET_MODULE, module)
    cache_config = getattr(cache_module, "CacheConfig", None)
    if not isinstance(cache_config, type):
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGET_MODULE}.CacheConfig is missing"
        )
    if getattr(cache_config, _CACHE_MARKER, False):
        return False

    setattr(cache_config, "swa_bounded_replay", _DEFAULT)

    # Always wrap __init__ so CacheConfig(swa_bounded_replay=False) is accepted
    # before the underlying dataclass/Pydantic constructor rejects the keyword.
    original_init = require_callable(
        cache_config, "__init__", f"{TARGET_MODULE}.CacheConfig.__init__"
    )

    @functools.wraps(original_init)
    def hcu_init(self, *args: Any, **kwargs: Any) -> None:
        if "swa_bounded_replay" in kwargs:
            value = kwargs.pop("swa_bounded_replay")
            if value is None:
                value = _DEFAULT
        else:
            value = None
        original_init(self, *args, **kwargs)
        if value is None:
            _ensure_cache_attr(self, _DEFAULT)
        else:
            object.__setattr__(self, "swa_bounded_replay", bool(value))

    setattr(cache_config, "__init__", hcu_init)

    post_init = getattr(cache_config, "__post_init__", None)
    if callable(post_init):

        @functools.wraps(post_init)
        def hcu_post_init(self, *args: Any, **kwargs: Any) -> Any:
            result = post_init(self, *args, **kwargs)
            _ensure_cache_attr(self, _DEFAULT)
            return result

        setattr(cache_config, "__post_init__", hcu_post_init)

    setattr(cache_config, _CACHE_MARKER, True)
    return True


def _install_engine_args(module: ModuleType | None = None) -> bool:
    arg_utils = load_exact_module(ENGINE_TARGET_MODULE, module)
    if getattr(arg_utils, _ENGINE_MARKER, False):
        return False

    engine_args = getattr(arg_utils, "EngineArgs", None)
    if not isinstance(engine_args, type):
        raise PatchCompatibilityError(
            f"required HCU patch target {ENGINE_TARGET_MODULE}.EngineArgs is missing"
        )

    setattr(engine_args, "swa_bounded_replay", _DEFAULT)

    original_init = vars(engine_args).get("__init__")
    if not callable(original_init):
        raise PatchCompatibilityError(
            f"required HCU patch target {ENGINE_TARGETS[0]} is missing"
        )

    @functools.wraps(original_init)
    def swa_init(self, *args: Any, **kwargs: Any) -> None:
        value = kwargs.pop("swa_bounded_replay", _DEFAULT)
        if value is None:
            value = _DEFAULT
        original_init(self, *args, **kwargs)
        object.__setattr__(self, "swa_bounded_replay", bool(value))

    setattr(engine_args, "__init__", swa_init)

    original_add_cli = getattr(engine_args, "add_cli_args", None)
    if not callable(original_add_cli):
        raise PatchCompatibilityError(
            f"required HCU patch target {ENGINE_TARGETS[1]} is missing"
        )

    @functools.wraps(original_add_cli)
    def swa_add_cli_args(parser: argparse.ArgumentParser):
        result = original_add_cli(parser)
        target_group = result
        for group in getattr(result, "_action_groups", ()):
            if getattr(group, "title", None) == "CacheConfig":
                target_group = group
                break
        existing = {
            opt
            for action in getattr(result, "_actions", ())
            for opt in getattr(action, "option_strings", ())
        }
        if "--swa-bounded-replay" not in existing:
            target_group.add_argument(
                "--swa-bounded-replay",
                action=argparse.BooleanOptionalAction,
                default=_DEFAULT,
                help=(
                    "Keep DeepSeek-V4.1 sliding-window KV out of prefix caching "
                    "and run decoder layers past the last KV source on each "
                    "request's trailing window in eager prefill (requires "
                    "model runner V2)."
                ),
            )
        return result

    setattr(engine_args, "add_cli_args", staticmethod(swa_add_cli_args))

    # Must preserve subclass type: AsyncEngineArgs.from_cli_args must keep
    # returning AsyncEngineArgs (which owns enable_log_requests). Grab the
    # unbound function via vars()/__func__, same as patch_engine_args.
    from_cli_descriptor = vars(engine_args).get("from_cli_args")
    if isinstance(from_cli_descriptor, classmethod):
        original_from_cli = from_cli_descriptor.__func__
    elif callable(from_cli_descriptor):
        original_from_cli = from_cli_descriptor
    else:
        raise PatchCompatibilityError(
            f"required HCU patch target {ENGINE_TARGETS[2]} is missing"
        )

    @classmethod
    @functools.wraps(original_from_cli)
    def swa_from_cli_args(cls, args: argparse.Namespace):
        instance = original_from_cli(cls, args)
        if hasattr(args, "swa_bounded_replay"):
            object.__setattr__(
                instance, "swa_bounded_replay", bool(args.swa_bounded_replay)
            )
        else:
            object.__setattr__(instance, "swa_bounded_replay", _DEFAULT)
        return instance

    setattr(engine_args, "from_cli_args", swa_from_cli_args)

    original_create = getattr(engine_args, "create_engine_config", None)
    if not callable(original_create):
        raise PatchCompatibilityError(
            f"required HCU patch target {ENGINE_TARGETS[3]} is missing"
        )

    @functools.wraps(original_create)
    def swa_create_engine_config(self, *args: Any, **kwargs: Any):
        config = original_create(self, *args, **kwargs)
        value = bool(getattr(self, "swa_bounded_replay", _DEFAULT))
        cache_config = getattr(config, "cache_config", None)
        if cache_config is not None:
            object.__setattr__(cache_config, "swa_bounded_replay", value)
        return config

    setattr(engine_args, "create_engine_config", swa_create_engine_config)
    setattr(arg_utils, _ENGINE_MARKER, True)
    return True


def apply_to_module(module: ModuleType) -> bool:
    """Exact-import callback for ``vllm.config.cache``."""

    # Coordinator already records the patch; do not nest apply_once/run_patch.
    return _install_cache_config(module)


def apply_engine_args_to_module(module: ModuleType) -> bool:
    """Exact-import / post-engine-args callback for ``vllm.engine.arg_utils``."""

    return _install_engine_args(module)


def apply(module: ModuleType | None = None) -> bool:
    cache_mod = load_exact_module(TARGET_MODULE, module)
    applied = apply_once(
        patch_id=PATCH_ID,
        targets=TARGETS,
        marker_owner=getattr(cache_mod, "CacheConfig", cache_mod),
        marker=_CACHE_MARKER,
        callback=lambda: _install_cache_config(cache_mod),
    )
    engine_mod = load_exact_module(ENGINE_TARGET_MODULE, None)
    applied = (
        apply_once(
            patch_id=ENGINE_PATCH_ID,
            targets=ENGINE_TARGETS,
            marker_owner=engine_mod,
            marker=_ENGINE_MARKER,
            callback=lambda: _install_engine_args(engine_mod),
        )
        or applied
    )
    return applied


__all__ = [
    "PATCH_ID",
    "ENGINE_PATCH_ID",
    "TARGETS",
    "ENGINE_TARGETS",
    "TARGET_MODULE",
    "ENGINE_TARGET_MODULE",
    "apply",
    "apply_to_module",
    "apply_engine_args_to_module",
]
