# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Port CompilationConfig decoder-replay CUDA-graph fields onto pinned vLLM 0.29.

0.29.1rc0.dev283 lacks ``cudagraph_decoder_replay``,
``decoder_replay_cudagraph_capture_sizes``, and ``decoder_replay_trim_threshold``
(upstream #59532). Defaults match upstream: graphs on, empty capture-size list
(auto), trim threshold 768.
"""

from __future__ import annotations

import functools
from types import ModuleType
from typing import Any

from ._common import (
    PatchCompatibilityError,
    apply_once,
    load_exact_module,
    require_callable,
)

TARGET_MODULE = "vllm.config.compilation"
PATCH_ID = "platform.core_fix.compilation.decoder_replay_cudagraph"
TARGETS = (f"{TARGET_MODULE}.CompilationConfig",)
_MARKER = "_vllm_hcu_decoder_replay_compilation_applied"

_DEFAULTS: dict[str, Any] = {
    "cudagraph_decoder_replay": True,
    "decoder_replay_cudagraph_capture_sizes": [],
    "decoder_replay_trim_threshold": 768,
}


def _coerce_field(name: str, value: Any) -> Any:
    default = _DEFAULTS[name]
    if value is None:
        return list(default) if isinstance(default, list) else default
    if isinstance(default, list):
        return list(value)
    return value


def _ensure_fields(compilation_config: object) -> None:
    # List defaults are installed on the class as a shared sentinel; always
    # materialize a per-instance copy so one engine cannot mutate another's.
    instance_dict = getattr(compilation_config, "__dict__", None)
    for name, default in _DEFAULTS.items():
        if isinstance(default, list):
            if instance_dict is None or name not in instance_dict:
                current = getattr(compilation_config, name, None)
                value = list(current) if isinstance(current, list) else list(default)
                object.__setattr__(compilation_config, name, value)
            continue
        if not hasattr(compilation_config, name):
            object.__setattr__(compilation_config, name, default)


def _pop_replay_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    extras: dict[str, Any] = {}
    for name in _DEFAULTS:
        if name in kwargs:
            extras[name] = kwargs.pop(name)
    return extras


def apply_to_module(module: ModuleType) -> bool:
    compilation_module = load_exact_module(TARGET_MODULE, module)
    compilation_config = getattr(compilation_module, "CompilationConfig", None)
    if not isinstance(compilation_config, type):
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGET_MODULE}.CompilationConfig is missing"
        )
    if getattr(compilation_config, _MARKER, False):
        return False

    for name, default in _DEFAULTS.items():
        if not hasattr(compilation_config, name):
            setattr(
                compilation_config,
                name,
                list(default) if isinstance(default, list) else default,
            )

    # Always wrap __init__ so CompilationConfig(decoder_replay_trim_threshold=...)
    # (and deserialization that hits __init__) can pass the new keywords.
    original_init = require_callable(
        compilation_config,
        "__init__",
        f"{TARGET_MODULE}.CompilationConfig.__init__",
    )

    @functools.wraps(original_init)
    def hcu_init(self, *args: Any, **kwargs: Any) -> None:
        extras = _pop_replay_kwargs(kwargs)
        original_init(self, *args, **kwargs)
        for name, value in extras.items():
            object.__setattr__(self, name, _coerce_field(name, value))
        _ensure_fields(self)

    setattr(compilation_config, "__init__", hcu_init)

    post_init = getattr(compilation_config, "__post_init__", None)
    if callable(post_init):

        @functools.wraps(post_init)
        def hcu_post_init(self, *args: Any, **kwargs: Any) -> Any:
            result = post_init(self, *args, **kwargs)
            _ensure_fields(self)
            return result

        setattr(compilation_config, "__post_init__", hcu_post_init)

    setattr(compilation_config, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    compilation_module = load_exact_module(TARGET_MODULE, module)
    compilation_config = getattr(compilation_module, "CompilationConfig", None)
    if not isinstance(compilation_config, type):
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGET_MODULE}.CompilationConfig is missing"
        )
    return apply_once(
        patch_id=PATCH_ID,
        targets=TARGETS,
        marker_owner=compilation_config,
        marker=_MARKER,
        callback=lambda: apply_to_module(compilation_module),
    )


__all__ = [
    "PATCH_ID",
    "TARGET_MODULE",
    "TARGETS",
    "apply",
    "apply_to_module",
]
