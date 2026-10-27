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


def _ensure_fields(compilation_config: object) -> None:
    for name, default in _DEFAULTS.items():
        if not hasattr(compilation_config, name):
            value = list(default) if isinstance(default, list) else default
            object.__setattr__(compilation_config, name, value)


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

    post_init = getattr(compilation_config, "__post_init__", None)
    if callable(post_init):

        @functools.wraps(post_init)
        def hcu_post_init(self, *args: Any, **kwargs: Any) -> Any:
            result = post_init(self, *args, **kwargs)
            _ensure_fields(self)
            return result

        setattr(compilation_config, "__post_init__", hcu_post_init)
    else:
        original_init = require_callable(
            compilation_config,
            "__init__",
            f"{TARGET_MODULE}.CompilationConfig.__init__",
        )

        @functools.wraps(original_init)
        def hcu_init(self, *args: Any, **kwargs: Any) -> None:
            original_init(self, *args, **kwargs)
            _ensure_fields(self)

        setattr(compilation_config, "__init__", hcu_init)

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
