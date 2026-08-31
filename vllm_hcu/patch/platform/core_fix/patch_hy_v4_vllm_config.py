# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Register HYV4 runtime defaults on vLLM v0.25.1."""

from __future__ import annotations

import functools
import os
from types import ModuleType

from ._common import (
    PatchCompatibilityError,
    apply_once,
    load_exact_module,
    require_callable,
    require_positional_signature,
)

TARGET_MODULE = "vllm.config.vllm"
PATCH_ID = "platform.core_fix.hy_v4_vllm_config"
TARGETS = (
    f"{TARGET_MODULE}.DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES",
    f"{TARGET_MODULE}.VllmConfig.__post_init__",
)
_MARKER = "_vllm_hcu_hy_v4_runtime_config_applied"
_TARGET_ARCHITECTURE = "HYV4ForCausalLM"
_BREAKABLE_ARCHITECTURES = frozenset({_TARGET_ARCHITECTURE, "HYV4MTPModel"})


def apply_to_module(module: ModuleType) -> bool:
    vllm_module = load_exact_module(TARGET_MODULE, module)
    if getattr(vllm_module, _MARKER, False):
        return False

    architectures = getattr(
        vllm_module,
        "DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES",
        None,
    )
    if not isinstance(architectures, frozenset):
        raise PatchCompatibilityError(
            "required HCU patch target "
            "vllm.config.vllm.DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES "
            "is incompatible"
        )
    config_cls = getattr(vllm_module, "VllmConfig", None)
    if not isinstance(config_cls, type):
        raise PatchCompatibilityError(
            "required HCU patch target vllm.config.vllm.VllmConfig is missing"
        )
    original_post_init = require_callable(config_cls, "__post_init__", TARGETS[1])
    require_positional_signature(original_post_init, TARGETS[1], ("self",))

    @functools.wraps(original_post_init)
    def hcu_post_init(self) -> None:
        model_config = getattr(self, "model_config", None)
        model_architectures = getattr(model_config, "architectures", ())
        if (
            "VLLM_USE_BREAKABLE_CUDAGRAPH" not in os.environ
            and any(
                architecture in _BREAKABLE_ARCHITECTURES
                for architecture in model_architectures
            )
        ):
            os.environ["VLLM_USE_BREAKABLE_CUDAGRAPH"] = "1"
        original_post_init(self)

    vllm_module.DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES = architectures | {
        _TARGET_ARCHITECTURE
    }
    setattr(config_cls, "_vllm_hcu_original_hy_v4_post_init", original_post_init)
    setattr(config_cls, "__post_init__", hcu_post_init)
    setattr(vllm_module, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    vllm_module = load_exact_module(TARGET_MODULE, module)
    return apply_once(
        patch_id=PATCH_ID,
        targets=TARGETS,
        marker_owner=vllm_module,
        marker=_MARKER,
        callback=lambda: apply_to_module(vllm_module),
    )


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
