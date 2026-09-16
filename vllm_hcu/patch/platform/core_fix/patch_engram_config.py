# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Allow upstream Engram configuration on CUDA-like HCU platforms."""

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

TARGET_MODULE = "vllm.config.engram"
PATCH_ID = "platform.core_fix.engram_config.hcu"
TARGETS = (f"{TARGET_MODULE}.EngramConfig.verify_model_config",)
_MARKER = "_vllm_hcu_engram_config_patch_applied"
_ORIGINAL = "_vllm_hcu_original_verify_model_config"
_SUPPORTED_ARCHITECTURES = frozenset(
    {
        "Qwen4ExpForCausalLM",
        "Qwen4ExpForConditionalGeneration",
    }
)
_REQUIRED_CODE_NAMES = frozenset(
    {
        "architecture",
        "hf_text_config",
        "is_cuda",
    }
)
_REQUIRED_CODE_CONSTANTS = frozenset(
    {
        "Qwen4ExpForCausalLM",
        "Qwen4ExpForConditionalGeneration",
        "ple_layer_ids",
    }
)


def _require_audited_contract(verify_model_config: object) -> None:
    code = getattr(verify_model_config, "__code__", None)
    if code is None:
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGETS[0]} has no inspectable code"
        )

    missing_names = _REQUIRED_CODE_NAMES.difference(code.co_names)
    missing_constants = _REQUIRED_CODE_CONSTANTS.difference(code.co_consts)
    if missing_names or missing_constants:
        missing = sorted((*missing_names, *missing_constants))
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGETS[0]} has incompatible source "
            f"contract; missing {missing}"
        )


def apply_to_module(module: ModuleType) -> bool:
    """Wrap Engram model validation for the HCU CUDA-like platform."""

    engram_module = load_exact_module(TARGET_MODULE, module)
    engram_config = getattr(engram_module, "EngramConfig", None)
    if not isinstance(engram_config, type):
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGET_MODULE}.EngramConfig is missing"
        )
    if getattr(engram_config, _MARKER, False):
        return False

    verify_model_config = require_callable(
        engram_config,
        "verify_model_config",
        TARGETS[0],
    )
    require_positional_signature(
        verify_model_config,
        TARGETS[0],
        ("self", "model_config"),
    )
    _require_audited_contract(verify_model_config)

    @functools.wraps(verify_model_config)
    def hcu_verify_model_config(self, model_config: object | None) -> None:
        from vllm.platforms import current_platform

        if (
            current_platform.is_cuda_alike()
            and model_config is not None
            and getattr(model_config, "architecture", None)
            in _SUPPORTED_ARCHITECTURES
            and getattr(
                getattr(model_config, "hf_text_config", None),
                "ple_layer_ids",
                None,
            )
        ):
            if getattr(self, "embedding_across_dp", False):
                raise ValueError(
                    "HCU Qwen4Exp PLE does not support "
                    "engram_config.embedding_across_dp; use the default "
                    "TP-sharded embedding with embedding_across_dp=false"
                )
            return None
        return verify_model_config(self, model_config)

    setattr(engram_config, _ORIGINAL, verify_model_config)
    setattr(engram_config, "verify_model_config", hcu_verify_model_config)
    setattr(engram_config, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    """Install the HCU Engram validation compatibility wrapper once."""

    engram_module = load_exact_module(TARGET_MODULE, module)
    engram_config = getattr(engram_module, "EngramConfig", None)
    if not isinstance(engram_config, type):
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGET_MODULE}.EngramConfig is missing"
        )
    return apply_once(
        patch_id=PATCH_ID,
        targets=TARGETS,
        marker_owner=engram_config,
        marker=_MARKER,
        callback=lambda: apply_to_module(engram_module),
    )


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
