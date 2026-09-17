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
# Fail closed for an architecture this adapter has not been reviewed against;
# official main owns the architecture-to-field mapping used below.
_SUPPORTED_ARCHITECTURES = frozenset(
    {
        "DeepseekV41ForCausalLM",
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


def _require_audited_contract(
    verify_model_config: object,
    engram_module: ModuleType,
) -> None:
    """Validate the official Engram contract this adapter relies on.

    Official main validates Engram through the module-level
    ``_NGRAM_LAYER_FIELDS`` mapping plus ``model_has_engram_layers`` instead of
    hard-coding one model family, so the adapter checks the mapping and the
    helper rather than byte-code constants.
    """

    code = getattr(verify_model_config, "__code__", None)
    if code is None:
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGETS[0]} has no inspectable code"
        )

    missing_names = _REQUIRED_CODE_NAMES.difference(code.co_names)
    if missing_names:
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGETS[0]} has incompatible source "
            f"contract; missing {sorted(missing_names)}"
        )

    layer_fields = getattr(engram_module, "_NGRAM_LAYER_FIELDS", None)
    if not isinstance(layer_fields, dict):
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGET_MODULE}._NGRAM_LAYER_FIELDS "
            "is missing"
        )
    undeclared = sorted(
        architecture
        for architecture in _SUPPORTED_ARCHITECTURES
        if architecture not in layer_fields
    )
    if undeclared:
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGET_MODULE}._NGRAM_LAYER_FIELDS "
            f"does not declare {undeclared}"
        )
    if not callable(getattr(engram_module, "model_has_engram_layers", None)):
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGET_MODULE}.model_has_engram_layers "
            "is missing"
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
    _require_audited_contract(verify_model_config, engram_module)

    has_engram_layers = getattr(engram_module, "model_has_engram_layers")

    @functools.wraps(verify_model_config)
    def hcu_verify_model_config(self, model_config: object | None) -> None:
        from vllm.platforms import current_platform

        if (
            current_platform.is_cuda_alike()
            and model_config is not None
            and getattr(model_config, "architecture", None)
            in _SUPPORTED_ARCHITECTURES
            and bool(has_engram_layers(model_config))
        ):
            if getattr(self, "embedding_across_dp", False):
                raise ValueError(
                    "HCU Engram does not support "
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
