# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Classify HYV4 target and MTP configs as MLA on vLLM v0.25.1."""

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

TARGET_MODULE = "vllm.transformers_utils.model_arch_config_convertor"
PATCH_ID = "platform.core_fix.hy_v4_model_arch_config"
TARGETS = (f"{TARGET_MODULE}.ModelArchConfigConvertorBase.is_deepseek_mla",)
_MARKER = "_vllm_hcu_hy_v4_mla_config_applied"
_MODEL_TYPES = frozenset({"hy_v4", "hy_v4_mtp"})


def apply_to_module(module: ModuleType) -> bool:
    convertor_module = load_exact_module(TARGET_MODULE, module)
    if getattr(convertor_module, _MARKER, False):
        return False

    convertor = getattr(convertor_module, "ModelArchConfigConvertorBase", None)
    if not isinstance(convertor, type):
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGET_MODULE}."
            "ModelArchConfigConvertorBase is missing"
        )
    original = require_callable(convertor, "is_deepseek_mla", TARGETS[0])
    require_positional_signature(original, TARGETS[0], ("self",))

    @functools.wraps(original)
    def hcu_is_deepseek_mla(self) -> bool:
        config = getattr(self, "hf_text_config", None)
        if getattr(config, "model_type", None) in _MODEL_TYPES:
            return getattr(config, "kv_lora_rank", None) is not None
        return original(self)

    setattr(convertor, "_vllm_hcu_original_is_deepseek_mla", original)
    setattr(convertor, "is_deepseek_mla", hcu_is_deepseek_mla)
    setattr(convertor_module, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    convertor_module = load_exact_module(TARGET_MODULE, module)
    return apply_once(
        patch_id=PATCH_ID,
        targets=TARGETS,
        marker_owner=convertor_module,
        marker=_MARKER,
        callback=lambda: apply_to_module(convertor_module),
    )


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
