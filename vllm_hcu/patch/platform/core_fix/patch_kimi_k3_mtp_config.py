# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Register the checkpoint-native Kimi-K3 draft configuration."""

from __future__ import annotations

import functools
from types import ModuleType
from typing import Literal, get_args

from ._common import (
    PatchCompatibilityError,
    apply_once,
    load_exact_module,
    require_callable,
    require_positional_signature,
)

TARGET_MODULE = "vllm.config.speculative"
PATCH_ID = "platform.core_fix.kimi_k3_mtp_config"
TARGETS = (f"{TARGET_MODULE}.SpeculativeConfig.hf_config_override",)
_MARKER = "_vllm_hcu_kimi_k3_mtp_config_applied"


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    if getattr(target, _MARKER, False):
        return False
    config_class = getattr(target, "SpeculativeConfig", None)
    original = require_callable(config_class, "hf_config_override", TARGETS[0])
    require_positional_signature(original, TARGETS[0], ("hf_config",))
    types = get_args(getattr(target, "MTPModelTypes", None))
    if "mtp" not in types:
        raise PatchCompatibilityError("SpeculativeConfig.MTPModelTypes is incompatible")

    @functools.wraps(original)
    def override(hf_config):
        if getattr(hf_config, "model_type", None) not in ("kimi_k3", "kimi_k3_mtp"):
            return original(hf_config)
        text_config = hf_config.get_text_config()
        count = getattr(text_config, "num_nextn_predict_layers", None)
        if type(count) is not int or count < 1:
            raise ValueError(
                "Kimi-K3 MTP requires positive text_config.num_nextn_predict_layers"
            )
        hf_config.model_type = "kimi_k3_mtp"
        hf_config.update({"n_predict": count, "architectures": ["KimiK3MTPModel"]})
        return hf_config

    if "kimi_k3_mtp" not in types:
        target.MTPModelTypes = Literal[(*types, "kimi_k3_mtp")]
    config_class.hf_config_override = staticmethod(override)
    setattr(target, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    return apply_once(
        patch_id=PATCH_ID, targets=TARGETS, marker_owner=target,
        marker=_MARKER, callback=lambda: apply_to_module(target),
    )


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
