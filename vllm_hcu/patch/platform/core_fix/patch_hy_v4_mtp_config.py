# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Register HYV4 as a native MTP target on vLLM v0.25.1."""

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
PATCH_ID = "platform.core_fix.hy_v4_mtp_config"
TARGETS = (f"{TARGET_MODULE}.SpeculativeConfig.hf_config_override",)
_MARKER = "_vllm_hcu_hy_v4_mtp_config_applied"
_MODEL_TYPE = "hy_v4_mtp"


def apply_to_module(module: ModuleType) -> bool:
    speculative = load_exact_module(TARGET_MODULE, module)
    if getattr(speculative, _MARKER, False):
        return False

    speculative_config = getattr(speculative, "SpeculativeConfig", None)
    if not isinstance(speculative_config, type):
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGET_MODULE}.SpeculativeConfig is missing"
        )
    original = require_callable(
        speculative_config,
        "hf_config_override",
        TARGETS[0],
    )
    require_positional_signature(original, TARGETS[0], ("hf_config",))

    mtp_model_types = getattr(speculative, "MTPModelTypes", None)
    existing_types = get_args(mtp_model_types)
    if not existing_types or "mtp" not in existing_types:
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGET_MODULE}.MTPModelTypes is incompatible"
        )
    if _MODEL_TYPE not in existing_types:
        speculative.MTPModelTypes = Literal[(*existing_types, _MODEL_TYPE)]

    @functools.wraps(original)
    def hcu_hf_config_override(hf_config):
        hf_config = original(hf_config)
        if getattr(hf_config, "model_type", None) != "hy_v4":
            return hf_config
        n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
        hf_config.model_type = _MODEL_TYPE
        hf_config.update(
            {
                "n_predict": n_predict,
                "architectures": ["HYV4MTPModel"],
                # PR #54160 widens autoregressive draft buffers by hc_mult.
                # HYV4's checkpoint-native MTP block is dense (its hnorm is
                # hidden_size and eh_proj is [hidden, 2 * hidden]), so it
                # consumes the target's post-iHC hidden state instead.
                "hc_mult": 1,
            }
        )
        return hf_config

    setattr(
        speculative_config,
        "_vllm_hcu_original_hf_config_override",
        original,
    )
    setattr(
        speculative_config,
        "hf_config_override",
        staticmethod(hcu_hf_config_override),
    )
    setattr(speculative, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    speculative = load_exact_module(TARGET_MODULE, module)
    return apply_once(
        patch_id=PATCH_ID,
        targets=TARGETS,
        marker_owner=speculative,
        marker=_MARKER,
        callback=lambda: apply_to_module(speculative),
    )


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
