# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
from __future__ import annotations
from types import ModuleType
from vllm_hcu.runtime_compat.step5_patch_contract import (
    already_applied,
    remember_targets,
    require_signatures,
)
from ._common import load_exact_module, require_callable

TARGET_MODULE = "vllm.config.speculative"
PATCH_ID = "platform.core_fix.step5_speculative_config"
TARGETS = (
    "vllm.config.speculative.SpeculativeConfig.hf_config_override",
    "vllm.config.speculative.SpeculativeConfig.__post_init__",
    "vllm.config.speculative.MTPModelTypes",
)
_MARKER = "_vllm_hcu_step5_speculative_config_applied"

SIGNATURES = (
    ("vllm.config.speculative.SpeculativeConfig.hf_config_override", ("hf_config",)),
    ("vllm.config.speculative.SpeculativeConfig.__post_init__", ("self",)),
)
REVIEWED_SOURCES = ()


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    if already_applied(target, _MARKER, TARGETS):
        return False
    require_signatures(target, SIGNATURES, REVIEWED_SOURCES)
    import functools
    from typing import Literal, get_args

    cls = getattr(target, "SpeculativeConfig", None)
    original = require_callable(cls, "hf_config_override", TARGETS[0])
    original_post_init = require_callable(cls, "__post_init__", TARGETS[1])
    if "step5_mtp" not in get_args(target.MTPModelTypes):
        target.MTPModelTypes = Literal[(*get_args(target.MTPModelTypes), "step5_mtp")]

    @functools.wraps(original)
    def hf_config_override(hf_config):
        architectures = getattr(hf_config, "architectures", None) or ()
        if (
            getattr(hf_config, "model_type", None) in ("step5", "step5_vl")
            or "Step5ForConditionalGeneration" in architectures
        ):
            quant = getattr(hf_config, "quantization_config", None)
            text = getattr(hf_config, "text_config", hf_config)
            if quant is not None and getattr(text, "quantization_config", None) is None:
                text.update({"quantization_config": quant})
            text.model_type = "step5_mtp"
            text.update(
                {
                    "n_predict": getattr(text, "num_nextn_predict_layers", 3),
                    "architectures": ["Step5MTP"],
                }
            )
            return text
        return original(hf_config)

    cls.hf_config_override = staticmethod(hf_config_override)

    @functools.wraps(original_post_init)
    def post_init(self):
        # Upstream makes an in-checkpoint DeepSeek-V4 DSpark draft inherit the
        # target quantization unconditionally. Preserve an explicitly selected
        # independent draft format so get_draft_quant_config dispatches to the
        # requested HCU facade (not the target's compressed-tensors config).
        explicit_quantization = getattr(self, "quantization", None)
        result = original_post_init(self)
        draft = getattr(self, "draft_model_config", None)
        hf_config = getattr(draft, "hf_config", None)
        if (
            explicit_quantization is not None
            and getattr(self, "method", None) == "dspark"
            and getattr(hf_config, "model_type", None) == "deepseek_v4"
            and "DSparkDraftModel" in (getattr(draft, "architectures", ()) or ())
        ):
            draft.quantization = explicit_quantization
        return result

    cls.__post_init__ = post_init
    remember_targets(target, _MARKER, TARGETS)
    return True


def apply(module=None):
    return apply_to_module(load_exact_module(TARGET_MODULE, module))
