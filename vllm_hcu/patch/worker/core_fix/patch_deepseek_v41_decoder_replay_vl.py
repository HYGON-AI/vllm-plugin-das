# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Expose decoder replay through the AMD multimodal wrapper."""

from __future__ import annotations

from types import ModuleType

from vllm_hcu.patch.worker.core_fix._common import load_exact_module, require_class

TARGET_MODULE = "vllm.models.deepseek_v41.amd.vl_model"
PATCH_ID = "worker.core_fix.deepseek_v41.decoder_replay_vl"
TARGETS = (f"{TARGET_MODULE}.DeepseekV41ForCausalLM.decoder_replay_layers",)
_MARKER = "_vllm_hcu_dsv41_decoder_replay_vl_applied"


def apply_to_module(module: ModuleType) -> bool:
    mod = load_exact_module(TARGET_MODULE, module)
    cls = require_class(
        mod, "DeepseekV41ForCausalLM", f"{TARGET_MODULE}.DeepseekV41ForCausalLM"
    )
    if getattr(cls, _MARKER, False):
        return False
    if not isinstance(getattr(cls, "decoder_replay_layers", None), property):

        @property
        def decoder_replay_layers(self):
            # VL wraps DeepseekV41LLMForCausalLM as language_model; that wrapper
            # exposes decoder_replay_layers via the inner DeepseekV4Model.
            return self.language_model.decoder_replay_layers

        cls.decoder_replay_layers = decoder_replay_layers
    setattr(cls, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGETS", "TARGET_MODULE", "apply", "apply_to_module"]
