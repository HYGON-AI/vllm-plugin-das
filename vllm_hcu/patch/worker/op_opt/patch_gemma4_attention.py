# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Inject the model-owned FA backend through Gemma4's local Attention binding."""

from __future__ import annotations

import functools
import inspect
from types import ModuleType

from ._common import PatchCompatibilityError, load_exact_module, require_class

TARGET_MODULE = "vllm.model_executor.models.gemma4"
PATCH_ID = "worker.op_opt.gemma4.fa_attention"
TARGETS = (f"{TARGET_MODULE}.Attention",)
_MARKER = "_vllm_hcu_gemma4_attention_injection"


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    attention = require_class(target, "Attention", TARGETS[0])
    if getattr(attention, _MARKER, False):
        return False
    signature = inspect.signature(attention.__init__)
    if "attn_backend" not in signature.parameters:
        raise PatchCompatibilityError(
            f"Gemma4 attention injection requires attn_backend in {signature}"
        )

    from vllm_hcu.models.gemma4.config import select_gemma4_hcu_attention_backend

    class Gemma4HcuAttentionLayer(attention):
        @functools.wraps(attention.__init__)
        def __init__(self, *args, **kwargs):
            bound = signature.bind(self, *args, **kwargs)
            bound.apply_defaults()
            if bound.arguments["attn_backend"] is None:
                bound.arguments["attn_backend"] = select_gemma4_hcu_attention_backend(
                    head_dim=bound.arguments["head_size"],
                    is_sliding=(
                        bound.arguments["per_layer_sliding_window"] is not None
                    ),
                )
            super().__init__(*bound.args[1:], **bound.kwargs)

    setattr(Gemma4HcuAttentionLayer, _MARKER, True)
    target.Attention = Gemma4HcuAttentionLayer
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGETS", "TARGET_MODULE", "apply", "apply_to_module"]
