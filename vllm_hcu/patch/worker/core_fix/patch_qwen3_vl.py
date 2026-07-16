# SPDX-License-Identifier: Apache-2.0
"""Normalize Qwen3-VL dense ``tie_word_embeddings`` before construction."""

from __future__ import annotations

import functools
from types import ModuleType

from ._common import (
    PatchCompatibilityError,
    get_hf_config,
    load_exact_module,
    require_class,
    require_model_init,
)

TARGET_MODULE = "vllm.model_executor.models.qwen3_vl"
PATCH_ID = "worker.core_fix.qwen3_vl.tie_word_embeddings"
TARGET_SYMBOL = f"{TARGET_MODULE}.Qwen3LLMForCausalLM.__init__"
_CLASS_MARKER = "_vllm_hcu_qwen3_vl_tie_embeddings_applied"
_WRAPPER_MARKER = "_vllm_hcu_qwen3_vl_tie_embeddings_wrapper"


def apply_to_module(module: ModuleType) -> bool:
    qwen3_vl = load_exact_module(TARGET_MODULE, module)
    model_class = require_class(
        qwen3_vl,
        "Qwen3LLMForCausalLM",
        f"{TARGET_MODULE}.Qwen3LLMForCausalLM",
    )
    current = require_model_init(model_class, TARGET_SYMBOL)
    if getattr(model_class, _CLASS_MARKER, False):
        if not getattr(current, _WRAPPER_MARKER, False):
            raise PatchCompatibilityError(
                f"required HCU patch marker for {TARGET_SYMBOL} is stale"
            )
        return False
    original = current

    @functools.wraps(original)
    def hcu_init(self, *, vllm_config, prefix=""):
        config = get_hf_config(vllm_config, TARGET_SYMBOL)
        if not hasattr(config, "tie_word_embeddings"):
            setattr(config, "tie_word_embeddings", False)
        return original(self, vllm_config=vllm_config, prefix=prefix)

    setattr(hcu_init, _WRAPPER_MARKER, True)
    setattr(model_class, "_vllm_hcu_original_init", original)
    setattr(model_class, "__init__", hcu_init)
    setattr(model_class, _CLASS_MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "apply", "apply_to_module"]
