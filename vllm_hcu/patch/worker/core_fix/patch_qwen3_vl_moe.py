# SPDX-License-Identifier: Apache-2.0
"""Atomically normalize both Qwen3-VL-MoE constructor contracts."""

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

TARGET_MODULE = "vllm.model_executor.models.qwen3_vl_moe"
PATCH_ID = "worker.core_fix.qwen3_vl_moe.tie_word_embeddings"
LLM_TARGET = f"{TARGET_MODULE}.Qwen3MoeLLMForCausalLM.__init__"
VL_TARGET = f"{TARGET_MODULE}.Qwen3VLMoeForConditionalGeneration.__init__"
_MODULE_MARKER = "_vllm_hcu_qwen3_vl_moe_tie_embeddings_applied"
_WRAPPER_MARKER = "_vllm_hcu_qwen3_vl_moe_tie_embeddings_wrapper"


def _is_wrapper(function: object) -> bool:
    return bool(getattr(function, _WRAPPER_MARKER, False))


def apply_to_module(module: ModuleType) -> bool:
    """Validate both targets before installing either constructor wrapper."""

    qwen3_vl_moe = load_exact_module(TARGET_MODULE, module)
    llm_class = require_class(
        qwen3_vl_moe,
        "Qwen3MoeLLMForCausalLM",
        f"{TARGET_MODULE}.Qwen3MoeLLMForCausalLM",
    )
    vl_class = require_class(
        qwen3_vl_moe,
        "Qwen3VLMoeForConditionalGeneration",
        f"{TARGET_MODULE}.Qwen3VLMoeForConditionalGeneration",
    )
    llm_current = require_model_init(llm_class, LLM_TARGET)
    vl_current = require_model_init(vl_class, VL_TARGET)

    if getattr(qwen3_vl_moe, _MODULE_MARKER, False):
        if not (_is_wrapper(llm_current) and _is_wrapper(vl_current)):
            raise PatchCompatibilityError(
                "required atomic Qwen3-VL-MoE patch marker is stale or partial"
            )
        return False
    if _is_wrapper(llm_current) or _is_wrapper(vl_current):
        raise PatchCompatibilityError(
            "refusing a partial Qwen3-VL-MoE constructor patch state"
        )

    llm_original = llm_current
    vl_original = vl_current

    @functools.wraps(llm_original)
    def hcu_llm_init(self, *, vllm_config, prefix=""):
        config = get_hf_config(vllm_config, LLM_TARGET)
        if not hasattr(config, "tie_word_embeddings"):
            setattr(config, "tie_word_embeddings", False)
        return llm_original(self, vllm_config=vllm_config, prefix=prefix)

    @functools.wraps(vl_original)
    def hcu_vl_init(self, *, vllm_config, prefix=""):
        config = get_hf_config(vllm_config, VL_TARGET)
        try:
            text_config = getattr(config, "text_config")
        except AttributeError as exc:
            raise PatchCompatibilityError(
                f"required HCU patch runtime input for {VL_TARGET} has no text_config"
            ) from exc
        tie_word_embeddings = getattr(config, "tie_word_embeddings", None)
        if tie_word_embeddings is not None:
            setattr(text_config, "tie_word_embeddings", tie_word_embeddings)
        return vl_original(self, vllm_config=vllm_config, prefix=prefix)

    setattr(hcu_llm_init, _WRAPPER_MARKER, True)
    setattr(hcu_vl_init, _WRAPPER_MARKER, True)

    # Class attribute assignment is normally infallible, but restore both
    # descriptors if a metaclass rejects any part of the atomic installation.
    try:
        setattr(llm_class, "__init__", hcu_llm_init)
        setattr(vl_class, "__init__", hcu_vl_init)
        setattr(qwen3_vl_moe, _MODULE_MARKER, True)
    except BaseException:
        setattr(llm_class, "__init__", llm_original)
        setattr(vl_class, "__init__", vl_original)
        if hasattr(qwen3_vl_moe, _MODULE_MARKER):
            delattr(qwen3_vl_moe, _MODULE_MARKER)
        raise

    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "apply", "apply_to_module"]
