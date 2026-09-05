# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Normalize `tie_word_embeddings` for Qwen4-Exp constructors."""

from __future__ import annotations

import inspect
import functools
from types import ModuleType

from ._common import (
    PatchCompatibilityError,
    get_hf_config,
    load_exact_module,
    require_class,
)

TARGET_MODULE = "vllm.model_executor.models.qwen4_exp"
TARGET_MODULES = (
    "vllm.models.qwen4_exp",
    TARGET_MODULE,
)
PATCH_ID = "worker.core_fix.qwen4_exp.tie_word_embeddings"
LLM_TARGET = "Qwen4ExpForCausalLM.__init__"
VL_TARGET = "Qwen4ExpForConditionalGeneration.__init__"
_MODULE_MARKER = "_vllm_hcu_qwen4_exp_tie_embeddings_applied"
_WRAPPER_MARKER = "_vllm_hcu_qwen4_exp_tie_embeddings_wrapper"


def _require_model_init(owner: type, target: str) -> tuple[object, object]:
    function = vars(owner).get("__init__")
    if not callable(function):
        raise PatchCompatibilityError(
            f"required HCU patch target {target} is missing"
        )
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError) as exc:
        raise PatchCompatibilityError(
            f"cannot inspect required HCU patch target {target}"
        ) from exc

    parameters = tuple(signature.parameters.values())
    if tuple(parameter.name for parameter in parameters) != (
        "self",
        "vllm_config",
        "prefix",
    ):
        raise PatchCompatibilityError(
            f"required HCU patch target {target} has incompatible signature "
            f"{signature}"
        )
    if (
        parameters[0].kind is not inspect.Parameter.POSITIONAL_OR_KEYWORD
        or parameters[1].kind is not inspect.Parameter.KEYWORD_ONLY
        or parameters[2].kind is not inspect.Parameter.KEYWORD_ONLY
    ):
        raise PatchCompatibilityError(
            f"required HCU patch target {target} has incompatible signature "
            f"{signature}"
        )
    if parameters[2].default not in ("", "model"):
        raise PatchCompatibilityError(
            f"required HCU patch target {target} has incompatible signature "
            f"{signature}"
        )
    return function, parameters[2].default


def _is_wrapper(function: object) -> bool:
    return bool(getattr(function, _WRAPPER_MARKER, False))


def _load_target_module(module: ModuleType | None) -> ModuleType:
    if module is not None:
        if module.__name__ not in TARGET_MODULES:
            expected = " or ".join(TARGET_MODULES)
            raise PatchCompatibilityError(
                f"expected module {expected}, got {module.__name__}"
            )
        return module
    last_error: PatchCompatibilityError | None = None
    for target in TARGET_MODULES:
        try:
            return load_exact_module(target, None)
        except PatchCompatibilityError as exc:
            last_error = exc
            continue
    raise last_error or PatchCompatibilityError(
        "required HCU patch target could not be imported from known qwen4_exp modules"
    )


def apply_to_module(module: ModuleType) -> bool:
    qwen4_exp = _load_target_module(module)
    llm_class = require_class(
        qwen4_exp,
        "Qwen4ExpForCausalLM",
        f"{qwen4_exp.__name__}.Qwen4ExpForCausalLM",
    )
    vl_class = require_class(
        qwen4_exp,
        "Qwen4ExpForConditionalGeneration",
        f"{qwen4_exp.__name__}.Qwen4ExpForConditionalGeneration",
    )
    llm_current, llm_prefix_default = _require_model_init(
        llm_class, f"{qwen4_exp.__name__}.{LLM_TARGET}"
    )
    vl_current, vl_prefix_default = _require_model_init(
        vl_class, f"{qwen4_exp.__name__}.{VL_TARGET}"
    )

    if getattr(qwen4_exp, _MODULE_MARKER, False):
        if not (_is_wrapper(llm_current) and _is_wrapper(vl_current)):
            raise PatchCompatibilityError(
                "required atomic Qwen4-Exp patch marker is stale or partial"
            )
        return False
    if _is_wrapper(llm_current) or _is_wrapper(vl_current):
        raise PatchCompatibilityError(
            "refusing a partial Qwen4-Exp constructor patch state"
        )

    llm_original = llm_current
    vl_original = vl_current

    @functools.wraps(llm_original)
    def hcu_llm_init(self, *, vllm_config, prefix=llm_prefix_default):
        config = get_hf_config(vllm_config, LLM_TARGET)
        if not hasattr(config, "tie_word_embeddings"):
            setattr(config, "tie_word_embeddings", False)
        return llm_original(self, vllm_config=vllm_config, prefix=prefix)

    @functools.wraps(vl_original)
    def hcu_vl_init(self, *, vllm_config, prefix=vl_prefix_default):
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

    # Keep constructor replacement atomic: either both wrappers are installed
    # or both remain untouched.
    try:
        setattr(llm_class, "__init__", hcu_llm_init)
        setattr(vl_class, "__init__", hcu_vl_init)
        setattr(qwen4_exp, _MODULE_MARKER, True)
    except BaseException:
        setattr(llm_class, "__init__", llm_original)
        setattr(vl_class, "__init__", vl_original)
        if hasattr(qwen4_exp, _MODULE_MARKER):
            delattr(qwen4_exp, _MODULE_MARKER)
        raise

    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(_load_target_module(module))


__all__ = [
    "PATCH_ID",
    "TARGET_MODULES",
    "TARGET_MODULE",
    "apply",
    "apply_to_module",
]
