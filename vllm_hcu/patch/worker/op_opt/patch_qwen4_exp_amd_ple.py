# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Validation for the HCU Qwen4Exp AMD PLE replacement."""

from __future__ import annotations

import importlib
import inspect
from types import ModuleType

from .moe._common import (
    PatchCompatibilityError,
    require_callable,
    require_class,
    require_replacement_module,
)

TARGET_MODULE = "vllm.models.qwen4_exp.amd.ple_layer"
REPLACEMENT_MODULE = "vllm_hcu.runtime_compat.qwen4_exp_amd_ple_layer"
PATCH_ID = "worker.op_opt.qwen4_exp.amd.ple_layer"
TARGETS = (
    TARGET_MODULE,
    f"{TARGET_MODULE}.Qwen4ExpNGramEmbedding._forward_impl",
    f"{TARGET_MODULE}.Qwen4ExpNGramEmbedding.forward",
    f"{TARGET_MODULE}.qwen4_exp_amd_ple_forward",
    f"{TARGET_MODULE}.qwen4_exp_amd_mtp_hidden_copy",
)
_MARKER = "_vllm_hcu_qwen4_exp_ple_replacement_validated"


def _parameter_names(function) -> tuple[str, ...]:
    return tuple(inspect.signature(function).parameters)


def apply_to_module(module: ModuleType) -> bool:
    require_replacement_module(module, REPLACEMENT_MODULE, TARGETS)
    if getattr(module, _MARKER, False):
        return False

    embedding = require_class(
        module,
        "Qwen4ExpNGramEmbedding",
        f"{REPLACEMENT_MODULE}.Qwen4ExpNGramEmbedding",
    )
    require_class(
        module,
        "Qwen4ExpPLELayer",
        f"{REPLACEMENT_MODULE}.Qwen4ExpPLELayer",
    )
    expected_methods = {
        "_forward_impl": (
            "self",
            "input_ids",
            "query_start_loc",
            "ngram_context",
        ),
        "forward": (
            "self",
            "input_ids",
            "query_start_loc",
            "ngram_context",
        ),
    }
    for name, expected in expected_methods.items():
        function = require_callable(embedding, name, f"{TARGETS[0]}.{name}")
        if _parameter_names(function) != expected:
            raise PatchCompatibilityError(
                f"HCU Qwen4Exp PLE replacement has incompatible {name} "
                f"signature {inspect.signature(function)}"
            )

    for name, expected in {
        "qwen4_exp_amd_ple_forward": (
            "input_ids",
            "query_start_loc",
            "ngram_context",
            "output",
            "layer_name",
        ),
        "qwen4_exp_amd_mtp_hidden_copy": ("hidden_states", "buffer"),
    }.items():
        function = require_callable(module, name, f"{REPLACEMENT_MODULE}.{name}")
        if _parameter_names(function) != expected:
            raise PatchCompatibilityError(
                f"HCU Qwen4Exp PLE replacement has incompatible {name} "
                f"signature {inspect.signature(function)}"
            )

    setattr(module, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    if module is None:
        module = importlib.import_module(REPLACEMENT_MODULE)
    return apply_to_module(module)


__all__ = [
    "PATCH_ID",
    "REPLACEMENT_MODULE",
    "TARGET_MODULE",
    "TARGETS",
    "apply",
    "apply_to_module",
]
