# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Correct the inactive base compressed-tensors capability fragment."""

from __future__ import annotations

from types import ModuleType

from ._common import (
    PatchCompatibilityError,
    already_applied,
    load_exact_module,
    require_callable,
    require_class,
    require_exact_signature,
)

TARGET_MODULE = (
    "vllm.model_executor.layers.quantization.compressed_tensors.schemes."
    "compressed_tensors_scheme"
)
PATCH_ID = "worker.op_opt.compressed_tensors.scheme_capability"
TARGETS = (
    f"{TARGET_MODULE}.CompressedTensorsScheme.supports_quanted_inputs",
)
_CLASS_MARKER = "_vllm_hcu_scheme_capability_applied"
_METHOD_MARKER = "_vllm_hcu_scheme_capability_method"


def apply_to_module(module: ModuleType) -> bool:
    scheme_module = load_exact_module(TARGET_MODULE, module)
    scheme_class = require_class(
        scheme_module,
        "CompressedTensorsScheme",
        f"{TARGET_MODULE}.CompressedTensorsScheme",
    )
    if already_applied(
        scheme_class,
        _CLASS_MARKER,
        ((scheme_class, "supports_quanted_inputs", TARGETS[0], _METHOD_MARKER),),
    ):
        return False

    process = require_callable(
        scheme_class,
        "process_weights_after_loading",
        f"{TARGET_MODULE}.CompressedTensorsScheme.process_weights_after_loading",
    )
    require_exact_signature(
        process,
        f"{TARGET_MODULE}.CompressedTensorsScheme.process_weights_after_loading",
        positional=("self", "layer"),
    )
    if "supports_quanted_inputs" in vars(scheme_class):
        raise PatchCompatibilityError(
            f"required HCU corrected target {TARGETS[0]} unexpectedly already exists"
        )

    def supports_quanted_inputs(self) -> bool:
        return False

    setattr(supports_quanted_inputs, _METHOD_MARKER, True)
    setattr(scheme_class, "supports_quanted_inputs", supports_quanted_inputs)
    setattr(scheme_class, _CLASS_MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = [
    "PATCH_ID",
    "TARGET_MODULE",
    "TARGETS",
    "apply",
    "apply_to_module",
]
