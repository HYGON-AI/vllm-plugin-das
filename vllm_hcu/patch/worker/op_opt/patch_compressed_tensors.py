# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Forward prequantized activations through compressed-tensors linear methods."""

from __future__ import annotations

import functools
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
    "vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors"
)
PATCH_ID = "worker.op_opt.compressed_tensors.prequantized_linear"
TARGETS = (
    f"{TARGET_MODULE}.CompressedTensorsLinearMethod.apply",
    f"{TARGET_MODULE}.CompressedTensorsLinearMethod.supports_quanted_inputs",
)
_CLASS_MARKER = "_vllm_hcu_prequantized_linear_applied"
_WRAPPER_MARKER = "_vllm_hcu_prequantized_linear_wrapper"


def apply_to_module(module: ModuleType) -> bool:
    compressed = load_exact_module(TARGET_MODULE, module)
    method_class = require_class(
        compressed,
        "CompressedTensorsLinearMethod",
        f"{TARGET_MODULE}.CompressedTensorsLinearMethod",
    )
    if already_applied(
        method_class,
        _CLASS_MARKER,
        ((method_class, "apply", TARGETS[0], _WRAPPER_MARKER),),
    ):
        return False

    original = require_callable(method_class, "apply", TARGETS[0])
    require_exact_signature(
        original,
        TARGETS[0],
        positional=("self", "layer", "x", "bias"),
        defaults={"bias": None},
    )
    if "supports_quanted_inputs" in vars(method_class):
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGETS[1]} unexpectedly already exists"
        )

    @functools.wraps(original)
    def hcu_apply(self, layer, x, bias=None, x_and_scale_quanted=None):
        if x_and_scale_quanted is None:
            return original(self, layer, x, bias)
        if not isinstance(x_and_scale_quanted, tuple) or len(x_and_scale_quanted) != 2:
            raise ValueError("x_and_scale_quanted must be a (tensor, scale) tuple")

        scheme = getattr(layer, "scheme", None)
        if scheme is None:
            raise ValueError("A scheme must be defined for each layer")
        supports = getattr(scheme, "supports_quanted_inputs", None)
        if callable(supports) and bool(supports()):
            return scheme.apply_weights(
                layer,
                x,
                bias=bias,
                x_and_scale_quanted=x_and_scale_quanted,
            )
        return original(self, layer, x, bias)

    def supports_quanted_inputs(self) -> bool:
        return True

    setattr(hcu_apply, _WRAPPER_MARKER, True)
    setattr(method_class, "_vllm_hcu_original_apply", original)
    setattr(method_class, "apply", hcu_apply)
    setattr(method_class, "supports_quanted_inputs", supports_quanted_inputs)
    setattr(method_class, _CLASS_MARKER, True)
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
