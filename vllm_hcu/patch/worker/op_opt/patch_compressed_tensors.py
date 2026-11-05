# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Adapt compressed-tensors linear inputs and channel W4A8 MoE dispatch."""

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
    f"{TARGET_MODULE}.CompressedTensorsConfig.get_quant_method",
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
    config_class = require_class(
        compressed,
        "CompressedTensorsConfig",
        f"{TARGET_MODULE}.CompressedTensorsConfig",
    )
    if already_applied(
        method_class,
        _CLASS_MARKER,
        (
            (method_class, "apply", TARGETS[0], _WRAPPER_MARKER),
            (config_class, "get_quant_method", TARGETS[2], _WRAPPER_MARKER),
        ),
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

    get_method = require_callable(
        config_class,
        "get_quant_method",
        TARGETS[2],
    )

    @functools.wraps(get_method)
    def hcu_get_quant_method(self, layer, prefix):
        from vllm.model_executor.layers.fused_moe import RoutedExperts

        if isinstance(layer, RoutedExperts):
            from vllm_hcu.model_executor.layers.quantization.compressed_tensors_w4a8 import (
                select_method,
            )

            method = select_method(self, layer, prefix)
            if method is not None:
                return method
        return get_method(self, layer, prefix)

    require_exact_signature(
        get_method,
        TARGETS[2],
        positional=("self", "layer", "prefix"),
    )
    setattr(hcu_get_quant_method, _WRAPPER_MARKER, True)
    config_class.get_quant_method = hcu_get_quant_method

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
