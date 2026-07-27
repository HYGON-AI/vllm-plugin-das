# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""HCU compressed-tensors INT8 hipBLASLt adapter."""

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
    "vllm.model_executor.layers.quantization.compressed_tensors.schemes."
    "compressed_tensors_w8a8_int8"
)
PATCH_ID = "worker.op_opt.compressed_tensors.w8a8_int8"
TARGETS = (
    f"{TARGET_MODULE}.apply_int8_linear",
    f"{TARGET_MODULE}.CompressedTensorsW8A8Int8.process_weights_after_loading",
    f"{TARGET_MODULE}.CompressedTensorsW8A8Int8.apply_weights",
    f"{TARGET_MODULE}.CompressedTensorsW8A8Int8.supports_quanted_inputs",
)
_CLASS_MARKER = "_vllm_hcu_w8a8_int8_applied"
_WRAPPER_MARKER = "_vllm_hcu_w8a8_int8_wrapper"


def _custom_quantization_enabled() -> bool:
    try:
        from vllm_hcu.platforms import envs as henvs

        return bool(henvs.VLLM_HCU_USE_CUSTOM_QUANTIZATION_GEMM)
    except (AttributeError, ImportError) as exc:
        raise PatchCompatibilityError(
            "required HCU flag VLLM_HCU_USE_CUSTOM_QUANTIZATION_GEMM is unavailable"
        ) from exc


def apply_to_module(module: ModuleType) -> bool:
    int8_module = load_exact_module(TARGET_MODULE, module)
    scheme_class = require_class(
        int8_module,
        "CompressedTensorsW8A8Int8",
        f"{TARGET_MODULE}.CompressedTensorsW8A8Int8",
    )
    wrapped = (
        (scheme_class, "process_weights_after_loading", TARGETS[1], _WRAPPER_MARKER),
        (scheme_class, "apply_weights", TARGETS[2], _WRAPPER_MARKER),
    )
    if already_applied(scheme_class, _CLASS_MARKER, wrapped):
        return False

    original_process = require_callable(
        scheme_class, "process_weights_after_loading", TARGETS[1]
    )
    require_exact_signature(
        original_process,
        TARGETS[1],
        positional=("self", "layer"),
    )
    original_apply = require_callable(scheme_class, "apply_weights", TARGETS[2])
    require_exact_signature(
        original_apply,
        TARGETS[2],
        positional=("self", "layer", "x", "bias"),
    )
    if "supports_quanted_inputs" in vars(scheme_class):
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGETS[3]} unexpectedly already exists"
        )

    from vllm_hcu.model_executor.layers.quantization.int8_runtime import (
        apply_int8_linear,
    )

    @functools.wraps(original_process)
    def hcu_process_weights_after_loading(self, layer) -> None:
        if not _custom_quantization_enabled():
            return original_process(self, layer)
        weight = getattr(layer, "weight", None)
        if getattr(weight, "ndim", None) != 2:
            raise RuntimeError("HCU W8A8 weight must be a 2D tensor before packing")
        original_weight = weight
        original_data = weight.data
        try:
            # Cancel the v0.25.1 target kernel's own transpose so the final
            # custom hipBLASLt operand remains [N, K].
            layer.weight.data = original_data.t()
            original_process(self, layer)
        except Exception:
            # ``original_weight`` is the same Parameter whose ``.data`` was
            # temporarily rebound above; assigning the object alone would
            # therefore leave the transposed shape behind after a failure.
            original_weight.data = original_data
            layer.weight = original_weight
            raise
        if not layer.weight.is_contiguous():
            layer.weight.data = layer.weight.data.contiguous()

    @functools.wraps(original_apply)
    def hcu_apply_weights(
        self,
        layer,
        x,
        bias,
        x_and_scale_quanted=None,
    ):
        if not _custom_quantization_enabled():
            return original_apply(self, layer, x, bias)
        return apply_int8_linear(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            params_dtype=layer.params_dtype,
            input_scale=getattr(layer, "input_scale", None),
            input_zero_point=getattr(layer, "input_zero_point", None),
            azp_adj=getattr(layer, "azp_adj", None),
            bias=bias,
            x_and_scale_quanted=x_and_scale_quanted,
        )

    def supports_quanted_inputs(self) -> bool:
        return True

    for function in (hcu_process_weights_after_loading, hcu_apply_weights):
        setattr(function, _WRAPPER_MARKER, True)
    setattr(int8_module, "apply_int8_linear", apply_int8_linear)
    setattr(scheme_class, "_vllm_hcu_original_process_weights", original_process)
    setattr(scheme_class, "_vllm_hcu_original_apply_weights", original_apply)
    setattr(
        scheme_class,
        "process_weights_after_loading",
        hcu_process_weights_after_loading,
    )
    setattr(scheme_class, "apply_weights", hcu_apply_weights)
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
