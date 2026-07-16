# SPDX-License-Identifier: Apache-2.0
"""HCU channelwise FP8 layout and prequantized-input adapter."""

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
    "compressed_tensors_w8a8_fp8"
)
PATCH_ID = "worker.op_opt.compressed_tensors.w8a8_fp8"
TARGETS = (
    f"{TARGET_MODULE}.CompressedTensorsW8A8Fp8.process_weights_after_loading",
    f"{TARGET_MODULE}.CompressedTensorsW8A8Fp8.apply_weights",
    f"{TARGET_MODULE}.CompressedTensorsW8A8Fp8.supports_quanted_inputs",
)
_CLASS_MARKER = "_vllm_hcu_w8a8_fp8_applied"
_WRAPPER_MARKER = "_vllm_hcu_w8a8_fp8_wrapper"


def _custom_quantization_enabled() -> bool:
    try:
        from vllm_hcu.platforms import envs as henvs

        return bool(henvs.VLLM_HCU_USE_CUSTOM_QUANTIZATION_GEMM)
    except (AttributeError, ImportError) as exc:
        raise PatchCompatibilityError(
            "required HCU flag VLLM_HCU_USE_CUSTOM_QUANTIZATION_GEMM is unavailable"
        ) from exc


def apply_to_module(module: ModuleType) -> bool:
    fp8_module = load_exact_module(TARGET_MODULE, module)
    scheme_class = require_class(
        fp8_module,
        "CompressedTensorsW8A8Fp8",
        f"{TARGET_MODULE}.CompressedTensorsW8A8Fp8",
    )
    wrapped = (
        (scheme_class, "process_weights_after_loading", TARGETS[0], _WRAPPER_MARKER),
        (scheme_class, "apply_weights", TARGETS[1], _WRAPPER_MARKER),
    )
    if already_applied(scheme_class, _CLASS_MARKER, wrapped):
        return False

    original_process = require_callable(
        scheme_class, "process_weights_after_loading", TARGETS[0]
    )
    require_exact_signature(
        original_process,
        TARGETS[0],
        positional=("self", "layer"),
    )
    original_apply = require_callable(scheme_class, "apply_weights", TARGETS[1])
    require_exact_signature(
        original_apply,
        TARGETS[1],
        positional=("self", "layer", "x", "bias"),
        defaults={"bias": None},
    )
    if "supports_quanted_inputs" in vars(scheme_class):
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGETS[2]} unexpectedly already exists"
        )

    @functools.wraps(original_process)
    def hcu_process_weights_after_loading(self, layer) -> None:
        custom_channel = (
            _custom_quantization_enabled()
            and self.strategy == fp8_module.QuantizationStrategy.CHANNEL
        )
        if custom_channel:
            kernel_class = type(getattr(self, "fp8_linear", None))
            if not getattr(kernel_class, "_hcu_fp8_patch_applied", False):
                raise RuntimeError(
                    "VLLM_HCU_USE_CUSTOM_QUANTIZATION_GEMM is enabled for "
                    "channelwise FP8, but the HCU channelwise scaled-mm kernel "
                    "adapter is not active"
                )

        original_process(self, layer)
        if custom_channel:
            # Upstream stores B as [K, N] for torch._scaled_mm.  The HCU
            # hipBLASLt adapter consumes a contiguous [N, K] weight.
            layer.weight.data = layer.weight.data.t().contiguous()

    @functools.wraps(original_apply)
    def hcu_apply_weights(
        self,
        layer,
        x,
        bias=None,
        x_and_scale_quanted=None,
    ):
        if x_and_scale_quanted is None:
            return original_apply(self, layer, x, bias)
        if not isinstance(x_and_scale_quanted, tuple) or len(x_and_scale_quanted) != 2:
            raise ValueError("x_and_scale_quanted must be a (tensor, scale) tuple")
        supports = getattr(self.fp8_linear, "supports_quanted_inputs", None)
        if callable(supports) and bool(supports()):
            return self.fp8_linear.apply_weights(
                layer,
                x,
                bias,
                x_and_scale_quanted=x_and_scale_quanted,
            )
        return original_apply(self, layer, x, bias)

    def supports_quanted_inputs(self) -> bool:
        return True

    for function in (hcu_process_weights_after_loading, hcu_apply_weights):
        setattr(function, _WRAPPER_MARKER, True)
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
