# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Validate Channel-FP8 routing and opt in to HCU INT8-W8A16 MoE."""

from __future__ import annotations

import functools
import os
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
    "vllm.model_executor.layers.quantization.compressed_tensors."
    "compressed_tensors_moe.compressed_tensors_moe_w8a8_fp8"
)
PATCH_ID = "worker.op_opt.compressed_tensors.moe_w8a8_fp8"
TARGETS = (
    f"{TARGET_MODULE}.CompressedTensorsW8A8Fp8MoEMethod.__init__",
    f"{TARGET_MODULE}.CompressedTensorsW8A8Fp8MoEMethod.process_weights_after_loading",
    f"{TARGET_MODULE}.CompressedTensorsW8A8Fp8MoEMethod.get_fused_moe_quant_config",
)
_CLASS_MARKER = "_vllm_hcu_moe_w8a8_fp8_applied"
_WRAPPER_MARKER = "_vllm_hcu_moe_w8a8_fp8_wrapper"
_EXPLICIT_BACKENDS = {
    "aiter": "AITER",
    "deep_gemm": "HCU_DEEPGEMM",
    "triton": "TRITON",
}


def _selected_backend_name(method) -> str | None:
    backend = getattr(method, "fp8_backend", None)
    value = getattr(backend, "value", backend)
    return value if isinstance(value, str) else None


def _is_channel_token_route(fp8_moe_module, weight_quant, input_quant) -> bool:
    strategy = getattr(fp8_moe_module, "QuantizationStrategy", None)
    try:
        return bool(
            weight_quant.strategy == strategy.CHANNEL
            and input_quant.strategy == strategy.TOKEN
        )
    except AttributeError as exc:
        raise PatchCompatibilityError(
            "vLLM compressed-tensors FP8 MoE quantization strategy contract "
            "is unavailable"
        ) from exc


def _required_selected_backend(moe) -> str:
    requested_backend = getattr(moe, "moe_backend", None)
    selected_backend = _EXPLICIT_BACKENDS.get(requested_backend)
    if selected_backend is None:
        raise RuntimeError(
            "Channel-FP8 MoE requires explicit --moe-backend "
            "aiter, deep_gemm, or triton"
        )
    return selected_backend


def apply_to_module(module: ModuleType) -> bool:
    fp8_moe_module = load_exact_module(TARGET_MODULE, module)
    method_class = require_class(
        fp8_moe_module,
        "CompressedTensorsW8A8Fp8MoEMethod",
        f"{TARGET_MODULE}.CompressedTensorsW8A8Fp8MoEMethod",
    )
    wrapped = (
        (method_class, "__init__", TARGETS[0], _WRAPPER_MARKER),
        (
            method_class,
            "process_weights_after_loading",
            TARGETS[1],
            _WRAPPER_MARKER,
        ),
        (
            method_class,
            "get_fused_moe_quant_config",
            TARGETS[2],
            _WRAPPER_MARKER,
        ),
    )
    if already_applied(method_class, _CLASS_MARKER, wrapped):
        return False

    original_init = require_callable(method_class, "__init__", TARGETS[0])
    original_process = require_callable(
        method_class,
        "process_weights_after_loading",
        TARGETS[1],
    )
    original_get_quant_config = require_callable(
        method_class,
        "get_fused_moe_quant_config",
        TARGETS[2],
    )
    require_exact_signature(
        original_init,
        TARGETS[0],
        positional=("self", "weight_quant", "input_quant", "moe", "layer_name"),
        defaults={"layer_name": None},
    )
    require_exact_signature(
        original_process,
        TARGETS[1],
        positional=("self", "layer"),
        defaults={},
    )
    require_exact_signature(
        original_get_quant_config,
        TARGETS[2],
        positional=("self", "layer"),
        defaults={},
    )

    @functools.wraps(original_init)
    def hcu_init(self, weight_quant, input_quant, moe, layer_name=None):
        channel_token = _is_channel_token_route(
            fp8_moe_module, weight_quant, input_quant
        )
        expected_backend = None
        if channel_token:
            expected_backend = _required_selected_backend(moe)
        original_init(self, weight_quant, input_quant, moe, layer_name)
        if channel_token:
            selected_backend = _selected_backend_name(self)
            if selected_backend != expected_backend:
                raise RuntimeError(
                    "vLLM v0.25.1 selected a Channel-FP8 MoE backend "
                    "different from the explicit request "
                    f"(expected={expected_backend!r}, selected={selected_backend!r})"
                )
        self._hcu_channel_fp8_w8a16 = bool(
            channel_token
            and expected_backend == "AITER"
            and os.environ.get(
                "VLLM_HCU_USE_CHANNEL_FP8_W8A16_MOE", "False"
            ).lower()
            in ("true", "1")
        )

    @functools.wraps(original_process)
    def hcu_process_weights_after_loading(self, layer):
        use_w8a16 = bool(getattr(self, "_hcu_channel_fp8_w8a16", False))
        if use_w8a16:
            original_backend = self.fp8_backend
            triton_backend = getattr(type(original_backend), "TRITON", None)
            if triton_backend is None:
                raise RuntimeError(
                    "Channel-FP8 W8A16 could not select the standard TRITON "
                    "weight layout before requantization"
                )
            self._hcu_channel_fp8_w8a16_backend = original_backend
            # The target AITER FP8 postprocessor shuffles weights eagerly.
            # W8A16 BoltOps consumes the standard layout, so let the target
            # finish all other postprocessing as TRITON, then restore AITER in
            # hcu_get_fused_moe_quant_config before the modular kernel is built.
            self.fp8_backend = triton_backend
            try:
                original_process(self, layer)
            finally:
                self.fp8_backend = original_backend
                del self._hcu_channel_fp8_w8a16_backend
        else:
            original_process(self, layer)
        if _selected_backend_name(self) != "HCU_DEEPGEMM":
            return
        moe_kernel = getattr(self, "moe_kernel", None)
        fused_experts = getattr(moe_kernel, "fused_experts", None)
        experts = getattr(fused_experts, "experts", fused_experts)
        process = getattr(experts, "process_weights_after_loading", None)
        if not callable(process):
            raise RuntimeError(
                "HCU_DEEPGEMM FP8 backend did not construct modular experts "
                "before weight postprocessing"
            )
        process(layer)

    @functools.wraps(original_get_quant_config)
    def hcu_get_fused_moe_quant_config(self, layer):
        if not bool(getattr(self, "_hcu_channel_fp8_w8a16", False)):
            return original_get_quant_config(self, layer)
        original_backend = getattr(
            self, "_hcu_channel_fp8_w8a16_backend", None
        )
        if original_backend is None:
            raise RuntimeError(
                "Channel-FP8 W8A16 quant config must be built during weight "
                "postprocessing"
            )
        self.fp8_backend = original_backend
        from vllm_hcu.model_executor.layers.quantization.compressed_tensors_moe_runtime import (
            requantize_channel_fp8_moe_weights,
        )

        requantize_channel_fp8_moe_weights(
            layer,
            replace_parameter=fp8_moe_module.replace_parameter,
        )
        return fp8_moe_module.FusedMoEQuantConfig.make(
            quant_dtype=None,
            weight_dtype=fp8_moe_module.torch.int8,
            per_out_ch_quant=True,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            gemm1_alpha=getattr(layer, "swiglu_alpha", None),
            gemm1_clamp_limit=getattr(layer, "swiglu_limit", None),
        )

    setattr(hcu_init, _WRAPPER_MARKER, True)
    setattr(hcu_process_weights_after_loading, _WRAPPER_MARKER, True)
    setattr(hcu_get_fused_moe_quant_config, _WRAPPER_MARKER, True)
    setattr(method_class, "_vllm_hcu_original_init", original_init)
    setattr(
        method_class,
        "_vllm_hcu_original_process_weights_after_loading",
        original_process,
    )
    setattr(method_class, "__init__", hcu_init)
    setattr(
        method_class,
        "process_weights_after_loading",
        hcu_process_weights_after_loading,
    )
    setattr(
        method_class,
        "get_fused_moe_quant_config",
        hcu_get_fused_moe_quant_config,
    )
    setattr(method_class, _CLASS_MARKER, True)
    setattr(method_class, "_vllm_hcu_fp8_moe_owner", "target-explicit")
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
