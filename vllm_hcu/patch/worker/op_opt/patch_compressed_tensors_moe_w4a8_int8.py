# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Connect compressed-tensors channel W4A8 to the existing HCU runtime."""

from __future__ import annotations

import functools
from types import ModuleType

from ._common import (
    already_applied,
    load_exact_module,
    require_callable,
    require_class,
    require_exact_signature,
)

TARGET_MODULE = (
    "vllm.model_executor.layers.quantization.compressed_tensors."
    "compressed_tensors_moe.compressed_tensors_moe"
)
PATCH_ID = "worker.op_opt.compressed_tensors.moe_w4a8_int8"
TARGETS = (f"{TARGET_MODULE}.CompressedTensorsMoEMethod.get_moe_method",)
_CLASS_MARKER = "_vllm_hcu_ct_w4a8_int8_applied"
_WRAPPER_MARKER = "_vllm_hcu_ct_w4a8_int8_wrapper"
_CPU_ONLY_ERROR = "W4A8 Int8 MoE is only supported on CPU platforms"


def _hcu_enabled(moe: object) -> bool:
    from vllm.platforms import current_platform
    from .patch_compressed_tensors_moe_wna16 import _aiter_requested

    return (
        type(current_platform).__module__.startswith("vllm_hcu.")
        and current_platform.device_type == "cuda"
    )


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    cls = require_class(target, "CompressedTensorsMoEMethod", TARGETS[0])
    if already_applied(
        cls, _CLASS_MARKER,
        ((cls, "get_moe_method", TARGETS[0], _WRAPPER_MARKER),),
    ):
        return False
    original = require_callable(cls, "get_moe_method", TARGETS[0])
    require_exact_signature(
        original, TARGETS[0], positional=("quant_config", "layer", "layer_name")
    )

    @functools.wraps(original)
    def hcu_get_moe_method(quant_config, layer, layer_name):
        try:
            # Prefer a future upstream implementation if it supports this path.
            return original(quant_config, layer, layer_name)
        except NotImplementedError as exc:
            if str(exc) != _CPU_ONLY_ERROR or not _hcu_enabled(layer.moe_config):
                raise
            from vllm_hcu.model_executor.layers.quantization import (
                compressed_tensors_w4a8_runtime as runtime,
            )

            scheme = quant_config.get_scheme_dict(layer, layer_name + ".0.down_proj")
            scheme_format = scheme.get("format") if scheme else None
            scheme_format = getattr(scheme_format, "value", scheme_format)
            if scheme_format != "pack-quantized":
                raise
            weight = scheme.get("weights")
            activation = scheme.get("input_activations")
            if (
                not runtime.supports_channel_w4a8(
                    weight, activation, layer.moe_config
                )
                or getattr(layer, "apply_router_weight_on_input", False)
            ):
                raise
            from vllm.logger import init_logger

            init_logger(__name__).info_once(
                "Using HCU compressed-tensors W4A8 MoE adapter "
                "(channel INT4 weights, dynamic per-token INT8 activations)"
            )
            return runtime.make_hcu_w4a8_method(
                weight, activation, layer.moe_config, layer_name
            )

    setattr(hcu_get_moe_method, _WRAPPER_MARKER, True)
    cls._vllm_hcu_ct_w4a8_original_get_moe_method = staticmethod(original)
    cls.get_moe_method = staticmethod(hcu_get_moe_method)
    setattr(cls, _CLASS_MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
