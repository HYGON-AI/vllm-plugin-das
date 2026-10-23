# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Route compressed-tensors channel-wise W4A8 MoE to the HCU owner."""

from __future__ import annotations

import functools
from types import ModuleType

from compressed_tensors import CompressionFormat
from compressed_tensors.quantization import (
    QuantizationStrategy,
    QuantizationType,
)
from vllm_hcu.model_executor.layers.quantization.hyv4_ownership import (
    HYV4_MODEL_MARKER,
)

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
_CLASS_MARKER = "_vllm_hcu_w4a8_int8_moe_applied"
_WRAPPER_MARKER = "_vllm_hcu_w4a8_int8_moe_wrapper"


def _scheme_dict(quant_config, layer, layer_name):
    quant_config._add_fused_moe_to_target_scheme_map()
    names = [
        layer_name + projection
        for projection in (".0.gate_proj", ".0.up_proj", ".0.down_proj")
    ]
    schemes = [quant_config.get_scheme_dict(layer, name) for name in names]
    scheme = schemes[-1]
    if not all(candidate == scheme for candidate in schemes[:-1]):
        return None
    return scheme


def _is_hcu_w4a8_scheme(weight_quant, input_quant, moe) -> bool:
    return bool(
        weight_quant is not None
        and input_quant is not None
        and weight_quant.num_bits == 4
        and weight_quant.type == QuantizationType.INT
        and weight_quant.strategy == QuantizationStrategy.CHANNEL
        and weight_quant.symmetric
        and not weight_quant.dynamic
        and weight_quant.group_size in (None, -1)
        and input_quant.num_bits == 8
        and input_quant.type == QuantizationType.INT
        and input_quant.strategy == QuantizationStrategy.TOKEN
        and input_quant.symmetric
        and input_quant.dynamic
        and getattr(moe, "has_bias", None) is False
    )


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    method_class = require_class(
        target,
        "CompressedTensorsMoEMethod",
        f"{TARGET_MODULE}.CompressedTensorsMoEMethod",
    )
    wrapped = (
        (method_class, "get_moe_method", TARGETS[0], _WRAPPER_MARKER),
    )
    if already_applied(method_class, _CLASS_MARKER, wrapped):
        return False

    original = require_callable(method_class, "get_moe_method", TARGETS[0])
    require_exact_signature(
        original,
        TARGETS[0],
        positional=("quant_config", "layer", "layer_name"),
    )

    @functools.wraps(original)
    def hcu_get_moe_method(quant_config, layer, layer_name):
        if not getattr(quant_config, HYV4_MODEL_MARKER, False):
            return original(quant_config, layer, layer_name)
        scheme = _scheme_dict(quant_config, layer, layer_name)
        if scheme is not None:
            weight_quant = scheme.get("weights")
            input_quant = scheme.get("input_activations")
            packed = (
                scheme.get("format")
                == CompressionFormat.pack_quantized.value
            )
            if packed and _is_hcu_w4a8_scheme(
                weight_quant, input_quant, layer.moe_config
            ):
                from vllm_hcu.model_executor.layers.quantization.compressed_tensors_w4a8_moe import (
                    HcuCompressedTensorsW4A8Int8MoEMethod,
                )

                return HcuCompressedTensorsW4A8Int8MoEMethod(
                    weight_quant,
                    input_quant,
                    layer.moe_config,
                    layer_name,
                )
        return original(quant_config, layer, layer_name)

    setattr(hcu_get_moe_method, _WRAPPER_MARKER, True)
    setattr(method_class, "_vllm_hcu_original_get_moe_method", original)
    setattr(method_class, "get_moe_method", staticmethod(hcu_get_moe_method))
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
