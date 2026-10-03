# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""HCU AITER W4A16 zero-point adapter for compressed-tensors MoE."""

from __future__ import annotations

import functools
from copy import copy
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
    "compressed_tensors_moe.compressed_tensors_moe_wna16"
)
QUANT_CONFIG_MODULE = "vllm.model_executor.layers.fused_moe.config"
PATCH_ID = "worker.op_opt.compressed_tensors.moe_wna16"
TARGETS = (
    f"{TARGET_MODULE}.CompressedTensorsWNA16MoEMethod.__init__",
    f"{TARGET_MODULE}.CompressedTensorsWNA16MoEMethod.create_weights",
    f"{TARGET_MODULE}.CompressedTensorsWNA16MoEMethod.get_fused_moe_quant_config",
)
_CLASS_MARKER = "_vllm_hcu_moe_wna16_applied"
_WRAPPER_MARKER = "_vllm_hcu_moe_wna16_wrapper"


def _aiter_requested(layer: object | None = None) -> bool:
    try:
        from vllm_hcu.platforms import envs as henvs
        from vllm_hcu.model_executor.layers.fused_moe.aiter_runtime import (
            is_aiter_moe_explicitly_disabled,
            is_aiter_moe_requested,
        )

        moe_config = getattr(layer, "moe_config", layer)
        if is_aiter_moe_explicitly_disabled(moe_config):
            return False
        return bool(henvs.VLLM_HCU_USE_CUSTOM_OPS) and bool(
            henvs.VLLM_HCU_USE_AITER_W4A16_MOE
            or is_aiter_moe_requested(moe_config)
        )
    except (AttributeError, ImportError) as exc:
        raise PatchCompatibilityError(
            "required HCU AITER W4A16 MoE flags are unavailable"
        ) from exc


def _resolve_config_builder(wna16_module: ModuleType):
    config_module = load_exact_module(QUANT_CONFIG_MODULE, None)
    builder = getattr(config_module, "int4_w4a16_moe_quant_config", None)
    if callable(builder):
        return builder
    return require_callable(
        wna16_module,
        "int4_w4a16_moe_quant_config",
        f"{TARGET_MODULE}.int4_w4a16_moe_quant_config",
    )


def apply_to_module(module: ModuleType) -> bool:
    wna16_module = load_exact_module(TARGET_MODULE, module)
    method_class = require_class(
        wna16_module,
        "CompressedTensorsWNA16MoEMethod",
        f"{TARGET_MODULE}.CompressedTensorsWNA16MoEMethod",
    )
    wrapped = (
        (method_class, "__init__", TARGETS[0], _WRAPPER_MARKER),
        (method_class, "create_weights", TARGETS[1], _WRAPPER_MARKER),
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
    require_exact_signature(
        original_init,
        TARGETS[0],
        positional=("self", "weight_quant", "input_quant", "moe", "layer_name"),
        defaults={"layer_name": None},
    )
    original_create = require_callable(method_class, "create_weights", TARGETS[1])
    require_exact_signature(
        original_create,
        TARGETS[1],
        positional=(
            "self",
            "layer",
            "num_experts",
            "hidden_size",
            "intermediate_size_per_partition",
            "params_dtype",
        ),
        var_keyword="extra_weight_attrs",
    )
    original_get_config = require_callable(
        method_class, "get_fused_moe_quant_config", TARGETS[2]
    )
    require_exact_signature(
        original_get_config,
        TARGETS[2],
        positional=("self", "layer"),
    )
    set_weight_attrs = require_callable(
        wna16_module,
        "set_weight_attrs",
        f"{TARGET_MODULE}.set_weight_attrs",
    )
    config_builder = _resolve_config_builder(wna16_module)

    from vllm_hcu.model_executor.layers.quantization import (
        compressed_tensors_moe_runtime as hcu_runtime,
    )

    @functools.wraps(original_init)
    def hcu_init(
        self,
        weight_quant,
        input_quant,
        moe,
        layer_name=None,
    ):
        strategy = getattr(weight_quant, "strategy", None)
        is_channel = str(getattr(strategy, "value", strategy)).lower() == "channel"
        if not (is_channel and moe is not None and _aiter_requested(moe)):
            return original_init(self, weight_quant, input_quant, moe, layer_name)

        # vLLM's WNA16 oracle has no AITER enum and rejects channel-wise
        # quantization after selecting its Triton storage/conversion path.
        # Select that compatible N-first storage path, then restore the actual
        # channel/AITER contract for the HCU runtime wrappers.
        oracle_weight_quant = copy(weight_quant)
        oracle_moe = copy(moe)
        object.__setattr__(oracle_weight_quant, "strategy", "group")
        object.__setattr__(oracle_moe, "moe_backend", "triton")
        original_init(
            self,
            oracle_weight_quant,
            input_quant,
            oracle_moe,
            layer_name,
        )
        self.weight_quant = weight_quant
        self.strategy = strategy
        self.group_size = getattr(weight_quant, "group_size", -1)
        self.moe = moe

    @functools.wraps(original_create)
    def hcu_create_weights(
        self,
        layer,
        num_experts,
        hidden_size,
        intermediate_size_per_partition,
        params_dtype,
        **extra_weight_attrs,
    ):
        result = original_create(
            self,
            layer,
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            params_dtype,
            **extra_weight_attrs,
        )
        if _aiter_requested(layer):
            hcu_runtime.create_aiter_w4a16_qzeros(
                self,
                layer,
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                extra_weight_attrs,
                set_weight_attrs,
            )
        return result

    @functools.wraps(original_get_config)
    def hcu_get_fused_moe_quant_config(self, layer):
        if not _aiter_requested(layer):
            return original_get_config(self, layer)
        return hcu_runtime.build_aiter_w4a16_quant_config(
            self,
            layer,
            config_builder,
        )

    for function in (hcu_init, hcu_create_weights, hcu_get_fused_moe_quant_config):
        setattr(function, _WRAPPER_MARKER, True)
    setattr(method_class, "_vllm_hcu_original_init", original_init)
    setattr(method_class, "_vllm_hcu_original_create_weights", original_create)
    setattr(method_class, "_vllm_hcu_original_get_quant_config", original_get_config)
    setattr(method_class, "__init__", hcu_init)
    setattr(method_class, "create_weights", hcu_create_weights)
    setattr(method_class, "get_fused_moe_quant_config", hcu_get_fused_moe_quant_config)
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
