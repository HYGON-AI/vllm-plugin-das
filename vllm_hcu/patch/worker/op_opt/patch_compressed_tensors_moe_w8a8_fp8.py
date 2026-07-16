# SPDX-License-Identifier: Apache-2.0
"""HCU AITER/DPSK adapter for compressed-tensors FP8-W8A8 MoE."""

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
    "vllm.model_executor.layers.quantization.compressed_tensors."
    "compressed_tensors_moe.compressed_tensors_moe_w8a8_fp8"
)
PATCH_ID = "worker.op_opt.compressed_tensors.moe_w8a8_fp8"
TARGETS = (
    f"{TARGET_MODULE}.CompressedTensorsW8A8Fp8MoEMethod.__init__",
    f"{TARGET_MODULE}.CompressedTensorsW8A8Fp8MoEMethod."
    "process_weights_after_loading",
    f"{TARGET_MODULE}.CompressedTensorsW8A8Fp8MoEMethod.apply",
    f"{TARGET_MODULE}.CompressedTensorsW8A8Fp8MoEMethod."
    "_get_aiter_moe_runtime_config",
    f"{TARGET_MODULE}.CompressedTensorsW8A8Fp8MoEMethod."
    "_get_aiter_weights_for_solution",
)
_CLASS_MARKER = "_vllm_hcu_moe_w8a8_fp8_applied"
_WRAPPER_MARKER = "_vllm_hcu_moe_w8a8_fp8_wrapper"


def _aiter_requested() -> bool:
    try:
        from vllm_hcu.platforms import envs as henvs

        return bool(henvs.VLLM_HCU_USE_CUSTOM_OPS) and bool(
            henvs.VLLM_HCU_USE_AITER_W8A8_FP8_MOE
        )
    except (AttributeError, ImportError) as exc:
        raise PatchCompatibilityError(
            "required HCU AITER FP8-W8A8 MoE flags are unavailable"
        ) from exc


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
        (method_class, "apply", TARGETS[2], _WRAPPER_MARKER),
        (
            method_class,
            "_get_aiter_moe_runtime_config",
            TARGETS[3],
            _WRAPPER_MARKER,
        ),
        (
            method_class,
            "_get_aiter_weights_for_solution",
            TARGETS[4],
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
    original_process = require_callable(
        method_class, "process_weights_after_loading", TARGETS[1]
    )
    require_exact_signature(
        original_process,
        TARGETS[1],
        positional=("self", "layer"),
    )
    original_apply = require_callable(method_class, "apply", TARGETS[2])
    require_exact_signature(
        original_apply,
        TARGETS[2],
        positional=(
            "self",
            "layer",
            "x",
            "topk_weights",
            "topk_ids",
            "shared_experts_input",
        ),
    )
    for name, target in (
        ("_get_aiter_moe_runtime_config", TARGETS[3]),
        ("_get_aiter_weights_for_solution", TARGETS[4]),
    ):
        if name in vars(method_class):
            raise PatchCompatibilityError(
                f"required HCU patch target {target} unexpectedly already exists"
            )

    from vllm_hcu.model_executor.layers.quantization import (
        compressed_tensors_moe_runtime as hcu_runtime,
    )

    @functools.wraps(original_init)
    def hcu_init(self, weight_quant, input_quant, moe, layer_name=None):
        original_init(self, weight_quant, input_quant, moe, layer_name)
        self._hcu_aiter_moe_config_cache = {}

    @functools.wraps(original_process)
    def hcu_process_weights_after_loading(self, layer) -> None:
        original_process(self, layer)
        hcu_runtime.process_dpsk_deepgemm_weights(self, layer)

    @functools.wraps(original_apply)
    def hcu_apply(
        self,
        layer,
        x,
        topk_weights,
        topk_ids,
        shared_experts_input,
        i_q=None,
        i_s=None,
    ):
        if not _aiter_requested():
            if i_q is not None or i_s is not None:
                raise RuntimeError(
                    "prequantized i_q/i_s inputs require the HCU AITER "
                    "FP8-W8A8 MoE backend"
                )
            return original_apply(
                self,
                layer,
                x,
                topk_weights,
                topk_ids,
                shared_experts_input,
            )
        return hcu_runtime.apply_aiter_w8a8_fp8_moe(
            self,
            layer,
            x,
            topk_weights,
            topk_ids,
            shared_experts_input,
            i_q,
            i_s,
        )

    def hcu_get_aiter_moe_runtime_config(self, layer, x, topk_ids):
        return hcu_runtime.get_aiter_w8a8_runtime_config(self, layer, x, topk_ids)

    def hcu_get_aiter_weights_for_solution(self, layer, solution_type):
        return hcu_runtime.get_aiter_weights_for_solution(layer, solution_type)

    for function in (
        hcu_init,
        hcu_process_weights_after_loading,
        hcu_apply,
        hcu_get_aiter_moe_runtime_config,
        hcu_get_aiter_weights_for_solution,
    ):
        setattr(function, _WRAPPER_MARKER, True)
    setattr(method_class, "_vllm_hcu_original_init", original_init)
    setattr(method_class, "_vllm_hcu_original_process_weights", original_process)
    setattr(method_class, "_vllm_hcu_original_apply", original_apply)
    setattr(method_class, "__init__", hcu_init)
    setattr(
        method_class,
        "process_weights_after_loading",
        hcu_process_weights_after_loading,
    )
    setattr(method_class, "apply", hcu_apply)
    setattr(
        method_class,
        "_get_aiter_moe_runtime_config",
        hcu_get_aiter_moe_runtime_config,
    )
    setattr(
        method_class,
        "_get_aiter_weights_for_solution",
        hcu_get_aiter_weights_for_solution,
    )
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
