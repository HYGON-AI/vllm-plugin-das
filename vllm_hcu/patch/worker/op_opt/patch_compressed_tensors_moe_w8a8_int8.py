# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Install self-developed AITER Int8 MoE layouts during weight post-load."""

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
    "compressed_tensors_moe.compressed_tensors_moe_w8a8_int8"
)
PATCH_ID = "worker.op_opt.compressed_tensors.moe_w8a8_int8"
TARGETS = (
    f"{TARGET_MODULE}.CompressedTensorsW8A8Int8MoEMethod."
    "process_weights_after_loading",
)
_CLASS_MARKER = "_vllm_hcu_moe_w8a8_int8_applied"
_WRAPPER_MARKER = "_vllm_hcu_moe_w8a8_int8_wrapper"


def _selected_backend_name(method) -> str:
    backend = getattr(method, "int8_backend", None)
    for value in (getattr(backend, "name", None), getattr(backend, "value", None), backend):
        if value is not None:
            return str(value).rsplit(".", 1)[-1].upper()
    return ""


def apply_to_module(module: ModuleType) -> bool:
    int8_moe_module = load_exact_module(TARGET_MODULE, module)
    method_class = require_class(
        int8_moe_module,
        "CompressedTensorsW8A8Int8MoEMethod",
        f"{TARGET_MODULE}.CompressedTensorsW8A8Int8MoEMethod",
    )
    if already_applied(
        method_class,
        _CLASS_MARKER,
        ((
            method_class,
            "process_weights_after_loading",
            TARGETS[0],
            _WRAPPER_MARKER,
        ),),
    ):
        return False

    original = require_callable(
        method_class,
        "process_weights_after_loading",
        TARGETS[0],
    )
    require_exact_signature(
        original,
        TARGETS[0],
        positional=("self", "layer"),
    )

    @functools.wraps(original)
    def hcu_process_weights_after_loading(self, layer) -> None:
        original(self, layer)
        if _selected_backend_name(self) != "AITER":
            return
        quant_config = getattr(self, "moe_quant_config", None)
        if quant_config is None:
            raise RuntimeError(
                "AITER Int8 MoE did not initialize its quantization config"
            )
        from vllm_hcu.model_executor.layers.quantization import (
            compressed_tensors_moe_runtime as hcu_runtime,
        )

        hcu_runtime.prewarm_aiter_quantized_moe(
            layer,
            self.moe,
            quant_config,
        )

    setattr(hcu_process_weights_after_loading, _WRAPPER_MARKER, True)
    setattr(method_class, "_vllm_hcu_original_process_weights_after_loading", original)
    setattr(
        method_class,
        "process_weights_after_loading",
        hcu_process_weights_after_loading,
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
