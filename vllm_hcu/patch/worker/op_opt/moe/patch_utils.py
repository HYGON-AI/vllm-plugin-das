# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Expert-aware INT8 quantization adapter for batched DeepEP activations."""

from __future__ import annotations

import functools
from types import ModuleType

from ._common import load_exact_module, require_callable, require_parameter_names

TARGET_MODULE = "vllm.model_executor.layers.fused_moe.utils"
PATCH_ID = "worker.op_opt.moe.utils.int8_expert_quant"
TARGETS = (f"{TARGET_MODULE}._int8_quantize",)
_MARKER = "_vllm_hcu_int8_expert_quant_applied"


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    if getattr(target, _MARKER, False):
        return False
    original = require_callable(target, "_int8_quantize", TARGETS[0])
    require_parameter_names(original, TARGETS[0], ("A", "A_scale", "per_act_token", "block_shape"))

    @functools.wraps(original)
    def hcu_int8_quantize(
        A,
        A_scale,
        per_act_token,
        block_shape=None,
        expert_num_tokens=None,
    ):
        if expert_num_tokens is None:
            return original(A, A_scale, per_act_token, block_shape)
        if block_shape is not None or not per_act_token:
            raise ValueError(
                "expert_num_tokens is supported only for per-token INT8 quantization "
                "without block_shape"
            )
        if A_scale is not None:
            raise ValueError("dynamic per-token INT8 quantization expects A_scale=None")
        from vllm_hcu.model_executor.layers.fused_moe.int8_quant_runtime import (
            per_token_quant_int8,
        )

        return per_token_quant_int8(A, expert_num_tokens)

    # The wrapper intentionally extends the public internal signature.
    del hcu_int8_quantize.__wrapped__
    target._vllm_hcu_original_int8_quantize = original
    target._int8_quantize = hcu_int8_quantize
    setattr(target, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
