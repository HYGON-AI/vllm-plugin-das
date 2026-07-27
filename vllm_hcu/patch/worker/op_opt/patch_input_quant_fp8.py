# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Lazily route eligible dynamic per-token FP8 quantization to LightOp."""

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

TARGET_MODULE = "vllm.model_executor.layers.quantization.input_quant_fp8"
PATCH_ID = "worker.op_opt.quantization.lightop_per_token_fp8"
TARGETS = (
    f"{TARGET_MODULE}.QuantFP8.forward_cuda",
    f"{TARGET_MODULE}.QuantFP8.forward_native",
    "vllm_hcu.model_executor.layers.quantization.lightop_fp8_runtime",
    "torch.ops.vllm.lightop_per_token_quant_fp8",
)
_CLASS_MARKER = "_vllm_hcu_lightop_fp8_applied"
_WRAPPER_MARKER = "_vllm_hcu_lightop_fp8_wrapper"
_NOT_ELIGIBLE = object()


def _lightop_requested() -> bool:
    try:
        from vllm_hcu.platforms import envs as henvs

        return bool(henvs.VLLM_HCU_USE_CUSTOM_OPS) and bool(
            henvs.VLLM_HCU_USE_LIGHTOP_PER_TOKEN_QUANT_FP8
        )
    except (AttributeError, ImportError) as exc:
        raise PatchCompatibilityError(
            "required HCU LightOp FP8 feature flags are unavailable"
        ) from exc


def _maybe_lightop(module, instance, x, scale, scale_ub):
    if not _lightop_requested():
        return _NOT_ELIGIBLE
    if not (
        scale is None
        and scale_ub is None
        and instance.group_shape == module.GroupShape.PER_TOKEN
        and instance.num_token_padding is None
        and x.is_contiguous()
    ):
        return _NOT_ELIGIBLE

    try:
        from vllm.utils.torch_utils import direct_register_custom_op
        from vllm_hcu.model_executor.layers.quantization import lightop_fp8_runtime
    except Exception as exc:
        raise RuntimeError(
            "HCU LightOp per-token FP8 was requested, but its runtime "
            "registration dependencies are unavailable"
        ) from exc
    return lightop_fp8_runtime.quantize(
        x,
        module._FP8_DTYPE,
        direct_register_custom_op,
    )


def apply_to_module(module: ModuleType) -> bool:
    input_quant = load_exact_module(TARGET_MODULE, module)
    quant_class = require_class(
        input_quant, "QuantFP8", f"{TARGET_MODULE}.QuantFP8"
    )
    wrapped = (
        (quant_class, "forward_cuda", TARGETS[0], _WRAPPER_MARKER),
        (quant_class, "forward_native", TARGETS[1], _WRAPPER_MARKER),
    )
    if already_applied(quant_class, _CLASS_MARKER, wrapped):
        return False

    original_cuda = require_callable(quant_class, "forward_cuda", TARGETS[0])
    original_native = require_callable(quant_class, "forward_native", TARGETS[1])
    method_defaults = {"scale": None, "scale_ub": None, "use_triton": False}
    for function, target in (
        (original_cuda, TARGETS[0]),
        (original_native, TARGETS[1]),
    ):
        require_exact_signature(
            function,
            target,
            positional=("self", "x", "scale", "scale_ub", "use_triton"),
            defaults=method_defaults,
        )
    if not hasattr(input_quant, "_FP8_DTYPE") or not hasattr(
        input_quant.GroupShape, "PER_TOKEN"
    ):
        raise PatchCompatibilityError(
            f"required HCU patch constants in {TARGET_MODULE} are missing"
        )

    @functools.wraps(original_cuda)
    def hcu_forward_cuda(
        self,
        x,
        scale=None,
        scale_ub=None,
        use_triton=False,
    ):
        result = _maybe_lightop(input_quant, self, x, scale, scale_ub)
        if result is not _NOT_ELIGIBLE:
            return result
        return original_cuda(self, x, scale, scale_ub, use_triton)

    @functools.wraps(original_native)
    def hcu_forward_native(
        self,
        x,
        scale=None,
        scale_ub=None,
        use_triton=False,
    ):
        result = _maybe_lightop(input_quant, self, x, scale, scale_ub)
        if result is not _NOT_ELIGIBLE:
            return result
        return original_native(self, x, scale, scale_ub, use_triton)

    for function in (hcu_forward_cuda, hcu_forward_native):
        setattr(function, _WRAPPER_MARKER, True)
    setattr(quant_class, "_vllm_hcu_original_forward_cuda", original_cuda)
    setattr(quant_class, "_vllm_hcu_original_forward_native", original_native)
    setattr(quant_class, "forward_cuda", hcu_forward_cuda)
    setattr(quant_class, "forward_native", hcu_forward_native)
    setattr(quant_class, _CLASS_MARKER, True)
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
