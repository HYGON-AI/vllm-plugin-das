# SPDX-License-Identifier: Apache-2.0
"""Allow FP8 scaled-mm kernels to consume an already quantized activation."""

from __future__ import annotations

import functools
from math import prod
from types import ModuleType

import torch

from ._common import (
    PatchCompatibilityError,
    already_applied,
    load_exact_module,
    require_callable,
    require_class,
    require_exact_signature,
)

TARGET_MODULE = "vllm.model_executor.kernels.linear.scaled_mm.ScaledMMLinearKernel"
PATCH_ID = "worker.op_opt.scaled_mm.prequantized_input"
TARGETS = (
    f"{TARGET_MODULE}.FP8ScaledMMLinearKernel.apply_weights",
    f"{TARGET_MODULE}.FP8ScaledMMLinearKernel.supports_quanted_inputs",
)
_CLASS_MARKER = "_vllm_hcu_prequantized_input_applied"
_WRAPPER_MARKER = "_vllm_hcu_prequantized_input_wrapper"


def apply_to_module(module: ModuleType) -> bool:
    scaled_mm = load_exact_module(TARGET_MODULE, module)
    kernel_class = require_class(
        scaled_mm,
        "FP8ScaledMMLinearKernel",
        f"{TARGET_MODULE}.FP8ScaledMMLinearKernel",
    )
    if already_applied(
        kernel_class,
        _CLASS_MARKER,
        ((kernel_class, "apply_weights", TARGETS[0], _WRAPPER_MARKER),),
    ):
        return False

    original = require_callable(kernel_class, "apply_weights", TARGETS[0])
    require_exact_signature(
        original,
        TARGETS[0],
        positional=("self", "layer", "x", "bias"),
        defaults={"bias": None},
    )
    if hasattr(kernel_class, "supports_quanted_inputs"):
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGETS[1]} unexpectedly already exists"
        )

    @functools.wraps(original)
    def hcu_apply_weights(
        self,
        layer,
        x,
        bias=None,
        x_and_scale_quanted=None,
    ):
        if x_and_scale_quanted is None:
            return original(self, layer, x, bias)
        if not supports_quanted_inputs(self):
            raise RuntimeError(
                "prequantized FP8 inputs require the reviewed Channel-FP8 "
                "target Triton route"
            )
        if not isinstance(x_and_scale_quanted, tuple) or len(x_and_scale_quanted) != 2:
            raise ValueError("x_and_scale_quanted must be a (tensor, scale) tuple")

        x_2d_q, x_scale = x_and_scale_quanted
        if not isinstance(x, torch.Tensor):
            raise TypeError("prequantized scaled-mm reference input must be a tensor")
        if not isinstance(x_2d_q, torch.Tensor) or not isinstance(
            x_scale, torch.Tensor
        ):
            raise TypeError("prequantized scaled-mm data and scale must be tensors")
        if x.ndim < 2 or x_2d_q.ndim != 2:
            raise ValueError("prequantized scaled-mm activation must be a 2D tensor")

        weight, weight_scale, _, _ = self._get_layer_params(layer)
        if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
            raise ValueError("prequantized scaled-mm weight must be a 2D tensor")
        num_tokens = prod(x.shape[:-1])
        if tuple(x_2d_q.shape) != (num_tokens, weight.shape[0]):
            raise ValueError(
                "prequantized scaled-mm activation shape does not match input "
                "tokens and weight K"
            )
        if (
            x_2d_q.dtype != weight.dtype
            or x_2d_q.device != weight.device
            or x.device != weight.device
        ):
            raise ValueError(
                "prequantized scaled-mm input and weight must share dtype/device"
            )
        # Comparing the scale shape with ``num_tokens`` while tracing creates a
        # SymBool.  vLLM's piecewise splitter can then thread the relation into
        # a standalone subgraph as a ``sympy.Equality`` input, which this Torch
        # Inductor does not support.  Preserve the friendly eager error here;
        # the target-Triton custom-op implementation repeats this contract with
        # concrete runtime dimensions before launching the backend.
        if not torch.compiler.is_compiling():
            if tuple(x_scale.shape) not in (
                (),
                (1,),
                (num_tokens,),
                (1, 1),
                (num_tokens, 1),
            ):
                raise ValueError(
                    "prequantized scaled-mm scale must be scalar or per-token"
                )
        if (
            x_scale.device != weight.device
            or not x_scale.is_floating_point()
            or not isinstance(weight_scale, torch.Tensor)
            or weight_scale.device != weight.device
        ):
            raise ValueError(
                "prequantized scaled-mm scales must be floating tensors on "
                "the weight device"
            )
        output_shape = [*x.shape[:-1], weight.shape[1]]
        out_dtype = x.dtype if self.config.out_dtype is None else self.config.out_dtype
        return self.apply_scaled_mm(
            A=x_2d_q,
            B=weight,
            out_dtype=out_dtype,
            As=x_scale,
            Bs=weight_scale,
            bias=bias,
            output_shape=output_shape,
        )

    def supports_quanted_inputs(self) -> bool:
        kernel_class = type(self)
        return bool(
            getattr(kernel_class, "_hcu_fp8_patch_applied", False)
            and getattr(kernel_class, "_hcu_fp8_backend", None)
            == "target-triton"
        )

    setattr(hcu_apply_weights, _WRAPPER_MARKER, True)
    setattr(kernel_class, "_vllm_hcu_original_apply_weights", original)
    setattr(kernel_class, "apply_weights", hcu_apply_weights)
    setattr(kernel_class, "supports_quanted_inputs", supports_quanted_inputs)
    setattr(kernel_class, _CLASS_MARKER, True)
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
