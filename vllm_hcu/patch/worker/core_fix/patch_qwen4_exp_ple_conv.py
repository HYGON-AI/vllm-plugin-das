# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Replace Qwen4Exp PLE's graph-unsafe HCU depthwise convolution."""

from __future__ import annotations

import functools
import importlib
import inspect
from types import ModuleType

import torch

from vllm.utils.torch_utils import direct_register_custom_op

from ._common import PatchCompatibilityError, load_exact_module, require_class

TARGET_MODULE = "vllm.models.qwen4_exp.amd.ple_layer"
PATCH_ID = "worker.core_fix.qwen4_exp.ple_depthwise_conv1d"
_MARKER = "_vllm_hcu_qwen4_exp_ple_conv_applied"
_WRAPPER = "_vllm_hcu_qwen4_exp_ple_fallback_wrapper"
_NGRAM_WRAPPER = "_vllm_hcu_qwen4_exp_ple_ngram_wrapper"


def _single_int(value) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, (tuple, list)) and len(value) == 1:
        return int(value[0])
    return None


def _can_use_hcu_depthwise_conv1d(
    inputs,
    weight,
    bias,
    stride,
    padding,
    dilation,
    groups,
) -> bool:
    stride_value = _single_int(stride)
    padding_value = _single_int(padding)
    dilation_value = _single_int(dilation)
    return bool(
        getattr(inputs, "ndim", None) == 3
        and getattr(weight, "ndim", None) == 3
        and weight.shape[1] == 1
        and inputs.shape[1] == weight.shape[0] == groups
        and (bias is None or tuple(bias.shape) == (groups,))
        and stride_value == 1
        and padding_value is not None
        and padding_value >= 0
        and dilation_value is not None
        and dilation_value > 0
    )


def _depthwise_conv1d(inputs, weight, bias, *, padding: int, dilation: int):
    from vllm_hcu.ops.depthwise_conv1d import depthwise_conv1d

    return depthwise_conv1d(
        inputs,
        weight,
        bias,
        padding=padding,
        dilation=dilation,
    )


class _HcuFunctionalProxy:
    def __init__(self, functional):
        self._functional = functional

    def __getattr__(self, name):
        return getattr(self._functional, name)

    def conv1d(
        self,
        inputs,
        weight,
        bias=None,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
    ):
        if _can_use_hcu_depthwise_conv1d(
            inputs, weight, bias, stride, padding, dilation, groups
        ):
            return _depthwise_conv1d(
                inputs,
                weight,
                bias,
                padding=_single_int(padding),
                dilation=_single_int(dilation),
            )
        return self._functional.conv1d(
            inputs,
            weight,
            bias,
            stride,
            padding,
            dilation,
            groups,
        )


def _hcu_qwen4_exp_ple_ngram(
    input_ids: torch.Tensor,
    query_start_loc: torch.Tensor,
    ngram_context: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    from vllm.forward_context import get_forward_context

    ple = importlib.import_module(TARGET_MODULE)
    original_forward = getattr(ple, "_vllm_hcu_original_ngram_forward")
    owner = get_forward_context().no_compile_layers[layer_name]
    result = original_forward(
        owner.ple_embedding,
        input_ids,
        query_start_loc,
        ngram_context,
    )
    output.copy_(result)


direct_register_custom_op(
    op_name="hcu_qwen4_exp_ple_ngram",
    op_func=_hcu_qwen4_exp_ple_ngram,
    mutates_args=["output"],
)


def _run_ple_ngram_custom_op(
    input_ids,
    query_start_loc,
    ngram_context,
    output,
    layer_name,
) -> None:
    torch.ops.vllm.hcu_qwen4_exp_ple_ngram(
        input_ids,
        query_start_loc,
        ngram_context,
        output,
        layer_name,
    )


def apply_to_module(module: ModuleType) -> bool:
    ple = load_exact_module(TARGET_MODULE, module)
    ple_class = require_class(ple, "Qwen4ExpPLELayer", f"{TARGET_MODULE}.Qwen4ExpPLELayer")
    ngram_class = require_class(
        ple,
        "Qwen4ExpNGramEmbedding",
        f"{TARGET_MODULE}.Qwen4ExpNGramEmbedding",
    )

    if getattr(ple, _MARKER, False):
        if not isinstance(getattr(ple, "F", None), _HcuFunctionalProxy) or not getattr(
            ple_class._short_conv_fallback, _WRAPPER, False
        ) or not getattr(
            ngram_class.forward, _NGRAM_WRAPPER, False
        ):
            raise PatchCompatibilityError(
                f"required HCU patch marker for {TARGET_MODULE} is stale"
            )
        return False

    functional = getattr(ple, "F", None)
    if functional is None or not callable(getattr(functional, "conv1d", None)):
        raise PatchCompatibilityError(f"required target {TARGET_MODULE}.F.conv1d is missing")
    fallback = getattr(ple_class, "_short_conv_fallback", None)
    if not callable(fallback) or tuple(inspect.signature(fallback).parameters) != (
        "self",
        "inputs",
    ):
        raise PatchCompatibilityError(
            f"required target {TARGET_MODULE}.Qwen4ExpPLELayer._short_conv_fallback "
            "has incompatible signature"
        )
    ngram_forward = getattr(ngram_class, "forward", None)
    if not callable(ngram_forward) or tuple(
        inspect.signature(ngram_forward).parameters
    ) != ("self", "input_ids", "query_start_loc", "ngram_context"):
        raise PatchCompatibilityError(
            f"required target {TARGET_MODULE}.Qwen4ExpNGramEmbedding.forward "
            "has incompatible signature"
        )

    @functools.wraps(fallback)
    def hcu_short_conv_fallback(self, inputs):
        inputs_t = inputs.transpose(0, 1).unsqueeze(0)
        output = _depthwise_conv1d(
            inputs_t,
            self.conv1d.weight,
            self.conv1d.bias,
            padding=self.conv_state_len,
            dilation=self.short_conv_dilation,
        )[..., : inputs_t.size(-1)]
        return functional.silu(output).squeeze(0).transpose(0, 1)

    @functools.wraps(ngram_forward)
    def hcu_ngram_forward(self, input_ids, query_start_loc, ngram_context):
        output = torch.empty(
            (input_ids.reshape(-1).shape[0], self.embedding_dim),
            dtype=self.ngram_embedding.params_dtype,
            device=input_ids.device,
        )
        _run_ple_ngram_custom_op(
            input_ids,
            query_start_loc,
            ngram_context,
            output,
            self.layer_name,
        )
        return output

    setattr(hcu_short_conv_fallback, _WRAPPER, True)
    setattr(hcu_ngram_forward, _NGRAM_WRAPPER, True)
    setattr(ple, "_vllm_hcu_original_functional", functional)
    setattr(ple, "_vllm_hcu_original_short_conv_fallback", fallback)
    setattr(ple, "_vllm_hcu_original_ngram_forward", ngram_forward)
    setattr(ple, "F", _HcuFunctionalProxy(functional))
    setattr(ple_class, "_short_conv_fallback", hcu_short_conv_fallback)
    setattr(ngram_class, "forward", hcu_ngram_forward)
    setattr(ple, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "apply", "apply_to_module"]
