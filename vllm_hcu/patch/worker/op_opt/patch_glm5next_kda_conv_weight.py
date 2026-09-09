# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Adapt GLM5Next KDA short-conv weights at the HCU NN-layout boundary."""

from __future__ import annotations

import functools
from types import ModuleType

import torch

from ._common import (
    PatchCompatibilityError,
    load_exact_module,
    require_callable,
    require_exact_signature,
)
from ._gdn_common import use_nn_layout


TARGET_MODULE = "vllm.models.glm5next.nvidia.kda"
TARGET_CLASS = "Glm5NextLinearAttention"
PATCH_ID = "worker.op_opt.mamba.glm5next_kda_conv_weight"
TARGET_SYMBOL = f"{TARGET_MODULE}.{TARGET_CLASS}._forward"
_PATCH_MARKER = "_vllm_hcu_glm5next_kda_conv_weight_applied"
_WRAPPER_MARKER = "_vllm_hcu_glm5next_kda_conv_weight_wrapper"


def _logical_conv_weight(module, projection_size: int, conv_size: int):
    weight = module.weight
    logical_shape = (projection_size, 1, conv_size)
    physical_shape = (conv_size, 1, projection_size)
    shape = tuple(weight.shape)
    if shape == logical_shape:
        return weight.reshape(projection_size, conv_size)
    if shape == physical_shape:
        return weight.permute(2, 1, 0).reshape(projection_size, conv_size)
    raise RuntimeError(
        "HCU GLM5Next KDA conv weight has an incompatible NN layout: "
        f"got {shape}, expected {physical_shape} or {logical_shape}"
    )


def _build_merged_conv_weight(layer) -> torch.Tensor:
    projection_size = int(layer.local_projection_size)
    conv_size = int(layer.conv_size)
    weights = (
        _logical_conv_weight(layer.q_conv1d, projection_size, conv_size),
        _logical_conv_weight(layer.k_conv1d, projection_size, conv_size),
        _logical_conv_weight(layer.v_conv1d, projection_size, conv_size),
    )
    return torch.cat(weights, dim=0).contiguous()


def apply_to_module(module: ModuleType) -> bool:
    kda = load_exact_module(TARGET_MODULE, module)
    layer_cls = vars(kda).get(TARGET_CLASS)
    if not isinstance(layer_cls, type):
        raise PatchCompatibilityError(
            f"required class {TARGET_MODULE}.{TARGET_CLASS} is missing"
        )

    original = require_callable(layer_cls, "_forward", TARGET_SYMBOL)
    if getattr(layer_cls, _PATCH_MARKER, False):
        if not getattr(original, _WRAPPER_MARKER, False):
            raise PatchCompatibilityError(
                f"required HCU patch marker for {TARGET_SYMBOL} is stale"
            )
        return False

    require_exact_signature(
        original,
        TARGET_SYMBOL,
        positional=(
            "self",
            "qkv_proj_states",
            "g1",
            "beta",
            "core_attn_out",
        ),
    )

    @functools.wraps(original)
    def hcu_forward(self, qkv_proj_states, g1, beta, core_attn_out):
        if use_nn_layout() and self._merged_conv_weight is None:
            # vLLM merges q/k/v only after viewing each [out, 1, kernel]
            # tensor. HCU NN storage reverses that shape to [kernel, 1, out],
            # so each projection must be restored before concatenation. A
            # transpose of the already merged [3*kernel, out] tensor cannot
            # recover the official [3*out, kernel] ordering.
            self._merged_conv_weight = _build_merged_conv_weight(self)
        return original(self, qkv_proj_states, g1, beta, core_attn_out)

    setattr(hcu_forward, _WRAPPER_MARKER, True)
    setattr(layer_cls, "_vllm_hcu_original_forward", original)
    setattr(layer_cls, "_forward", hcu_forward)
    setattr(layer_cls, _PATCH_MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = [
    "PATCH_ID",
    "TARGET_CLASS",
    "TARGET_MODULE",
    "TARGET_SYMBOL",
    "apply",
    "apply_to_module",
]
