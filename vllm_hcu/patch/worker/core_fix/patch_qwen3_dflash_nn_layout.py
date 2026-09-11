# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Adapt Qwen3 DSpark context-KV fusion to the HCU NN weight layout."""

from __future__ import annotations

import functools
from types import ModuleType

import torch

from ._common import (
    PatchCompatibilityError,
    load_exact_module,
    require_callable,
    require_class,
    require_exact_signature,
)

TARGET_MODULE = "vllm.model_executor.models.qwen3_dflash"
PATCH_ID = "worker.core_fix.qwen3_dflash.nn_layout"
TARGET_SYMBOL = f"{TARGET_MODULE}.DFlashQwen3Model._build_context_kv_buffers"
_CLASS_MARKER = "_vllm_hcu_qwen3_dflash_nn_layout_applied"
_WRAPPER_MARKER = "_vllm_hcu_qwen3_dflash_nn_layout_wrapper"


def _use_nn_layout() -> bool:
    try:
        from vllm_hcu.platforms import envs as henvs

        return bool(henvs.VLLM_USE_NN)
    except (AttributeError, ImportError) as exc:
        raise PatchCompatibilityError(
            "required HCU NN weight-layout feature flag is unavailable"
        ) from exc


def apply_to_module(module: ModuleType) -> bool:
    qwen3_dflash = load_exact_module(TARGET_MODULE, module)
    model_class = require_class(
        qwen3_dflash,
        "DFlashQwen3Model",
        f"{TARGET_MODULE}.DFlashQwen3Model",
    )
    original = require_callable(
        model_class,
        "_build_context_kv_buffers",
        TARGET_SYMBOL,
    )
    require_exact_signature(
        original,
        TARGET_SYMBOL,
        positional=("self", "layers_attn", "has_bias"),
    )
    if getattr(model_class, _CLASS_MARKER, False):
        if not getattr(original, _WRAPPER_MARKER, False):
            raise PatchCompatibilityError(
                f"required HCU patch marker for {TARGET_SYMBOL} is stale"
            )
        return False

    @functools.wraps(original)
    def hcu_build_context_kv_buffers(self, layers_attn, has_bias):
        if not _use_nn_layout():
            return original(self, layers_attn, has_bias)

        try:
            quantized = [
                bool(attention.qkv_proj.is_quantization)
                for attention in layers_attn
            ]
        except AttributeError as exc:
            raise PatchCompatibilityError(
                f"required HCU NN weight metadata for {TARGET_SYMBOL} is missing"
            ) from exc
        if any(quantized):
            if not all(quantized):
                raise PatchCompatibilityError(
                    f"required HCU QKV quantization for {TARGET_SYMBOL} is inconsistent"
                )
            return original(self, layers_attn, has_bias)

        self._hidden_norm_weight = self.hidden_norm.weight.data
        kv_weights = []
        for attention in layers_attn:
            projection = attention.qkv_proj
            weight = projection.weight
            expected_shape = (
                projection.input_size,
                attention.q_size + 2 * attention.kv_size,
            )
            if tuple(weight.shape) != expected_shape:
                raise PatchCompatibilityError(
                    f"required HCU NN weight layout for {TARGET_SYMBOL} "
                    f"expected {expected_shape}, got {tuple(weight.shape)}"
                )
            kv_weights.append(weight[:, attention.q_size :].transpose(0, 1))
        self._fused_kv_weight = torch.cat(kv_weights, dim=0).contiguous()

        if has_bias:
            self._fused_kv_bias = torch.cat(
                [
                    attention.qkv_proj.bias[attention.q_size :]
                    for attention in layers_attn
                ],
                dim=0,
            )
        else:
            self._fused_kv_bias = None
        self._k_norm_weights = torch.stack(
            [attention.k_norm.weight.data for attention in layers_attn], dim=0
        ).contiguous()

    setattr(hcu_build_context_kv_buffers, _WRAPPER_MARKER, True)
    setattr(
        model_class,
        "_vllm_hcu_original_build_context_kv_buffers",
        original,
    )
    setattr(
        model_class,
        "_build_context_kv_buffers",
        hcu_build_context_kv_buffers,
    )
    setattr(model_class, _CLASS_MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "apply", "apply_to_module"]
