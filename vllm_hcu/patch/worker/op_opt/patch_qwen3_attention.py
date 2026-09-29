# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

"""Fuse Qwen3 query/key RMSNorm and rotary embedding on HCU."""

from __future__ import annotations

import functools
import importlib
from types import ModuleType

import torch

from vllm_hcu.platforms import envs as henvs

from ._common import (
    already_applied,
    load_exact_module,
    require_callable,
    require_class,
    require_exact_signature,
)

TARGET_MODULE = "vllm.model_executor.models.qwen3"
PATCH_ID = "worker.op_opt.qwen3.fused_rms_rope"
TARGETS = (f"{TARGET_MODULE}.Qwen3Attention.forward",)
_CLASS_MARKER = "_vllm_hcu_fused_rms_rope_applied"
_WRAPPER_MARKER = "_vllm_hcu_fused_rms_rope_wrapper"


def fused_rms_rotary_embedding(*args):
    """Load the custom-op owner only when the fused route actually runs."""

    from vllm_hcu.ops.rms_rope import fused_rms_rotary_embedding as fused

    return fused(*args)


def _register_fused_op_owner() -> None:
    """Register custom-op schemas before an AOT cache is loaded."""

    importlib.import_module("vllm_hcu.ops.rms_rope")


def _supports_fused_path(
    attention: object,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
) -> bool:
    if positions.ndim != 1 or hidden_states.ndim != 2:
        return False
    head_dim = getattr(attention, "head_dim", None)
    rotary_emb = getattr(attention, "rotary_emb", None)
    cos_sin_cache = getattr(rotary_emb, "cos_sin_cache", None)
    if (
        not isinstance(head_dim, int)
        or getattr(attention, "dual_chunk_attention_config", None) is not None
        or rotary_emb is None
        or getattr(rotary_emb, "head_size", None) != head_dim
        or getattr(rotary_emb, "rotary_dim", None) != head_dim
        or not isinstance(cos_sin_cache, torch.Tensor)
        or cos_sin_cache.ndim != 2
        or cos_sin_cache.shape[-1] != head_dim
        or getattr(rotary_emb, "update_cache", False)
        or not callable(getattr(rotary_emb, "_match_cos_sin_cache_dtype", None))
        or not isinstance(getattr(rotary_emb, "is_neox_style", None), bool)
    ):
        return False
    q_norm = getattr(attention, "q_norm", None)
    k_norm = getattr(attention, "k_norm", None)
    q_epsilon = getattr(q_norm, "variance_epsilon", None)
    return (
        isinstance(getattr(q_norm, "weight", None), torch.Tensor)
        and isinstance(getattr(k_norm, "weight", None), torch.Tensor)
        and isinstance(q_epsilon, float)
        and getattr(k_norm, "variance_epsilon", None) == q_epsilon
    )


def apply_to_module(module: ModuleType) -> bool:
    qwen3 = load_exact_module(TARGET_MODULE, module)
    attention_class = require_class(
        qwen3, "Qwen3Attention", f"{TARGET_MODULE}.Qwen3Attention"
    )
    if already_applied(
        attention_class,
        _CLASS_MARKER,
        (
            (
                attention_class,
                "forward",
                TARGETS[0],
                _WRAPPER_MARKER,
            ),
        ),
    ):
        return False

    original = require_callable(attention_class, "forward", TARGETS[0])
    require_exact_signature(
        original,
        TARGETS[0],
        positional=("self", "positions", "hidden_states"),
    )
    feature_enabled = henvs.fused_qwen3_rms_rope_enabled()
    if feature_enabled:
        _register_fused_op_owner()

    @functools.wraps(original)
    def hcu_forward(self, positions, hidden_states):
        if not feature_enabled or not _supports_fused_path(
            self, positions, hidden_states
        ):
            return original(self, positions, hidden_states)

        qkv, _ = self.qkv_proj(hidden_states)
        query, key, value = qkv.split(
            [self.q_size, self.kv_size, self.kv_size], dim=-1
        )
        query = query.contiguous()
        key = key.contiguous()
        rotary_emb = self.rotary_emb
        cos_sin_cache = rotary_emb._match_cos_sin_cache_dtype(query)
        query, key = fused_rms_rotary_embedding(
            positions,
            query,
            key,
            self.head_dim,
            cos_sin_cache,
            rotary_emb.is_neox_style,
            self.q_norm.weight,
            self.k_norm.weight,
            self.q_norm.variance_epsilon,
        )
        attn_output = self.attn(query, key, value)
        output, _ = self.o_proj(attn_output)
        return output

    setattr(hcu_forward, _WRAPPER_MARKER, True)
    setattr(attention_class, "_vllm_hcu_original_forward", original)
    setattr(attention_class, "forward", hcu_forward)
    setattr(attention_class, _CLASS_MARKER, True)
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
