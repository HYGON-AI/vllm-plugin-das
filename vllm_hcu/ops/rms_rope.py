# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

"""Public-LightOp fused Q/K RMSNorm and rotary embedding wrapper."""

from __future__ import annotations

import torch
from vllm.utils.torch_utils import direct_register_custom_op


def _hcu_rms_rotary_embedding_impl(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    weight_q: torch.Tensor,
    weight_k: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    from lightop.attention import rms_rotary_embedding_fuse

    output_q, output_k = rms_rotary_embedding_fuse(
        positions,
        query,
        key,
        head_size,
        cos_sin_cache,
        is_neox,
        weight_q,
        weight_k,
        None,
        None,
        epsilon,
    )
    if output_k is None:
        raise RuntimeError("LightOp fused Qwen3 RMS+RoPE returned no key tensor")
    return output_q, output_k


def _hcu_rms_rotary_embedding_fake(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    weight_q: torch.Tensor,
    weight_k: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return query, key


direct_register_custom_op(
    op_name="hcu_fused_rms_rotary_embedding",
    op_func=_hcu_rms_rotary_embedding_impl,
    mutates_args=["query", "key"],
    fake_impl=_hcu_rms_rotary_embedding_fake,
)


def fused_rms_rotary_embedding(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    weight_q: torch.Tensor,
    weight_k: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.vllm.hcu_fused_rms_rotary_embedding(
        positions,
        query,
        key,
        head_size,
        cos_sin_cache,
        is_neox,
        weight_q,
        weight_k,
        epsilon,
    )


__all__ = ["fused_rms_rotary_embedding"]
