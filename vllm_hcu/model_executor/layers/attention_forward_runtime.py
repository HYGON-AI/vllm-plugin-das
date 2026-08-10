# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Dependency-light HCU attention forward implementation.

This module intentionally depends only on PyTorch.  The complete attention
runtime imports vLLM backend classes and native accelerator extensions, while
this forward wrapper can be validated independently in portable CPU tests.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any

import torch


def quantize_attention_query(self: Any, query: torch.Tensor) -> torch.Tensor:
    """Match the query dtype to a quantized KV cache when supported."""

    if self.query_quant is None:
        return query
    if self.kv_cache_dtype not in {
        "fp8",
        "fp8_e4m3",
        "fp8_e5m2",
        "nvfp4",
    }:
        raise ValueError(
            "unsupported HCU quantized attention KV-cache dtype "
            f"{self.kv_cache_dtype!r}"
        )
    if self.impl.supports_quant_query_input:
        query, _ = self.query_quant(query, self._q_scale)
    return query


def attention_forward(
    upstream: ModuleType,
    self: Any,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output_shape: torch.Size | None = None,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Run attention while preserving HCU's custom split KV-cache semantics."""

    if self.calculate_kv_scales:
        torch.ops.vllm.maybe_calc_kv_scales(
            query,
            key,
            value,
            upstream._encode_layer_name(self.layer_name),
        )
    if output_dtype is None:
        output_dtype = query.dtype
    query = quantize_attention_query(self, query)

    if output_shape is None:
        num_tokens = query.shape[0]
        output_shape = torch.Size((num_tokens, self.num_heads * self.head_size_v))
    output = torch.empty(output_shape, dtype=output_dtype, device=query.device)
    hidden_size = output_shape[-1]
    query = query.view(-1, self.num_heads, self.head_size)
    output = output.view(-1, self.num_heads, self.head_size_v)
    if key is not None:
        key = key.view(-1, self.num_kv_heads, self.head_size)
    if value is not None:
        value = value.view(-1, self.num_kv_heads, self.head_size_v)

    kv_cache_dummy_dep = None
    if (
        not self.attn_backend.forward_includes_kv_cache_update
        and self.kv_sharing_target_layer_name is None
        and key is not None
        and value is not None
    ):
        layer_name = upstream._resolve_layer_name(self.layer_name)
        _, attn_layer, kv_cache, layer_slot_mapping = upstream.get_attention_context(
            layer_name
        )
        if layer_slot_mapping is not None:
            update = getattr(attn_layer.impl, "do_kv_cache_update", None)
            if not callable(update):
                raise RuntimeError(
                    f"{attn_layer.impl.__class__.__name__} does not support KV cache update"
                )
            update(attn_layer, key, value, kv_cache, layer_slot_mapping)
        # HCU's custom cache is a (key, value) pair and has no ``.device``.
        kv_cache_dummy_dep = torch.empty(0, device=key.device, dtype=key.dtype)
    upstream.unified_attention_with_output(
        query,
        key,
        value,
        output,
        self.layer_name,
        kv_cache_dummy_dep=kv_cache_dummy_dep,
    )
    return output.view(-1, hidden_size)


__all__ = ["attention_forward", "quantize_attention_query"]
