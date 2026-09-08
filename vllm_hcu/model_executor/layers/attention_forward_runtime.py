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
    if self.query_quant is not None:
        if self.kv_cache_dtype not in {"fp8", "fp8_e4m3", "fp8_e5m2", "nvfp4"}:
            raise ValueError(
                "unsupported HCU quantized attention KV-cache dtype "
                f"{self.kv_cache_dtype!r}"
            )
        if self.impl.supports_quant_query_input:
            query, _ = self.query_quant(query, self._q_scale)

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
    needs_kv_cache_update = (
        not self.attn_backend.forward_includes_kv_cache_update
        and self.kv_sharing_target_layer_name is None
        and key is not None
        and value is not None
    )
    if self.use_direct_call:
        if needs_kv_cache_update:
            kv_cache_dummy_dep = upstream.unified_kv_cache_update(
                key, value, self.layer_name
            )
        upstream.unified_attention_with_output(
            query,
            key,
            value,
            output,
            self.layer_name,
            kv_cache_dummy_dep=kv_cache_dummy_dep,
        )
    else:
        encoded = upstream._encode_layer_name(self.layer_name)
        if needs_kv_cache_update:
            kv_cache_dummy_dep = torch.ops.vllm.unified_kv_cache_update(
                key, value, encoded
            )
        torch.ops.vllm.unified_attention_with_output(
            query,
            key,
            value,
            output,
            encoded,
            kv_cache_dummy_dep=kv_cache_dummy_dep,
        )
    return output.view(-1, hidden_size)


__all__ = ["attention_forward"]
