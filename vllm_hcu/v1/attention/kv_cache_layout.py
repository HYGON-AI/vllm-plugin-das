# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Compatibility helpers for vLLM 0.28 KV-cache layout descriptors."""

from __future__ import annotations


# Each worker serves one engine. Cache allocation runs under the engine config
# context, but model forwards and Graph replay need not retain that context.
_worker_kv_cache_layout: str | None = None


def get_kv_cache_layout() -> str:
    """Return the active per-layer cache layout in flash-attn terminology."""
    from vllm.config import get_current_vllm_config_or_none

    global _worker_kv_cache_layout

    vllm_config = get_current_vllm_config_or_none()
    if vllm_config is None:
        return _worker_kv_cache_layout or "NHD"
    if vllm_config.cache_config.kv_cache_layout is None:
        # Do not let a temporary, unresolved config replace the worker layout.
        return "NHD"
    layout = vllm_config.cache_config.get_resolved_kv_cache_layout()
    order = layout.layer_view_order
    _worker_kv_cache_layout = "HND" if order.index(1) < order.index(2) else "NHD"
    return _worker_kv_cache_layout


__all__ = ["get_kv_cache_layout"]
