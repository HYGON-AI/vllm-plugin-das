# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Compatibility helpers for vLLM 0.28 KV-cache layout descriptors."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.kv_cache_layout import KVCacheLayout


def prefer_hnd_layouts(
    supported: tuple[KVCacheLayout, ...] | None,
) -> tuple[KVCacheLayout, ...] | None:
    """Express a gfx938 MLA preference without changing supported layouts.

    Only the HCU FLASHMLA and FLASHMLA_SPARSE backends call this helper. The
    upstream resolver still owns explicit settings, connector preferences,
    and cross-backend checks. Other architectures retain the inherited
    declaration, including None.
    """
    from vllm_hcu.platforms.hcu import on_gfx938

    if not on_gfx938():
        return supported

    from vllm.v1.kv_cache_layout import KVCacheLayout

    # None means unrestricted, so retain every layout rather than narrowing
    # the set to HND/NHD and potentially rejecting a KV transfer connector.
    layouts = tuple(KVCacheLayout) if supported is None else supported
    hnd = KVCacheLayout.LBHNC
    if hnd not in layouts:
        return supported
    return (hnd, *(layout for layout in layouts if layout != hnd))


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


__all__ = ["get_kv_cache_layout", "prefer_hnd_layouts"]
