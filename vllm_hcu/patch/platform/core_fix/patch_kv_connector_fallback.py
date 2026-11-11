# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Sparse-aware kv-connector fallback backend resolution.

``vllm.distributed.kv_transfer.kv_connector.utils.get_current_attn_backends``
falls back to a blind dense resolution when the calling process has no
layers in the static forward context (the connector/engine-core side).
With DSA sparse MLA (``--kv-cache-dtype fp8_e4m3`` normalized to
``fp8_ds_mla``), that fallback resolves FLASHMLA/TRITON_MLA and aborts with
"No valid attention backend found ... use_sparse=False".  The wrapper below
keeps the layered path untouched and only adds a ``ds_mla``-aware
``use_sparse`` flag to the fallback resolution.
"""

from __future__ import annotations

import functools
from types import ModuleType
from typing import Any

from ._common import load_exact_module

TARGET_MODULE = "vllm.distributed.kv_transfer.kv_connector.utils"
PATCH_ID = "platform.core_fix.kv_connector_fallback_sparse"
TARGETS = (f"{TARGET_MODULE}.get_current_attn_backends",)
_MARKER = "_vllm_hcu_sparse_fallback_applied"

_DS_MLA_DTYPES = ("fp8_ds_mla", "nvfp4_ds_mla")


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    if getattr(target, _MARKER, False):
        return False

    original = target.get_current_attn_backends

    @functools.wraps(original)
    def get_current_attn_backends(
        vllm_config: Any, layer_names: list[str] | None = None
    ) -> list[type]:
        from vllm.config.vllm import get_layers_from_vllm_config
        from vllm.model_executor.layers.attention_layer_base import (
            AttentionLayerBase,
        )

        layers = get_layers_from_vllm_config(
            vllm_config, AttentionLayerBase, layer_names
        )
        if layers:
            # Layered path: real layers report their own backends.
            return original(vllm_config, layer_names)
        # Fallback (no layers in this process): mirror the upstream body but
        # resolve sparse when the (already normalized) cache dtype is a
        # ds_mla sparse layout.
        from vllm.v1.attention.selector import get_attn_backend
        from vllm.config import set_current_vllm_config

        cache_dtype = vllm_config.cache_config.cache_dtype
        with set_current_vllm_config(vllm_config):
            return [
                get_attn_backend(
                    head_size=vllm_config.model_config.get_head_size(),
                    dtype=vllm_config.model_config.dtype,
                    kv_cache_dtype=cache_dtype,
                    use_mla=vllm_config.model_config.use_mla,
                    use_sparse=cache_dtype in _DS_MLA_DTYPES,
                )
            ]

    target.get_current_attn_backends = get_current_attn_backends
    setattr(target, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))
