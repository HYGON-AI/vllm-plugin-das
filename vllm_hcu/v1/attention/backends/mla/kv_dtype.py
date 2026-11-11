# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Per-group KV-cache dtype resolution for the HCU dense MLA builders.

The dense MLA metadata builders historically keyed their FP8 planning off the
run-level ``cache_config.cache_dtype``.  That is exact while every dense MLA
group shares the run dtype, but a Kimi-K3.1 DSpark draft keeps BF16 latent
pages (see ``patch_kimi_k3_dspark_draft_attention``) while the run dtype is
the sparse-only ``fp8_ds_mla``: planning FP8 scheduler metadata or the FP8
non-causal fast path for BF16 storage is wrong.

The resolution only ever DOWNGRADES a quantized run dtype to ``"auto"`` when
every layer of the group demonstrably stores an unquantized (>=16-bit
floating) dtype; every other case returns the run dtype unchanged, so groups
that follow the run dtype keep their exact previous behavior.

Stdlib-only on purpose: importable (and unit-testable) without a device.
"""

from __future__ import annotations

from typing import Any

_UNQUANTIZED_CACHE_DTYPES = frozenset(
    {"auto", "bfloat16", "float16", "half", "float", "float32"}
)


def is_quantized_kv_cache_name(cache_dtype: str | None) -> bool:
    """Name-level mirror of vLLM's ``is_quantized_kv_cache``."""

    return cache_dtype is not None and cache_dtype not in _UNQUANTIZED_CACHE_DTYPES


def _is_unquantized_storage(dtype: Any) -> bool:
    # torch.bfloat16/float16/float32 -> True; uint8 / float8_* -> False.
    return bool(getattr(dtype, "is_floating_point", False)) and int(
        getattr(dtype, "itemsize", 0) or 0
    ) >= 2


def _group_storage_dtypes(kv_cache_spec: Any) -> list[Any]:
    dtype = getattr(kv_cache_spec, "dtype", None)
    if dtype is not None:
        return [dtype]
    # UniformTypeKVCacheSpecs carries per-layer specs instead of one dtype.
    per_layer = getattr(kv_cache_spec, "kv_cache_specs", None)
    if isinstance(per_layer, dict) and per_layer:
        dtypes = [getattr(spec, "dtype", None) for spec in per_layer.values()]
        if all(item is not None for item in dtypes):
            return dtypes
    return []


def effective_group_kv_cache_dtype(kv_cache_spec: Any, run_cache_dtype: str) -> str:
    """KV dtype the group's pages actually hold, for builder planning."""

    if not is_quantized_kv_cache_name(run_cache_dtype):
        return run_cache_dtype
    dtypes = _group_storage_dtypes(kv_cache_spec)
    if dtypes and all(_is_unquantized_storage(dtype) for dtype in dtypes):
        return "auto"
    return run_cache_dtype


__all__ = ["effective_group_kv_cache_dtype", "is_quantized_kv_cache_name"]
