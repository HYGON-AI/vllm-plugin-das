# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Expose the AITER cache writer expected by MiniMax-M3's sparse path."""

from __future__ import annotations

import importlib
from types import ModuleType

from ._common import (
    PatchCompatibilityError,
    load_exact_module,
    require_callable,
    require_class,
)

TARGET_MODULE = "vllm.models.minimax_m3.amd.model"
PATCH_ID = "worker.core_fix.minimax_m3.aiter_cache_export"
_MARKER = "_vllm_hcu_minimax_m3_aiter_cache_export_applied"


def apply_to_module(module: ModuleType) -> bool:
    model = load_exact_module(TARGET_MODULE, module)
    attention = require_class(
        model,
        "MiniMaxM3SparseAttention",
        f"{TARGET_MODULE}.MiniMaxM3SparseAttention",
    )
    require_callable(
        attention,
        "_insert_aiter_sparse_pa_kv",
        f"{TARGET_MODULE}.MiniMaxM3SparseAttention._insert_aiter_sparse_pa_kv",
    )

    aiter = importlib.import_module("aiter")
    if getattr(model, _MARKER, False):
        if not callable(getattr(aiter, "reshape_and_cache", None)):
            raise PatchCompatibilityError(
                "required HCU MiniMax-M3 AITER reshape_and_cache export is stale"
            )
        return False

    if not hasattr(aiter, "reshape_and_cache"):
        cache_ops = importlib.import_module("aiter.ops.cache")
        reshape_and_cache = require_callable(
            cache_ops,
            "reshape_and_cache",
            "aiter.ops.cache.reshape_and_cache",
        )
        # vLLM v0.28.1's MiniMax-M3 sparse writer imports this symbol from the
        # package root.  Current HCU AITER ships the same operator in
        # aiter.ops.cache but does not re-export it.
        aiter.reshape_and_cache = reshape_and_cache
    elif not callable(aiter.reshape_and_cache):
        raise PatchCompatibilityError(
            "required HCU patch target aiter.reshape_and_cache is not callable"
        )

    setattr(model, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "apply", "apply_to_module"]
