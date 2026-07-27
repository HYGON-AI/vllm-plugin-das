# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Preserve prefix-cache semantics for combined MTP/indexer KV groups."""

from __future__ import annotations

import functools
import inspect
from types import ModuleType

from ._common import (
    PatchCompatibilityError,
    already_applied,
    load_exact_module,
    require_callable,
    require_class,
)

TARGET_MODULE = "vllm.v1.core.kv_cache_coordinator"
PATCH_ID = "platform.framework_opt.mtp_indexer_kv_cache_coordinator"
TARGETS = (f"{TARGET_MODULE}.KVCacheCoordinator.__init__",)
_MARKER = "_vllm_hcu_mtp_indexer_coordinator_applied"
_WRAPPER = "_vllm_hcu_mtp_indexer_coordinator_wrapper"

_PARAMETERS = (
    "self",
    "kv_cache_config",
    "max_model_len",
    "max_num_batched_tokens",
    "use_eagle",
    "enable_caching",
    "enable_kv_cache_events",
    "dcp_world_size",
    "pcp_world_size",
    "scheduler_block_size",
    "hash_block_size",
    "metrics_collector",
)


def _has_unmarked_mtp_indexer_group(kv_cache_config: object) -> bool:
    groups = getattr(kv_cache_config, "kv_cache_groups", ())
    if any(bool(getattr(group, "is_eagle_group", False)) for group in groups):
        return False
    for group in groups:
        layer_names = tuple(
            name.lower()
            for name in getattr(group, "layer_names", ())
            if isinstance(name, str)
        )
        has_mtp = any("mtp" in name or "nextn" in name for name in layer_names)
        has_indexer = any(
            "indexer" in name or "k_cache" in name for name in layer_names
        )
        if has_mtp and has_indexer:
            return True
    return False


def apply_to_module(module: ModuleType) -> bool:
    coordinator = load_exact_module(TARGET_MODULE, module)
    coordinator_cls = require_class(
        coordinator, "KVCacheCoordinator", f"{TARGET_MODULE}.KVCacheCoordinator"
    )
    if already_applied(
        coordinator,
        _MARKER,
        ((coordinator_cls, "__init__", _WRAPPER),),
    ):
        return False

    original = require_callable(coordinator_cls, "__init__", TARGETS[0])
    signature = inspect.signature(original)
    if tuple(signature.parameters) != _PARAMETERS:
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGETS[0]} has incompatible "
            f"signature {signature}"
        )
    if signature.parameters["metrics_collector"].default is not None:
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGETS[0]} has incompatible "
            f"signature {signature}"
        )

    @functools.wraps(original)
    def hcu_init(
        self,
        kv_cache_config,
        max_model_len,
        max_num_batched_tokens,
        use_eagle,
        enable_caching,
        enable_kv_cache_events,
        dcp_world_size,
        pcp_world_size,
        scheduler_block_size,
        hash_block_size,
        metrics_collector=None,
    ):
        # vLLM's all-group fallback is correct for generic unmarked EAGLE
        # models.  Combined GLM/DeepSeek MTP+indexer groups are the exception:
        # dropping the last block from every group makes a prefix-cache hit
        # diverge from cold prefill.
        effective_use_eagle = use_eagle and not _has_unmarked_mtp_indexer_group(
            kv_cache_config
        )
        return original(
            self,
            kv_cache_config,
            max_model_len,
            max_num_batched_tokens,
            effective_use_eagle,
            enable_caching,
            enable_kv_cache_events,
            dcp_world_size,
            pcp_world_size,
            scheduler_block_size,
            hash_block_size,
            metrics_collector,
        )

    setattr(hcu_init, _WRAPPER, True)
    setattr(coordinator_cls, "_vllm_hcu_original_init", original)
    setattr(coordinator_cls, "__init__", hcu_init)
    setattr(coordinator, _MARKER, True)
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
