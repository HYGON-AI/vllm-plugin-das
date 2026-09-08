# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Preserve prefix-cache semantics for hybrid MTP KV groups."""

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
TARGETS = (
    f"{TARGET_MODULE}.KVCacheCoordinator.__init__",
    f"{TARGET_MODULE}.MambaManager.cache_blocks",
)
_MARKER = "_vllm_hcu_mtp_indexer_coordinator_applied"
_WRAPPER = "_vllm_hcu_mtp_indexer_coordinator_wrapper"
_MAMBA_WRAPPER = "_vllm_hcu_mamba_eagle_replay_wrapper"

_PARAMETERS = (
    "self",
    "kv_cache_config",
    "max_model_len",
    "max_in_flight_tokens",
    "use_eagle",
    "enable_caching",
    "enable_kv_cache_events",
    "dcp_world_size",
    "pcp_world_size",
    "scheduler_block_size",
    "hash_block_size",
    "metrics_collector",
    "num_prefill_lookahead",
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
    mamba_cls = require_class(coordinator, "MambaManager", TARGETS[1])
    if already_applied(
        coordinator,
        _MARKER,
        (
            (coordinator_cls, "__init__", _WRAPPER),
            (mamba_cls, "cache_blocks", _MAMBA_WRAPPER),
        ),
    ):
        return False

    original = require_callable(coordinator_cls, "__init__", TARGETS[0])
    signature = inspect.signature(original)
    if tuple(signature.parameters) != _PARAMETERS:
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGETS[0]} has incompatible "
            f"signature {signature}"
        )
    if (
        signature.parameters["metrics_collector"].default is not None
        or signature.parameters["num_prefill_lookahead"].default != 0
    ):
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGETS[0]} has incompatible "
            f"signature {signature}"
        )

    original_mamba_cache_blocks = require_callable(
        mamba_cls, "cache_blocks", TARGETS[1]
    )

    @functools.wraps(original)
    def hcu_init(
        self,
        kv_cache_config,
        max_model_len,
        max_in_flight_tokens,
        use_eagle,
        enable_caching,
        enable_kv_cache_events,
        dcp_world_size,
        pcp_world_size,
        scheduler_block_size,
        hash_block_size,
        metrics_collector=None,
        num_prefill_lookahead=0,
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
            max_in_flight_tokens,
            effective_use_eagle,
            enable_caching,
            enable_kv_cache_events,
            dcp_world_size,
            pcp_world_size,
            scheduler_block_size,
            hash_block_size,
            metrics_collector,
            num_prefill_lookahead,
        )

    setattr(hcu_init, _WRAPPER, True)
    @functools.wraps(original_mamba_cache_blocks)
    def cache_mamba_eagle_replay_boundary(
        self, request, num_tokens, retention_interval=None
    ):
        result = original_mamba_cache_blocks(
            self,
            request,
            num_tokens,
            retention_interval=retention_interval,
        )
        if (
            not self.drop_eagle_checkpoint_block
            or retention_interval is None
            or self.mamba_cache_mode != "align"
        ):
            return result

        alignment = self.cache_hit_alignment_tokens
        if not isinstance(alignment, int) or alignment <= 0:
            return result
        replay_boundary = (
            (request.num_prompt_tokens - 1) // alignment * alignment
        ) - alignment
        if replay_boundary <= 0 or num_tokens < replay_boundary:
            return result

        block_idx = replay_boundary // self.block_size - 1
        blocks = self.req_to_blocks.get(request.request_id, ())
        if block_idx < 0 or block_idx >= len(blocks):
            return result
        block = blocks[block_idx]
        if block.is_null or block.block_hash is not None:
            return result

        # The scheduler has materialized this exact recurrent state, but
        # upstream retention=0 masks the pre-Eagle replay boundary. Register
        # only that existing state; do not make Mamba caching dense.
        self.block_pool.cache_full_blocks(
            request=request,
            blocks=blocks,
            num_cached_blocks=block_idx,
            num_full_blocks=block_idx + 1,
            block_size=self.block_size,
            kv_cache_group_id=self.kv_cache_group_id,
        )
        if block.block_hash is not None:
            self.cached_blocks_this_step.add(block.block_hash)
        return result

    setattr(cache_mamba_eagle_replay_boundary, _MAMBA_WRAPPER, True)
    setattr(coordinator_cls, "_vllm_hcu_original_init", original)
    setattr(
        mamba_cls,
        "_vllm_hcu_original_cache_blocks",
        original_mamba_cache_blocks,
    )
    setattr(coordinator_cls, "__init__", hcu_init)
    setattr(mamba_cls, "cache_blocks", cache_mamba_eagle_replay_boundary)
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
