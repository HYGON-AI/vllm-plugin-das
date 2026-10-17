# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Zero newly allocated KV blocks only inside their owning cache group.

Mixed-precision KV caches (DeepSeek V4) make Model Runner V2 zero every new
block before use.  The upstream ``KVBlockZeroer`` clears a block id in the
segments of *every* cache group, although a block id belongs to exactly one
group and only that group's layers read or write it until it is freed.  With
V4's block-outermost packing this rewrites ~1.8 MB per block for 13k blocks a
32k-token step (mostly 4/8-token compressor-state blocks), ~59 ms on BW1100.

This adapter builds one upstream ``KVBlockZeroer`` per cache group (same
segment rules, restricted to that group's layers) and zeroes each new block
with its owner's zeroer.  Ownership comes from the per-group block ids in the
same scheduler output; any id it cannot attribute keeps the upstream all-group
zeroing.  Zeroing still happens before the original ``update_requests`` runs,
so it precedes copy-on-write block copies and the forward pass.
"""

from __future__ import annotations

import functools
from collections import defaultdict
from types import ModuleType

from ._common import (
    already_applied,
    load_exact_module,
    require_callable,
    require_class,
    require_exact_signature,
)

TARGET_MODULE = "vllm.v1.worker.gpu.model_runner"
PATCH_ID = "worker.framework_opt.kv_zero.by_group"
TARGETS = (
    f"{TARGET_MODULE}.GPUModelRunner._init_kv_zero_meta",
    f"{TARGET_MODULE}.GPUModelRunner.update_requests",
)
_MARKER = "_vllm_hcu_kv_zero_by_group_applied"
_WRAPPER = "_vllm_hcu_kv_zero_by_group_wrapper"
_ZEROERS = "_vllm_hcu_group_kv_block_zeroers"


def _enabled() -> bool:
    from vllm_hcu.platforms import envs as henvs

    return henvs.VLLM_HCU_KV_ZERO_BY_GROUP


def build_group_zeroers(zeroer_cls, device, attn_groups, kernel_block_sizes,
                        static_forward_context, num_blocks) -> dict[int, object]:
    """One upstream zeroer per KV cache group that owns zeroable segments."""
    by_group = defaultdict(list)
    for group in attn_groups:
        by_group[group.kv_cache_group_id].append(group)
    zeroers = {}
    for group_id, groups in by_group.items():
        zeroer = zeroer_cls(device, groups, kernel_block_sizes,
                            static_forward_context, num_blocks)
        if zeroer._meta is not None:
            zeroers[group_id] = zeroer
    return zeroers


def block_owners(scheduler_output) -> dict[int, int]:
    """Map this step's newly scheduled block ids to their cache group."""
    owners: dict[int, int] = {}

    def record(groups) -> None:
        if groups is not None:
            for group_id, block_ids in enumerate(groups):
                owners.update(dict.fromkeys(block_ids, group_id))

    for request in scheduler_output.scheduled_new_reqs:
        record(request.block_ids)
    for groups in scheduler_output.scheduled_cached_reqs.new_block_ids:
        record(groups)
    return owners


def zero_by_group(fallback, zeroers: dict[int, object], block_ids, owners) -> None:
    buckets: dict[int, list[int]] = defaultdict(list)
    unowned = []
    for block_id in block_ids:
        group_id = owners.get(block_id)
        if group_id in zeroers:
            buckets[group_id].append(block_id)
        else:
            unowned.append(block_id)
    for group_id, ids in buckets.items():
        zeroers[group_id].zero_block_ids(ids)
    if unowned:
        fallback.zero_block_ids(unowned)


def apply_to_module(module: ModuleType) -> bool:
    model_runner = load_exact_module(TARGET_MODULE, module)
    runner_cls = require_class(
        model_runner, "GPUModelRunner", f"{TARGET_MODULE}.GPUModelRunner"
    )
    wrapped = (
        (runner_cls, "_init_kv_zero_meta", TARGETS[0], _WRAPPER),
        (runner_cls, "update_requests", TARGETS[1], _WRAPPER),
    )
    if already_applied(model_runner, _MARKER, wrapped):
        return False

    original_init = require_callable(runner_cls, "_init_kv_zero_meta", TARGETS[0])
    original_update = require_callable(runner_cls, "update_requests", TARGETS[1])
    require_exact_signature(original_init, TARGETS[0], positional=("self",))
    require_exact_signature(
        original_update, TARGETS[1], positional=("self", "scheduler_output")
    )

    @functools.wraps(original_init)
    def init_kv_zero_meta(self):
        original_init(self)
        zeroer = getattr(self, "kv_block_zeroer", None)
        zeroers = {}
        if zeroer is not None and zeroer._meta is not None:
            zeroers = build_group_zeroers(
                type(zeroer), self.device,
                [group for groups in self.attn_groups for group in groups],
                self.kernel_block_sizes,
                self.compilation_config.static_forward_context,
                self.kv_cache_config.num_blocks,
            )
        setattr(self, _ZEROERS, zeroers)

    @functools.wraps(original_update)
    def update_requests(self, scheduler_output):
        block_ids = scheduler_output.new_block_ids_to_zero
        zeroers = getattr(self, _ZEROERS, None)
        if not block_ids or not zeroers or not _enabled():
            return original_update(self, scheduler_output)
        zero_by_group(self.kv_block_zeroer, zeroers, block_ids,
                      block_owners(scheduler_output))
        scheduler_output.new_block_ids_to_zero = None
        try:
            return original_update(self, scheduler_output)
        finally:
            scheduler_output.new_block_ids_to_zero = block_ids

    for function in (init_kv_zero_meta, update_requests):
        setattr(function, _WRAPPER, True)
    runner_cls._init_kv_zero_meta = init_kv_zero_meta
    runner_cls.update_requests = update_requests
    setattr(model_runner, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
