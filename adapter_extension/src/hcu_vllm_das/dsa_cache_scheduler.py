# SPDX-License-Identifier: Apache-2.0
"""Native request lifetimes with an independently bounded pool per KV group."""
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_coordinator import (
    HybridKVCacheCoordinator, KVCacheCoordinatorNoPrefixCache,
)
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.sched.async_scheduler import AsyncScheduler

from .dsa_cache_pools import IndependentKVCacheConfig


class PoolCollection:
    """Only the aggregate operations consumed by the native KVCacheManager.

    IDs are pool-local. Deferred frees carry their originating pool. The Core
    error API supplies unqualified IDs, so invalidation conservatively evicts
    that ID in every pool, without freeing or modifying any live page.
    """
    def __init__(self, pools, pitches):
        self.pools = tuple(pools)
        self.pitches = tuple(pitches)
        self.num_gpu_blocks = sum(p.num_gpu_blocks for p in pools)

    def get_num_free_blocks(self):
        return sum(p.get_num_free_blocks() for p in self.pools)

    def get_usage(self):
        total = sum((p.num_gpu_blocks - 1) * size
                    for p, size in zip(self.pools, self.pitches))
        free = sum(p.get_num_free_blocks() * size
                   for p, size in zip(self.pools, self.pitches))
        return 1 - free / total if total else 0.0

    def free_blocks(self, blocks):
        by_pool = {}
        for block in blocks:
            if block.pool not in self.pools:
                raise ValueError('Cannot free a block from a foreign KV pool')
            by_pool.setdefault(block.pool, []).append(block)
        for pool, selected in by_pool.items():
            pool.free_blocks(selected)

    def evict_blocks(self, block_ids):
        for pool in self.pools:
            pool.evict_blocks({i for i in block_ids if 0 <= i < pool.num_gpu_blocks})

    def take_events(self):
        return [event for pool in self.pools for event in pool.take_events()]


class IndependentAdmission:
    def get_num_blocks_to_allocate(
        self, request_id, num_tokens, new_computed_blocks, num_encoder_tokens,
        total_computed_tokens, num_local_computed_tokens, num_tokens_main_model,
        apply_admission_cap=False, prefill_end=0,
    ):
        # All supported DSA groups use the native attention/state signatures.
        # Preflight *every* pool before Core can mutate any request allocation.
        if self.retention_interval != 0:
            prefill_end = 0
        total = 0
        for i, manager in enumerate(self.single_type_managers):
            need = manager.get_num_blocks_to_allocate(
                request_id, num_tokens, new_computed_blocks[i],
                total_computed_tokens, num_local_computed_tokens,
                num_tokens_main_model, apply_admission_cap=apply_admission_cap,
                prefill_end=prefill_end)
            if need > manager.block_pool.get_num_free_blocks():
                return self.block_pool.get_num_free_blocks() + 1
            total += need
        return total


class IndependentNoPrefix(IndependentAdmission, KVCacheCoordinatorNoPrefixCache):
    pass


class IndependentHybrid(IndependentAdmission, HybridKVCacheCoordinator):
    pass


def install_independent_pools(manager, config):
    if not isinstance(config, IndependentKVCacheConfig):
        raise TypeError('Independent scheduler requires an independent KV plan')
    previous = manager.coordinator
    classes = {KVCacheCoordinatorNoPrefixCache: IndependentNoPrefix,
               HybridKVCacheCoordinator: IndependentHybrid}
    if type(previous) not in classes:
        raise ValueError('Unsupported independent-pool coordinator')
    if len(previous.single_type_managers) != len(config.pool_blocks):
        raise ValueError('Independent pool/group count mismatch')
    if any(m.req_to_blocks for m in previous.single_type_managers):
        raise ValueError('Cannot replace KV pools after request admission')
    old = previous.block_pool
    pools = [BlockPool(n, old.enable_caching, old.hash_block_size,
                       old.enable_kv_cache_events, old.metrics_collector,
                       old.num_hash_lookahead_tokens, old.medium)
             for n in config.pool_blocks]
    coordinator = object.__new__(classes[type(previous)])
    coordinator.__dict__.update(previous.__dict__)
    for single, pool in zip(coordinator.single_type_managers, pools):
        single.block_pool = pool
        single._null_block = pool.null_block
    coordinator.block_pool = PoolCollection(pools, config.pool_pitches)
    if isinstance(coordinator, HybridKVCacheCoordinator):
        coordinator.verify_and_split_kv_cache_groups()
    manager.coordinator = coordinator
    manager.block_pool = coordinator.block_pool


class IndependentScheduler(Scheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        install_independent_pools(self.kv_cache_manager, self.kv_cache_config)


class IndependentAsyncScheduler(AsyncScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        install_independent_pools(self.kv_cache_manager, self.kv_cache_config)
