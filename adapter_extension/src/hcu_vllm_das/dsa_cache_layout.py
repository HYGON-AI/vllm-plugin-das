# SPDX-License-Identifier: Apache-2.0
"""Native KV groups with rank-local, owner-plus-scratch physical placement."""
from __future__ import annotations

from dataclasses import asdict, replace
import json
from math import lcm
import re

from vllm.v1.core.kv_cache_planning import (
    DefaultKVCacheConfigBuilder,
    _get_layer_block_stride,
)
from vllm.v1.kv_cache_interface import (
    KVCacheConfig, KVCacheTensor, MLAAttentionSpec, SlidingWindowMLASpec, UniformTypeKVCacheSpecs,
    compute_layout_strides,
)

from .dsa_cache_layer_split import (
    kv_source_owner_rank, record_layer_split_state, register_source_block_sizes,
    split_config,
    swa_cache_owner_rank,
    logger,
)


def _specs(group):
    spec = group.kv_cache_spec
    for name in group.layer_names:
        yield name, spec.kv_cache_specs[name] if isinstance(spec, UniformTypeKVCacheSpecs) else spec


def _owner(name, ids, width, *, is_draft=False):
    # Native DSpark drafts use the restored global batch (PCP1). Their short
    # sliding-window caches remain replicated and cannot alias target scratch.
    if is_draft:
        return None
    if name.endswith('.swa_cache'):
        return swa_cache_owner_rank(name, width)
    match = re.search(r'(?:^|\.)layers\.(\d+)\.attn(?:\.indexer\.k_cache)?$', name)
    if match is None:
        return None
    layer = int(match[1])
    if layer not in ids:
        raise ValueError(f'Unexpected DSA source cache {name}')
    return kv_source_owner_rank(layer, ids, width)


def _placements(config, groups):
    """Return every rank's offsets in bytes per block and the common budget.

    Groups overlay because the native block pool assigns a block ID to one
    group at a time. The transport MUST write only live IDs of that group,
    never a block-ID prefix or padded/stale table entries. Each group's
    remote sources share a receive slot after that group's persistent data.
    """
    width, ids = split_config(config)
    if any(group.host_resident for group in groups):
        raise ValueError('DSA layer split does not support host KV groups')
    specs = [(name, spec) for group in groups for name, spec in _specs(group)]
    # Every per-block offset must be legal for every represented dtype.
    alignment = lcm(*(spec.dtype.itemsize for _, spec in specs)) if specs else 1
    pitches = {name: _get_layer_block_stride(spec) for name, spec in specs}
    for name, spec in specs:
        if not spec.has_layer_views:
            raise ValueError(f'DSA layer split needs per-layer views for {name}')
        if _owner(name, ids, width, is_draft=spec.is_draft) is not None and not isinstance(spec, (MLAAttentionSpec, SlidingWindowMLASpec)):
            raise ValueError(f'DSA source must have an MLA cache spec: {name}')
        pitches[name] = (pitches[name] + alignment - 1) // alignment * alignment
    ranks = []
    pool_pitch = 0
    for rank in range(width):
        offsets = {}
        for group in groups:
            cursor = 0
            remote = []
            for name, spec in _specs(group):
                owner = _owner(name, ids, width, is_draft=spec.is_draft)
                if owner is not None and owner != rank:
                    remote.append(name)
                else:
                    offsets[name] = cursor
                    cursor += pitches[name]
            for name in remote:
                offsets[name] = cursor
            receive_pitch = max((pitches[name] for name in remote), default=0)
            pool_pitch = max(pool_pitch, cursor + receive_pitch)
        ranks.append(offsets)
    block_alignment = lcm(alignment, *(spec.get_block_stride_alignment() for _, spec in specs))
    pool_pitch = (pool_pitch + block_alignment - 1) // block_alignment * block_alignment
    return ranks, pitches, pool_pitch


def _materialize(config, groups, num_blocks, rank):
    offsets, pitches, pool_pitch = _placements(config, groups)
    layout = config.cache_config.get_resolved_kv_cache_layout()
    # Every block ID must address the same physical block across groups.
    # Keep block-outer packing even for a single-layer view. A trailing wire
    # region supplies the live-ID bitmap, ID staging and dense broadcast chunks;
    # all three are budgeted linearly, including for a one-block pool.
    # Bitmap + aligned int64 IDs need at most 9*N + 7 bytes. Reserve
    # 256*N in total, leaving a positive bounded payload even for N=1.
    # The runtime splits rows and columns to fit this actual wire capacity.
    # Reserving 4096*N made the wire larger than the physical cache savings
    # when the real model's replicated SWA groups set the common block stride.
    wire_bytes_per_block = 256
    tensors = []
    for group in groups:
        for name, spec in _specs(group):
            layer_stride, block_stride, _, _, _ = compute_layout_strides(
                spec, num_blocks, 1, layout,
                fixed_strides=(pitches[name], pool_pitch, None, None, None),
            )
            tensors.append(KVCacheTensor(
                size=(pool_pitch + wire_bytes_per_block) * num_blocks, layers=[name],
                layer_stride=layer_stride, block_stride=block_stride,
                offset=offsets[rank][name],
            ))
    return KVCacheConfig(
        num_blocks=num_blocks, kv_cache_tensors=tensors, kv_cache_groups=groups,
        prefix_cache_retention_interval=config.cache_config.prefix_cache_retention_interval,
    )


class DsaLayerSplitCacheBuilder(DefaultKVCacheConfigBuilder):
    # Keep native groups: every additional group consumes block IDs per request.
    # Source ownership changes only the physical placement within those groups.
    def get_kv_cache_config_from_groups(self, vllm_config, kv_cache_groups, num_blocks):
        if split_config(vllm_config) is None or not kv_cache_groups:
            return super().get_kv_cache_config_from_groups(vllm_config, kv_cache_groups, num_blocks)
        # The engine has no PCP rank. The common worst-rank budget is exact
        # for every placement. Workers rematerialize offsets before allocation.
        return _materialize(vllm_config, kv_cache_groups, num_blocks, rank=0)


def builder_for_config(config):
    if split_config(config) is None:
        transfer = config.kv_transfer_config
        if (transfer is not None and transfer.kv_role == 'kv_consumer'
                and transfer.kv_connector == 'HcuLayerSplitMooncakeConnector'):
            from .mooncake_layer_split import draft_context_enabled, validate_config
            if draft_context_enabled(config):
                validate_config(config)
                return 'hcu_vllm_das.dsa_cache_pools.DSparkConsumerCacheBuilder'
        return None
    from .dsa_cache_pools import select_scheduler
    select_scheduler(config)
    return 'hcu_vllm_das.dsa_cache_pools.IndependentDSACacheBuilder'


def localize_kv_cache(config, cache_config):
    state = record_layer_split_state(config)
    if state is None:
        return cache_config
    width, ids, _ = state
    if state[2] == 0:
        logger.info('DSA layer split cache groups %s', json.dumps([
            {'layers': {name: {'kind': type(spec).__name__, 'spec': asdict(spec)}
                        for name, spec in _specs(group)}}
            for group in cache_config.kv_cache_groups
        ], default=str))
    register_source_block_sizes({
        name: spec.block_size
        for group in cache_config.kv_cache_groups for name, spec in _specs(group)
        if _owner(name, ids, width, is_draft=spec.is_draft) is not None
    })
    owned_swa = [name for group in cache_config.kv_cache_groups
                 for name, spec in _specs(group)
                 if name.endswith('.swa_cache') and _owner(name, ids, width, is_draft=spec.is_draft) == state[2]]
    logger.info('DSA layer split PCP rank %d/%d owns SWA layers %s',
                state[2], width, owned_swa)
    from .dsa_cache_pools import IndependentKVCacheConfig, localize
    if isinstance(cache_config, IndependentKVCacheConfig):
        local = localize(config, cache_config, state[2])
        logger.info('DSA independent KV pools rank %d: pages=%s pitches_bytes=%s '
                    'pool_offsets=%s wire_bytes=%d total_bytes=%d', state[2],
                    local.pool_blocks, local.pool_pitches, local.pool_offsets,
                    local.wire_bytes, local.kv_cache_tensors[0].size)
        return local
    from .dsa_cache_layer_split import register_transport_regions
    register_transport_regions({})
    local = _materialize(config, cache_config.kv_cache_groups, cache_config.num_blocks, state[2])
    # Retain Core-owned metadata resolved after planning, including transfer
    # roles, hash alignment, layout, and shared-pool block counts.
    return replace(cache_config, kv_cache_tensors=local.kv_cache_tensors)
