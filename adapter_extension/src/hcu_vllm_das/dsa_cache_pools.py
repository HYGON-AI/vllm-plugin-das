# SPDX-License-Identifier: Apache-2.0
"""Separately sized, owner-local KV pools in one native backing allocation.

``num_blocks`` is an admission accounting unit, not a tensor's page count.
Each pool has its own native block IDs and an explicitly sized byte interval.
One reservation bundle holds one maximum-length request in every pool. This
keeps Core's group-aware capacity formula exact without changing cache specs,
window lifetimes, or its cross-worker minimum-capacity convergence.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import lcm

from vllm.v1.core.kv_cache_planning import DefaultKVCacheConfigBuilder, _get_layer_block_stride
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec, KVCacheConfig, KVCacheTensor, MLAAttentionSpec,
    SlidingWindowMLASpec, UniformTypeKVCacheSpecs, compute_layout_strides,
)

from .dsa_cache_layout import _owner, _specs
from .dsa_cache_layer_split import split_config

WIRE_BYTES_PER_UNIT = 256
BACKING_ALIGNMENT = 4096  # native ROCm allocator/RDMA registration granularity


@dataclass
class IndependentKVCacheConfig(KVCacheConfig):
    pool_blocks: tuple[int, ...] = ()
    pool_pitches: tuple[int, ...] = ()
    pool_demands: tuple[int, ...] = ()
    pool_offsets: tuple[int, ...] = ()
    wire_offset: int = 0
    wire_bytes: int = 0
    layer_pool: dict[str, int] = field(default_factory=dict)

    @property
    def needs_kv_cache_zeroing(self):
        # Core zeroes mixed precision because a shared block can change dtype
        # when it moves between groups. Here IDs never move between pools.
        # Every persistent layer keeps its own fixed offset and storage dtype;
        # remote scratch is overwritten by a complete page before each read.
        # Main/indexer quantization tags differ, but their persistent regions
        # never exchange bytes. There is no Mamba read-before-write state.
        # Do not invoke Core's global-ID zeroer on pool-local IDs.
        for group in self.kv_cache_groups:
            precisions = {s.dtype for _, s in _specs(group)}
            if len(precisions) != 1:
                raise ValueError('Independent KV pool contains mixed precisions')
        if self.has_mamba_layers:
            raise ValueError('Independent DSA pools do not support Mamba zeroing')
        return False

    def num_blocks_of(self, tensor):
        if tensor.host_resident:
            raise ValueError('Independent DSA pools cannot contain host tensors')
        return self.pool_blocks[self.layer_pool[tensor.layers[0]]]


def merge_swa_groups(groups):
    """Merge only identical SWA semantics; never group solely by byte size."""
    result = []
    for group in groups:
        specs = dict(_specs(group))
        first = next(iter(specs.values()))
        if not isinstance(first, SlidingWindowMLASpec) or any(s != first for s in specs.values()):
            result.append(group)
            continue
        for i, previous in enumerate(result):
            previous_specs = dict(_specs(previous))
            if (all(s == first for s in previous_specs.values())
                    and (group.is_eagle_group, group.host_resident, group.enable_kv_transfer, group.role)
                    == (previous.is_eagle_group, previous.host_resident, previous.enable_kv_transfer, previous.role)):
                combined = previous_specs | specs
                result[i] = replace(previous, layer_names=list(combined),
                    kv_cache_spec=UniformTypeKVCacheSpecs.from_specs(combined))
                break
        else:
            result.append(group)
    return result


def pool_geometry(config, groups):
    selected = split_config(config)
    if selected is None:
        raise ValueError('Independent DSA pools require Layer Split')
    if getattr(getattr(config, 'kernel_config', None), 'enable_jit_warmup', False):
        raise ValueError('Independent DSA pools require the resolved eager policy: JIT warmup disabled')
    width, ids = selected
    all_specs = [(name, spec) for g in groups for name, spec in _specs(g)]
    if any(g.host_resident for g in groups) or not all_specs:
        raise ValueError('Independent DSA pools require nonempty device-only groups')
    if any(not isinstance(s, (MLAAttentionSpec, SlidingWindowMLASpec, CircularBufferSpec))
           or not s.has_layer_views for _, s in all_specs):
        raise ValueError('Unsupported cache spec for independent DSA pools')
    alignment = lcm(8, *(s.dtype.itemsize for _, s in all_specs),
                    *(s.get_block_stride_alignment() for _, s in all_specs))
    pitches = {n: _get_layer_block_stride(s) for n, s in all_specs}
    offsets = [{} for _ in range(width)]
    costs, demands = [], []
    for group in groups:
        if len({s.dtype for _, s in _specs(group)}) != 1:
            raise ValueError('Independent KV pool contains mixed precisions')
        cost = 0
        for rank in range(width):
            cursor, remote = 0, []
            for name, spec in _specs(group):
                owner = _owner(name, ids, width, is_draft=spec.is_draft)
                if owner is not None and owner != rank:
                    remote.append(name)
                else:
                    offsets[rank][name] = cursor
                    cursor += pitches[name]
            for name in remote:
                offsets[rank][name] = cursor
            cost = max(cost, cursor + max((pitches[n] for n in remote), default=0))
        costs.append((cost + alignment - 1) // alignment * alignment)
        spec = group.kv_cache_spec
        demand_bytes = spec.max_memory_usage_bytes(config)
        demands.append((demand_bytes + spec.page_size_bytes - 1) // spec.page_size_bytes)
    if min(demands) <= 0:
        raise ValueError('Invalid cache admission demand')
    return offsets, pitches, tuple(costs), tuple(demands)


def unit_cost(config, groups):
    _, _, costs, demands = pool_geometry(config, groups)
    # Include one null page per pool in *every* bundle's budget. Only one set
    # is allocated, so this is conservative, linear, and replan-idempotent.
    bundle_bytes = sum(c * (d + 1) for c, d in zip(costs, demands)) + BACKING_ALIGNMENT - 1
    units = sum(demands)
    return (bundle_bytes + units - 1) // units + WIRE_BYTES_PER_UNIT


def materialize(config, groups, num_blocks, rank):
    offsets, pitches, costs, demands = pool_geometry(config, groups)
    units = sum(demands)
    bundles = num_blocks // units
    if bundles < 1:
        raise ValueError('Independent DSA pools cannot hold one maximum-length request')
    num_blocks = bundles * units
    counts = tuple(bundles * d + 1 for d in demands)  # a distinct null page per pool
    starts, cursor = [], 0
    for count, cost in zip(counts, costs):
        starts.append(cursor)
        cursor += count * cost
    wire_offset = cursor
    wire_bytes = WIRE_BYTES_PER_UNIT * num_blocks
    total = (cursor + wire_bytes + BACKING_ALIGNMENT - 1) // BACKING_ALIGNMENT * BACKING_ALIGNMENT
    assert total <= num_blocks * unit_cost(config, groups)
    tensors, layer_pool = [], {}
    layout = config.cache_config.get_resolved_kv_cache_layout()
    for pool, group in enumerate(groups):
        for name, spec in _specs(group):
            layer_stride, block_stride, _, _, _ = compute_layout_strides(
                spec, counts[pool], 1, layout,
                fixed_strides=(pitches[name], costs[pool], None, None, None))
            tensors.append(KVCacheTensor(size=total, layers=[name],
                layer_stride=layer_stride, block_stride=block_stride,
                offset=starts[pool] + offsets[rank][name]))
            layer_pool[name] = pool
    return IndependentKVCacheConfig(num_blocks=num_blocks, kv_cache_tensors=tensors,
        kv_cache_groups=groups, prefix_cache_retention_interval=config.cache_config.prefix_cache_retention_interval,
        pool_blocks=counts, pool_pitches=costs, pool_demands=demands,
        pool_offsets=tuple(starts), wire_offset=wire_offset, wire_bytes=wire_bytes,
        layer_pool=layer_pool)


class IndependentDSACacheBuilder(DefaultKVCacheConfigBuilder):
    def get_kv_cache_groups(self, vllm_config, kv_cache_spec):
        draft_names = {name for name, spec in kv_cache_spec.items() if spec.is_draft}
        groups = super().get_kv_cache_groups(vllm_config, kv_cache_spec)
        if split_config(vllm_config) is None:
            return groups
        if draft_names:
            # Native equality ignores is_draft. Separate producers before
            # merging SWA pools so replicated drafts cannot alias target scratch.
            separated = []
            for group in groups:
                for is_draft in (False, True):
                    specs = {name: replace(spec, is_draft=is_draft)
                        for name, spec in _specs(group) if (name in draft_names) == is_draft}
                    if not specs:
                        continue
                    spec = (UniformTypeKVCacheSpecs.from_specs(specs)
                        if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
                        else next(iter(specs.values())))
                    separated.append(replace(group, layer_names=list(specs),
                        kv_cache_spec=spec, is_eagle_group=is_draft))
            groups = separated
        return merge_swa_groups(groups)

    def _get_pool_bytes_per_block(self, vllm_config, kv_cache_groups):
        return unit_cost(vllm_config, kv_cache_groups)

    def get_kv_cache_config_from_groups(self, vllm_config, kv_cache_groups, num_blocks):
        return materialize(vllm_config, kv_cache_groups, num_blocks, rank=0)


def select_scheduler(config):
    scheduler = config.scheduler_config
    name = ('IndependentAsyncScheduler' if getattr(scheduler, 'async_scheduling', False)
            else 'IndependentScheduler')
    selected = 'hcu_vllm_das.dsa_cache_scheduler.' + name
    current = getattr(scheduler, 'scheduler_cls', None)
    if current not in (None, selected):
        raise ValueError('Independent DSA pools cannot replace a foreign scheduler')
    scheduler.scheduler_cls = selected


def localize(config, cache_config, rank):
    from .dsa_cache_layer_split import register_transport_regions
    local = materialize(config, cache_config.kv_cache_groups, cache_config.num_blocks, rank)
    if (local.pool_blocks, local.pool_offsets) != (cache_config.pool_blocks, cache_config.pool_offsets):
        raise ValueError('Worker independent-pool geometry differs from engine plan')
    register_transport_regions({n: (local.wire_offset, local.wire_bytes,
        local.pool_blocks[p]) for n, p in local.layer_pool.items()})
    return replace(cache_config, kv_cache_tensors=local.kv_cache_tensors)
