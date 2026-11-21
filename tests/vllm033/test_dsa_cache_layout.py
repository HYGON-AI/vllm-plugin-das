"""CPU regressions through native v0.33 planning, allocation and byte views."""
import copy
import logging
from dataclasses import replace
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest
import torch

from hcu_vllm_das import dsa_cache_layer_split as runtime
from hcu_vllm_das import dsa_cache_layout as placement
from vllm.v1.core.kv_cache_planning import (
    DefaultKVCacheConfigBuilder, _pool_bytes_per_block_from_builder,
)
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec, KVCacheGroupSpec, MLAAttentionSpec,
    SlidingWindowMLASpec, UniformTypeKVCacheSpecs,
)
from vllm.v1.kv_cache_layout import KVCacheLayout
from vllm.v1.attention.backend import MultipleOf
from vllm.v1.worker import utils


def config(width=8, layout='BLHNC'):
    return NS(
        parallel_config=NS(prefill_context_parallel_size=width, tensor_parallel_size=1,
                           pipeline_parallel_size=1, data_parallel_size=1,
                           decode_context_parallel_size=1),
        model_config=NS(enforce_eager=True, supports_kv_cache_grouping=False,
                        hf_config=NS(kv_source_layer_ids=[2, 8, 14, 20])),
        speculative_config=None, kv_transfer_config=None,
        scheduler_config=NS(disable_hybrid_kv_cache_manager=False),
        cache_config=NS(prefix_cache_retention_interval=None, num_gpu_blocks_override=None,
                        disable_hybrid_kv_cache_manager=False,
                        get_resolved_kv_cache_layout=lambda: KVCacheLayout[layout]),
        attention_config=NS(hisparse_config=None),
    )


def groups():
    specs = {}
    for i in (2, 8, 14, 20):
        for suffix, head in (('', 512), ('.indexer.k_cache', 132)):
            ratio = 1 if i == 20 else 2
            specs[f'model.layers.{i}.attn{suffix}'] = MLAAttentionSpec(
                block_size=64, num_kv_heads=1, head_size=head, dtype=torch.uint8,
                tokens_per_state=ratio, alignment=576,
            )
    uniform = UniformTypeKVCacheSpecs.from_specs(specs)
    assert uniform is not None
    return [KVCacheGroupSpec(list(specs), uniform)]


@pytest.fixture(autouse=True)
def source_block_sizes(monkeypatch):
    monkeypatch.setattr(runtime, '_source_block_sizes', {
        name: spec.block_size for name, spec in placement._specs(groups()[0])
    }, raising=False)


def root_metadata(table, lengths, *, count=None):
    return NS(block_table=torch.tensor(table, dtype=torch.int64),
              seq_lens=torch.tensor(lengths, dtype=torch.int32),
              num_decodes=0, num_prefills=len(lengths) if count is None else count)


@pytest.mark.parametrize('main', [False, True])
def test_native_zero_query_pcp_placeholder_has_no_local_blocks(main):
    from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerMetadata
    prefix = 'model.layers.2.attn'
    indexer = DeepseekV32IndexerMetadata(
        seq_lens=torch.tensor([65], dtype=torch.int32), max_seq_len=65,
        slot_mapping=torch.tensor([-1], dtype=torch.int64),
        num_decodes=0, num_decode_tokens=0, num_prefills=0, num_prefill_tokens=0,
    )
    assert indexer.block_table is None
    metadata = {prefix + '.indexer.k_cache': indexer}
    if main:
        metadata[prefix] = NS(num_reqs=1, block_table=torch.tensor([[5, 6]]))
    else:
        prefix += '.indexer.k_cache'
    assert runtime._live_block_ids(metadata, prefix, 7) == []


def test_real_fp8_groups_gain_capacity_including_transport_budget():
    import json
    from pathlib import Path
    from vllm.v1 import kv_cache_interface as interface
    from vllm.v1.core.kv_cache_utils import get_kv_cache_capacity

    fixture = Path(__file__).with_name('fixtures') / 'dsv41_fp8_native_cache_groups.json'
    real_groups = []
    for group in json.loads(fixture.read_text()):
        specs = {}
        for name, record in group['layers'].items():
            kwargs = dict(record['spec'])
            kwargs['dtype'] = getattr(torch, kwargs['dtype'].split('.')[-1])
            kwargs['kv_quant_mode'] = interface.KVQuantMode(kwargs['kv_quant_mode'])
            if isinstance(kwargs['block_stride_alignment'], dict):
                kwargs['block_stride_alignment'] = MultipleOf(**kwargs['block_stride_alignment'])
            specs[name] = getattr(interface, record['kind'])(**kwargs)
        uniform = UniformTypeKVCacheSpecs.from_specs(specs)
        assert uniform is not None
        real_groups.append(KVCacheGroupSpec(list(specs), uniform))

    cfg = config()
    cfg.model_config.max_model_len = 16384
    cfg.model_config.original_max_model_len = 65536
    cfg.num_prefill_lookahead_tokens = 0
    # The serving default permits two concurrent batches.
    cfg.max_in_flight_tokens = 2 * 8192
    budget = 10 * 1024**3
    native = DefaultKVCacheConfigBuilder()
    with patch.object(cfg, 'additional_config', {'hcu_layer_split': {'enabled': True}}, create=True):
        split = placement.DsaLayerSplitCacheBuilder()
        native_cost = _pool_bytes_per_block_from_builder(native, cfg, real_groups)
        split_cost = _pool_bytes_per_block_from_builder(split, cfg, real_groups)
        assert native_cost == 116928  # Bound to the recorded FP8 layout.
        assert split_cost < native_cost
        native_plan = native.get_kv_cache_config_from_groups(cfg, real_groups, budget // native_cost)
        split_plan = split.get_kv_cache_config_from_groups(cfg, real_groups, budget // split_cost)
        assert split_plan.kv_cache_groups is real_groups
        assert get_kv_cache_capacity(cfg, split_plan)[0] > get_kv_cache_capacity(cfg, native_plan)[0]
        assert max(t.size for t in split_plan.kv_cache_tensors) <= budget


@pytest.mark.parametrize('width', [2, 4, 8, 16])
@pytest.mark.parametrize('layout', ['BLHNC', 'BLNHC', 'LBHNC', 'LBNHC'])
def test_rank_placement_preserves_groups_and_owner_data(width, layout):
    cfg, original = config(width, layout), groups()
    with patch.object(cfg, 'additional_config', {'hcu_layer_split': {'enabled': True}}, create=True):
        builder = placement.DsaLayerSplitCacheBuilder()
        planned = builder.get_kv_cache_config_from_groups(cfg, original, 7)
        budget = _pool_bytes_per_block_from_builder(builder, cfg, original)
        assert {t.size for t in planned.kv_cache_tensors} == {budget * 7}
        assert planned.kv_cache_groups is original
        # Native allocator must find a group for every tensor, even on ranks 4-7.
        for rank in range(width):
            with patch.object(runtime, 'get_pcp_group', return_value=NS(rank_in_group=rank)):
                local = placement.localize_kv_cache(cfg, planned)
            assert planned.kv_cache_tensors[0].offset == 0  # caller not mutated
            assert local.kv_cache_groups is original
            assert {t.size for t in local.kv_cache_tensors} == {budget * 7}
            with patch.object(utils.current_platform, 'is_rocm', return_value=False):
                views = utils.allocate_kv_cache(local, torch.device('cpu'), KVCacheLayout[layout])
            assert set(views) == set(original[0].layer_names)
            spans = {}
            for tensor in local.kv_cache_tensors:
                name = tensor.layers[0]
                pitch = placement._placements(cfg, original)[1][name]
                spans[name] = (tensor.offset, tensor.offset + pitch)
                assert spans[name][1] + 6 * tensor.block_stride <= tensor.size
                # Synthetic independent content models writes across all layers.
                if placement._owner(name, (2, 8, 14, 20), width) == rank:
                    views[name].fill_(17 + original[0].layer_names.index(name))
            for name, view in views.items():
                if placement._owner(name, (2, 8, 14, 20), width) != rank:
                    view.fill_(231)
            for name, view in views.items():
                owner = placement._owner(name, (2, 8, 14, 20), width)
                if owner == rank:
                    assert torch.all(view == 17 + original[0].layer_names.index(name))
                    for other, (start, end) in spans.items():
                        if name != other:
                            assert spans[name][1] <= start or end <= spans[name][0]


@pytest.mark.parametrize('alignment', [
    None, 256, MultipleOf(16), MultipleOf(64), MultipleOf(128),
])
@pytest.mark.parametrize('layout', ['BLHNC', 'BLNHC'])
def test_block_stride_resolves_native_alignment_constraints(alignment, layout):
    cfg = config(layout=layout)
    original = groups()
    specs = {
        name: replace(spec, block_stride_alignment=alignment)
        for name, spec in placement._specs(original[0])
    }
    uniform = UniformTypeKVCacheSpecs.from_specs(specs)
    assert uniform is not None
    original = [KVCacheGroupSpec(list(specs), uniform)]
    with patch.object(cfg, 'additional_config', {'hcu_layer_split': {'enabled': True}}, create=True):
        budget = _pool_bytes_per_block_from_builder(
            placement.DsaLayerSplitCacheBuilder(), cfg, original)
        for rank in (0, 7):
            plan = placement._materialize(cfg, original, 7, rank)
            with patch.object(utils.current_platform, 'is_rocm', return_value=False):
                views = utils.allocate_kv_cache(
                    plan, torch.device('cpu'), KVCacheLayout[layout])
            assert set(views) == set(specs)
            for tensor in plan.kv_cache_tensors:
                spec = specs[tensor.layers[0]]
                assert tensor.block_stride % spec.get_block_stride_alignment() == 0
                assert tensor.size == budget * 7


def test_ownership_logging_uses_vllm_logging_configuration(monkeypatch, caplog):
    monkeypatch.setattr(runtime, '_state', None)
    monkeypatch.setattr(runtime, 'get_pcp_group', lambda: NS(rank_in_group=7))
    cfg = config()
    with patch.object(cfg, 'additional_config', {'hcu_layer_split': {'enabled': True}}, create=True), \
         caplog.at_level(logging.INFO, logger='vllm'):
        runtime.record_layer_split_state(cfg)
    assert 'DSA layer split PCP rank 7/8 owns KV sources []' in caplog.text


def test_disabled_uses_native_placement_and_policy_guard():
    cfg, original = config(), groups()
    with patch.object(cfg, 'additional_config', {}, create=True):
        got = placement.DsaLayerSplitCacheBuilder().get_kv_cache_config_from_groups(cfg, original, 3)
        expected = DefaultKVCacheConfigBuilder().get_kv_cache_config_from_groups(cfg, original, 3)
        assert got == expected
        assert placement.localize_kv_cache(cfg, got) is got
    with patch.object(cfg, 'additional_config', {'hcu_layer_split': {'enabled': True}}, create=True):
        cfg.kv_transfer_config = object()
        with pytest.raises(ValueError, match='standalone'):
            placement.builder_for_config(cfg)


def test_smaller_budget_counts_scratch_and_converges_linearly():
    cfg, original = config(), groups()
    with patch.object(cfg, 'additional_config', {'hcu_layer_split': {'enabled': True}}, create=True):
        builder = placement.DsaLayerSplitCacheBuilder()
        one = _pool_bytes_per_block_from_builder(builder, cfg, original)
        native = _pool_bytes_per_block_from_builder(DefaultKVCacheConfigBuilder(), cfg, original)
        assert 0 < one < native
        for blocks in (1, 13, 100003):
            plan = builder.get_kv_cache_config_from_groups(cfg, original, blocks)
            assert {t.size for t in plan.kv_cache_tensors} == {one * blocks}
        assert cfg.cache_config.num_gpu_blocks_override is None


def test_broadcast_writes_bound_scratch_with_padding_and_no_receive_allocation():
    cfg, original = config(), groups()
    with patch.object(cfg, 'additional_config', {'hcu_layer_split': {'enabled': True}}, create=True):
        plan = placement._materialize(cfg, original, 7, rank=7)
    with patch.object(utils.current_platform, 'is_rocm', return_value=False):
        caches = utils.allocate_kv_cache(plan, torch.device('cpu'), KVCacheLayout.BLHNC)
    name = 'model.layers.20.attn.indexer.k_cache'
    cache = caches[name].squeeze(1)
    cache.fill_(0)
    storage = cache.untyped_storage()
    seen = []

    def broadcast(wire, src):
        assert src == 3 and wire.is_contiguous()
        assert wire.untyped_storage().data_ptr() == storage.data_ptr()
        seen.append(wire.numel())
        wire.fill_(123)

    # Another PCP rank needs block 5. Neither holes nor a stale table tail
    # (block 6) may be written, even though its ID is within the pool bounds.
    def all_reduce(bitmap, **kwargs):
        assert bitmap.tolist() == [0, 1, 0, 1, 0, 0, 0]
        bitmap[5] = 1

    metadata = {name: root_metadata([[1, 3, 6]], [128])}
    with patch.object(runtime, 'get_forward_context', return_value=NS(attn_metadata=metadata)), \
         patch.object(runtime, 'get_pcp_group', return_value=NS(rank_in_group=7, device_group=object(), broadcast=broadcast)), \
         patch.object(runtime.dist, 'all_reduce', side_effect=all_reduce), \
         patch.object(torch.cuda, 'is_current_stream_capturing', return_value=False):
        assert runtime.broadcast_kv_source_cache(name, 3, cache) is cache
    assert sum(seen) == 3 * cache.shape[1] * cache.shape[2]
    assert torch.all(cache[[1, 3, 5]] == 123)
    assert torch.all(cache[[0, 2, 4, 6]] == 0)


def test_empty_local_queries_still_participate_in_collectives():
    backing = torch.zeros(4 * (32 * 132 + 4096), dtype=torch.uint8)
    cache = backing[:4 * 32 * 132].reshape(4, 32, 132)
    name = 'model.layers.2.attn.indexer.k_cache'
    context = NS(attn_metadata={name: NS(
        block_table=torch.empty((0, 0), dtype=torch.int64),
        seq_lens=torch.empty(0, dtype=torch.int32), num_decodes=0, num_prefills=0)})
    calls = []

    def remote_max(bitmap, **kwargs):
        calls.append('reduce')
        bitmap[2] = 1

    group = NS(rank_in_group=1, device_group=None, broadcast=lambda wire, src: calls.append(('broadcast', src)))
    with patch.object(runtime, 'get_forward_context', return_value=context), \
         patch.object(runtime, 'get_pcp_group', return_value=group), \
         patch.object(runtime.dist, 'all_reduce', side_effect=remote_max), \
         patch.object(torch.cuda, 'is_current_stream_capturing', return_value=False):
        runtime.broadcast_kv_source_cache(name, 0, cache)
    assert calls == ['reduce', ('broadcast', 0)]


def test_live_ids_use_root_uncompressed_lengths_and_planner_block_size(monkeypatch):
    name = 'model.layers.2.attn.indexer.k_cache'
    monkeypatch.setitem(runtime._source_block_sizes, name, 128)
    meta = root_metadata([[1, 3, 6], [5, 2, 4], [6, 6, 6]], [129, 0, 384], count=2)
    # Neither compressed decode nor PCP-padded prefill lengths are authoritative.
    meta.decode = NS(block_table=torch.tensor([[4, 6]]), seq_lens=torch.tensor([64]))
    meta.prefill = NS(chunks=[NS(block_table=torch.tensor([[2, 6]]))])
    assert runtime._live_block_ids({name: meta}, name, 7) == [1, 3]
    meta.seq_lens[0] = 128
    assert runtime._live_block_ids({name: meta}, name, 7) == [1]


def test_main_source_uses_own_table_with_paired_indexer_lengths(monkeypatch):
    name = 'model.layers.2.attn'
    monkeypatch.setitem(runtime._source_block_sizes, name, 128)
    meta = NS(block_table=torch.tensor([[2, 4, 6]]), num_reqs=1)
    sibling = root_metadata([[1, 3, 5]], [129])
    metadata = {name: meta, name + '.indexer.k_cache': sibling}
    assert runtime._live_block_ids(metadata, name, 7) == [2, 4]
    sibling.num_prefills = 0
    with pytest.raises(RuntimeError, match='paired indexer request count'):
        runtime._live_block_ids(metadata, name, 7)


@pytest.mark.parametrize('table,lengths,count,error', [
    ([[1]], [129], 1, 'shorter'),
    ([[7]], [1], 1, 'out of range'),
    ([[-1]], [1], 1, 'out of range'),
    ([[1]], [-1], 1, 'length invalid'),
    ([[1]], [1], 2, 'count invalid'),
])
def test_bad_live_metadata_is_rejected(table, lengths, count, error):
    name = 'model.layers.2.attn.indexer.k_cache'
    with pytest.raises(RuntimeError, match=error):
        runtime._live_block_ids({name: root_metadata(table, lengths, count=count)}, name, 7)


def test_null_empty_and_padded_entries_are_never_selected():
    name = 'model.layers.2.attn.indexer.k_cache'
    meta = root_metadata([[0, 1, -1, 999], [5, 6, 4, 3]], [128, 0])
    assert runtime._live_block_ids({name: meta}, name, 7) == [1]
    meta.seq_lens.zero_()
    assert runtime._live_block_ids({name: meta}, name, 7) == []


def test_missing_lengths_fails_collectively_without_payload():
    name = 'model.layers.2.attn.indexer.k_cache'
    backing = torch.zeros(4 * (32 * 132 + 4096), dtype=torch.uint8)
    cache = backing[:4 * 32 * 132].reshape(4, 32, 132)
    context = NS(attn_metadata={name: NS(block_table=torch.tensor([[1]]))})
    def reduce(bitmap, **kwargs):
        assert bitmap.tolist() == [1, 0, 0, 0]
    group = NS(rank_in_group=1, device_group=None, broadcast=lambda *a, **k: pytest.fail('payload'))
    with patch.object(runtime, 'get_forward_context', return_value=context), \
         patch.object(runtime, 'get_pcp_group', return_value=group), \
         patch.object(runtime.dist, 'all_reduce', side_effect=reduce), \
         patch.object(torch.cuda, 'is_current_stream_capturing', return_value=False):
        with pytest.raises(RuntimeError, match='invalid live-block metadata') as caught:
            runtime.broadcast_kv_source_cache(name, 0, cache)
    assert 'lengths missing' in str(caught.value.__cause__)
    assert torch.count_nonzero(cache) == 0


def test_tiny_pool_transport_covers_full_physical_page_including_records():
    cfg = config()
    name = 'model.layers.20.attn'
    with patch.object(cfg, 'additional_config', {'hcu_layer_split': {'enabled': True}}, create=True):
        plan = placement._materialize(cfg, groups(), 2, 7)
    with patch.object(utils.current_platform, 'is_rocm', return_value=False):
        views = utils.allocate_kv_cache(plan, torch.device('cpu'), KVCacheLayout.BLHNC)
    cache = views[name].squeeze(1)
    cache.fill_(0)
    context = NS(attn_metadata={name: root_metadata([[1]], [64])})
    widths = []
    def broadcast(wire, src):
        widths.append(wire.numel())
        wire.fill_(79)
    group = NS(rank_in_group=7, device_group=None, broadcast=broadcast)
    with patch.object(runtime, 'get_forward_context', return_value=context), \
         patch.object(runtime, 'get_pcp_group', return_value=group), \
         patch.object(runtime.dist, 'all_reduce'), \
         patch.object(torch.cuda, 'is_current_stream_capturing', return_value=False):
        runtime.broadcast_kv_source_cache(name, 3, cache)
    assert len(widths) > 1
    assert sum(widths) == 1 + sum((n - 1) * s for n, s in zip(cache.shape[1:], cache.stride()[1:]))
    assert torch.all(cache[1] == 79) and torch.all(cache[0] == 0)


@pytest.mark.parametrize('rank', [0, 3, 7])
def test_live_broadcast_holes_and_stale_tail_preserve_other_groups(rank):
    cfg = config()
    swa_names = ['model.layers.0.attn.swa_cache', 'model.layers.1.attn.swa_cache']
    swa = SlidingWindowMLASpec(
        block_size=64, num_kv_heads=1, head_size=512, dtype=torch.uint8,
        sliding_window=128, alignment=576, model_version='deepseek_v4',
        state_content_bytes=584, cache_dtype_str='fp8_ds_mla')
    group_list = groups() + [KVCacheGroupSpec(swa_names, swa)]
    with patch.object(cfg, 'additional_config', {'hcu_layer_split': {'enabled': True}}, create=True):
        plan = placement._materialize(cfg, group_list, 7, rank)
    with patch.object(utils.current_platform, 'is_rocm', return_value=False):
        views = utils.allocate_kv_cache(plan, torch.device('cpu'), KVCacheLayout.BLHNC)
    for name in swa_names:
        views[name].fill_(42)
    prefix = 'model.layers.8.attn.indexer.k_cache'
    cache = views[prefix].squeeze(1)
    metadata = {prefix: root_metadata([[3, 1, 5]], [64])}
    group = NS(rank_in_group=rank, device_group=None,
               broadcast=lambda wire, src: wire.fill_(123))
    with patch.object(runtime, 'get_forward_context', return_value=NS(attn_metadata=metadata)), \
         patch.object(runtime, 'get_pcp_group', return_value=group), \
         patch.object(runtime.dist, 'all_reduce'), \
         patch.object(torch.cuda, 'is_current_stream_capturing', return_value=False):
        runtime.broadcast_kv_source_cache(prefix, 1, cache)
    assert torch.all(cache[3] == 123)
    # Only block 3 belongs to the source group. Holes AND stale tail IDs are SWA.
    for name in swa_names:
        assert torch.all(views[name][[0, 1, 2, 4, 5, 6]] == 42)


def test_adapter_platform_and_worker_use_registered_layout(monkeypatch):
    from hcu_adapter.bridges import vllm as bridge, vllm_kv_cache as registry
    from hcu_adapter.bridges.vllm_worker import Worker, OfficialWorker

    monkeypatch.setattr(registry, '_provider', None)
    monkeypatch.setattr(bridge, 'require_runtime_healthy', lambda: None)
    selector, localizer = placement.builder_for_config, placement.localize_kv_cache
    registry.register_kv_cache_layout('test-dsa', selector, localizer)
    registry.register_kv_cache_layout('test-dsa', selector, localizer)
    with pytest.raises(RuntimeError, match='different owner'):
        registry.register_kv_cache_layout('foreign', selector, localizer)
    cfg = config()
    with patch.object(cfg, 'additional_config', {'hcu_layer_split': {'enabled': True}}, create=True):
        assert bridge.HcuPlatform.get_kv_cache_config_builder_cls(cfg) == placement.builder_for_config(cfg)
        plan = placement.DsaLayerSplitCacheBuilder().get_kv_cache_config_from_groups(cfg, groups(), 7)
        original = copy.deepcopy(plan)
        worker = object.__new__(Worker)
        worker.vllm_config = cfg
        captured = []
        with patch.object(runtime, 'get_pcp_group', return_value=NS(rank_in_group=7)), \
             patch.object(OfficialWorker, 'initialize_from_config', lambda self, value: captured.append(value)):
            worker.initialize_from_config(plan)
        assert plan == original
        assert captured[0].kv_cache_tensors != plan.kv_cache_tensors
        assert captured[0].kv_cache_groups == plan.kv_cache_groups
        assert captured[0].num_blocks == plan.num_blocks
    with patch.object(cfg, 'additional_config', {}, create=True):
        assert bridge.HcuPlatform.get_kv_cache_config_builder_cls(cfg).endswith('DefaultKVCacheConfigBuilder')
    monkeypatch.setattr(registry, '_provider', None)
    assert registry.localize_kv_cache_config(cfg, plan) is plan


def _distributed_peer(rank, rendezvous):
    import torch.distributed as dist
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=2)
    try:
        cfg = config(width=2)
        swa_name = 'model.layers.0.attn.swa_cache'
        swa = SlidingWindowMLASpec(
            block_size=64, num_kv_heads=1, head_size=512, dtype=torch.uint8,
            sliding_window=128, alignment=576, model_version='deepseek_v4',
            state_content_bytes=584, cache_dtype_str='fp8_ds_mla')
        group_list = groups() + [KVCacheGroupSpec([swa_name], swa)]
        with patch.object(cfg, 'additional_config', {'hcu_layer_split': {'enabled': True}}, create=True):
            plan = placement._materialize(cfg, group_list, 7, rank)
        runtime.register_source_block_sizes({
            name: spec.block_size for name, spec in placement._specs(groups()[0])})
        with patch.object(utils.current_platform, 'is_rocm', return_value=False):
            caches = utils.allocate_kv_cache(plan, torch.device('cpu'), KVCacheLayout.BLHNC)
        # Each rank owns sources. Repeated remote reads must leave them intact.
        sources = [name for name in caches if name != swa_name]
        own = [name for name in sources if placement._owner(name, (2, 8, 14, 20), 2) == rank]
        for name in own:
            caches[name].fill_(11 + list(caches).index(name))
        caches[swa_name][0].fill_(37)
        caches[swa_name][4].fill_(37)
        group = NS(rank_in_group=rank, device_group=dist.group.WORLD,
                   broadcast=lambda wire, src: dist.broadcast(wire, src=src))
        for iteration in range(3):
            for name in sources:
                cache = caches[name]
                owner = placement._owner(name, (2, 8, 14, 20), 2)
                # Owner has no local queries. Remote needs 1/3/5 then 2/6;
                # stale IDs remain in the table tail across iterations.
                selected = [1, 3, 5] if iteration == 0 else [2, 6]
                local_selected = selected
                if iteration == 2:
                    local_selected = [1, 3] if rank == 0 else [2, 6]
                    selected = [1, 2, 3, 6]
                meta = root_metadata([local_selected + [4]], [64 * len(local_selected)])
                if rank == owner and iteration < 2:
                    meta = NS(block_table=torch.empty((0, 0), dtype=torch.int64),
                              seq_lens=torch.empty(0, dtype=torch.int32),
                              num_decodes=0, num_prefills=0)
                context = NS(attn_metadata={name: meta})
                with patch.object(runtime, 'get_forward_context', return_value=context), \
                     patch.object(runtime, 'get_pcp_group', return_value=group), \
                     patch.object(torch.cuda, 'is_current_stream_capturing', return_value=False):
                    runtime.broadcast_kv_source_cache(name, owner, cache.squeeze(1))
                assert torch.all(cache[selected] == 11 + list(caches).index(name))
            for name in own:
                assert torch.all(caches[name][[1, 2, 3, 5, 6]] == 11 + list(caches).index(name))
            assert torch.all(caches[swa_name][[0, 4]] == 37)
        assert not torch.cuda.is_initialized()
    finally:
        dist.destroy_process_group()


def test_real_gloo_broadcast_owner_without_local_queries(tmp_path):
    import torch.multiprocessing as mp
    from datetime import timedelta

    rendezvous = (tmp_path / 'gloo').as_uri()
    mp.spawn(_distributed_peer, args=(rendezvous,), nprocs=2, join=True)


def test_multiple_groups_keep_distinct_block_ids_disjoint():
    cfg = config()
    source = groups()[0]
    names = source.layer_names
    # Two logical groups with unequal page families. Independent block IDs
    # must not alias even when the groups' layouts overlay the same backing.
    group_list = []
    for selected in (names[:4], names[4:]):
        spec = UniformTypeKVCacheSpecs.from_specs({n: source.kv_cache_spec.kv_cache_specs[n] for n in selected})
        group_list.append(KVCacheGroupSpec(selected, spec))
    with patch.object(cfg, 'additional_config', {'hcu_layer_split': {'enabled': True}}, create=True):
        for rank in range(8):
            plan = placement._materialize(cfg, group_list, 7, rank)
            with patch.object(utils.current_platform, 'is_rocm', return_value=False):
                views = utils.allocate_kv_cache(plan, torch.device('cpu'), KVCacheLayout.BLHNC)
            for index, group in enumerate(group_list):
                for name in group.layer_names:
                    if placement._owner(name, (2, 8, 14, 20), 8) == rank:
                        views[name][index::2].fill_(50 + index)
            for index, group in enumerate(group_list):
                for name in group.layer_names:
                    if placement._owner(name, (2, 8, 14, 20), 8) == rank:
                        assert torch.all(views[name][index::2] == 50 + index)


@pytest.mark.parametrize('swa_block_size', [32, 64])
def test_native_hybrid_groups_account_for_receive_budget_and_capacity(swa_block_size):
    from vllm.v1.core.kv_cache_utils import get_kv_cache_capacity

    cfg = config()
    cfg.model_config.max_model_len = 16384
    cfg.model_config.original_max_model_len = 16384
    cfg.num_prefill_lookahead_tokens = 0
    cfg.max_in_flight_tokens = 8192
    specs = dict(groups()[0].kv_cache_spec.kv_cache_specs)
    for i in range(40):
        specs[f'model.layers.{i}.attn.swa_cache'] = SlidingWindowMLASpec(
            block_size=swa_block_size, num_kv_heads=1, head_size=512, dtype=torch.uint8,
            sliding_window=128, alignment=576, model_version='deepseek_v4',
            state_content_bytes=584, cache_dtype_str='fp8_ds_mla',
        )
    for i in (2, 8, 14):
        specs[f'model.layers.{i}.attn.compressor'] = CircularBufferSpec(
            block_size=1, num_kv_heads=1, head_size=512, dtype=torch.float32,
        )
    native = DefaultKVCacheConfigBuilder()
    native_groups = native.get_kv_cache_groups(cfg, specs)
    with patch.object(cfg, 'additional_config', {'hcu_layer_split': {'enabled': True}}, create=True):
        builder = placement.DsaLayerSplitCacheBuilder()
        split_groups = builder.get_kv_cache_groups(cfg, specs)
        assert split_groups == native_groups
        assert sorted(n for g in split_groups for n in g.layer_names) == sorted(specs)
        budget = 1024**3
        native_cost = _pool_bytes_per_block_from_builder(native, cfg, native_groups)
        split_cost = _pool_bytes_per_block_from_builder(builder, cfg, split_groups)
        native_plan = native.get_kv_cache_config_from_groups(
            cfg, native_groups, budget // native_cost)
        split_plan = builder.get_kv_cache_config_from_groups(
            cfg, split_groups, budget // split_cost)
        # Per-group scratch reuses space at IDs owned by that group only.
        # Same groups + lower physical cost must increase scheduler capacity.
        assert split_cost < native_cost
        assert split_cost * split_plan.num_blocks <= budget
        assert {tensor.size for tensor in split_plan.kv_cache_tensors} == {
            split_cost * split_plan.num_blocks}
        _, native_concurrency = get_kv_cache_capacity(cfg, native_plan)
        _, split_concurrency = get_kv_cache_capacity(cfg, split_plan)
        assert split_concurrency > native_concurrency
        assert split_concurrency == pytest.approx(
            native_concurrency * split_plan.num_blocks / native_plan.num_blocks)
        for rank in range(8):
            plan = placement._materialize(cfg, split_groups, 7, rank)
            with patch.object(utils.current_platform, 'is_rocm', return_value=False):
                views = utils.allocate_kv_cache(plan, torch.device('cpu'), KVCacheLayout.BLHNC)
            assert set(views) == set(specs)
            offsets, pitches, _ = placement._placements(cfg, split_groups)
            for group in split_groups:
                persistent_end = max((
                    offsets[rank][name] + pitches[name] for name in group.layer_names
                    if placement._owner(name, (2, 8, 14, 20), 8) in (None, rank)), default=0)
                for name in group.layer_names:
                    if placement._owner(name, (2, 8, 14, 20), 8) not in (None, rank):
                        assert offsets[rank][name] >= persistent_end


def test_core_planner_converges_all_eight_workers_with_full_specs():
    from vllm.v1.core import kv_cache_config_builder as resolution, kv_cache_planning as planning

    cfg = config()
    cfg.model_config.original_max_model_len = 256
    cfg.model_config.max_model_len = 256
    cfg.num_prefill_lookahead_tokens = 0
    specs = dict(groups()[0].kv_cache_spec.kv_cache_specs)
    builder = placement.DsaLayerSplitCacheBuilder()
    with patch.object(cfg, 'additional_config', {'hcu_layer_split': {'enabled': True}}, create=True), \
         patch.object(resolution, 'get_kv_cache_config_builder', return_value=builder):
        global_groups = planning.get_global_kv_cache_groups(cfg, [specs] * 8)
        cost = _pool_bytes_per_block_from_builder(builder, cfg, global_groups)
        configurations = planning.get_kv_cache_configs(
            cfg, [specs] * 8, [cost * (300 + rank) for rank in range(8)], global_groups,
        )
        assert len(configurations) == 8
        for rank, plan in enumerate(configurations):
            assert plan.num_blocks == 300
            assert {t.size for t in plan.kv_cache_tensors} == {cost * 300}
            with patch.object(runtime, 'get_pcp_group', return_value=NS(rank_in_group=rank)):
                local = placement.localize_kv_cache(cfg, plan)
            assert sorted(n for t in local.kv_cache_tensors for n in t.layers) == sorted(specs)
            assert local.kv_cache_groups == global_groups


def test_attention_zero_query_entry_point_broadcasts_before_returning():
    from hcu_vllm_das import v41_pcp_layer_split as attention
    from hcu_adapter.kernels import pcp_fork
    from vllm.models.deepseek_v41.amd import rocm

    obj = object.__new__(attention.HcuV41PcpAttention)
    torch.nn.Module.__init__(obj)
    swa_prefix = 'model.layers.20.attn.swa_cache'
    obj.swa_cache_layer = NS(prefix=swa_prefix, kv_cache=torch.zeros((4, 32, 584), dtype=torch.uint8))
    obj.compressed_cache_prefix = 'model.layers.20.attn'
    obj.is_kv_source = True
    obj.kv_cache = torch.zeros((4, 64, 584), dtype=torch.uint8)
    context = NS(attn_metadata={swa_prefix: NS(num_decode_tokens=0, num_prefill_tokens=0)})
    calls = []
    with patch.object(runtime, '_state', (8, (2, 8, 14, 20), 7)), \
         patch.object(attention, 'get_forward_context', return_value=context), \
         patch.object(pcp_fork, '_forward_context', return_value=context), \
         patch.object(attention, 'broadcast_kv_source_cache', side_effect=lambda prefix, owner, cache: calls.append((prefix, owner))), \
         patch.object(rocm.DeepseekV41ROCMAiterMLAAttention, 'forward_mqa') as native:
        obj.forward_mqa(torch.zeros((2, 1, 4)), torch.zeros((2, 4)), torch.arange(2), torch.ones((2, 1, 4)))
    assert calls == [('model.layers.20.attn', 3), (swa_prefix, 4)]
    native.assert_not_called()


def test_indexer_shortcut_entry_point_still_broadcasts():
    from hcu_vllm_das import v41_pcp_layer_split as attention
    from vllm.models.deepseek_v41.attention import DeepseekV4Indexer

    obj = object.__new__(attention.HcuV41PcpIndexer)
    torch.nn.Module.__init__(obj)
    obj.k_cache = NS(prefix='model.layers.20.attn.indexer.k_cache', kv_cache=torch.zeros((4, 64, 132)))
    calls = []
    with patch.object(runtime, '_state', (8, (2, 8, 14, 20), 7)), \
         patch.object(DeepseekV4Indexer, 'forward', return_value=(None, None, None)), \
         patch.object(attention, 'broadcast_kv_source_cache', side_effect=lambda prefix, owner, cache: calls.append((prefix, owner))):
        assert obj.forward() == (None, None, None)
    assert calls == [('model.layers.20.attn.indexer.k_cache', 3)]
