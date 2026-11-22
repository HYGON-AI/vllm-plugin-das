"""Independent pool regression against the installed Core, with GPU hidden."""
import copy
import pickle
from dataclasses import replace
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest
import torch

from hcu_vllm_das import dsa_cache_pools as pools
from hcu_vllm_das import dsa_cache_layout as placement
from hcu_vllm_das import dsa_cache_layer_split as runtime
from hcu_vllm_das.dsa_cache_scheduler import install_independent_pools
from test_swa_cache_layer_split import config, real_groups
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import get_kv_cache_capacity, generate_scheduler_kv_cache_config
from vllm.v1.core.kv_cache_planning import (
    DefaultKVCacheConfigBuilder,
)
from vllm.v1.kv_cache_layout import KVCacheLayout
from vllm.v1.worker import utils
from vllm.v1.request import Request
from vllm.sampling_params import SamplingParams


@pytest.fixture(autouse=True)
def enabled(monkeypatch):
    monkeypatch.setattr(runtime, '_transport_regions', {})


def setup(replay=False, budget=256 * 1024**2):
    cfg = config()
    cfg.additional_config = {'hcu_layer_split': {'enabled': True}}
    cfg.model_config.max_model_len = 16384
    cfg.model_config.original_max_model_len = 65536
    cfg.num_prefill_lookahead_tokens = 0
    cfg.max_in_flight_tokens = 16384
    cfg.cache_config.swa_bounded_replay = replay
    gs = real_groups()
    if replay:
        gs = [replace(g, kv_cache_spec=replace(g.kv_cache_spec,
            kv_cache_specs={n: replace(s, bounded_replay=True) if n.endswith('.swa_cache') else s
                            for n, s in placement._specs(g)})) for g in gs]
    merged = pools.merge_swa_groups(gs)
    plan = pools.materialize(cfg, merged, budget // pools.unit_cost(cfg, merged), 0)
    return cfg, gs, plan


@pytest.mark.parametrize('replay', [False, True])
def test_real_schema_budget_capacity_and_serialization(replay):
    cfg, original, plan = setup(replay)
    assert len(original) == 9 and len(plan.kv_cache_groups) == 3
    assert {n: s for g in original for n, s in placement._specs(g)} == {
        n: s for g in plan.kv_cache_groups for n, s in placement._specs(g)}
    budget = 256 * 1024**2
    assert plan.kv_cache_tensors[0].size <= budget
    native = DefaultKVCacheConfigBuilder().get_kv_cache_config_from_groups(
        cfg, original, budget // 116928)
    assert native.needs_kv_cache_zeroing
    assert not plan.needs_kv_cache_zeroing
    assert not generate_scheduler_kv_cache_config([plan]).needs_kv_cache_zeroing
    assert get_kv_cache_capacity(cfg, plan)[0] > get_kv_cache_capacity(cfg, native)[0]
    for clone in (copy.deepcopy(plan), pickle.loads(pickle.dumps(plan)), replace(plan)):
        assert clone.pool_blocks == plan.pool_blocks
        assert clone.num_blocks_of(clone.kv_cache_tensors[0]) == plan.pool_blocks[0]
    again = pools.materialize(cfg, plan.kv_cache_groups, plan.num_blocks, 7)
    assert again.num_blocks == plan.num_blocks
    assert again.pool_offsets == plan.pool_offsets
    assert again.pool_blocks == plan.pool_blocks
    with pytest.raises(ValueError, match='one maximum-length'):
        pools.materialize(cfg, plan.kv_cache_groups, sum(plan.pool_demands) - 1, 0)


@pytest.mark.parametrize('rank', range(8))
def test_native_views_isolate_groups_and_persistent_owners(rank):
    cfg, _, plan = setup(budget=80 * 1024**2)
    local = pools.localize(cfg, plan, rank)
    with patch.object(utils.current_platform, 'is_rocm', return_value=False):
        views = utils.allocate_kv_cache(local, torch.device('cpu'), KVCacheLayout.BLHNC)
    names = [next(iter(g.layer_names)) for g in local.kv_cache_groups]
    for i, name in enumerate(names):
        views[name][1].fill_(i + 1)
    for i, name in enumerate(names):
        assert (views[name][1] == i + 1).all()
    owned = [n for n in views if placement._owner(n, (2, 8, 14, 20), 8) in (rank, None)]
    for n in owned:
        views[n][1].fill_(17)
    for n in views:
        if n not in owned:
            views[n][1].fill_(91)
    assert all((views[n][1] == 17).all() for n in owned)
    for n, view in views.items():
        assert view.shape[0] == local.pool_blocks[local.layer_pool[n]]
    assert not torch.cuda.is_initialized()


def manager_for(plan, caching=False, pcp=1):
    scheduler_plan = generate_scheduler_kv_cache_config([plan])
    manager = KVCacheManager(scheduler_plan, 16384, 64, 8,
        max_in_flight_tokens=16384, enable_caching=caching, pcp_world_size=pcp)
    install_independent_pools(manager, scheduler_plan)
    return manager


def request(name, length):
    return Request(request_id=name, prompt_token_ids=[1] * length,
                   sampling_params=SamplingParams(max_tokens=16), pooling_params=None)


def test_real_manager_atomic_exhaustion_and_deferred_free():
    _, _, plan = setup()
    manager = manager_for(plan)
    initial = manager.block_pool.get_num_free_blocks()
    req = request('a', 4096)
    assert manager.allocate_slots(req, 4096) is not None
    assert manager.usage > 0
    # Empty the state pool while the other pools still have space.
    state_pool = manager.coordinator.single_type_managers[-1].block_pool
    held = state_pool.get_new_blocks(state_pool.get_num_free_blocks())
    before = [p.get_num_free_blocks() for p in manager.block_pool.pools]
    assert manager.allocate_slots(request('b', 128), 128) is None
    assert before == [p.get_num_free_blocks() for p in manager.block_pool.pools]
    state_pool.free_blocks(held)
    blocks = manager.coordinator.pop_blocks_for_free('a')
    manager.block_pool.free_blocks(reversed(blocks))
    assert manager.block_pool.get_num_free_blocks() == initial
    assert manager.usage == 0


def test_reported_capacity_is_backed_by_native_admission():
    cfg, _, plan = setup()
    manager = manager_for(plan)
    tokens, concurrent = get_kv_cache_capacity(cfg, plan)
    assert concurrent == int(concurrent)
    admitted = []
    for i in range(int(concurrent)):
        req = request(f'full-{i}', 16384)
        assert manager.allocate_slots(req, 16384, full_sequence_must_fit=True) is not None
        admitted.append(req)
    assert manager.take_new_block_ids() == []
    assert manager.take_kv_cache_block_copies() == ([], [])
    assert tokens == sum(req.num_tokens for req in admitted)
    assert manager.allocate_slots(request('overflow', 16384), 16384) is None
    for req in admitted:
        manager.free(req)
    assert manager.usage == 0


@pytest.mark.parametrize('replay', [False, True])
def test_chunked_request_recycling_and_reuse(replay):
    _, _, plan = setup(replay)
    manager = manager_for(plan)
    initial = manager.block_pool.get_num_free_blocks()
    req = request('chunks', 12288)
    for end in range(1024, 12289, 1024):
        assert manager.allocate_slots(req, 1024) is not None
        req.num_computed_tokens = end
        manager.coordinator.new_step_starts()
    manager.free(req)
    assert manager.block_pool.get_num_free_blocks() == initial
    req2 = request('reuse', 128)
    assert manager.allocate_slots(req2, 128) is not None
    manager.free(req2)
    assert manager.block_pool.get_num_free_blocks() == initial


def test_builder_selects_sync_async_and_leaves_off_unchanged(monkeypatch):
    cfg, _, _ = setup()
    for asynchronous in (False, True):
        cfg.scheduler_config.scheduler_cls = None
        cfg.scheduler_config.async_scheduling = asynchronous
        assert placement.builder_for_config(cfg).endswith('IndependentDSACacheBuilder')
        assert cfg.scheduler_config.scheduler_cls.endswith(
            'IndependentAsyncScheduler' if asynchronous else 'IndependentScheduler')
    cfg.scheduler_config.scheduler_cls = 'foreign.Scheduler'
    with pytest.raises(ValueError, match='foreign scheduler'):
        placement.builder_for_config(cfg)
    cfg.additional_config = {}
    assert placement.builder_for_config(cfg) is None
    assert cfg.scheduler_config.scheduler_cls == 'foreign.Scheduler'


def scheduler_for(asynchronous, replay, caching=False):
    from vllm.config import CacheConfig, SchedulerConfig, ObservabilityConfig
    from hcu_vllm_das.dsa_cache_scheduler import IndependentScheduler, IndependentAsyncScheduler
    cfg, _, plan = setup(replay)
    cfg.scheduler_config = SchedulerConfig(max_num_seqs=32, max_num_batched_tokens=8192,
                                          max_model_len=16384, is_encoder_decoder=False,
                                          async_scheduling=asynchronous)
    cfg.cache_config = CacheConfig(block_size=64, enable_prefix_caching=caching,
                                  swa_bounded_replay=replay)
    cfg.cache_config.num_gpu_blocks = plan.num_blocks
    cfg.observability_config = ObservabilityConfig()
    cfg.parallel_config.data_parallel_index = 0
    cfg.model_config.uses_mrope = False
    cfg.model_config.is_encoder_decoder = False
    cfg.model_config.runner_type = 'generate'
    cfg.model_config.is_diffusion = False
    cfg.model_config.supports_multimodal_inputs = False
    cfg.model_config.return_sampling_mask = False
    cfg.lora_config = cfg.kv_events_config = cfg.ec_transfer_config = None
    cfg.is_mm_encoder_only = False
    cfg.ec_manager_config = NS(get_encoder_cache_manager_obj=lambda: None)
    cfg.aux_output_config = NS(enabled=False)
    cfg.num_speculative_tokens = cfg.num_lookahead_tokens = cfg.num_hash_lookahead_tokens = 0
    cfg.use_v2_model_runner = True
    cfg.max_concurrent_batches = 2 if asynchronous else 1
    cls = IndependentAsyncScheduler if asynchronous else IndependentScheduler
    scheduler = cls(cfg, generate_scheduler_kv_cache_config([plan]), NS(), 64, 8)
    return scheduler


@pytest.mark.parametrize('asynchronous', [False, True])
@pytest.mark.parametrize('replay', [False, True])
def test_native_scheduler_constructor_and_first_request(asynchronous, replay):
    scheduler = scheduler_for(asynchronous, replay)
    assert len(scheduler.kv_cache_manager.block_pool.pools) == 3
    assert scheduler.prefix_replay_tokens == (128 if replay else 0)
    req = request('scheduler', 128)
    scheduler.add_request(req)
    output = scheduler.schedule()
    assert output.num_scheduled_tokens[req.request_id] == 128
    assert not output.new_block_ids_to_zero and not output.kv_cache_block_copies


@pytest.mark.parametrize('asynchronous', [False, True])
def test_native_scheduler_rewinds_swa_after_main_prefix_hit(asynchronous):
    from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
    from vllm.utils.hashing import sha256
    init_none_hash(sha256)
    hasher = get_request_block_hasher(8, sha256)
    scheduler = scheduler_for(asynchronous, True, caching=True)
    manager = scheduler.kv_cache_manager
    seed = request('seed', 512)
    seed.block_hashes = hasher(seed)
    assert manager.allocate_slots(seed, 512) is not None
    manager.free(seed)
    req = request('replay-hit', 512)
    req.block_hashes = hasher(req)
    _, hit_tokens, _ = manager.get_computed_blocks(req)
    assert hit_tokens == 448
    scheduler.add_request(req)
    output = scheduler.schedule()
    assert output.num_scheduled_tokens[req.request_id] == 512 - hit_tokens + 128


def test_prefix_cache_hit_uses_each_pool_and_reset():
    from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
    from vllm.utils.hashing import sha256
    init_none_hash(sha256)
    _, _, plan = setup(replay=True)
    manager = manager_for(plan, caching=True)
    hasher = get_request_block_hasher(8, sha256)
    req = request('prefix', 512)
    req.block_hashes = hasher(req)
    assert manager.allocate_slots(req, 512) is not None
    manager.free(req)
    other = request('prefix2', 512)
    other.block_hashes = hasher(other)
    hit, tokens, *_ = manager.get_computed_blocks(other)
    assert tokens > 0
    assert manager.allocate_slots(other, 512 - tokens, num_new_computed_tokens=tokens,
                                  new_computed_blocks=hit) is not None
    manager.free(other)
    assert manager.reset_prefix_cache()
    assert manager.get_computed_blocks(request('cold', 512))[1] == 0


def test_independent_pools_gloo_transport_preserves_other_pools(tmp_path):
    import torch.multiprocessing as mp
    from test_swa_cache_layer_split import _swa_peer
    mp.spawn(_swa_peer, args=((tmp_path / 'independent_gloo').as_uri(), True),
             nprocs=2, join=True)


@pytest.mark.parametrize('override', [None, 1540])
def test_core_planning_cross_rank_minimum_and_override(override):
    from vllm.v1.core.kv_cache_planning import get_kv_cache_configs, get_global_kv_cache_groups
    cfg, _, plan = setup()
    cfg.cache_config.num_gpu_blocks_override = override
    groups = plan.kv_cache_groups
    specs = {n: s for g in groups for n, s in placement._specs(g)}
    budgets = [256 * 1024**2 + i * 1024**2 for i in range(8)]
    with patch('vllm.v1.core.kv_cache_config_builder.get_kv_cache_config_builder',
               return_value=pools.IndependentDSACacheBuilder()):
        groups = get_global_kv_cache_groups(cfg, [specs] * 8)
        assert len(groups) == 3
        plans = get_kv_cache_configs(cfg, [specs] * 8, budgets, groups)
    assert len({p.num_blocks for p in plans}) == 1
    assert len({p.pool_blocks for p in plans}) == 1
    for p, budget in zip(plans, budgets):
        limit = budget if override is None else override * pools.unit_cost(cfg, groups)
        assert p.kv_cache_tensors[0].size <= limit
        if override is not None:
            assert p.num_blocks == override // sum(p.pool_demands) * sum(p.pool_demands)


def test_core_auto_fit_fits_actual_backing():
    from vllm.v1.core.kv_cache_planning import get_kv_cache_configs
    cfg, _, plan = setup()
    cfg.model_config.original_max_model_len = -1
    cfg.device_config = NS(device_type='cpu')
    groups = plan.kv_cache_groups
    specs = {n: s for g in groups for n, s in placement._specs(g)}
    budget = 40 * 1024**2
    with patch('vllm.v1.core.kv_cache_config_builder.get_kv_cache_config_builder',
               return_value=pools.IndependentDSACacheBuilder()):
        result, = get_kv_cache_configs(cfg, [specs], [budget], groups)
    assert 0 < cfg.model_config.max_model_len < 16384
    assert result.kv_cache_tensors[0].size <= budget
    assert get_kv_cache_capacity(cfg, result)[0] >= cfg.model_config.max_model_len


def test_pd_destination_rejects_out_of_pool_ids():
    from test_mooncake_layer_split import bare_worker
    _, _, plan = setup()
    worker = bare_worker()
    worker.kv_cache_config = plan
    worker._validate_destination_ids([[1], [1], [1]])
    with pytest.raises(ValueError, match='count mismatch'):
        worker._validate_destination_ids([[1]])
    with pytest.raises(ValueError, match='exceeds'):
        worker._validate_destination_ids([[1], [1], [plan.pool_blocks[2]]])


def test_reject_global_id_jit_warmup_and_mixed_precision_pool():
    cfg, _, plan = setup()
    cfg.kernel_config = NS(enable_jit_warmup=True)
    with pytest.raises(ValueError, match='JIT warmup disabled'):
        pools.materialize(cfg, plan.kv_cache_groups, plan.num_blocks, 0)
    cfg.kernel_config.enable_jit_warmup = False
    groups = copy.deepcopy(plan.kv_cache_groups)
    specs = groups[1].kv_cache_spec.kv_cache_specs
    name = next(iter(specs))
    specs[name] = replace(specs[name], dtype=torch.int8)
    with pytest.raises(ValueError, match='mixed precisions'):
        pools.materialize(cfg, groups, plan.num_blocks, 0)


def test_pd_different_groups_native_byte_transfer():
    import asyncio
    import ctypes
    from test_mooncake_layer_split import bare_worker
    from hcu_vllm_das import mooncake_layer_split as m
    async def run():
        cfg, original, plan = setup(budget=80 * 1024**2)
        dst_plan = DefaultKVCacheConfigBuilder().get_kv_cache_config_from_groups(cfg, original, 24)
        with patch.object(utils.current_platform, 'is_rocm', return_value=False):
            dst = utils.allocate_kv_cache(dst_plan, torch.device('cpu'), KVCacheLayout.BLHNC)
        dst_regions, dst_manifest = m.describe_caches(dst_plan, dst)
        dst_blocks = [[i * 2 + 1, i * 2 + 2] for i in range(8)] + [[17]]
        src_blocks = [[1, 3], [1, 3], [1]]
        storage = next(iter(dst.values())).untyped_storage()
        storage.fill_(173)
        touched = torch.zeros(storage.nbytes(), dtype=torch.bool)
        expected = {}
        registrations = []
        for rank in range(8):
            local_plan = pools.materialize(cfg, plan.kv_cache_groups, plan.num_blocks, rank)
            with patch.object(utils.current_platform, 'is_rocm', return_value=False):
                src = utils.allocate_kv_cache(local_plan, torch.device('cpu'), KVCacheLayout.BLHNC)
            regions, manifest = m.describe_caches(local_plan, src)
            worker = bare_worker()
            worker.kv_cache_config, worker._manifest = local_plan, manifest
            worker._regions = [r for r in regions if m.transfer_owner(r.layer_name, (2,8,14,20), 8, True) == rank]
            worker._get_sender_transfer_plan = lambda **kw: (True, 0, 0, kw['local_kv_block_len'])
            registrations.append(dict(protocol=m.PROTOCOL, engine_id='p', pcp_rank=rank,
                pcp_size=8, split=True, source_ids=[2,8,14,20], addr=f'tcp://localhost:{1000+rank}',
                sends=[r.layer_name for r in worker._regions], manifest=manifest))
            for idx, r in enumerate(worker._regions):
                for block in src_blocks[r.group_index]:
                    src[r.layer_name][block].fill_(19 + idx + rank)
                    expected[r.layer_name] = src[r.layer_name][block].clone()
            meta = m.core.MooncakeXferMetadata(remote_hostname='localhost', remote_port=1,
                remote_tp_size=1, remote_tp_rank=0, req_blocks={'d': ('x', dst_blocks)},
                kv_caches_base_addr=[r.base_addr for r in dst_regions],
                block_lens=[r.block_len for r in dst_regions], kv_block_lens=[r.kv_block_len for r in dst_regions],
                registered_layer_names=[r.layer_name for r in dst_regions],
                registered_layer_indices=[r.layer_index for r in dst_regions],
                registered_group_indices=[r.group_index for r in dst_regions],
                registered_shared_group_ids=[[] for _ in dst_regions],
                registered_row_offsets=[-1 for _ in dst_regions], layout_id=1)
            local, remote, error = worker._prepare_transfer_regions(meta)
            assert error is None
            send = m.core.SendBlockMeta('p', 'x', src_blocks, asyncio.Event())
            a, b, sizes, errors, error = await worker._build_transfer_params([('d', send)], meta, local, remote)
            assert not errors and error is None
            compact = copy.copy(meta)
            compact.kv_caches_base_addr = compact.block_lens = compact.kv_block_lens = []
            compact.registered_group_indices = compact.registered_layer_names = []
            cached_local, cached_remote = worker._lookup_peer_layout(compact)
            repeat = await worker._build_transfer_params([('d', send)], compact, cached_local, cached_remote)
            assert repeat == (a, b, sizes, errors, error)
            assert send.local_block_ids is src_blocks and meta.req_blocks['d'][1] is dst_blocks
            for source, target, size in zip(a, b, sizes):
                begin = target - storage.data_ptr()
                assert not touched[begin:begin+size].any()
                touched[begin:begin+size] = True
                ctypes.memmove(target, source, size)
        assert len(m.validate_peers(dict(protocol=m.PROTOCOL, workers=registrations), dst_manifest)[1]) == 8
        with pytest.raises(ValueError, match='owner-aware'):
            m.validate_peers(dict(protocol='hcu-v41-pcp-owners-v2', workers=registrations), dst_manifest)
        for r in dst_regions:
            for block in dst_blocks[r.group_index]:
                assert torch.equal(dst[r.layer_name][block], expected[r.layer_name])
        raw = torch.empty(0, dtype=torch.uint8).set_(storage, 0, (storage.nbytes(),), (1,))
        assert torch.all(raw[~touched] == 173)
    asyncio.run(run())
