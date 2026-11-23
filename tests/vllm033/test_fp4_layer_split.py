"""FP4 + Layer Split CPU integration; device leaves are explicit recorders."""
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest
import torch

from hcu_vllm_das import fp4_indexer as impl, v41_pcp as binding
from hcu_vllm_das import v41_pcp_layer_split as split_impl
from hcu_vllm_das import dsa_cache_layer_split as runtime, dsa_cache_pools as pools
from hcu_vllm_das import dsa_cache_layout as placement, mooncake_layer_split as mooncake
from hcu_vllm_das.models import v41_pcp as model
from test_fp4_full_chain import chain
from test_fp4_pcp_workspace import pcp_config
from test_swa_cache_layer_split import real_groups
import test_dsa_cache_pools as pool_tests
from vllm.v1.kv_cache_interface import KVCacheGroupSpec, UniformTypeKVCacheSpecs
from vllm.v1.kv_cache_layout import KVCacheLayout
from vllm.v1.worker import utils


@pytest.fixture
def split_chain(chain, monkeypatch):
    monkeypatch.setenv('VLLM_HCU_DSV41_PCP_ATTENTION', '1')
    monkeypatch.setenv('HCU_VLLM_PCP_IMPL', 'native')
    pcp_config(chain.cfg)
    chain.cfg.additional_config = {'hcu_layer_split': {'enabled': True}}
    monkeypatch.setattr(runtime, '_state', (8, (0,), 0))
    return chain


def attention(chain, ratio=1):
    chain.cfg.model_config.hf_config.compress_ratios = [ratio, 0]
    return model._fp4_attention(chain.cfg)(vllm_config=chain.cfg,
        prefix='model.layers.0.self_attn',
        topk_indices_buffer=torch.empty(16, 4, dtype=torch.int32))


@pytest.mark.parametrize('replay', [False, True])
def test_real_selector_native_constructor_and_off_path(split_chain, monkeypatch, replay):
    split_chain.cfg.cache_config.swa_bounded_replay = replay
    obj = attention(split_chain)
    assert type(obj) is impl.Fp4LayerSplitPcpAttention
    assert type(obj.indexer) is impl.Fp4LayerSplitIndexer
    assert type(obj.indexer.k_cache) is impl.Fp4IndexerCache
    assert impl.Fp4LayerSplitPcpAttention.forward is impl.Fp4PcpAttention.forward
    assert impl.Fp4LayerSplitIndexer.__init__ is impl.Fp4Indexer.__init__
    assert model._FP4_PCP_DECODER_INIT.__globals__['_select_dsv4_attn_cls'] is model._fp4_attention
    assert getattr(impl.common.DeepseekV4Indexer._produce_k, '_hcu_layer_split_wrapper', False)
    split_chain.cfg.additional_config = {}
    assert model._fp4_attention(split_chain.cfg) is impl.Fp4PcpAttention


@pytest.mark.parametrize('rank', range(8))
@pytest.mark.parametrize('ratio', [1, 2])
def test_only_owner_runs_fp4_writer_and_empty_shortcut_broadcasts(split_chain, monkeypatch, rank, ratio):
    source = attention(split_chain, ratio).indexer
    monkeypatch.setattr(runtime, '_state', (8, (0,), rank))
    cache = torch.full((2, 1, 64 // ratio, 68), 97, dtype=torch.uint8)
    source.k_cache.bind_kv_cache(cache)
    latent = torch.ones(4, 128, dtype=torch.bfloat16)
    positions = torch.arange(4)
    slots = torch.where((positions + 1) % ratio == 0, positions // ratio, -1)
    split_chain.context.attn_metadata = {source.k_cache.prefix: NS(
        slot_mapping=slots, max_seq_len=1, num_decode_tokens=0, num_prefill_tokens=0)}
    monkeypatch.setattr(impl.common, 'get_forward_context', lambda: split_chain.context)
    monkeypatch.setattr(source.wk, 'forward', lambda x: (x, None))
    broadcasts = []
    monkeypatch.setattr(split_impl, 'broadcast_kv_source_cache',
                        lambda name, owner, value: broadcasts.append((name, owner, value)))
    result = source(latent, latent, torch.ones(4, 32), positions, split_chain.rope)
    assert result == (None, None, None)  # Real native short-context/zero-query branch.
    assert len(broadcasts) == 1 and broadcasts[0][1] == 0
    assert broadcasts[0][2] is source.k_cache.kv_cache
    assert len(split_chain.kernel.calls) == int(rank == 0)
    if rank == 0:
        grid, args, options = split_chain.kernel.calls[0]
        assert grid == (4,) and args[8] is slots
        assert options['USE_FP4'] and options['TOKEN_STRIDE'] == 64
        assert options['SCALE_DIM'] == 4 and options['COMPRESS_RATIO'] == ratio
    assert torch.all(cache == 97)  # The GPU store is a recorder, not simulated math.


def test_pd_requires_matching_owner_protocol_and_retains_fp4_guards(split_chain, monkeypatch):
    cfg = split_chain.cfg
    cfg.kv_transfer_config = NS(kv_connector=mooncake.CONNECTOR,
        kv_connector_module_path=mooncake.MODULE, kv_role='kv_producer')
    binding.bind_adapter_classes()
    assert model._fp4_attention(cfg) is impl.Fp4LayerSplitPcpAttention
    for attr, value in [('kv_connector', 'MooncakeConnector'), ('kv_role', 'kv_consumer'),
                        ('kv_connector_module_path', 'foreign.module')]:
        previous = getattr(cfg.kv_transfer_config, attr)
        setattr(cfg.kv_transfer_config, attr, value)
        with pytest.raises(ValueError):
            model._fp4_attention(cfg)
        setattr(cfg.kv_transfer_config, attr, previous)
    cfg.speculative_config = NS(method='dspark', num_speculative_tokens=5,
                                enable_adaptive_verification=False)
    with pytest.raises(ValueError, match='requires an eager draft model'):
        model._fp4_attention(cfg)
    cfg.speculative_config = None
    cfg.additional_config = {}
    with pytest.raises(ValueError, match='KV transfer is not admitted'):
        model._fp4_attention(cfg)


@pytest.fixture
def fp4_groups(split_chain):
    # Take index specs from the real FP4 constructors, not a guessed 68-byte formula.
    index_specs = {}
    for ratio in (1, 2):
        split_chain.cfg.compilation_config.static_forward_context.clear()
        obj = attention(split_chain, ratio)
        index_specs[ratio] = obj.indexer.k_cache.get_kv_cache_spec(split_chain.cfg)
    groups = []
    for group in real_groups():
        specs = {name: (index_specs[spec.tokens_per_state] if name.endswith('.indexer.k_cache')
                        else spec) for name, spec in placement._specs(group)}
        groups.append(KVCacheGroupSpec(list(specs), UniformTypeKVCacheSpecs.from_specs(specs)))
    return groups


@pytest.mark.parametrize('rank', range(8))
def test_fp4_native_pool_views_and_live_capacity(split_chain, fp4_groups, monkeypatch, rank):
    monkeypatch.setattr(pool_tests, 'real_groups', lambda: fp4_groups)
    cfg, original, plan = pool_tests.setup(budget=80 * 1024**2)
    local = pools.localize(cfg, plan, rank)
    with patch.object(utils.current_platform, 'is_rocm', return_value=False):
        views = utils.allocate_kv_cache(local, torch.device('cpu'), KVCacheLayout.BLHNC)
    for name, view in views.items():
        if name.endswith('.indexer.k_cache'):
            normalized = impl.cache_view(view)
            assert normalized.shape[-1] == 68
            assert normalized.shape[0] == local.pool_blocks[local.layer_pool[name]]
    pool_tests.test_reported_capacity_is_backed_by_native_admission()


def test_fp4_pd_transfers_exact_pages_across_different_group_numbers(split_chain, fp4_groups, monkeypatch):
    monkeypatch.setattr(pool_tests, 'real_groups', lambda: fp4_groups)
    pool_tests.test_pd_different_groups_native_byte_transfer()


def test_fp8_peer_schema_cannot_receive_fp4_pages(split_chain, fp4_groups, monkeypatch):
    cfg, _, _ = pool_tests.setup(budget=80 * 1024**2)
    manifests = []
    for groups in (real_groups(), fp4_groups):
        builder = pool_tests.DefaultKVCacheConfigBuilder()
        plan = builder.get_kv_cache_config_from_groups(cfg, groups, 24)
        with patch.object(utils.current_platform, 'is_rocm', return_value=False):
            views = utils.allocate_kv_cache(plan, torch.device('cpu'), KVCacheLayout.BLHNC)
        manifests.append(mooncake.describe_caches(plan, views)[1])
    registrations = [dict(protocol=mooncake.PROTOCOL, engine_id='p', pcp_rank=rank,
        pcp_size=8, split=True, source_ids=[2, 8, 14, 20], addr=str(rank),
        manifest=manifests[1], sends=[n for n in manifests[1]
            if mooncake.transfer_owner(n, (2, 8, 14, 20), 8, True) == rank]) for rank in range(8)]
    with pytest.raises(ValueError, match='schemas differ'):
        mooncake.validate_peers(dict(protocol=mooncake.PROTOCOL, workers=registrations), manifests[0])


def _fp4_cache_peer(rank, rendezvous, groups):
    """Real Gloo bytes on native allocated FP4 pages; no device math doubles."""
    import torch.distributed as dist
    from test_dsa_cache_layout import config, root_metadata

    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=2)
    try:
        cfg = config(width=2)
        cfg.model_config.max_model_len = 512
        cfg.max_in_flight_tokens = 128
        cfg.num_prefill_lookahead_tokens = 0
        merged = pools.merge_swa_groups(groups)
        with patch.object(cfg, 'additional_config', {'hcu_layer_split': {'enabled': True}}, create=True):
            demands = pools.pool_geometry(cfg, merged)[3]
            plan = pools.materialize(cfg, merged, sum(demands) * 10, rank)
            local = pools.localize(cfg, plan, rank)
        runtime.register_source_block_sizes({n: s.block_size for g in merged
                                             for n, s in placement._specs(g)})
        with patch.object(utils.current_platform, 'is_rocm', return_value=False):
            caches = utils.allocate_kv_cache(local, torch.device('cpu'), KVCacheLayout.BLHNC)
        names = sorted(n for n in caches if n.endswith('.indexer.k_cache'))
        protected = [n for n in caches if local.layer_pool[n] != local.layer_pool[names[0]]]
        for n in protected:
            caches[n].fill_(197)
        references = {}
        for i, name in enumerate(names):
            view = impl.cache_view(caches[name])
            # Distinct FP4 payload and scale bytes, plus distinct rows/layers.
            ref = ((torch.arange(view[1].numel()).reshape(view[1].shape)
                    + i * 31) % 251).to(torch.uint8)
            references[name] = ref
            if runtime.indexer_cache_owner_rank(name, (2, 8, 14, 20), 2) == rank:
                view[[1, 3, 5, 7]] = ref
        group = NS(rank_in_group=rank, device_group=dist.group.WORLD,
                   broadcast=lambda wire, src: dist.broadcast(wire, src=src))
        for live in ([1, 3, 5, 7], [5, 7], [1, 7]):
            for name in names:
                owner = runtime.indexer_cache_owner_rank(name, (2, 8, 14, 20), 2)
                meta = root_metadata([list(live) + [999]], [64 * len(live)])
                if rank == owner:
                    meta = NS(num_decodes=0, num_prefills=0,
                              num_decode_tokens=0, num_prefill_tokens=0)
                with patch.object(runtime, 'get_forward_context', return_value=NS(attn_metadata={name: meta})), \
                     patch.object(runtime, 'get_pcp_group', return_value=group), \
                     patch.object(torch.cuda, 'is_current_stream_capturing', return_value=False):
                    runtime.broadcast_kv_source_cache(name, owner, caches[name])
                view = impl.cache_view(caches[name])
                assert all(torch.equal(view[b], references[name]) for b in live)
                # Remote source pages share scratch; persistent owners must survive reuse.
                for own in names:
                    if runtime.indexer_cache_owner_rank(own, (2, 8, 14, 20), 2) == rank:
                        own_view = impl.cache_view(caches[own])
                        assert all(torch.equal(own_view[b], references[own]) for b in [1, 3, 5, 7])
                assert all(torch.all(caches[n] == 197) for n in protected)
        assert not torch.cuda.is_initialized()
    finally:
        runtime.register_source_block_sizes({})
        runtime.register_transport_regions({})
        dist.destroy_process_group()


def test_fp4_real_gloo_sparse_pages_empty_owner_and_scratch_reuse(fp4_groups, tmp_path):
    import torch.multiprocessing as mp
    mp.spawn(_fp4_cache_peer, args=((tmp_path / 'fp4_gloo').as_uri(), fp4_groups),
             nprocs=2, join=True)
