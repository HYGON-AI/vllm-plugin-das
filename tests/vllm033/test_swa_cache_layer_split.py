"""CPU coverage for SWA ownership, exact live pages and decoder replay.

Real native layouts, Gloo and replay scheduling execute here. HIP kernels are
explicit doubles; these tests do not claim GPU numerical/performance coverage.
"""
import copy
import json
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest
import torch

from hcu_vllm_das import dsa_cache_layer_split as runtime
from hcu_vllm_das import dsa_cache_layout as placement
from vllm.v1 import kv_cache_interface as interface
from vllm.v1.attention.backend import MultipleOf
from vllm.v1.core.kv_cache_planning import (
    DefaultKVCacheConfigBuilder, _pool_bytes_per_block_from_builder,
)
from vllm.v1.core.kv_cache_utils import get_kv_cache_capacity
from vllm.v1.kv_cache_interface import KVCacheGroupSpec, SlidingWindowMLASpec, UniformTypeKVCacheSpecs
from vllm.v1.kv_cache_layout import KVCacheLayout
from vllm.v1.worker import utils
from test_dsa_cache_layout import config, groups


def real_groups():
    fixture = Path(__file__).with_name('fixtures') / 'dsv41_fp8_native_cache_groups.json'
    result = []
    for group in json.loads(fixture.read_text()):
        specs = {}
        for name, record in group['layers'].items():
            kwargs = dict(record['spec'])
            kwargs['dtype'] = getattr(torch, kwargs['dtype'].split('.')[-1])
            kwargs['kv_quant_mode'] = interface.KVQuantMode(kwargs['kv_quant_mode'])
            if isinstance(kwargs['block_stride_alignment'], dict):
                kwargs['block_stride_alignment'] = MultipleOf(**kwargs['block_stride_alignment'])
            specs[name] = getattr(interface, record['kind'])(**kwargs)
        result.append(KVCacheGroupSpec(list(specs), UniformTypeKVCacheSpecs.from_specs(specs)))
    return result


@pytest.mark.parametrize('replay', [False, True])
def test_real_fp8_capacity_beats_main_only_without_changing_logical_specs(replay):
    cfg, gs = config(), real_groups()
    cfg.cache_config.swa_bounded_replay = replay
    cfg.model_config.max_model_len = 16384
    cfg.model_config.original_max_model_len = 65536
    cfg.max_in_flight_tokens = 16384
    cfg.num_prefill_lookahead_tokens = 0
    budget = 10 * 1024**3
    with patch.object(cfg, 'additional_config', {'hcu_layer_split': {'enabled': True}}, create=True):
        builder = placement.DsaLayerSplitCacheBuilder()
        cost = _pool_bytes_per_block_from_builder(builder, cfg, gs)
        assert cost < 114304  # previous main-only split, including its wire
        assert cost == 98560  # unchanged compressor-state group now dominates
        plan = builder.get_kv_cache_config_from_groups(cfg, gs, budget // cost)
        old = builder.get_kv_cache_config_from_groups(cfg, gs, budget // 114304)
        native = DefaultKVCacheConfigBuilder().get_kv_cache_config_from_groups(
            cfg, gs, budget // 116928)
        assert plan.kv_cache_groups is gs
        assert get_kv_cache_capacity(cfg, plan)[0] > get_kv_cache_capacity(cfg, old)[0]
        assert get_kv_cache_capacity(cfg, old)[0] > get_kv_cache_capacity(cfg, native)[0]
        offsets, pitches, _ = placement._placements(cfg, gs)
        swa = [n for g in gs for n in g.layer_names if n.endswith('.swa_cache')]
        assert len(swa) == 40
        assert [sum(runtime.swa_cache_owner_rank(n, 8) == r for n in swa) for r in range(8)] == [5] * 8
        for rank in range(8):
            for group in gs[:7]:
                for name in group.layer_names:
                    if placement._owner(name, (2, 8, 14, 20), 8) != rank:
                        continue
                    # Persistent SWA may never overlap another layer's receive slot.
                    a, b = offsets[rank][name], offsets[rank][name] + pitches[name]
                    for other in group.layer_names:
                        if other != name:
                            x, y = offsets[rank][other], offsets[rank][other] + pitches[other]
                            assert b <= x or y <= a


def swa_meta(table=None, end=0, gather=0, *, decode=False):
    table = [] if table is None else table
    return NS(block_size=32, block_table=torch.tensor([table], dtype=torch.int64),
              num_decodes=int(decode), num_prefills=int(not decode),
              num_decode_tokens=int(decode), num_prefill_tokens=0 if decode else gather,
              prefill_seq_lens=torch.tensor([end]), prefill_gather_lens=torch.tensor([gather]),
              decode_swa_indices=torch.empty((1, 0), dtype=torch.int64),
              decode_swa_lens=torch.tensor([0]))


@pytest.fixture
def prefix(monkeypatch):
    name = 'model.layers.9.attn.swa_cache'
    monkeypatch.setattr(runtime, '_source_block_sizes', {name: 32})
    return name


def test_prefill_and_replay_read_only_native_gather_range(prefix):
    # Old prefix and padded tail IDs are deliberately invalid. They must not
    # be selected, even when the block table still contains them after reuse.
    meta = swa_meta([999, -5, 2, 3, 5, 6, 7, 999], end=224, gather=130)
    assert runtime._live_block_ids({prefix: meta}, prefix, 8) == [2, 3, 5, 6, 7]
    # Native bounded replay metadata can trim the visible prefix further.
    meta.prefill_gather_lens[0] = 64
    assert runtime._live_block_ids({prefix: meta}, prefix, 8) == [6, 7]


@pytest.mark.parametrize('native_shape', [False, True])
def test_decode_uses_actual_paged_indices_not_old_table(prefix, native_shape):
    meta = swa_meta([999, -8], decode=True)
    meta.decode_swa_indices = torch.tensor([[32, 33, 159, -1, 999]])
    meta.decode_swa_lens = torch.tensor([4])
    if native_shape:
        from vllm.models.deepseek_v41.amd.rocm import DeepseekV41ROCMAiterSparseSWAMetadata
        meta.decode_swa_indices = meta.decode_swa_indices.unsqueeze(1)
        meta = DeepseekV41ROCMAiterSparseSWAMetadata(slot_mapping=torch.tensor([32]), **vars(meta))
    assert runtime._live_block_ids({prefix: meta}, prefix, 8) == [1, 4]


def test_mixed_prefill_decode_uses_separate_extents(prefix):
    meta = swa_meta([5, 6, 999], end=64, gather=32)
    meta.num_decodes = meta.num_decode_tokens = 1
    meta.block_table = torch.tensor([[999, 999, 999], [5, 6, 999]])
    meta.decode_swa_indices = torch.tensor([[2 * 32, 999]])
    meta.decode_swa_lens = torch.tensor([1])
    assert runtime._live_block_ids({prefix: meta}, prefix, 8) == [2, 6]


@pytest.mark.parametrize('field,value', [
    ('block_size', 64), ('num_prefill_tokens', -1),
    ('prefill_gather_lens', torch.tensor([66])),
    ('prefill_seq_lens', torch.tensor([512])),
    ('block_table', torch.tensor([[1, 8]])),
])
def test_bad_swa_metadata_rejected(prefix, field, value):
    meta = swa_meta([1, 2], 64, 64)
    setattr(meta, field, value)
    with pytest.raises(RuntimeError, match='DSA SWA'):
        runtime._live_block_ids({prefix: meta}, prefix, 8)


def test_empty_pcp_placeholder_needs_no_read_table(prefix):
    meta = NS(block_size=32, num_decode_tokens=0, num_prefill_tokens=0)
    assert runtime._live_block_ids({prefix: meta}, prefix, 8) == []


@pytest.mark.parametrize('rank', [0, 1, 7])
def test_write_mask_preserves_original_and_query_metadata(prefix, rank):
    slots = torch.tensor([32, 33, -1, 161])
    meta = NS(cache_slot_mapping=slots, slot_mapping=torch.tensor([32]),
              cache_positions=torch.arange(4), replay_start=torch.tensor([128]))
    original = {prefix: meta, 'other-layer': meta}
    with patch.object(runtime, '_state', (8, (2, 8, 14, 20), rank)):
        result = runtime.owner_swa_write_metadata(original, prefix)
    assert original[prefix] is meta and slots.tolist() == [32, 33, -1, 161]
    assert result['other-layer'] is meta
    assert result[prefix].slot_mapping is meta.slot_mapping
    assert result[prefix].replay_start is meta.replay_start
    if rank == 1:
        assert result is original
    else:
        assert result[prefix].cache_slot_mapping.tolist() == [-1] * 4


def _swa_peer(rank, rendezvous, independent=False):
    import torch.distributed as dist
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=2)
    try:
        cfg = config(width=2)
        names = [f'model.layers.{i}.attn.swa_cache' for i in range(4)]
        spec = SlidingWindowMLASpec(block_size=32, num_kv_heads=1, head_size=16,
                                    dtype=torch.uint8, sliding_window=128)
        gs = groups() + [KVCacheGroupSpec(names, spec)]
        with patch.object(cfg, 'additional_config', {'hcu_layer_split': {'enabled': True}}, create=True):
            if independent:
                from hcu_vllm_das import dsa_cache_pools as pools
                cfg.model_config.max_model_len = 512
                cfg.max_in_flight_tokens = 128
                cfg.num_prefill_lookahead_tokens = 0
                demands = pools.pool_geometry(cfg, gs)[3]
                plan = pools.materialize(cfg, gs, sum(demands) * 9, rank)
                plan = pools.localize(cfg, plan, rank)
            else:
                plan = placement._materialize(cfg, gs, 9, rank)
        runtime.register_source_block_sizes({n: 32 for n in names})
        with patch.object(utils.current_platform, 'is_rocm', return_value=False):
            caches = utils.allocate_kv_cache(plan, torch.device('cpu'), KVCacheLayout.BLHNC)
        # Odd IDs are SWA. Even IDs belong to another group; broadcasts must
        # leave those bytes untouched. Each layer has an independent CPU ref.
        main = next(iter(caches.values()))
        if independent:
            main.fill_(197)
        else:
            main[[0, 2, 4, 6, 8]] = 197
        references = {}
        for i, name in enumerate(names):
            ref = torch.full_like(caches[name][1], 20 + i)
            references[name] = ref
            if runtime.swa_cache_owner_rank(name, 2) == rank:
                caches[name][[1, 3, 5, 7]] = ref
        group = NS(rank_in_group=rank, device_group=dist.group.WORLD,
                   broadcast=lambda wire, src: dist.broadcast(wire, src=src))
        # Full prefill, bounded replay sub-batch, then decode. Remote sources
        # share a slot and are revisited; owners must survive each overwrite.
        for step in range(3):
            for name in names:
                owner = runtime.swa_cache_owner_rank(name, 2)
                meta = swa_meta([1, 3, 5, 7, 999], 128, 128 if step == 0 else 64)
                expected = [1, 3, 5, 7] if step == 0 else [5, 7]
                if step == 2:
                    meta = swa_meta(decode=True)
                    meta.decode_swa_indices = torch.tensor([[5 * 32, 7 * 32, 999]])
                    meta.decode_swa_lens = torch.tensor([2])
                if rank == owner:
                    meta = NS(block_size=32, num_decode_tokens=0, num_prefill_tokens=0)
                with patch.object(runtime, 'get_forward_context', return_value=NS(attn_metadata={name: meta})), \
                     patch.object(runtime, 'get_pcp_group', return_value=group), \
                     patch.object(torch.cuda, 'is_current_stream_capturing', return_value=False):
                    runtime.broadcast_kv_source_cache(name, owner, caches[name])
                for block in expected:
                    assert torch.equal(caches[name][block], references[name])
                for own in names:
                    if runtime.swa_cache_owner_rank(own, 2) == rank:
                        assert all(torch.equal(caches[own][b], references[own]) for b in [1, 3, 5, 7])
                assert torch.all(main[[0, 2, 4, 6, 8]] == 197)
                if independent:
                    assert torch.all(main == 197)
        assert not torch.cuda.is_initialized()
    finally:
        runtime.register_transport_regions({})
        dist.destroy_process_group()


def test_real_gloo_swa_owner_without_queries_and_reused_scratch(tmp_path):
    import torch.multiprocessing as mp
    mp.spawn(_swa_peer, args=((tmp_path / 'swa_gloo').as_uri(),), nprocs=2, join=True)


@pytest.mark.parametrize('rank', [0, 1])
@pytest.mark.parametrize('fp4', [False, True])
def test_native_replay_full_batch_write_then_trimmed_write(rank, fp4):
    from vllm.forward_context import get_forward_context, override_forward_context
    from vllm.models.deepseek_v41 import attention as common
    from vllm.models.deepseek_v41.decoder_replay_layers import DecoderReplayLayers, ReplayBatch
    from hcu_vllm_das.v41_pcp_layer_split import HcuV41PcpAttention
    from hcu_vllm_das.fp4_indexer import Fp4LayerSplitPcpAttention

    obj = object.__new__(Fp4LayerSplitPcpAttention if fp4 else HcuV41PcpAttention)
    torch.nn.Module.__init__(obj)
    obj._initialize_q_rope = obj._prepare_q_rope = None
    name = 'model.layers.20.attn.swa_cache'  # cyclic SWA owner is 0, not main owner 3
    obj.swa_cache_layer = NS(prefix=name, kv_cache=torch.zeros(8, 32, 4, dtype=torch.uint8), block_size=32)
    obj.use_pcp = True
    obj.rotary_emb = NS(cos_sin_cache=torch.zeros(1))
    obj.kv_mxfp8 = False
    obj.compressor = None
    obj._run_parallel_input_projections = lambda hidden: (hidden, None, None)
    obj._split_qkv_and_norm = lambda values: (None, None, values)
    original = NS(cache_slot_mapping=torch.arange(8) + 32, slot_mapping=torch.arange(4) + 32,
                  cache_positions=torch.arange(8), block_size=32)
    replay = copy.copy(original)
    replay.cache_slot_mapping = torch.tensor([38, 39, -1, -1])
    replay.cache_positions = torch.arange(4)
    outer = NS(attn_metadata={name: original})
    inner = NS(attn_metadata={name: replay})
    writes, gathers = [], []

    def insert(kv, cache, slots, *args):
        writes.append(slots.clone())
        for slot in slots.tolist():
            if slot >= 0:
                cache[slot // 32, slot % 32].fill_(71)

    def gather(tensor, dim):
        gathers.append(tensor.shape[0])
        return torch.cat((tensor, tensor), dim=dim)

    def trimmed(hidden):
        # Execute the real owned fused-preparation wrapper; GPU Q/KV op is
        # doubled, checking the exact metadata sent to the inherited method.
        obj._fused_qnorm_rope_kv_insert(hidden, hidden, torch.arange(len(hidden)),
                                       get_forward_context().attn_metadata)
        return (hidden,)

    def fused(self, q, kv, positions, metadata):
        slots = metadata[name].cache_slot_mapping
        insert(kv, self.swa_cache_layer.kv_cache, slots)
        assert metadata[name].slot_mapping is original.slot_mapping
        return q

    runner = DecoderReplayLayers(128, trimmed,
        lambda hidden: obj.forward_kv(torch.arange(len(hidden)), hidden), {name}, name)
    runner.replay_batch = ReplayBatch(torch.tensor([2, 3]), inner)
    with patch.object(runtime, '_state', (2, (2, 8, 14, 20), rank)), \
         patch.object(common, 'get_pcp_group', return_value=NS(all_gather=gather)), \
         patch.object(common.DeepseekV4Attention, '_fused_qnorm_rope_kv_insert', fused), \
         patch.object(torch.ops._C, 'fused_deepseek_v4_kv_rope_insert', insert, create=True), \
         patch('vllm.models.deepseek_v41.decoder_replay_layers.in_piecewise_cudagraph', return_value=False), \
         override_forward_context(outer):
        hidden = torch.arange(16).reshape(4, 4).float()
        output, = runner(hidden)
        assert get_forward_context() is outer
    assert gathers == [4]  # non-owner still joins the native full-batch gather
    assert len(writes) == 2
    assert original.cache_slot_mapping.tolist() == list(range(32, 40))
    assert replay.cache_slot_mapping.tolist() == [38, 39, -1, -1]
    if rank == 0:
        assert writes[0].tolist() == list(range(32, 40))
        assert writes[1].tolist() == [38, 39, -1, -1]
        assert torch.all(obj.swa_cache_layer.kv_cache[1, :8] == 71)
    else:
        assert all(torch.all(slots == -1) for slots in writes)
        assert torch.count_nonzero(obj.swa_cache_layer.kv_cache) == 0
    assert torch.equal(output[2:], hidden[2:]) and torch.count_nonzero(output[:2]) == 0


@pytest.mark.parametrize('split,rank', [(False, 1), (True, 0), (True, 1)])
@pytest.mark.parametrize('fp4', [False, True])
@pytest.mark.parametrize('q_rope', [False, True])
def test_native_fused_preparation_keeps_gather_and_query_work(split, rank, fp4, q_rope):
    from vllm import distributed
    from vllm.models.deepseek_v41 import attention as common
    from hcu_vllm_das.v41_pcp_layer_split import HcuV41PcpAttention
    from hcu_vllm_das.fp4_indexer import Fp4LayerSplitPcpAttention

    obj = object.__new__(Fp4LayerSplitPcpAttention if fp4 else HcuV41PcpAttention)
    torch.nn.Module.__init__(obj)
    name = 'model.layers.20.attn.swa_cache'
    obj.swa_cache_layer = NS(prefix=name, kv_cache=torch.zeros(4, 32, 584, dtype=torch.uint8))
    obj.use_pcp = True
    obj.rotary_emb = NS(cos_sin_cache=torch.zeros(1), is_neox_style=False)
    obj.kv_mxfp8 = False
    obj.kv_cache_torch_dtype = torch.uint8
    obj.rope_head_dim = 64
    obj.n_local_heads = obj.padded_heads = 2
    obj.eps = 1e-6
    meta = NS(block_size=32, num_prefill_tokens=2, num_decode_tokens=0, slot_mapping=torch.tensor([32, 33]),
              cache_slot_mapping=torch.tensor([32, 33, 34, -1]), cache_positions=torch.arange(4))
    calls = []

    def gather(kv, dim):
        calls.append('gather')
        return torch.cat((kv, kv), dim=dim)

    def insert(kv, cache, slots, *args):
        calls.append('insert')
        assert kv.shape[0] == 4
        assert slots.tolist() == ([-1] * 4 if split and rank != 0 else [32, 33, 34, -1])

    def query_op(q, kv, cache, slots, *args):
        calls.append('query')
        assert slots.numel() == 0  # native PCP already handled the KV insert
        return q + 1

    def prepare(q, positions, table):
        def rotate():
            calls.append('hcu_query')
            q.add_(1)
        return rotate

    obj._initialize_q_rope = None
    obj._prepare_q_rope = prepare if q_rope else None
    with patch.object(runtime, '_state', (2, (2, 8, 14, 20), rank) if split else None), \
         patch.object(common, 'get_pcp_group', return_value=NS(all_gather=gather)), \
         patch.object(distributed, 'get_pcp_group', return_value=NS(all_gather=gather)), \
         patch.object(torch.ops._C, 'fused_deepseek_v4_kv_rope_insert', insert, create=True), \
         patch.object(torch.ops._C, 'fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert', query_op, create=True):
        q = torch.ones(2, 2, 4)
        result = obj._fused_qnorm_rope_kv_insert(q, torch.ones(2, 4), torch.arange(2), {name: meta})
    assert calls == ['gather', 'insert', 'hcu_query' if q_rope else 'query']
    assert torch.equal(result, torch.full_like(q, 2))
    assert meta.cache_slot_mapping.tolist() == [32, 33, 34, -1]


def test_old_pd_owner_protocol_rejected_before_transfer():
    from hcu_vllm_das import mooncake_layer_split as connector
    with pytest.raises(ValueError, match='owner-aware'):
        connector.validate_peers({'protocol': 'hcu-v41-pcp-owners-v1', 'workers': []}, {})
