"""PCP speculative ring retention through native compressor forward, CPU/Gloo."""
from datetime import timedelta
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from replay_v41_static_eplb import cfg as static_cfg
from hcu_vllm_das import pcp_dspark as owned
from hcu_vllm_das import static_eplb_runtime as static
from hcu_vllm_das import v41_pcp
from hcu_vllm_das.models import v41_pcp as models
from vllm.models.deepseek_v41 import compressor as native
from vllm.v1.kv_cache_interface import CircularBufferSpec


def config(width=8, speculative=5):
    cfg = static_cfg(width=width)
    cfg.speculative_config = NS(
        method='dspark', num_speculative_tokens=speculative,
        draft_model_config=NS(enforce_eager=True, hf_config=NS(
            model_type='deepseek_v41', sample_from_anchor=True,
            sliding_window=128, layer_types=['sliding_attention'] * 3)))
    cfg.scheduler_config = NS(max_num_seqs=32, max_num_batched_tokens=32768)
    cfg.num_speculative_tokens = speculative
    cfg.compilation_config = NS(static_forward_context={})
    return cfg


def builder(width=8, capacity=8, speculative=5):
    spec = CircularBufferSpec(
        block_size=capacity, num_kv_heads=1, head_size=1024,
        head_size_v=0, dtype=torch.float32)
    return owned.HcuPcpDSparkCompressorMetadataBuilder(
        spec, ['test.state_cache'], config(width, speculative), torch.device('cpu'))


def test_static_target_and_native_draft_are_separate(monkeypatch):
    monkeypatch.setenv('HCU_VLLM_PCP_IMPL', 'native')
    cfg = config()
    assert v41_pcp.validate_config(cfg)
    assert static.validate_config(cfg)
    assert static.moe_eplb_kwargs(cfg, 'model.layers.2.ffn.experts') == {
        'enable_eplb': True, 'num_redundant_experts': 2}
    assert static.moe_eplb_kwargs(cfg, 'model.layers.4.ffn.experts') == {}
    assert models.select_attention(cfg) is owned.HcuV41PcpDSparkAttention
    cfg.speculative_config = None
    assert models.select_attention(cfg) is v41_pcp.HcuV41PcpAttention
    # The namespace substitutions are private and keep native executable code.
    assert native.DeepseekCompressor.__init__.__globals__['CompressorStateCache'] is (
        native.CompressorStateCache)
    assert owned.HcuPcpDSparkCompressor.__init__.__code__ is (
        native.DeepseekCompressor.__init__.__code__)
    assert owned.HcuV41PcpDSparkAttention.__mro__.index(owned._PcpDSparkAttentionBase) < (
        owned.HcuV41PcpDSparkAttention.__mro__.index(owned.attention.DeepseekV4Attention))


@pytest.mark.parametrize('method', ['mtp', 'eagle3', 'dflash'])
def test_other_speculators_remain_rejected(method):
    cfg = config()
    cfg.speculative_config.method = method
    with pytest.raises(ValueError, match='DSpark'):
        static.validate_config(cfg)


def test_draft_capacity_and_replay_window_are_checked():
    cfg = config()
    cfg.scheduler_config.max_num_batched_tokens = 16
    with pytest.raises(ValueError, match='capacity'):
        v41_pcp.validate_config(cfg)
    assert cfg.scheduler_config.max_num_batched_tokens == 16
    cfg = config()
    cfg.parallel_config.tensor_parallel_size = 2
    with pytest.raises(ValueError, match='TP1'):
        v41_pcp.validate_config(cfg)
    cfg = config()
    cfg.parallel_config.use_ubatching = False
    backbone = NS(start_layer=0, end_layer=40, use_sequence_parallel=False,
                  config=NS(num_hidden_layers=40, sliding_window=128, engram_layer_ids=[]))
    gate = models.HcuV41EplbBackbone._decoder_replay_supported
    assert gate(backbone, cfg, 20)
    cfg.speculative_config.draft_model_config.hf_config.sliding_window = 256
    assert not gate(backbone, cfg, 20)


@pytest.mark.parametrize('speculative,capacity', [(1, 8), (5, 8), (13, 16)])
def test_public_metadata_backend_uses_real_ring_capacity(speculative, capacity):
    cfg = config(speculative=speculative)
    with patch.object(native, 'get_current_vllm_config', return_value=cfg):
        cache = owned.HcuPcpDSparkStateCache(1024, torch.float32, 'test.state_cache')
    assert cache.block_size == capacity
    backend = cache.get_attn_backend()
    instance = backend.get_builder_cls()(
        cache.get_kv_cache_spec(cfg), [cache.prefix], cfg, torch.device('cpu'))
    assert instance.capacity == capacity
    assert instance.vllm_config is cfg and cfg.speculative_config is not None
    # Core's original restriction continues to apply to its own backend.
    with pytest.raises(AssertionError):
        native.CompressorMetadataBuilder(
            instance.kv_cache_spec, [cache.prefix], cfg, torch.device('cpu'))


def test_undersized_ring_is_rejected():
    with pytest.raises(ValueError, match='verification window'):
        builder(capacity=4)


def inputs(chunks, padded, step, capacity=8):
    positions = torch.zeros(padded, dtype=torch.long)
    slots = torch.full((padded,), -1, dtype=torch.long)
    raw = torch.zeros(padded, 1024)
    boundaries = [0]
    values = {}
    cursor = 0
    for block, token_positions in chunks:
        for position in token_positions:
            value = float(step * 100000 + block * 10000 + position)
            positions[cursor] = position
            slots[cursor] = block * capacity + position % capacity
            raw[cursor].fill_(value)
            values[block, position] = value
            cursor += 1
        boundaries.append(cursor)
    metadata = native.CompressorMetadata(
        slots, torch.tensor(boundaries, dtype=torch.int32), torch.zeros(padded, dtype=torch.int32))
    return metadata, positions, raw, values


def expected_keys(chunks, padded, capacity=8):
    rows = min(padded, 64)
    keys = torch.full((rows, capacity), -1, dtype=torch.long)
    index = 0
    for block, token_positions in chunks:
        tail = token_positions[-capacity:]
        if tail:
            keys[index, capacity-len(tail):] = torch.tensor(
                [(block << 32) | position for position in tail])
            index += 1
    return keys.flatten()


def run_native_forward(instance, metadata, positions, raw, ring, gathered_raw, reference):
    class Group:
        def all_gather(self, rows, dim=0):
            return gathered_raw

    obj = native.DeepseekCompressor.__new__(native.DeepseekCompressor)
    torch.nn.Module.__init__(obj)
    obj.pcp_size = instance.pcp_size
    obj.state_cache = NS(prefix='state', kv_cache=ring)
    obj.head_dim = 512
    obj.compress_ratio = 2
    obj.norm = NS(weight=torch.ones(512))
    obj.rms_norm_eps = 1e-5

    def compress_double(raw, positions, state, slots, starts, req_ids, *args,
                        prev_rows, prev_row_indices):
        for index in range(starts.numel()-1):
            start, end = int(starts[index]), int(starts[index+1])
            if start == end or slots[start] < 0 or positions[start] % 2 != 1:
                continue
            block, position = int(slots[start]) // instance.capacity, int(positions[start])
            previous = int(prev_row_indices[index])
            row = prev_rows[previous] if previous >= 0 else state[
                block, (position-1) % instance.capacity]
            assert torch.all(row == reference[block, position-1])

    def save_double(rows, slots, state):
        for index, slot in enumerate(slots.tolist()):
            if slot >= 0:
                state.view(-1, 1024)[slot].copy_(rows[index])

    with (patch.object(native, 'get_pcp_group', return_value=Group()),
          patch.object(native, 'get_forward_context', return_value=NS(attn_metadata={'state': metadata})),
          patch.object(native, 'fused_save_compress_norm', compress_double),
          patch.object(native, 'save_ring_rows', save_double)):
        obj.forward(raw, positions)


def step_all_ranks(chunks_by_rank, rings, reference, step, capacity=8, speculative=5):
    width = len(chunks_by_rank)
    padded = max(1, max(sum(len(p) for _, p in chunks) for chunks in chunks_by_rank))
    all_keys = torch.cat([expected_keys(c, padded, capacity) for c in chunks_by_rank])
    data = [inputs(c, padded, step, capacity) for c in chunks_by_rank]
    current = dict(reference)
    for _, _, _, values in data:
        current.update(values)
    class Group:
        def all_gather(self, keys, dim=0):
            return all_keys
    for chunks, (metadata, positions, _, _) in zip(chunks_by_rank, data):
        instance = builder(width, capacity, speculative)
        with patch.object(native, 'get_pcp_group', return_value=Group()):
            instance._link_pcp_chunks(metadata, len(chunks), positions)
    gathered_raw = torch.cat([raw[m.gather_tokens] for m, _, raw, _ in data])
    for index, (metadata, positions, raw, _) in enumerate(data):
        run_native_forward(builder(width, capacity, speculative), metadata, positions, raw,
                           rings[index], gathered_raw, current)
    for block in {block for chunks in chunks_by_rank for block, _ in chunks}:
        positions = [p for b, p in current if b == block]
        newest = max(positions)
        for position in range(max(0, newest-capacity+1), newest+1):
            if (block, position) in current:
                for ring in rings:
                    assert torch.all(ring[block, position % capacity] == current[block, position])
    return current


@pytest.mark.parametrize('accepted,speculative,capacity',
                         [(k, 5, 8) for k in range(6)]
                         + [(k, 13, 16) for k in (0, 1, 6, 12, 13)])
def test_pcp8_rejection_retains_every_possible_predecessor(accepted, speculative, capacity):
    width, chunk = 8, 17
    prefill = []
    for rank in range(width):
        segments = []
        for index in (rank, 2*width-1-rank):
            positions = list(range(index*chunk, min((index+1)*chunk, 257)))
            if positions:
                segments.append((3, positions))
        prefill.append(segments)
    rings = [torch.full((8, capacity, 1024), -999.0) for _ in range(width)]
    reference = step_all_ranks(prefill, rings, {}, 1, capacity, speculative)
    verify = [[] for _ in range(width)]
    verify[3] = [(3, list(range(257, 258+speculative)))]
    reference = step_all_ranks(verify, rings, reference, 2, capacity, speculative)
    verify = [[] for _ in range(width)]
    verify[6] = [(3, list(range(258+accepted, 259+accepted+speculative)))]
    step_all_ranks(verify, rings, reference, 3, capacity, speculative)


def test_multiple_requests_keep_separate_histories_when_decode_owners_change():
    prefill = [[(3, list(range(rank*2, rank*2+2))),
                (4, list(range(rank*3, rank*3+3)))] for rank in range(8)]
    rings = [torch.full((8, 8, 1024), -999.0) for _ in range(8)]
    reference = step_all_ranks(prefill, rings, {}, 1)
    verify = [[] for _ in range(8)]
    verify[1] = [(3, list(range(16, 22)))]
    verify[6] = [(4, list(range(24, 30)))]
    reference = step_all_ranks(verify, rings, reference, 2)
    verify = [[] for _ in range(8)]
    verify[6] = [(3, list(range(17, 23)))]
    verify[1] = [(4, list(range(25, 31)))]
    step_all_ranks(verify, rings, reference, 3)


def test_active_request_after_empty_chunks_is_not_truncated():
    prefill = [[(3, list(range(8)))], [(3, list(range(8, 16)))]]
    rings = [torch.full((8, 8, 1024), -999.0) for _ in range(2)]
    reference = step_all_ranks(prefill, rings, {}, 1)
    # One real token with seven earlier empty chunks. Its request index is
    # greater than the padded local token count and must still retain history.
    verify = [[(3, [])] * 7 + [(3, [16])], [(3, [])] * 8]
    reference = step_all_ranks(verify, rings, reference, 2)
    rollback = [[(3, [])] * 8, [(3, [])] * 7 + [(3, [17])]]
    step_all_ranks(rollback, rings, reference, 3)


def test_zero_token_metadata_has_empty_gathers():
    instance = builder(width=2)
    metadata, positions, _, _ = inputs([(3, []), (4, [])], 0, 1)
    instance._link_pcp_chunks(metadata, 2, positions)
    assert metadata.gather_tokens.numel() == metadata.ring_write_slots.numel() == 0
    assert metadata.prev_row_indices.tolist() == [-1, -1]


def test_adaptive_verification_is_rejected():
    cfg = config()
    cfg.speculative_config.enable_adaptive_verification = True
    with pytest.raises(ValueError, match='nonadaptive'):
        owned.validate_dspark_config(cfg)


def test_fp4_constructor_uses_scoped_compressor_factory():
    from hcu_vllm_das import fp4_indexer
    from vllm.models.deepseek_v41 import attention
    assert fp4_indexer._COMMON_INIT.__globals__['DeepseekCompressor'] is fp4_indexer._compressor_factory
    assert attention.DeepseekV4Attention.__init__.__globals__['DeepseekCompressor'] is native.DeepseekCompressor


def test_fp4_compressor_factory_preserves_native_fallback(monkeypatch):
    from hcu_vllm_das import fp4_indexer
    calls = []
    monkeypatch.setattr(owned, 'HcuPcpDSparkCompressor', lambda *args, **kwargs: calls.append('pcp_dspark'))
    monkeypatch.setattr(fp4_indexer.common, 'DeepseekCompressor', lambda *args, **kwargs: calls.append('native'))
    cfg = config()
    fp4_indexer._compressor_factory(cfg, prefix='target')
    cfg.speculative_config = None
    fp4_indexer._compressor_factory(cfg, prefix='target')
    cfg.speculative_config = config().speculative_config
    cfg.parallel_config.prefill_context_parallel_size = 1
    fp4_indexer._compressor_factory(cfg, prefix='target')
    assert calls == ['pcp_dspark', 'native', 'native']


class GlooGroup:
    def all_gather(self, value, dim=0):
        outputs = [torch.empty_like(value) for _ in range(dist.get_world_size())]
        dist.all_gather(outputs, value)
        return torch.cat(outputs, dim=dim)


def gloo_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=60))
    try:
        group = GlooGroup()
        ring = torch.full((8, 8, 1024), -999.0)
        stages = [
            [[(3, list(range(3))), (3, list(range(9, 12)))],
             [(3, list(range(3, 6))), (3, list(range(6, 9)))]],
            [[(3, list(range(12, 18)))], []],
            [[], [(3, list(range(13, 19)))]],
        ]
        reference = {}
        for step, chunks_by_rank in enumerate(stages, 1):
            padded = max(sum(len(p) for _, p in c) for c in chunks_by_rank)
            for chunks in chunks_by_rank:
                reference.update(inputs(chunks, padded, step)[3])
            metadata, positions, raw, _ = inputs(chunks_by_rank[rank], padded, step)
            instance = builder(width=2)
            with patch.object(native, 'get_pcp_group', return_value=group):
                instance._link_pcp_chunks(metadata, len(chunks_by_rank[rank]), positions)
            gathered_raw = group.all_gather(raw[metadata.gather_tokens])
            run_native_forward(instance, metadata, positions, raw, ring,
                               gathered_raw, reference)
            newest = max(position for block, position in reference if block == 3)
            for position in range(max(0, newest-7), newest+1):
                assert torch.all(ring[3, position % 8] == reference[3, position])
            dist.barrier()
    finally:
        dist.destroy_process_group()


def test_real_gloo_collectives_with_empty_rank_and_rejection(tmp_path):
    rendezvous = (tmp_path / 'pcp-dspark-rendezvous').as_uri()
    mp.spawn(gloo_worker, args=(rendezvous,), nprocs=2, join=True)
