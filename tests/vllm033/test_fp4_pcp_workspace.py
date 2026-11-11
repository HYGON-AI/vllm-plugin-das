"""CPU integration: native PCP constructors, FP4 row windows and lifetimes.

GPU kernels/transport are explicit doubles. Passing does not certify device
quantization, collective execution, full-model accuracy or OOM elimination.
"""
from types import SimpleNamespace as NS
import weakref

import pytest
import torch

from test_fp4_full_chain import chain, make_attention, bind_cache, metadata, query
from hcu_adapter.fp4_indexer import validate_config
from hcu_vllm_das import fp4_indexer as impl, fp4_indexer_lightop as reader
from hcu_vllm_das import v41_pcp, v41_indexer as scorer


def pcp_config(cfg):
    cfg.parallel_config.prefill_context_parallel_size = 8
    cfg.parallel_config.enable_expert_parallel = True
    cfg.parallel_config.all2all_backend = 'deepep_high_throughput'
    cfg.attention_config.backend = None
    return cfg


def test_pcp8_without_speculation_and_fp8_route_unchanged(chain, monkeypatch):
    cfg = pcp_config(chain.cfg)
    assert validate_config(cfg) and v41_pcp.validate_config(cfg)
    from hcu_vllm_das.models import v41_pcp as models
    assert models._fp4_attention(cfg) is impl.Fp4PcpAttention
    assert models._FP4_PCP_DECODER_INIT.__globals__['Engram'] is models.HcuV41PcpEngram
    monkeypatch.setenv('HCU_VLLM_FP4_INDEXER_IMPL', 'stock')
    cfg.attention_config.resolve_indexer_kv_dtype = lambda _: 'fp8'
    assert validate_config(cfg) is None
    assert models.select_attention(cfg) is v41_pcp.HcuV41PcpAttention


@pytest.mark.parametrize('field,value', [
    ('data_parallel_size', 8), ('tensor_parallel_size', 2),
    ('prefill_context_parallel_size', 4), ('decode_context_parallel_size', 2),
    ('enable_eplb', True), ('enable_expert_parallel', False),
    ('all2all_backend', 'deepep_low_latency')])
def test_pcp_domain_remains_narrow(chain, field, value):
    cfg = pcp_config(chain.cfg)
    setattr(cfg.parallel_config, field, value)
    with pytest.raises(ValueError):
        validate_config(cfg)


@pytest.mark.parametrize('field,value', [
    ('speculative_config', NS(method='dspark', num_speculative_tokens=5,
                             enable_adaptive_verification=False)),
    ('kv_transfer_config', object())])
def test_pcp_does_not_admit_draft_or_transfer(chain, field, value):
    cfg = pcp_config(chain.cfg)
    setattr(cfg, field, value)
    with pytest.raises(ValueError):
        validate_config(cfg)


@pytest.mark.parametrize('ratio', [1, 2])
def test_real_pcp_attention_keeps_native_schedule_and_source_cache(chain, ratio):
    cfg = pcp_config(chain.cfg)
    cfg.model_config.hf_config.compress_ratios = [ratio, 0]
    attention = impl.Fp4PcpAttention(
        vllm_config=cfg, prefix='model.layers.0.self_attn',
        topk_indices_buffer=torch.full((16, 4), -77, dtype=torch.int32))
    assert attention.use_pcp and attention.aux_stream_list is None
    assert attention.forward.__func__ is impl.common.DeepseekV4Attention.forward
    assert attention.backend_cls.supports_pcp()
    assert attention.swa_backend_cls.supports_pcp()
    assert isinstance(attention.indexer, impl.Fp4Indexer)
    assert attention.indexer.indexer_op.use_pcp
    assert attention.indexer.k_cache.head_dim == 68
    assert attention.indexer.k_cache.get_kv_cache_spec(cfg).num_states == 64 // ratio


def test_prefill_subchunks_keep_windows_candidates_and_release_scores(chain, monkeypatch):
    layer = make_attention(chain).indexer.indexer_op
    bind_cache(layer.k_cache)
    meta = metadata(chain, layer.k_cache, prefill=True)
    chunk = meta.prefill.chunks[0]
    # Two independent requests, nonzero window starts, partial final subchunk.
    meta.num_prefill_tokens = 5
    chunk.token_end = 5
    chunk.cu_seqlen_ks = torch.tensor([0, 0, 2, 2, 2], dtype=torch.int32)
    chunk.cu_seqlen_ke = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32)
    layer.fp4_query_chunk_rows = 2
    layer.candidate_blocks = torch.arange(10, dtype=torch.int32).reshape(5, 2)
    layer.candidate_block_size = 2
    layer.candidate_write = False
    original_select = scorer._select
    seen, refs = [], []
    def prefill(q, kv, weights, ks, ke, clean, width, *args):
        assert not refs or refs[-1]() is None, 'previous logits still live'
        seen.append((ks.clone(), ke.clone(), weights.clone()))
        result = torch.full((len(ks), width), -float('inf'))
        for i, count in enumerate((ke - ks).tolist()):
            result[i, :count] = torch.arange(count, dtype=torch.float32)
        refs.append(weakref.ref(result))
        return result
    banks = []
    def select(logits, starts, ends, out, bank, block, write):
        banks.append(bank.clone())
        # Independently check candidate row alignment, then run production topk.
        assert block == 2 and not write
        original_select(logits, starts, ends, out, None, block, write)
    monkeypatch.setattr(reader, '_providers', reader.Providers(
        reader.providers().gather, prefill, reader.providers().decode, object()))
    monkeypatch.setattr(scorer, '_select', select)
    weights = torch.arange(5 * 32, dtype=torch.float32).reshape(5, 32)
    output = reader.execute(layer, torch.empty(5, 128), query(5), weights)
    assert [len(x[0]) for x in seen] == [2, 2, 1]
    assert torch.equal(torch.cat([x[0] for x in seen]), chunk.cu_seqlen_ks)
    assert torch.equal(torch.cat([x[1] for x in seen]), chunk.cu_seqlen_ke)
    assert torch.equal(torch.cat([x[2] for x in seen]), weights)
    assert torch.equal(torch.cat(banks), layer.candidate_blocks)
    assert output[:5].tolist() == [[0,-1,-1,-1], [0,1,-1,-1],
                                   [0,-1,-1,-1], [0,1,-1,-1], [0,1,2,-1]]
    assert all(ref() is None for ref in refs)


def test_profile_uses_frozen_capacity_without_current_config(chain, monkeypatch):
    layer = make_attention(chain).indexer.indexer_op
    assert layer.fp4_decode_rows == 4
    chain.cfg.scheduler_config.max_num_seqs = 99999
    chain.context.attn_metadata = None
    import vllm.config
    from vllm.v1.worker import workspace
    def forbidden():
        raise AssertionError('profile must not consult construction config')
    monkeypatch.setattr(vllm.config, 'get_current_vllm_config', forbidden)
    # The estimate must raise the profile peak without retaining a duplicate
    # persistent workspace that runtime does not consume.
    monkeypatch.setattr(workspace, 'current_workspace_manager', forbidden)
    calls, allocations = [], []
    original_empty = torch.empty
    hidden = original_empty(16, 128)
    def allocate(*args, **kwargs):
        calls.append((args, kwargs))
        tensor = original_empty(*args, **kwargs)
        allocations.append(weakref.ref(tensor))
        return tensor
    monkeypatch.setattr(torch, 'empty', allocate)
    reader.execute(layer, hidden, None, None)
    expected = reader.workspace_bytes(
        query_rows=16, decode_rows=4, max_model_len=layer.max_model_len,
        total_seq_lens=layer.max_total_seq_len, topk_tokens=4, num_query_heads=32,
        max_logits_bytes=layer.fp4_max_logits_bytes, physical_page_size=64)
    assert calls == [(((expected,),), dict(dtype=torch.uint8, device=hidden.device))]
    assert allocations[0]() is None
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('ratio', [1, 2])
def test_pcp_reindex_layer_shares_source_fp4_cache(chain, ratio):
    cfg = pcp_config(chain.cfg)
    cfg.model_config.hf_config.compress_ratios = [ratio, ratio]
    cfg.model_config.hf_config.index_source_layer_ids = [0, 1]
    layers = [impl.Fp4PcpAttention(vllm_config=cfg,
              prefix=f'model.layers.{i}.self_attn',
              topk_indices_buffer=torch.empty(16, 4, dtype=torch.int32))
              for i in range(2)]
    assert layers[0].indexer.owns_k and not layers[1].indexer.owns_k
    assert layers[0].indexer.k_cache is layers[1].indexer.k_cache
    assert not hasattr(layers[1].indexer, 'wk')


@pytest.mark.parametrize('actual', [0, 1, 3])
def test_pcp_fp4_preserves_empty_rank_and_padding(chain, monkeypatch, actual):
    attn = object.__new__(impl.Fp4PcpAttention)
    torch.nn.Module.__init__(attn)
    attn.swa_cache_layer = NS(prefix='swa')
    chain.context.attn_metadata = {'swa': NS(num_decode_tokens=actual, num_prefill_tokens=0)}
    monkeypatch.setattr(v41_pcp, 'get_forward_context', lambda: chain.context)
    calls = []
    def native(self, q, kv, positions, output):
        calls.append(q.shape[0])
        output.copy_(q * 2)
    monkeypatch.setattr(impl.rocm.DeepseekV41ROCMAiterMLAAttention, 'forward_mqa', native)
    q = torch.ones(5, 1, 128)
    output = torch.full_like(q, 777)
    attn.forward_mqa(q, torch.ones(5, 128), torch.arange(5), output)
    assert calls == ([actual] if actual else [])
    assert output[:actual].eq(2).all() and output[actual:].eq(0).all()


@pytest.mark.parametrize('ratio', [1, 2])
def test_eight_rank_native_preparation_keeps_global_k_local_q(chain, monkeypatch, ratio):
    """Run the real common preparation with a recorded rank-major gather.

    The producer uses the real FP4 method; its final Triton store is captured
    by the existing fixture, so this verifies positions/slots, not GPU math.
    """
    common = impl.common
    cfg = pcp_config(chain.cfg)
    cfg.model_config.hf_config.compress_ratios = [ratio, 0]
    rows = [torch.tensor([r * 2, r * 2 + 1, (15-r)*2, (15-r)*2+1]) for r in range(8)]
    global_positions = torch.cat(rows)
    global_latent = global_positions[:, None].expand(-1, 128).bfloat16()
    global_slots = torch.where((global_positions + 1) % ratio == 0,
                               global_positions // ratio, -1)
    source = impl.Fp4PcpAttention(vllm_config=cfg, prefix='model.layers.0.self_attn',
              topk_indices_buffer=torch.empty(16, 4, dtype=torch.int32)).indexer
    source.k_cache.bind_kv_cache(torch.zeros((2, 1, 64 // ratio, 68), dtype=torch.uint8))
    for rank, positions in enumerate(rows):
        local = positions[:, None].expand(-1, 128).bfloat16()
        monkeypatch.setattr(source.wk, 'forward', lambda x: (x, None))
        events = []
        attn = object.__new__(impl.Fp4PcpAttention)
        torch.nn.Module.__init__(attn)
        attn.use_pcp = True
        attn.aux_stream_list = None
        attn.ln_events = [None, None]
        attn.swa_cache_layer = NS(prefix='swa')
        attn.rotary_emb = attn.indexer_rotary_emb = chain.rope
        attn.n_local_heads, attn.head_dim = 1, 128
        attn._wq_b_proj = lambda qr, scale: qr
        attn._fused_qnorm_rope_kv_insert = lambda q, kv, pos, meta: q
        class Compressor:
            def __call__(self, scores, pos):
                events.append('compress')
                return local
            def insert_cache(self, latent, pos, rope):
                events.append('main_store')
                assert torch.equal(latent, global_latent)
                assert torch.equal(pos, global_positions)
        attn.compressor = Compressor()
        def forward_q(qr, qr_scale, weights, pos, rope):
            events.append('query')
            assert torch.equal(qr, local) and torch.equal(pos, positions)
            return query(4), None, weights
        monkeypatch.setattr(source, 'forward_q', forward_q)
        attn.indexer = source
        def gather(x, dim):
            events.append('gather')
            assert dim == 0 and torch.equal(x, local)
            return global_latent
        def finish(hidden, iq, iqs, iw, q, kv, pos, out):
            events.append('attention')
            assert iq.bf16.shape[0] == 4 and torch.equal(pos, positions)
        attn._sparse_indexer_and_attn = finish
        chain.context.attn_metadata = {
            'swa': NS(cache_positions=global_positions),
            source.k_cache.prefix: NS(slot_mapping=global_slots, max_seq_len=4096)}
        monkeypatch.setattr(common, 'get_forward_context', lambda: chain.context)
        monkeypatch.setattr(common, 'get_pcp_group', lambda: NS(all_gather=gather))
        common.DeepseekV4Attention._prepare_and_attn(
            attn, local, local, local, None, local, torch.ones(4,32), positions,
            torch.empty(4,1,128))
        grid, args, options = chain.kernel.calls[-1]
        assert grid == (32,)
        assert torch.equal(args[0], global_latent)
        assert args[2] is global_positions and args[8] is global_slots
        assert options['USE_FP4'] and options['COMPRESS_RATIO'] == ratio
        assert events == ['compress', 'gather', 'query', 'main_store', 'attention']
