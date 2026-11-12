"""Only HCU bridge contracts; PCP scheduling and Engram stay in Core."""
from types import SimpleNamespace as NS
import inspect

import pytest
import torch
from hcu_vllm_das.dsv41 import pcp_attention as bridge
from hcu_vllm_das.models import dsv41_pcp as model
from vllm.models.deepseek_v41.amd import model as native, rocm, vl_model
from vllm.models.deepseek_v41.attention import DeepseekV4Attention


def config():
    return NS(model_config=NS(enforce_eager=True, hf_config=NS(model_type="deepseek_v41")),
        parallel_config=NS(tensor_parallel_size=1, data_parallel_size=1,
            pipeline_parallel_size=1, prefill_context_parallel_size=8,
            decode_context_parallel_size=1, enable_expert_parallel=True,
            enable_eplb=False, enable_dbo=False), use_v2_model_runner=True,
        kernel_config=NS(moe_backend="deep_gemm_mega_moe", monokernel=False),
        speculative_config=None, lora_config=None, kv_transfer_config=None,
        cache_config=NS(swa_bounded_replay=False),
        attention_config=NS(backend=None, indexer_sparse_logits=False,
                            resolve_indexer_kv_dtype=lambda _: "fp8"))


def test_only_target_combination_is_admitted():
    cfg = config()
    assert bridge.validate_config(cfg)
    cfg.kernel_config.moe_backend = "deep_gemm"
    assert bridge.validate_config(cfg)
    cfg.kernel_config.moe_backend = "deep_gemm_mega_moe"
    for owner, field, value in [(cfg, "speculative_config", object()),
        (cfg.parallel_config, "prefill_context_parallel_size", 1),
        (cfg.parallel_config, "enable_eplb", True),
        (cfg.kernel_config, "moe_backend", "triton")]:
        previous = getattr(owner, field)
        setattr(owner, field, value)
        with pytest.raises(ValueError):
            bridge.validate_config(cfg)
        setattr(owner, field, previous)


def test_native_constructors_preparation_and_engram_are_reused():
    for owned, core in [(model._DECODER_INIT, native.DeepseekV4DecoderLayer.__init__),
        (model._BACKBONE_INIT, native.DeepseekV4Model.__init__),
        (model._OUTER_INIT, vl_model.DeepseekV41ForCausalLM.__init__)]:
        assert owned.__code__ is core.__code__
        assert inspect.signature(owned) == inspect.signature(core)
    assert model._DECODER_INIT.__globals__["Engram"] is model.HcuDSV41PcpEngram
    assert model.HcuDSV41PcpEngram._create_embedding.__code__ is native.Engram._create_embedding.__code__
    assert model.HcuDSV41PcpEngramEmbedding._allocate_weights is model.native_engram.ParallelEngramEmbedding._allocate_weights
    assert model.HcuDSV41PcpEngramEmbedding.lookup is model.native_engram.ParallelEngramEmbedding.lookup
    assert model._PCP_GATHER_ROWS.__code__ is model.native_engram._gather_engram_rows.__code__
    assert model._DECODER_INIT.__globals__['MHCPreDelayedOp'] is model.HcuDSV41PcpMHCPreDelayedOp
    assert model._BACKBONE_INIT.__globals__['MHCPostOp'] is model.HcuDSV41PcpMHCPostOp
    assert model.HcuDSV41PcpMHCPreDelayedOp.__init__.__code__ is model.native_mhc.MHCPreDelayedOp.__init__.__code__
    assert model.HcuDSV41PcpMHCPreDelayedOp.forward_native is model.native_mhc.MHCPreDelayedOp.forward_native
    assert model.HcuDSV41PcpMHCPostOp.forward_native is model.native_mhc.MHCPostOp.forward_native
    assert bridge.HcuDSV41PcpAttention._prepare_and_attn is DeepseekV4Attention._prepare_and_attn
    assert bridge.HcuDSV41PcpAttention.forward is rocm.DeepseekV41ROCMAiterMLAAttention.forward
    assert bridge.HcuDSV41PcpAttention._alloc_attn_out is rocm.DeepseekV41ROCMAiterMLAAttention._alloc_attn_out
    assert bridge.HcuDSV41PcpBackend.get_builder_cls() is rocm.DeepseekV41ROCMAiterMLASparseMetadataBuilder
    assert bridge.HcuDSV41PcpBackend.supports_pcp()


@pytest.mark.parametrize("actual", [0, 2, 5])
def test_only_actual_local_queries_enter_native_attention(monkeypatch, actual):
    obj = object.__new__(bridge.HcuDSV41PcpAttention)
    torch.nn.Module.__init__(obj)
    obj.swa_cache_layer = NS(prefix="swa")
    monkeypatch.setattr(bridge, "get_forward_context", lambda: NS(
        attn_metadata={"swa": NS(num_decode_tokens=actual, num_prefill_tokens=0)}))
    calls = []
    def core(self, q, kv, positions, output):
        calls.append(q.shape[0])
        output.copy_(q * 2)
    monkeypatch.setattr(rocm.DeepseekV41ROCMAiterMLAAttention, "forward_mqa", core)
    q = torch.ones(5, 2, 4)
    out = torch.full_like(q, 999)
    obj.forward_mqa(q, torch.ones(5, 4), torch.arange(5), out)
    assert calls == ([actual] if actual else [])
    assert (out[:actual] == 2).all() and (out[actual:] == 0).all()


def test_native_rocm_serial_forward_and_bf16_allocation(monkeypatch):
    obj = object.__new__(bridge.HcuDSV41PcpAttention)
    torch.nn.Module.__init__(obj)
    obj.indexer = object()
    obj._prepare_and_attn_fn = object()
    obj.aux_stream_list = None
    obj.padded_heads, obj.head_dim = 2, 4
    monkeypatch.setattr(rocm, "get_forward_context", lambda: NS(attn_metadata={}))
    monkeypatch.setattr(rocm, "current_platform", NS(
        enable_multi_stream_overlap=lambda streams, metadata: streams is not None))
    calls = []
    def core(self, positions, hidden, scaling=None):
        calls.append(self.aux_stream_list)
        return hidden
    monkeypatch.setattr(DeepseekV4Attention, "forward", core)
    hidden = torch.empty(5, 8, dtype=torch.bfloat16)
    assert obj.forward(torch.arange(5), hidden) is hidden
    assert calls == [None]
    out = obj._alloc_attn_out(5, hidden)
    assert out.shape == (5, 2, 4) and out.dtype == torch.bfloat16


@pytest.mark.parametrize('rank', range(8))
def test_pcp_engram_shards_cover_all_hash_heads(monkeypatch, rank):
    obj = object.__new__(model.HcuDSV41PcpEngramEmbedding)
    torch.nn.Module.__init__(obj)
    obj.dp_size = obj.tp_size = 1
    monkeypatch.setattr(model, 'get_pcp_group', lambda: NS(world_size=8, rank_in_group=rank))
    assert obj._get_shard_info() == (8, rank)


@pytest.mark.parametrize('counts,rank', [([0]*8, 0), ([0,2,1,0,3,0,1,2], 0), ([0,2,1,0,3,0,1,2], 4)])
def test_pcp_engram_stages_uneven_and_empty_ranks(monkeypatch, counts, rank):
    heads, local_heads, dim = 24, 3, 2
    hashes = [torch.arange(n * heads, dtype=torch.int32).reshape(n, heads) + 1000*r for r,n in enumerate(counts)]
    slot = max(counts)
    def gather(value, dim=0):
        if value.dtype == torch.int64:
            return torch.tensor(counts, dtype=torch.int64)
        expected = torch.full((slot,heads), model.native_engram.DEAD_ID, dtype=torch.int32)
        expected[:counts[rank]] = hashes[rank]
        assert torch.equal(value, expected)
        padded = []
        for h in hashes:
            p = torch.full((slot,heads), model.native_engram.DEAD_ID, dtype=torch.int32)
            p[:len(h)] = h
            padded.append(p)
        return torch.cat(padded)
    monkeypatch.setattr(model, 'get_pcp_group', lambda: NS(world_size=8, rank_in_group=rank, all_gather=gather))
    obj = object.__new__(model.HcuDSV41PcpEngram)
    torch.nn.Module.__init__(obj)
    obj.embed_tokens = NS(n_hash_cols=heads, dim=dim)
    obj.staged_rows = torch.empty(8*8, local_heads, dim)
    obj._pcp_prepared = None
    staged = []
    monkeypatch.setattr(model.native_engram.Engram, 'prepare_embeddings', lambda self,h: staged.append(h.clone()))
    monkeypatch.setattr(model.native_engram.Engram, '_ready_rows', lambda self,n: self.staged_rows[:n])
    selected = torch.zeros(counts[rank], heads, dim)
    calls = []
    def rows(value,n):
        calls.append((value.shape,n))
        return selected
    monkeypatch.setattr(model, '_PCP_GATHER_ROWS', rows)
    obj.prepare_embeddings(hashes[rank])
    actual = obj._ready_rows(counts[rank])
    assert actual.shape == (counts[rank],heads,dim)
    assert len(staged) == bool(slot)
    assert calls == ([(torch.Size([slot*8,local_heads,dim]),counts[rank])] if slot else [])
    with pytest.raises(RuntimeError, match='matching prepare'):
        obj._ready_rows(counts[rank])


@pytest.mark.parametrize('rank', range(8))
def test_native_engram_constructor_and_loader_preserve_pcp_head_buckets(monkeypatch, rank):
    core = model.native_engram
    monkeypatch.setattr(model, 'get_pcp_group', lambda: NS(world_size=8, rank_in_group=rank))
    monkeypatch.setattr(core, 'get_engram_dp_size', lambda: 1)
    monkeypatch.setattr(core, 'get_tensor_model_parallel_world_size', lambda: 1)
    monkeypatch.setattr(core, 'is_uva_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'get_device_properties', lambda _: NS(multi_processor_count=1))
    monkeypatch.setattr(torch.accelerator, 'current_device_index', lambda: 0)
    heads = tuple(range(1,25))
    with torch.device('meta'):
        obj = model.HcuDSV41PcpEngramEmbedding(sum(heads), 256, heads, cpu_offload=True)
    start = sum(heads[:3*rank])
    stop = sum(heads[:3*(rank+1)])
    assert obj.head_start == rank*3 and obj.part_n_hash_cols == 3
    assert obj.vocab_start_idx == start and obj.vocab_end_idx == stop
    assert obj.weight.shape == (stop-start,256)
    assert obj.weight_scale_inv.shape == (stop-start,8)
    assert obj.dp_size == obj.tp_size == 1
    param = torch.nn.Parameter(torch.empty(stop-start,4), requires_grad=False)
    param.engram_vocab_start = start
    loaded = torch.arange(sum(heads)*4).reshape(sum(heads),4).float()
    obj._weight_loader(param, loaded)
    assert torch.equal(param, loaded[start:stop])


@pytest.mark.parametrize('rank', range(8))
@pytest.mark.parametrize('slot,count', [(3,3),(3,0),(3,1)])
def test_core_head_exchange_selects_pcp_local_tokens(monkeypatch, rank, slot, count):
    size, local_heads, dim = 8,3,2
    complete = torch.arange(size*slot*size*local_heads*dim).reshape(size,slot*size,local_heads,dim).float()
    group = NS(world_size=size,rank_in_group=rank,all_gather=lambda staged,dim: complete.flatten(0,1))
    monkeypatch.setitem(model._PCP_GATHER_ROWS.__globals__, 'get_engram_dp_group', lambda: group)
    def select(gathered,out,group_tokens,start,width):
        # Independent logical reference: concatenate the rank-owned heads.
        logical = gathered.reshape(size,group_tokens,local_heads,dim).permute(1,0,2,3).reshape(group_tokens,size*local_heads,dim)
        out.copy_(logical[start:start+len(out)])
    monkeypatch.setitem(model._PCP_GATHER_ROWS.__globals__, '_engram_select_rows', select)
    staged = complete[rank]
    actual = model._PCP_GATHER_ROWS(staged,count)
    expected = torch.cat([complete[r,rank*slot:rank*slot+count] for r in range(size)],dim=1)
    assert torch.equal(actual,expected)
