"""PCP behavior after unifying the model and adapter-owned attention.

Transport/device kernels are doubles. Constructors, loaders and the owned
Engram prepare/restore logic are real; this is not GPU acceptance.
"""
from types import SimpleNamespace as NS
import inspect

import pytest
import torch
from hcu_vllm_das import v41_pcp as bridge, v41_engram as engram
from hcu_vllm_das.models import v41_pcp as model
from vllm.models.deepseek_v41.amd import model as native, rocm, vl_model
from vllm.model_executor.layers import mhc
from vllm.models.deepseek_v41.attention import DeepseekV4Attention
from test_pcp_merge_contracts import config


def test_attention_admission_preserves_existing_widths_and_exclusions():
    for width in (2, 4, 8, 16):
        assert bridge.validate_config(config(width))
    assert not bridge.validate_config(config(1))
    cfg = config()
    for owner, field, value in [
        (cfg, "speculative_config", object()),
        (cfg.parallel_config, "enable_eplb", True),
        (cfg.model_config, "enforce_eager", False),
        (cfg.attention_config, "indexer_sparse_logits", True),
    ]:
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
    assert model._DECODER_INIT.__globals__["Engram"] is engram.HcuV41PcpEngram
    assert model._DECODER_INIT.__globals__["DeepseekV4MoE"] is model.HcuV41PcpMoE
    assert engram.HcuV41PcpEngram._create_embedding.__code__ is engram.native.Engram._create_embedding.__code__
    assert engram.HcuV41PcpEmbedding.lookup is engram.native.ParallelEngramEmbedding.lookup
    assert model._DECODER_INIT.__globals__["MHCPreDelayedOp"] is mhc.MHCPreDelayedOp
    assert model._BACKBONE_INIT.__globals__["MHCPostOp"] is mhc.MHCPostOp
    assert bridge.HcuV41PcpAttention._prepare_and_attn is DeepseekV4Attention._prepare_and_attn
    assert bridge.HcuV41PcpAttention.forward is DeepseekV4Attention.forward
    assert bridge.HcuV41PcpBackend.get_builder_cls() is rocm.DeepseekV41ROCMAiterMLASparseMetadataBuilder
    assert bridge.HcuV41PcpBackend.supports_pcp()


@pytest.mark.parametrize("actual", [0, 2, 5])
def test_only_actual_local_queries_enter_native_attention(monkeypatch, actual):
    obj = object.__new__(bridge.HcuV41PcpAttention)
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


def test_native_serial_forward_and_bf16_allocation():
    obj = object.__new__(bridge.HcuV41PcpAttention)
    torch.nn.Module.__init__(obj)
    obj.padded_heads, obj.head_dim = 2, 4
    calls = []
    hidden = torch.ones(5, 8, dtype=torch.bfloat16)
    obj._run_parallel_input_projections = lambda value: (value, None, None)
    obj._split_qkv_and_norm = lambda value: (value, None, value)
    def prepare(*args):
        calls.append("prepare")
        args[-1].fill_(3)
    obj._prepare_and_attn_fn = prepare
    obj._o_proj = lambda value, positions: value
    out = obj.forward(torch.arange(5), hidden)
    assert calls == ["prepare"]
    assert out.shape == (5, 2, 4) and out.dtype == torch.bfloat16
    assert (out == 3).all()


@pytest.mark.parametrize("rank", range(8))
def test_pcp_engram_shards_cover_all_hash_heads(monkeypatch, rank):
    obj = object.__new__(engram.HcuV41PcpEmbedding)
    torch.nn.Module.__init__(obj)
    obj.dp_size = obj.tp_size = 1
    obj._hcu_pcp_size, obj._hcu_pcp_rank = 8, rank
    monkeypatch.setattr(engram, "get_tensor_model_parallel_rank", lambda: 0)
    assert obj._get_shard_info() == (8, rank)


@pytest.mark.parametrize("worker,budget,max_reqs,slot", [
    ("hcu_vllm_das.dsv41.worker.HcuDSV41PcpWorker", 32768, 128, 4352),
    ("hcu_vllm_das.dsv41.worker.HcuDSV41PcpWorker", 32769, 128, 4353),
    ("hcu_vllm_das.dsv41.worker.HcuDSV41PcpWorker", 7, 4, 7),
    ("hcu_adapter.bridges.vllm_worker.Worker", 32768, 128, 32768),
    ("example.CustomWorker", 32768, 128, 32768),
])
def test_engram_staging_capacity_matches_padded_pcp_gather(monkeypatch, worker, budget, max_reqs, slot):
    obj = object.__new__(engram.HcuV41PcpEngram)
    torch.nn.Module.__init__(obj)
    lookups = []
    obj.embed_tokens = NS(_hcu_pcp_size=8, _hcu_pcp_rank=0, part_n_hash_cols=3,
        n_hash_cols=24, dim=256, lookup=lambda ids, rows: lookups.append((len(ids), len(rows))))
    counts = torch.tensor([slot] * 8)
    def gather(value, dim):
        if value.dtype == torch.int64:
            return counts
        return value.new_empty((len(value) * 8, value.shape[1]))
    group = NS(world_size=8, rank_in_group=0, all_gather=gather)
    cfg = config(worker=worker)
    cfg.scheduler_config.max_num_batched_tokens = budget
    cfg.scheduler_config.max_num_seqs = max_reqs
    monkeypatch.setattr(engram, "get_pcp_group", lambda: group)
    monkeypatch.setattr(engram, "get_current_vllm_config", lambda: cfg)
    with torch.device("meta"):
        obj._init_staging(budget, 256)
        obj.prepare_embeddings(torch.empty(slot, 24, dtype=torch.int32))
    assert obj.staged_rows.shape == (slot * 8, 3, 256)
    assert obj.staged_rows.dtype == torch.bfloat16
    assert lookups == [(slot * 8, slot * 8)]
    counts[0] += 1
    with pytest.raises(ValueError, match="staging capacity"):
        obj.prepare_embeddings(torch.empty(0, 24, dtype=torch.int32))


@pytest.mark.parametrize("counts,rank", [([0]*8, 0), ([0,2,1,0,3,0,1,2], 0), ([0,2,1,0,3,0,1,2], 4)])
def test_pcp_engram_stages_uneven_and_empty_ranks(monkeypatch, counts, rank):
    heads, local_heads, dim = 24, 3, 2
    hashes = [torch.arange(n * heads, dtype=torch.int32).reshape(n, heads) + 1000*r for r,n in enumerate(counts)]
    slot = max(counts)
    def gather(value, dim=0):
        if value.dtype == torch.int64:
            return torch.tensor(counts, dtype=torch.int64)
        if dim == 1:
            return torch.zeros(slot * 8, heads, 2)
        expected = torch.full((slot, heads), engram.native.DEAD_ID, dtype=torch.int32)
        expected[:counts[rank]] = hashes[rank]
        assert torch.equal(value, expected)
        padded = []
        for h in hashes:
            p = torch.full((slot, heads), engram.native.DEAD_ID, dtype=torch.int32)
            p[:len(h)] = h
            padded.append(p)
        return torch.cat(padded)
    monkeypatch.setattr(engram, "get_pcp_group", lambda: NS(world_size=8, rank_in_group=rank, all_gather=gather))
    obj = object.__new__(engram.HcuV41PcpEngram)
    torch.nn.Module.__init__(obj)
    obj.embed_tokens = NS(n_hash_cols=heads, part_n_hash_cols=local_heads, dim=dim,
                         _hcu_pcp_size=8, _hcu_pcp_rank=rank, tp_size=1)
    obj.staged_rows = torch.empty(8*8, local_heads, dim)
    obj._hcu_prepared = None
    staged = []
    monkeypatch.setattr(engram.native.Engram, "prepare_embeddings", lambda self,h: staged.append(h.clone()))
    monkeypatch.setattr(engram.native.Engram, "_ready_rows", lambda self,n: self.staged_rows[:n])
    obj.prepare_embeddings(hashes[rank])
    actual = obj.embed(hashes[rank])
    assert actual.shape == (counts[rank], heads, dim)
    assert len(staged) == bool(slot)
    with pytest.raises(RuntimeError, match="successful prepare"):
        obj.embed(hashes[rank])


@pytest.mark.parametrize("rank", range(8))
def test_native_engram_constructor_and_loader_preserve_pcp_head_buckets(monkeypatch, rank):
    core = engram.native
    cfg = config()
    cfg.engram_config = NS(cpu_offload=True, dp_shared_memory=False, embedding_across_dp=False, use_thp=False)
    monkeypatch.setattr(engram, "get_current_vllm_config", lambda: cfg)
    monkeypatch.setattr(engram, "get_pcp_group", lambda: NS(world_size=8, rank_in_group=rank))
    monkeypatch.setattr(engram, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(core, "get_engram_dp_size", lambda: 1)
    monkeypatch.setattr(core, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(core, "is_uva_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _: NS(multi_processor_count=1))
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 0)
    heads = tuple(range(1, 25))
    with torch.device("meta"):
        obj = engram.HcuV41PcpEmbedding(sum(heads), 256, heads, cpu_offload=True)
    start, stop = sum(heads[:3*rank]), sum(heads[:3*(rank+1)])
    assert obj.head_start == rank*3 and obj.part_n_hash_cols == 3
    assert obj.vocab_start_idx == start and obj.vocab_end_idx == stop
    assert obj.weight.shape == (stop-start, 256)
    assert obj.weight_scale_inv.shape == (stop-start, 8)
    assert obj.dp_size == obj.tp_size == 1
    param = torch.nn.Parameter(torch.empty(stop-start, 4), requires_grad=False)
    param.engram_vocab_start = start
    loaded = torch.arange(sum(heads)*4).reshape(sum(heads), 4).float()
    obj._weight_loader(param, loaded)
    assert torch.equal(param, loaded[start:stop])


@pytest.mark.parametrize("rank", range(8))
@pytest.mark.parametrize("slot,count", [(3,3), (3,0), (3,1)])
def test_head_exchange_selects_pcp_local_tokens(monkeypatch, rank, slot, count):
    size, local_heads, dim = 8, 3, 2
    complete = torch.arange(size*slot*size*local_heads*dim).reshape(size, slot*size, local_heads, dim).float()
    heads = torch.cat(list(complete), dim=1)
    obj = object.__new__(engram.HcuV41PcpEngram)
    torch.nn.Module.__init__(obj)
    obj.embed_tokens = NS(n_hash_cols=24, part_n_hash_cols=3, dim=2, tp_size=1,
                         _hcu_pcp_size=8, _hcu_pcp_rank=rank)
    obj._hcu_prepared = (count, slot)
    monkeypatch.setattr(engram, "get_pcp_group", lambda: NS(
        world_size=size, rank_in_group=rank, all_gather=lambda value, dim: heads))
    monkeypatch.setattr(engram.native.Engram, "_ready_rows", lambda self,n: complete[rank])
    actual = obj.embed(torch.zeros(count, 24, dtype=torch.int32))
    assert torch.equal(actual, heads[rank*slot:rank*slot+count])
