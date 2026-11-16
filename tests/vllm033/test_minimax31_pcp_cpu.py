"""PCP ownership, empty-rank metadata and cache layout without GPU execution."""
from dataclasses import dataclass
from types import SimpleNamespace as NS
import inspect

import numpy as np
import pytest
import torch

from hcu_vllm_das.parallel import minimax31_pcp as pcp


def config():
    return NS(
        parallel_config=NS(tensor_parallel_size=1,pipeline_parallel_size=1,
            data_parallel_size=1,prefill_context_parallel_size=8,
            decode_context_parallel_size=1,enable_expert_parallel=True,
            enable_eplb=False,expert_placement_strategy="linear",
            all2all_backend="allgather_reducescatter"),
        model_config=NS(hf_config=NS(architectures=[pcp.ARCH]),
                        enforce_eager=True,multimodal_config=NS(language_model_only=True)),
        use_v2_model_runner=True,lora_config=None,speculative_config=None,
        cache_config=NS(cache_dtype="auto"))


def test_scope_rejects_fake_topology_and_media_capability():
    cfg=config()
    pcp.validate_scope(cfg)
    cfg.parallel_config.data_parallel_size=8
    with pytest.raises(ValueError,match="TP1"):
        pcp.validate_scope(cfg)
    cfg=config()
    cfg.model_config.multimodal_config.language_model_only=False
    with pytest.raises(ValueError,match="language-model-only"):
        pcp.validate_scope(cfg)


def test_legacy_ep_router_uses_rank_major_global_padding():
    manager=pcp.MiniMax31PCPManager(pcp_world_size=8,pcp_rank=0,
        device=torch.device("cpu"),shard_decode_requests=True,
        max_num_reqs=2,max_num_tokens=16,moe_gathers_tokens=False)
    assert manager._moe_is_padding is not None
    actual=manager.moe_is_padding(torch.tensor([False,True]))
    assert actual.tolist()==[False,True]*8


def test_real_planner_covers_uneven_prefill_once_and_shards_decode():
    planner=object.__new__(pcp.MiniMax31PCPManager)
    planner.pcp_world_size=8
    planner.dcp_world_size=1
    planner.shard_decode_requests=True
    lengths=np.array([257,1,31],dtype=np.int32)
    prefills=np.array([True,False,True])
    seen={i:[] for i in range(3)}
    per_rank=[]
    for rank in range(8):
        chunks=list(planner._iter_rank_chunks(rank,lengths,prefills))
        per_rank.append(sum(n for _,_,n in chunks))
        for request,start,n in chunks:
            seen[request].extend(range(start,start+n))
    for request,length in enumerate(lengths):
        assert sorted(seen[request])==list(range(int(length)))
    assert min(per_rank)>0
    assert max(per_rank)<sum(lengths)


def test_cache_gather_checks_rank_major_slots(monkeypatch):
    rows=torch.arange(6,dtype=torch.float32).reshape(2,3)
    monkeypatch.setattr(pcp,"get_pcp_group",
        lambda:NS(world_size=8,all_gather=lambda value,dim:torch.cat([value+i*10 for i in range(8)],dim=dim)))
    actual=pcp.gather_cache_rows(rows,torch.arange(16))
    assert actual.shape==(16,3)
    torch.testing.assert_close(actual[14:],rows+70)
    with pytest.raises(ValueError,match="slot"):
        pcp.gather_cache_rows(rows,torch.arange(15))


def test_native_wrapper_keywords_match_current_public_signature():
    signature=inspect.signature(pcp.stock.ops.fused_minimax_m3_qknorm_rope_kv_insert)
    signature.bind(*([None]*9),index_q_norm_weight=None,index_k_norm_weight=None,
                   num_index_heads=4,q_out=None,index_q_out=None,kv_cache_dtype="auto")


@dataclass
class EmptyBatch:
    num_tokens:int=0
    num_tokens_after_padding:int=1
    num_reqs:int=1
    num_scheduled_tokens:object=None
    query_start_loc:object=None
    query_start_loc_np:object=None
    seq_lens:object=None
    seq_lens_cpu_upper_bound:object=None
    num_computed_tokens_np:object=None
    num_computed_prefill_tokens_np:object=None
    is_prefilling_np:object=None
    has_prefill:bool=False
    max_query_len:object=None
    is_padding:object=None


def test_empty_rank_attn_view_preserves_original_and_padding():
    original=EmptyBatch(query_start_loc=torch.zeros(2,dtype=torch.int32),
                        seq_lens=torch.zeros(1,dtype=torch.int32),
                        is_padding=torch.ones(1,dtype=torch.bool))
    local=pcp.empty_rank_attention_batch(original)
    assert original.num_tokens==0 and original.seq_lens.tolist()==[0]
    assert local.num_tokens==1 and local.query_start_loc_np.tolist()==[0,1]
    assert local.is_padding is original.is_padding and local.is_padding.all()
