"""Static-only channel W4A8 must preserve logical weight/scale conversion."""
from types import SimpleNamespace
import json

import pytest
import torch
from compressed_tensors.compressors.pack_quantized.helpers import pack_to_int32
from compressed_tensors.quantization import QuantizationArgs

from tests.models.static_eplb_test_utils import GenericMoE
from vllm_hcu.model_executor.layers.quantization.compressed_tensors_w4a8_moe import (
    HcuCompressedTensorsW4A8Int8MoEMethod,
    repack_compressed_tensors_int4,
)
from vllm_hcu.model_executor.layers.fused_moe.static_eplb import bind_static_eplb_plan


def method(monkeypatch, *, path=None, redundant=0, backend='deepep_high_throughput', topology=None):
    parallel=SimpleNamespace(enable_eplb=True,enable_expert_parallel=True,
        _vllm_hcu_expert_map_path=path,tensor_parallel_size=1,eplb_config=SimpleNamespace(num_redundant_experts=redundant))
    config=SimpleNamespace(parallel_config=parallel)
    monkeypatch.setattr('vllm.config.get_current_vllm_config_or_none',lambda:config)
    monkeypatch.setattr('vllm_hcu.model_executor.layers.fused_moe.deepep_runtime.slimquant_w4a8_uses_deepep_ht',lambda moe:True)
    moe=SimpleNamespace(moe_backend='deep_gemm',has_bias=False,
        hidden_dim_unpadded=8,intermediate_size_per_partition_unpadded=8,
        num_logical_experts=4,num_experts=4,tp_rank=0,tp_size=1,is_act_and_mul=True,
        moe_parallel_config=SimpleNamespace(tp_size=1,all2all_backend=backend,use_ep=True,pcp_size=16,ep_size=16,dp_size=1))
    for name,value in (topology or {}).items():setattr(moe.moe_parallel_config,name,value)
    result=HcuCompressedTensorsW4A8Int8MoEMethod(
        QuantizationArgs(num_bits=4,type='int',strategy='channel',symmetric=True,dynamic=False,group_size=-1),
        QuantizationArgs(num_bits=8,type='int',strategy='token',symmetric=True,dynamic=True),moe)
    return result,moe,parallel


def test_static_capability_is_enabled_only_for_guarded_direct_load(monkeypatch):
    assert method(monkeypatch,path='/tmp/static-map.json')[0].supports_eplb


@pytest.mark.parametrize('kwargs',[{},dict(path='/tmp/map',redundant=16),dict(path='/tmp/map',backend='deepep_auto')])
def test_dynamic_or_unvalidated_static_modes_remain_unsupported(monkeypatch,kwargs):
    assert not method(monkeypatch,**kwargs)[0].supports_eplb


@pytest.mark.parametrize('topology',[dict(pcp_size=2),dict(ep_size=8),dict(tp_size=2),dict(dp_size=2)])
def test_other_topologies_do_not_advertise_static_support(monkeypatch,topology):
    assert not method(monkeypatch,path='/tmp/static-map.json',topology=topology)[0].supports_eplb


def test_static_capability_survives_initialization_config_scope(monkeypatch):
    quant=method(monkeypatch,path='/tmp/static-map.json')[0]
    monkeypatch.setattr('vllm.config.get_current_vllm_config_or_none',lambda:None)
    assert quant.supports_eplb


@pytest.mark.parametrize('rank',[0,1])
def test_static_permutation_loads_packed_weights_and_scales_without_changing_values(tmp_path,monkeypatch,rank):
    path=tmp_path/'map.json';rows=[[2,0,3,1],[1,3,0,2]]
    path.write_text(json.dumps({'model_maps':{'GenericMoE':{'physical_to_logical_map':rows}}}))
    model=GenericMoE(rank)
    model.num_logical_experts=model.num_routed_experts=4;model.num_redundant_experts=0
    for layer in model.moe_layers:
        owner=layer.routed_experts
        quant,moe,parallel=method(monkeypatch,path=str(path))
        del owner.quant_method;owner.quant_method=quant;owner.moe_config=moe
        for name in ['w13_weight','w2_weight','w13_weight_scale','w2_weight_scale']:delattr(owner,name)
        quant.create_weights(owner,num_experts=2,hidden_size=8,intermediate_size_per_partition=8,
            params_dtype=torch.bfloat16,weight_loader=owner.weight_loader)
        for name in ['w13_weight','w2_weight','w13_weight_scale','w2_weight_scale']:
            loader=getattr(owner,name).weight_loader
            assert getattr(loader,'__self__',None) is owner
            assert callable(getattr(loader,'__func__',None))
    parallel.enable_ep_weight_filter=False;parallel.pipeline_parallel_size=1
    bind_static_eplb_plan(SimpleNamespace(parallel_config=parallel),model)
    for index,layer in enumerate(model.moe_layers):
        owner=layer.routed_experts
        for logical in range(4):
            for shard,name,height,width in [('w1','w13_weight',8,8),('w3','w13_weight',8,8),('w2','w2_weight',8,8)]:
                values=torch.full((height,width),logical-2,dtype=torch.int8)
                packed=pack_to_int32(values,4)
                getattr(owner,name).weight_loader(param=getattr(owner,name),loaded_weight=packed,
                    weight_name=name,shard_id=shard,expert_id=logical,return_success=True)
                scale=torch.full((height,1),float(logical+1))
                getattr(owner,name+'_scale').weight_loader(param=getattr(owner,name+'_scale'),loaded_weight=scale,
                    weight_name=name+'_scale',shard_id=shard,expert_id=logical,return_success=True)
        for local,logical in enumerate(rows[index][rank*2:rank*2+2]):
            expected=repack_compressed_tensors_int4(pack_to_int32(torch.full((8,8),logical-2,dtype=torch.int8),4))
            torch.testing.assert_close(owner.w13_weight[local,:8],expected)
            torch.testing.assert_close(owner.w13_weight[local,8:],expected)
            torch.testing.assert_close(owner.w2_weight[local],expected)
            torch.testing.assert_close(owner.w13_weight_scale[local],torch.full((16,1),float(logical+1)/16))
            torch.testing.assert_close(owner.w2_weight_scale[local],torch.full((8,1),float(logical+1)/16))
