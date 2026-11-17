# SPDX-License-Identifier: Apache-2.0
"""Verify target packed expert layout with actual upstream loader methods."""
from types import SimpleNamespace
import torch
from torch import nn
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from hcu_vllm_das.quantization.slimquant_load_only import SlimQuantW4A8LoadOnlyExperts


def test_actual_expert_weight_loader_tp16_no_padding():
    inter, hidden, tp = 64, 12, 16
    torch.manual_seed(321)
    w1=torch.randint(-128,128,(inter,hidden//2),dtype=torch.int8)
    w3=torch.randint(-128,128,(inter,hidden//2),dtype=torch.int8)
    w2=torch.randint(-128,128,(hidden,inter//2),dtype=torch.int8)
    s1,s3,s2=torch.rand(inter,1),torch.rand(inter,1),torch.rand(hidden,1)
    for rank in range(tp):
        e=RoutedExperts.__new__(RoutedExperts); nn.Module.__init__(e)
        e.moe_config=SimpleNamespace(num_experts=1,tp_rank=rank,tp_size=tp,
            is_act_and_mul=True,tp_shard_with_padding=False,
            moe_parallel_config=SimpleNamespace(tp_size=tp))
        e.expert_map_manager=SimpleNamespace(num_fused_shared_experts=0,
            map_global_to_local=lambda x:x)
        e.quant_config=SimpleNamespace(get_name=lambda:"slimquant_w4a8")
        e.quant_method=SlimQuantW4A8LoadOnlyExperts(e.moe_config)
        e._fused_shared_expert_quantizer=None
        e.layer_name="model.layers.1.block_sparse_moe.experts"
        e.quant_method.create_weights(e,1,hidden,inter//tp,torch.bfloat16,weight_loader=e.weight_loader)
        for shard,w,s in (("w1",w1,s1),("w3",w3,s3),("w2",w2,s2)):
            stem="w2" if shard=="w2" else "w13"
            e.weight_loader(getattr(e,stem+"_weight"),w,stem+"_weight",shard,0)
            e.weight_loader(getattr(e,stem+"_weight_scale"),s,stem+"_weight_scale",shard,0)
        n=inter//tp
        assert torch.equal(e.w13_weight[0,:n],w1[rank*n:(rank+1)*n])
        assert torch.equal(e.w13_weight[0,n:],w3[rank*n:(rank+1)*n])
        assert torch.equal(e.w2_weight[0],w2[:,rank*n//2:(rank+1)*n//2])
        assert torch.equal(e.w13_weight_scale[0,:n],s1[rank*n:(rank+1)*n])
        assert torch.equal(e.w13_weight_scale[0,n:],s3[rank*n:(rank+1)*n])
        assert torch.equal(e.w2_weight_scale[0],s2)
