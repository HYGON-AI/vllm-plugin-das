# SPDX-License-Identifier: Apache-2.0
"""Real upstream TP loader methods; fake constructors avoid device allocation."""
import torch
from torch import nn
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.models.kimi_k3.nvidia.kda import _KimiGDNMergedColumnParallelLinear
from hcu_vllm_das.quantization.mixed_kda import MixedKDAProjection


def parameter(shape,dtype,loader):
    p=nn.Parameter(torch.zeros(shape,dtype=dtype),requires_grad=False)
    p.output_dim=0; p.weight_loader=loader
    return p


def test_mixed_kda_real_upstream_tp_loaders_all_ranks():
    torch.manual_seed(42)
    width, head_dim, heads, hidden, pad, tp=32,8,16,4,3,16
    weights={}
    for branch in ("q_proj","k_proj","v_proj","g_proj","f_a_proj","b_proj"):
        rows = head_dim if branch == "f_a_proj" else heads if branch == "b_proj" else width
        quant=branch in ("q_proj","k_proj","v_proj")
        weights[branch,"weight"]=(torch.randint(-12,12,(rows,hidden),dtype=torch.int8)
                                   if quant else torch.randn(rows,hidden).bfloat16())
        if quant: weights[branch,"weight_scale"]=torch.rand(rows,1)
    for rank in range(tp):
        class Column(nn.Module):
            weight_loader=ColumnParallelLinear.weight_loader
            def __init__(self,n,m,**kwargs):
                super().__init__(); self.tp_rank=rank
                self.weight=parameter((m//tp,n),torch.int8,self.weight_loader)
                self.weight_scale=parameter((m//tp,1),torch.float32,self.weight_loader)
        class Merged(_KimiGDNMergedColumnParallelLinear):
            def __init__(self,n,sizes,replicated_shard_id,tp_size,**kwargs):
                nn.Module.__init__(self)
                self.tp_rank,self.tp_size=rank,tp_size
                self.padding_size=kwargs["padding_size"]
                self.replicated_shard_id=replicated_shard_id
                self.output_sizes=list(sizes); self.output_sizes[replicated_shard_id]*=tp_size
                self.weight=parameter((sum(self.output_sizes)//tp_size,n),torch.bfloat16,self.weight_loader)
        obj=MixedKDAProjection(hidden,[width]*4+[head_dim,heads,pad*tp],4,tp,column_linear=Column,merged_linear=Merged,quant_config=object(),prefix="layers.0.self_attn.in_proj_qkvgfab",padding_size=pad,bias=False)
        for (branch,suffix),value in weights.items(): obj.load_branch(branch,suffix,value)
        for branch in ("q_proj","k_proj","v_proj"):
            for suffix in ("weight","weight_scale"):
                actual=getattr(getattr(obj,branch),suffix)
                expected=weights[branch,suffix].chunk(tp,dim=0)[rank]
                torch.testing.assert_close(actual,expected,atol=0,rtol=0)
        expected=torch.cat([weights["g_proj","weight"].chunk(tp)[rank],
                            weights["f_a_proj","weight"],
                            weights["b_proj","weight"].chunk(tp)[rank],
                            torch.zeros(pad,hidden,dtype=torch.bfloat16)])
        torch.testing.assert_close(obj.fp_gate.weight,expected,atol=0,rtol=0)


def test_weight_stream_composes_with_actual_kimi_model_loader():
    from types import SimpleNamespace
    from vllm.models.kimi_k3.nvidia.model import KimiLinearModel
    from hcu_vllm_das.quantization.mixed_kda import load_mixed_kda_weights
    tp,rank,width,hidden,head_dim,heads=16,7,32,4,8,16
    class Column(nn.Module):
        weight_loader=ColumnParallelLinear.weight_loader
        def __init__(self,n,m,**kwargs):
            super().__init__(); self.tp_rank=rank
            self.weight=parameter((m//tp,n),torch.int8,self.weight_loader)
            self.weight_scale=parameter((m//tp,1),torch.float32,self.weight_loader)
    class Merged(_KimiGDNMergedColumnParallelLinear):
        def __init__(self,n,sizes,replicated_shard_id,tp_size,**kwargs):
            nn.Module.__init__(self); self.tp_rank,self.tp_size=rank,tp_size
            self.padding_size=kwargs["padding_size"]; self.replicated_shard_id=replicated_shard_id
            self.output_sizes=list(sizes); self.output_sizes[replicated_shard_id]*=tp_size
            self.weight=parameter((sum(self.output_sizes)//tp_size,n),torch.bfloat16,self.weight_loader)
    model=KimiLinearModel.__new__(KimiLinearModel); nn.Module.__init__(model)
    model.config=SimpleNamespace(linear_attn_config={"use_full_rank_gate":True},q_lora_rank=None,
                                 dsa_config=None,is_moe=False,is_linear_attn=False)
    model.attn=nn.Module()
    model.attn.in_proj_qkvgfab=MixedKDAProjection(hidden,[width]*4+[head_dim,heads],4,tp,
        column_linear=Column,merged_linear=Merged,quant_config=object(),
        prefix="attn.in_proj_qkvgfab",padding_size=0,bias=False)
    model.direct=nn.Module()
    model.direct.weight=parameter((1,hidden),torch.float32,lambda p,t:p.data.copy_(t))
    entries=[]
    for branch in ("q_proj","k_proj","v_proj","g_proj","f_a_proj","b_proj"):
        n=head_dim if branch=="f_a_proj" else heads if branch=="b_proj" else width
        quant=branch in ("q_proj","k_proj","v_proj")
        w=torch.arange(n*hidden).reshape(n,hidden).remainder(100).to(torch.int8 if quant else torch.bfloat16)
        entries.append((f"attn.{branch}.weight",w))
        if quant: entries.append((f"attn.{branch}.weight_scale",torch.ones(n,1)))
    entries.append(("direct.weight",torch.ones(1,hidden)))
    actual=load_mixed_kda_weights(model,iter(entries),model.load_weights)
    assert actual == set(dict(model.named_parameters()))
