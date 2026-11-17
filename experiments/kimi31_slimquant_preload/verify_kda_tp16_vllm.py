"""Real checkpoint KDA TP16 parameter bytes via official vLLM loaders on CPU."""
import os
import json
from pathlib import Path
import torch
from torch import nn
from safetensors import safe_open
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.models.kimi_k3.nvidia.kda import _KimiGDNMergedColumnParallelLinear
from hcu_vllm_das.quantization.mixed_kda import MixedKDAProjection
root=Path(os.environ["KIMI_K31_CHECKPOINT"])
prefix='language_model.model.layers.0.self_attn.'
branches=('q_proj','k_proj','v_proj','g_proj','f_a_proj','b_proj')
keys=[prefix+b+'.weight' for b in branches]+[prefix+b+'.weight_scale' for b in branches[:3]]
index=json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
values={}
for shard in set(index[k] for k in keys):
    with safe_open(str(root/shard),framework='pt',device='cpu') as f:
        for k in keys:
            if index[k]==shard: values[k]=f.get_tensor(k).clone()

def param(shape,dtype,loader):
    p=nn.Parameter(torch.zeros(shape,dtype=dtype),requires_grad=False)
    p.output_dim=0; p.weight_loader=loader
    return p
width,head_dim,heads,hidden,tp=12288,128,96,7168,16
pad=(-(4*(width//tp)+head_dim+heads//tp))%16
for rank in range(tp):
    class Column(nn.Module):
        weight_loader=ColumnParallelLinear.weight_loader
        def __init__(self,n,m,**kwargs):
            super().__init__(); self.tp_rank=rank
            self.weight=param((m//tp,n),torch.int8,self.weight_loader)
            self.weight_scale=param((m//tp,1),torch.float32,self.weight_loader)
    class Merged(_KimiGDNMergedColumnParallelLinear):
        def __init__(self,n,sizes,replicated_shard_id,tp_size,**kwargs):
            nn.Module.__init__(self); self.tp_rank,self.tp_size=rank,tp_size
            self.padding_size=kwargs['padding_size']; self.replicated_shard_id=replicated_shard_id
            self.output_sizes=list(sizes); self.output_sizes[replicated_shard_id]*=tp_size
            self.weight=param((sum(self.output_sizes)//tp_size,n),torch.bfloat16,self.weight_loader)
    obj=MixedKDAProjection(hidden,[width]*4+[head_dim,heads,pad*tp],4,tp,
        column_linear=Column,merged_linear=Merged,quant_config=object(),
        prefix='language_model.model.layers.0.self_attn.in_proj_qkvgfab',padding_size=pad,bias=False)
    for b in branches:
        obj.load_branch(b,'weight',values[prefix+b+'.weight'])
        if b in branches[:3]: obj.load_branch(b,'weight_scale',values[prefix+b+'.weight_scale'])
    for b in branches[:3]:
        for suffix in ('weight','weight_scale'):
            v=values[prefix+b+'.'+suffix]; assert torch.equal(getattr(getattr(obj,b),suffix),v.chunk(tp,0)[rank])
    expected=torch.cat((values[prefix+'g_proj.weight'].chunk(tp)[rank],values[prefix+'f_a_proj.weight'],
                        values[prefix+'b_proj.weight'].chunk(tp)[rank],torch.zeros(pad,hidden,dtype=torch.bfloat16)))
    assert torch.equal(obj.fp_gate.weight,expected)
result={'passed':True,'rank_count':tp,'kda_local_qkv_width':width//tp,'kda_gate_alignment_rows':pad,
        'scope':'one real KDA layer, CPU fake constructors with official vLLM weight_loader methods; no full model or IPC',
        'checkpoint_shards':{k:index[k] for k in keys}}
Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
