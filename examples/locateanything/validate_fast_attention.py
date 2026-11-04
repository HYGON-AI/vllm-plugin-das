# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Paged Fast attention vs FP32 dense reference; execute candidates via graph."""
import json
import math
from pathlib import Path
import torch
from vllm_hcu import hcu_platform_plugin
hcu_platform_plugin()
from vllm.v1.attention.ops.triton_unified_attention import unified_attention

torch.manual_seed(7)
device='cuda'
results=[]
cases=[([q],[c]) for c in (0,31,511,8192) for q in range(7,13)]
cases += [([38],[0]),([7,12],[17,63]),([7+i%6 for i in range(8)],[100+i*13 for i in range(8)]),
          ([7+i%6 for i in range(16)],[200+i*13 for i in range(16)])]
for qlens,contexts in cases:
    lengths=[q+c for q,c in zip(qlens,contexts)]
    n=len(qlens);maxlen=max(lengths);blocks=math.ceil(maxlen/16)+1
    table=torch.arange(n*blocks,device=device,dtype=torch.int32).view(n,blocks)
    k=torch.randn((n*blocks,16,2,128),device=device,dtype=torch.bfloat16)
    v=torch.randn_like(k)
    q=torch.randn((sum(qlens),16,128),device=device,dtype=torch.bfloat16)
    out=torch.empty_like(q)
    starts=torch.tensor([0]+list(torch.tensor(qlens).cumsum(0).tolist()),device=device,dtype=torch.int32)
    seq=torch.tensor(lengths,device=device,dtype=torch.int32)
    def run():
        unified_attention(q,k,v,out,starts,max(qlens),seq,maxlen,128**-.5,True,(-1,-1),
                          table,0,None,None,None)
    for _ in range(3):run()
    torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):run()
    for replay in range(2):
        if replay:
            q.normal_();k.normal_();v.normal_()
        graph.replay();torch.cuda.synchronize()
        refs=[]
        offset=0
        for row,(ql,c,l) in enumerate(zip(qlens,contexts,lengths)):
            kr=k[row*blocks:(row+1)*blocks].reshape(-1,2,128)[:l].repeat_interleave(8,dim=1).transpose(0,1).float()
            vr=v[row*blocks:(row+1)*blocks].reshape(-1,2,128)[:l].repeat_interleave(8,dim=1).transpose(0,1).float()
            qr=q[offset:offset+ql].transpose(0,1).float();offset+=ql
            query=c+torch.arange(ql,device=device)[:,None];keys=torch.arange(l,device=device)[None,:]
            real=l-6
            mask=torch.where(query<real,keys<=query,(keys<real-1)|(keys>=real))
            scores=(qr@kr.transpose(-1,-2))*(128**-.5)
            ref=(scores.masked_fill(~mask,float('-inf')).softmax(-1)@vr).transpose(0,1)
            refs.append(ref)
        ref=torch.cat(refs)
        diff=out.float()-ref
        rel=float(diff.norm()/ref.norm());maximum=float(diff.abs().max())
        assert rel < .007 and maximum < .04,(qlens,contexts,rel,maximum)
        results.append({'q':qlens,'context':contexts,'replay':replay,'relative_l2':rel,'max_abs':maximum})
record={'status':'passed','graph_replays':len(results),'worst_relative_l2':max(r['relative_l2'] for r in results),'cases':results}
Path('attention-validation.json').write_text(json.dumps(record,indent=2))
print(json.dumps({k:v for k,v in record.items() if k!='cases'}))
