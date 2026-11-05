import torch
from vllm.config import set_current_vllm_config
from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphWrapper

def metric(a,b):
    a=a.float();b=b.float();d=a-b
    return {'exact':torch.equal(a,b), 'nonzero':int(torch.count_nonzero(d)), 'numel':d.numel(), 'max_abs':d.abs().max().item(), 'relative_l2':(d.norm()/a.norm()).item()}
model=worker.model_runner.model
result={'rank':worker.rank,'cases':[]}
with torch.inference_mode(),set_current_vllm_config(worker.vllm_config):
    for index,n in [(119,22),(94,43)]:
        snapshot=torch.load(f'/root/hy4-perf-stage4/snapshot-{index}-0-decode_only.pt',weights_only=False,map_location='cpu')
        for name,path in [('model.layers.0.hc_attn_layer.hc_pre:1',('input',0)),('model.layers.0.self_attn.linear_gate:1',('input',0))]:
            value=snapshot[name]
            for k in path:value=value[k]
            samples=value['value'].to(worker.device)
            shape=(n,)+tuple(samples.shape[1:])
            x=samples[torch.arange(n,device=worker.device)%3].clone()
            xp=samples[torch.arange(128,device=worker.device)%3].clone()
            xp[:n].copy_(x)
            layer=dict(model.named_modules())[name.rsplit(':',1)[0]]
            outputs=[layer(v) for v in (x,xp,x)]
            if not isinstance(outputs[0],tuple):outputs=[(v,) for v in outputs]
            entry={'index':index,'module':name,'valid_tokens':n,'padded_tokens':128,'outputs':[]}
            for j in range(len(outputs[0])):
                if outputs[0][j] is not None:
                    entry['outputs'].append({'component':j,'shape_change':metric(outputs[0][j],outputs[1][j][:n]),'repeat_same_shape':metric(outputs[0][j],outputs[2][j])})
            result['cases'].append(entry)
