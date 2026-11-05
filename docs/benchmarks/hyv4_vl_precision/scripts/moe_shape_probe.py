import torch
from vllm.config import set_current_vllm_config, CUDAGraphMode
from vllm.forward_context import set_forward_context
from boltops.ihc.tilelang.configs.pre.common import get_config
for h in getattr(worker, '_stage4_moe_hooks', []): h.remove()
worker._stage4_moe_hooks=[]
def metrics(a,b):
    a=a.float();b=b.float();d=a-b
    return {'exact':torch.equal(a,b), 'nonzero':int(torch.count_nonzero(d)), 'numel':d.numel(), 'max_abs':d.abs().max().item(), 'relative_l2':(d.norm()/a.norm()).item()}
result={'rank':worker.rank,'layers':{},'ihc_configs':{str(n):str(get_config(n,24576)) for n in (22,43,128,1024)}}
with torch.inference_mode(),set_current_vllm_config(worker.vllm_config):
    for idx in (0,1):
        a=torch.load(f'/root/hy4-perf-stage4/moe-input-decode_only-layer{idx}-rank{worker.rank}.pt',weights_only=False)
        b=torch.load(f'/root/hy4-perf-stage4/moe-input-padded_eager-layer{idx}-rank{worker.rank}.pt',weights_only=False)
        n=a['real_tokens']
        x=a['input'].to(worker.device)
        xp=b['input'].to(worker.device)
        xp[:n].copy_(x)
        layer=worker.model_runner.model.model.layers[idx].mlp
        outputs=[];gates=[]
        for value in (x,xp,x):
            with set_forward_context(None, worker.vllm_config, num_tokens=value.shape[0], cudagraph_runtime_mode=CUDAGraphMode.NONE):
                outputs.append(layer(value).clone()[:n])
                if idx==1:gates.append(layer.gate(value)[0][:n].clone())
        entry={'same_input_shape_comparison':metrics(outputs[0],outputs[1]), 'repeat_same_shape':metrics(outputs[0],outputs[2])}
        if gates:
            entry['gate_logits']=metrics(gates[0],gates[1])
            topks=[torch.topk(g.sigmoid()+layer.expert_bias,8,dim=-1).indices.sort(-1).values for g in gates]
            entry['changed_expert_sets']=int((topks[0]!=topks[1]).any(-1).sum())
        result['layers'][str(idx)]=entry
