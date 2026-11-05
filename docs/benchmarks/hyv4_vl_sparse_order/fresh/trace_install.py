import torch
from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphWrapper
from vllm.forward_context import get_forward_context,is_forward_context_available
from vllm_hcu.models.hy_v4.attention import Indexer
worker._fresh_records=[];worker._fresh_indices=[];worker._fresh_trace=False;worker._fresh_context=None
worker._fresh_topk_buffer=next(m.topk_indices_buffer for m in worker.model_runner.model.modules() if isinstance(m,Indexer))
original_call=BreakableCUDAGraphWrapper.__call__
def call(wrapper,*args,**kwargs):
    if is_forward_context_available():
        ctx=get_forward_context();worker._fresh_context={'runtime_mode':str(ctx.cudagraph_runtime_mode),'descriptor':str(ctx.batch_descriptor),'requested_tokens':worker._stage4_real}
    return original_call(wrapper,*args,**kwargs)
BreakableCUDAGraphWrapper.__call__=call
model=worker.model_runner.model;original_logits=model.compute_logits
def compute_logits(hidden_states):
    logits=original_logits(hidden_states)
    if worker._fresh_trace:
        worker._fresh_indices.append(worker._fresh_topk_buffer[:1].detach().cpu().clone())
        if worker.rank==0 and logits is not None:
            worker._fresh_records.append({'context':dict(worker._fresh_context or {}),'hidden_states':hidden_states.detach().cpu().clone(),'logits':logits.detach().cpu().clone()})
    return logits
model.compute_logits=compute_logits
result={'rank':worker.rank,'installed':True}
