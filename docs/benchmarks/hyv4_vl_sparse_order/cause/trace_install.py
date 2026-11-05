import types
import torch
from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphWrapper
from vllm.forward_context import get_forward_context,is_forward_context_available
worker._ready_records=[]
worker._ready_trace=False
worker._ready_context=None
original_call=BreakableCUDAGraphWrapper.__call__
def call(wrapper,*args,**kwargs):
    if is_forward_context_available():
        ctx=get_forward_context()
        worker._ready_context={'runtime_mode':str(ctx.cudagraph_runtime_mode),'descriptor':str(ctx.batch_descriptor),'requested_tokens':worker._stage4_real}
    return original_call(wrapper,*args,**kwargs)
BreakableCUDAGraphWrapper.__call__=call
model=worker.model_runner.model
original_logits=model.compute_logits
def compute_logits(hidden_states):
    logits=original_logits(hidden_states)
    if worker._ready_trace and worker.rank==0 and logits is not None:
        worker._ready_records.append({'context':dict(worker._ready_context or {}),'hidden_states':hidden_states.detach().cpu().clone(),'logits':logits.detach().cpu().clone()})
    return logits
model.compute_logits=compute_logits
result={'rank':worker.rank,'installed':True}
