"""Untimed eager attribution, independent of the HIP profiler callbacks."""
import json
import time
import torch
from minimax31_validation import finite


def diagnose(*,runtime,forward,destination,rank,world):
    # The benchmark's lexical target is the exact model used by forward.
    bindings=dict(zip(forward.__code__.co_freevars,
                      (cell.cell_contents for cell in forward.__closure__)))
    model=bindings['target']
    selected=[]
    for name,module in model.named_modules():
        kind=type(module).__name__
        if kind in {'MiniMaxM3DecoderLayer','MiniMaxM3SparseAttention','MiniMax31DecodeMoERunner'} or name.endswith(
                ('.qkv_proj','.o_proj','.gate_up_proj','.down_proj','.gate')):
            selected.append((name,module,kind))
    # Pre-create every event before any collective work is queued. Lazy event
    # initialization must not introduce allocation waits into attribution.
    events=[(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True))
            for _ in range(len(selected)+60*5+57*7+17)]
    for first,last in events:first.record();last.record()
    torch.cuda.synchronize();world.barrier()
    cursor=0
    records=[];pending={};handles=[];restores=[]
    origin=time.perf_counter()

    def begin(name):
        nonlocal cursor
        if cursor>=len(events):raise RuntimeError('unexpected diagnostic scope count')
        event,last=events[cursor];cursor+=1;event.record()
        return name,time.perf_counter(),event,last

    def end(token):
        name,started,event,last=token
        finished=time.perf_counter();last.record()
        records.append((name,(started-origin)*1000,(finished-started)*1000,event,last))

    class Proxy:
        def __init__(self,base,names):self.base=base;self.names=names
        def __getattr__(self,key):
            value=getattr(self.base,key)
            if key not in self.names:return value
            def invoke(*args,**kwargs):
                token=begin('native/'+key)
                try:return value(*args,**kwargs)
                finally:end(token)
            return invoke

    def replace(obj,field,value):
        restores.append((obj,field,getattr(obj,field),field in vars(obj)))
        setattr(obj,field,value)

    def wrap(value,label,recv_hook=False):
        def invoke(*args,**kwargs):
            token=begin(label)
            try:result=value(*args,**kwargs)
            finally:end(token)
            if recv_hook:
                if not isinstance(result,tuple) or not callable(result[-1]):
                    raise RuntimeError('expected the qualified receive-hook ABI')
                return result[:-1]+(wrap(result[-1],label+'_receive'),)
            return result
        return invoke

    try:
        for obj,field,names in [(runtime,'index_native',{'msa128_stage1_page_max','msa128_stage2_topk_select'}),
                                (runtime,'decode_native',{'msa128_decode_fp8'})]+[
                                (state,'native',{'plan_headwise','swap_headwise'}) for state in runtime.states]:
            replace(obj,field,Proxy(getattr(obj,field),names))
        from hcu_vllm_das.parallel import minimax31_decode
        executor=minimax31_decode._executor
        replace(executor.buffer,'low_latency_dispatch',wrap(executor.buffer.low_latency_dispatch,'moe/dispatch_send',True))
        replace(executor.buffer,'low_latency_combine',wrap(executor.buffer.low_latency_combine,'moe/combine_send',True))
        replace(executor.experts,'gemm',wrap(executor.experts.gemm,'moe/masked_w4a8_gemm'))
        replace(executor.experts,'activation',wrap(executor.experts.activation,'moe/masked_oai'))
        replace(model,'compute_logits',wrap(model.compute_logits,'model/compute_logits'))
        for name,module,kind in selected:
            def before(module,inputs,_name=name,_kind=kind):pending[id(module)]=begin('module/'+_kind+'/'+_name)
            def after(module,inputs,output):end(pending.pop(id(module)))
            handles.extend((module.register_forward_pre_hook(before),module.register_forward_hook(after)))
        torch.cuda.synchronize();world.barrier();origin=time.perf_counter()
        hidden,logits=forward();torch.cuda.synchronize()
        total=(time.perf_counter()-origin)*1000
        if not finite(hidden) or not finite(logits):
            raise RuntimeError('nonfinite diagnostic forward')
        result={'rank':rank,'scope':'instrumented eager forward; excludes primary timing; nested event spans overlap and include CPU gaps; not kernel sums',
                'event_pairs_precreated':len(events),
                'total_wall_ms':total,'records':[{'name':name,'start_cpu_ms':start,'cpu_ms':cpu,
                'gpu_event_span_ms':first.elapsed_time(last)} for name,start,cpu,first,last in records]}
        destination.mkdir(parents=True,exist_ok=True)
        (destination/('rank%d.scopes.json'%rank)).write_text(json.dumps(result,indent=2)+'\n')
        print('MINIMAX_EAGER_DIAGNOSED '+json.dumps({'rank':rank,'wall_ms':total,'records':len(records)}),flush=True)
    finally:
        for handle in handles:handle.remove()
        for obj,field,old,had_instance in reversed(restores):
            if had_instance:setattr(obj,field,old)
            else:delattr(obj,field)
    world.barrier()
