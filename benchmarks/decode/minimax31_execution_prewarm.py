"""Prime real processed-weight execution while module-loading headroom is free."""
from contextlib import nullcontext
import json
import torch
from minimax31_validation import finite


def run(*,target,runner,executor,pool,rank,world):
    records=[]
    with torch.inference_mode(),pool.context() if pool else nullcontext():
        # Materialize the exact validation primitives on each ordinal before
        # 48.6GB of caches is allocated. Serialize first use across ranks.
        for owner in range(8):
            if owner==rank:
                for dtype in (torch.bfloat16,torch.float32):
                    x=torch.full((128,6144),.25,device='cuda',dtype=dtype)
                    assert finite(x)
                    x[-1,-1]=float('nan');assert not finite(x)
                    x[-1,-1]=float('inf');assert not finite(x)
                    del x
                torch.cuda.current_stream().synchronize()
                print('MINIMAX_VALIDATION_KERNELS_WARMED '+json.dumps({'rank':rank}),flush=True)
            world.barrier()
        seen=set()
        for name,layer in target.named_modules():
            kernel=getattr(getattr(layer,'scheme',None),'kernel',None)
            weight=getattr(layer,'weight',None)
            if kernel is None or not isinstance(weight,torch.Tensor) or weight.ndim!=2 or weight.dtype!=torch.int8:
                continue
            k,n=weight.shape;identity=(type(kernel).__name__,k,n)
            if identity in seen:continue
            seen.add(identity)
            source=torch.full((128,k),.2,device=weight.device,dtype=torch.bfloat16)
            result=kernel.apply_weights(layer,source,getattr(layer,'bias',None))
            torch.cuda.current_stream().synchronize()
            if not finite(result):raise RuntimeError('nonfinite processed-weight linear prewarm: '+name)
            records.append({'module':name,'kernel':identity[0],'k':k,'n':n,'m':128})
            print('MINIMAX_LINEAR_WARMED '+json.dumps({'rank':rank,**records[-1]}),flush=True)
            del source,result
        world.barrier()
        layer=runner.routed_experts
        source=torch.full((128,6144),.2,device=layer.w13_weight.device,dtype=torch.bfloat16)
        ids=((torch.arange(128,device=source.device)[:,None]+torch.tensor([0,32,64,96],device=source.device)[None,:])%128).long().contiguous()
        probabilities=torch.full((128,4),.25,device=source.device,dtype=torch.float32)
        weights=(layer.w13_weight,layer.w2_weight,layer.w13_weight_scale,layer.w2_weight_scale)
        # Warm both transport slots; even count preserves the model's initial
        # buffer parity. These are execution temporaries, never weight writes.
        for _ in range(2):
            result=executor(source,ids,probabilities,weights)
            torch.cuda.current_stream().synchronize()
            if not finite(result):raise RuntimeError('nonfinite real-weight MoE prewarm')
            world.barrier()
        del result,source,ids,probabilities
    record={'rank':rank,'linear_shapes':records,'real_weight_moe_warmups':2,
            'validation_nan_inf_checks':True,'pool':pool.check() if pool else None,
            'gpu_free_bytes':torch.cuda.mem_get_info()[0]}
    print('MINIMAX_EXECUTION_PREFLIGHT_PASSED '+json.dumps(record),flush=True)
    return record
