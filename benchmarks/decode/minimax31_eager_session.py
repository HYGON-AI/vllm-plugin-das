"""Resident, graph-free full VERIFY measurement over existing IPC/cache state."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
import torch
import torch.distributed as dist
from minimax31_validation import finite

MODULE_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def export_trace(profiler,path):
    profiler.stop()
    raw=path.with_suffix('')
    profiler.export_chrome_trace(str(raw))
    with raw.open('rb') as source,gzip.open(path,'wb') as target:
        while chunk:=source.read(1024*1024):target.write(chunk)
    raw.unlink()


def primitive_probe(root,rank):
    """Separate stream/device waits and tiny transfers from model execution."""
    records=[]
    def step(name,operation):
        print('MINIMAX_PRIMITIVE_BEGIN '+json.dumps({'rank':rank,'operation':name}),flush=True)
        started=time.perf_counter();value=operation();elapsed=(time.perf_counter()-started)*1000
        records.append({'operation':name,'wall_ms':elapsed})
        print('MINIMAX_PRIMITIVE_END '+json.dumps({'rank':rank,**records[-1]}),flush=True)
        return value
    x=step('allocate_64_bytes',lambda:torch.empty(16,device='cuda',dtype=torch.int32))
    x.fill_(7)
    step('current_stream_sync',torch.cuda.current_stream().synchronize)
    step('device_sync',torch.cuda.synchronize)
    host=step('D2H_64_bytes',x.cpu)
    if host.tolist()!=[7]*16:raise RuntimeError('primitive D2H mismatch')
    source=torch.arange(16,dtype=torch.int32)
    step('H2D_64_bytes',lambda:x.copy_(source))
    step('post_copy_stream_sync',torch.cuda.current_stream().synchronize)
    (root/('rank%d.primitive.json'%rank)).write_text(json.dumps(records,indent=2)+'\n')


def run(args,*,cfg,runtime,ids,forward,base_result,profiler=None):
    from vllm.distributed.parallel_state import get_world_group
    world=get_world_group();rank=int(os.environ['LOCAL_RANK'])
    root=args.output;control=root/'commands';control.mkdir(exist_ok=True)
    identity={'pid':os.getpid(),'rank':rank,'host_backing_pointer':hex(runtime.backing.storage.data_ptr()),
        'first_gpu_cache_pointer':hex(runtime.main[0].data_ptr()),'cuda_graph':False,
        'session_source_sha256':MODULE_SHA256}
    vocab=cfg.model_config.get_vocab_size()
    def set_input(pattern):
        if pattern=='constant':ids.fill_(1000+rank)
        elif pattern=='grouped':ids.copy_((1000+rank*4096+torch.arange(16,device=ids.device).repeat_interleave(8)*37)%vocab)
        elif pattern=='distinct':ids.copy_((1000+rank*4096+torch.arange(128,device=ids.device)*37)%vocab)
        else:raise ValueError('unknown explicit token pattern')
        # No history is committed in this isolated VERIFY phase. At a fixed
        # prefix every forward overwrites all eight fresh K/V/index entries;
        # the zero past and valid page cache can persist across token patterns.
        runtime.prefix.fill_(147456);runtime.prepare()
    def measure(name,pattern,iterations,trials,profile=False,initial_profiler=None):
        destination=root if name=='baseline' else root/'runs'/name
        destination.mkdir(parents=True,exist_ok=True)
        print('MINIMAX_EAGER_MEASURE_BEGIN '+json.dumps({'rank':rank,'name':name,'pattern':pattern}),flush=True)
        set_input(pattern);torch.cuda.synchronize();world.barrier()
        for _ in range(2):hidden,logits=forward()
        torch.cuda.synchronize();world.barrier()
        print('MINIMAX_EAGER_WARMED '+json.dumps({'rank':rank,'name':name}),flush=True)
        active_profiler=initial_profiler
        if profile and active_profiler is None:
            active_profiler=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA],record_shapes=False,with_stack=False)
            active_profiler.start()
        if active_profiler is not None:
            for step in range(2):
                with torch.profiler.record_function('VERIFY_EAGER_STEADY_BS128_T147456_STEP%d'%step):
                    hidden,logits=forward();torch.cuda.synchronize()
            export_trace(active_profiler,destination/('rank%d.trace.json.gz'%rank))
        # Primary timings are collected only after the profiler has stopped.
        # Materialize event resources while communication is quiescent, rather
        # than lazily creating an end event after dispatches have been queued.
        events=[(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True))
                for _ in range(iterations*trials)]
        for begin,end in events:begin.record();end.record()
        torch.cuda.synchronize();world.barrier()
        pool_before=runtime.execution_pool_owner.check() if hasattr(runtime,'execution_pool_owner') else None
        gpu_times=[];wall_times=[]
        for trial in range(trials):
            world.barrier()
            for step in range(iterations):
                begin,end=events[trial*iterations+step]
                begin.record();started=time.perf_counter()
                hidden,logits=forward()
                end.record();end.synchronize()
                wall_times.append((time.perf_counter()-started)*1000)
                gpu_times.append(begin.elapsed_time(end))
        print('MINIMAX_EAGER_TIMED '+json.dumps({'rank':rank,'name':name,
            'gpu_span_ms':gpu_times,'wall_ms':wall_times}),flush=True)
        pool_after=runtime.execution_pool_owner.check() if hasattr(runtime,'execution_pool_owner') else None
        if not finite(hidden) or not finite(logits):
            raise RuntimeError('nonfinite eager VERIFY outputs')
        misses=torch.stack([state.miss_count.sum() for state in runtime.states]).cpu().tolist()
        # Read actual local-expert loads in a separate untimed forward. The
        # executor's numerical implementation and all weight storage stay fixed.
        from hcu_vllm_das.parallel import minimax31_decode
        executor=minimax31_decode._executor;original=executor.experts;loads=[]
        def observe(source,counts,weights,expected_m):
            loads.append(counts.clone())
            return original(source,counts,weights,expected_m)
        try:
            executor.experts=observe
            forward();torch.cuda.synchronize()
        finally:executor.experts=original
        if len(loads)!=57:raise RuntimeError('expected 57 real MoE load observations')
        received=torch.stack(loads).cpu().tolist()
        result=dict(base_result,**identity,input_token_pattern=pattern,profiler_active_during_timing=False,
            eager_layer_sync=bool(getattr(runtime,'eager_sync_handles',[])),
            shared_zero_host_history=getattr(runtime.backing,'zero_history_layer_alias',False),
            registered_host_bytes=runtime.backing.registered_bytes,
            bootstrap_registered_host_bytes=getattr(getattr(runtime,'bootstrap_history_backing',None),'registered_bytes',0),
            execution_pool_stats=runtime.execution_pool_owner.check() if hasattr(runtime,'execution_pool_owner') else None,
            raw_allocations_during_primary_timing=pool_after['allocations']-pool_before['allocations'] if pool_before else None,
            eager_step_gpu_span_ms=gpu_times,eager_step_wall_ms=wall_times,
            median_eager_gpu_span_ms=statistics.median(gpu_times),
            median_eager_wall_ms=statistics.median(wall_times),
            iterations=iterations,trials=trials,
            last_timed_step_host_miss_pages_per_layer=misses,
            last_timed_step_host_swap_bytes=sum(misses)*2*128*128,
            input_ids_sha256=hashlib.sha256(json.dumps(ids.cpu().tolist(),separators=(',',':')).encode()).hexdigest(),
            cache_reset_between_patterns=False,
            moe_received_rows_per_local_expert=received,
            gpu_free_bytes=torch.cuda.mem_get_info()[0],
            consumer_allocated_bytes=torch.cuda.memory_allocated(),
            consumer_reserved_bytes=torch.cuda.memory_reserved(),
            consumer_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            timing_scope='full real target eager VERIFY; GPU span includes CPU launch gaps; not graph latency')
        (destination/('rank%d.json'%rank)).write_text(json.dumps(result,indent=2)+'\n')
        print('MINIMAX_EAGER_RESULT '+json.dumps(result),flush=True)
        world.barrier()
    primitive_probe(root,rank);world.barrier()
    settings=[json.loads((root/'reregister_host.json').read_text()) if rank==0 and
              (root/'reregister_host.json').exists() else None]
    dist.broadcast_object_list(settings,src=0,group=world.cpu_group)
    if settings[0] is not None:
        from minimax31_reregister_host import reregister
        reregister(runtime=runtime,root=root,rank=rank,world=world,chunk_bytes=settings[0]['chunk_bytes'])
        destination=root/'post_reregister';destination.mkdir(exist_ok=True)
        primitive_probe(destination,rank);world.barrier()
    measure('baseline','constant',args.eager_iterations,args.eager_trials,
            profile=False,initial_profiler=profiler)
    if not args.resident_eager:
        from minimax31_release_history import release
        release(runtime=runtime,rank=rank,world=world)
        return
    (root/('rank%d.ready.json'%rank)).write_text(json.dumps(identity,indent=2)+'\n')
    print('MINIMAX_EAGER_SESSION_READY '+json.dumps(identity),flush=True)
    seen=set();deadline=time.monotonic()+args.resident_idle_seconds
    while True:
        command=None
        if rank==0:
            while command is None:
                pending=sorted(p for p in control.glob('*.json') if p.name not in seen)
                if pending:
                    path=pending[0];seen.add(path.name)
                    try:
                        command=json.loads(path.read_text())
                        name=command.get('name','')
                        if command.get('op') not in ('measure','profile','diagnose','graph','layer_sync','attach_history','stop'):raise ValueError('unknown eager command')
                        if command['op']!='stop':
                            if not name.replace('_','').isalnum():raise ValueError('explicit safe result name required')
                            if command.get('pattern','constant') not in ('constant','distinct','grouped'):raise ValueError('unknown pattern')
                            for key,default,upper in [('iterations',5,100),('trials',3,10)]:
                                value=command.get(key,default)
                                if type(value) is not int or not 1<=value<=upper:raise ValueError('invalid '+key)
                    except Exception as exc:command={'op':'error','message':str(exc)}
                elif time.monotonic()>deadline:command={'op':'stop','reason':'idle lease expired'}
                else:command={'op':'wait'}
        payload=[command];dist.broadcast_object_list(payload,src=0,group=world.cpu_group);command=payload[0]
        if command['op']=='wait':
            time.sleep(.2)
            continue
        if command['op']=='stop':break
        if command['op']=='error':raise RuntimeError(command['message'])
        if command['op']=='attach_history':
            from minimax31_attach_history import attach
            attach(args=args,runtime=runtime,forward=forward,rank=rank,world=world,destination=root/'runs'/command['name'])
        elif command['op']=='layer_sync':
            enabled=command.get('enabled')
            if type(enabled) is not bool:raise ValueError('layer_sync requires explicit boolean')
            if not hasattr(runtime,'eager_sync_layers'):raise RuntimeError('session was not initialized for layer-sync control')
            torch.cuda.synchronize();world.barrier()
            for handle in getattr(runtime,'eager_sync_handles',[]):handle.remove()
            runtime.eager_sync_handles=[]
            if enabled:
                def synchronize_layer(module,inputs,output):torch.cuda.current_stream().synchronize()
                runtime.eager_sync_handles=[m.register_forward_hook(synchronize_layer) for m in runtime.eager_sync_layers]
            destination=root/'runs'/command['name'];destination.mkdir(parents=True,exist_ok=True)
            (destination/('rank%d.json'%rank)).write_text(json.dumps({'rank':rank,'eager_layer_sync':enabled})+'\n')
            print('MINIMAX_LAYER_SYNC_CHANGED '+json.dumps({'rank':rank,'enabled':enabled}),flush=True)
            world.barrier()
        elif command['op']=='graph':
            if getattr(runtime.backing,'zero_history_layer_alias',False):raise RuntimeError('attach full history before graph validation')
            if getattr(runtime,'eager_sync_handles',[]):raise RuntimeError('disable diagnostic layer sync before graph validation')
            from minimax31_resident_graph import validate
            set_input(command.get('pattern','constant'));torch.cuda.synchronize();world.barrier()
            validate(runtime=runtime,ids=ids,forward=forward,destination=root/'runs'/command['name'],rank=rank,
                     world=world,iterations=command.get('iterations',8),trials=command.get('trials',3))
        elif command['op']=='diagnose':
            from minimax31_eager_diagnose import diagnose
            set_input(command.get('pattern','constant'));torch.cuda.synchronize();world.barrier()
            diagnose(runtime=runtime,forward=forward,destination=root/'runs'/command['name'],rank=rank,world=world)
        else:
            measure(command['name'],command.get('pattern','constant'),command.get('iterations',5),
                    command.get('trials',3),profile=command['op']=='profile')
        deadline=time.monotonic()+args.resident_idle_seconds
    print('MINIMAX_EAGER_SESSION_STOP '+json.dumps(identity),flush=True)
    world.barrier()
    from minimax31_release_history import release
    release(runtime=runtime,rank=rank,world=world)
