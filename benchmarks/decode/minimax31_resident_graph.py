"""Explicit later-stage graph validation over an already qualified eager session."""
import json
import hashlib
from pathlib import Path
import statistics
import time
import torch
from minimax31_validation import finite
MODULE_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def validate(*,runtime,ids,forward,destination,rank,world,iterations,trials):
    if not runtime.q_lens.is_cuda or not runtime.kv_lens.is_cuda:
        raise RuntimeError('resident GPU index metadata is required before graph validation')
    destination.mkdir(parents=True,exist_ok=True)
    from minimax31_full_history_check import check
    cold=check(runtime=runtime,forward=forward,destination=destination,rank=rank,world=world,compare=bounded_compare)
    stream=torch.cuda.current_stream();torch.cuda.synchronize();world.barrier()
    for _ in range(2):forward()
    torch.cuda.synchronize();world.barrier()
    expected_prefix=runtime.prefix.cpu().clone();expected_positions=runtime.positions.cpu().clone()
    if not bool((expected_prefix==147456).all()):raise RuntimeError('unexpected benchmark prefix before capture')
    graph=torch.cuda.CUDAGraph()
    print('MINIMAX_RESIDENT_CAPTURE_BEGIN '+json.dumps({'rank':rank}),flush=True)
    pool=getattr(runtime,'execution_pool_owner',None)
    with (pool.capture(graph,stream=stream) if pool else torch.cuda.graph(graph,stream=stream)):
        hidden,logits=forward()
    # DeepEP must not replay while another rank is still capturing.
    world.barrier()
    print('MINIMAX_RESIDENT_CAPTURE_END '+json.dumps({'rank':rank}),flush=True)
    graph.replay();torch.cuda.synchronize();world.barrier()
    if not torch.equal(runtime.prefix.cpu(),expected_prefix) or not torch.equal(runtime.positions.cpu(),expected_positions):
        raise RuntimeError('first graph replay corrupted request metadata')
    print('MINIMAX_RESIDENT_FIRST_REPLAY_END '+json.dumps({'rank':rank}),flush=True)
    for step in range(4):
        graph.replay();torch.cuda.synchronize();world.barrier()
        if not torch.equal(runtime.prefix.cpu(),expected_prefix) or not torch.equal(runtime.positions.cpu(),expected_positions):
            raise RuntimeError('graph replay corrupted request metadata')
        print('MINIMAX_RESIDENT_SERIAL_WARM_END '+json.dumps({'rank':rank,'step':step}),flush=True)
    events=[(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)) for _ in range(trials*iterations)]
    for first,last in events:first.record();last.record()
    torch.cuda.synchronize();world.barrier()
    gpu=[];wall=[]
    for first,last in events:
        world.barrier()
        first.record();started=time.perf_counter()
        graph.replay()
        last.record();last.synchronize()
        wall.append((time.perf_counter()-started)*1000)
        gpu.append(first.elapsed_time(last))
    preliminary={'rank':rank,'verify_graph_ms':gpu,'verify_graph_wall_ms':wall,
                 'median_verify_graph_ms':statistics.median(gpu),
                 'correctness_validation_pending':True,'profiler_active_during_timing':False,
                 'replay_synchronized_each_step':True,'cpu_barrier_outside_timing_per_step':True}
    (destination/('rank%d.graph_timing_pending.json'%rank)).write_text(json.dumps(preliminary,indent=2)+'\n')
    print('MINIMAX_RESIDENT_GRAPH_TIMED '+json.dumps({'rank':rank,'median_ms':statistics.median(gpu),'validation_pending':True}),flush=True)
    if not finite(hidden) or not finite(logits):
        raise RuntimeError('nonfinite full graph outputs')
    misses=torch.stack([state.miss_count.sum() for state in runtime.states]).cpu().tolist()
    # Changing actual token values and positions checks replay consumes current
    # input/cache state, rather than accepting a frozen finite capture output.
    # The graph reuses its intermediate addresses. Templates allocated in that
    # same MemPool may overlap freed capture temporaries and be overwritten by
    # replay. Keep persistent validation inputs on CPU instead.
    saved_ids=ids.cpu().clone();saved_prefix=runtime.prefix.cpu().clone();cases=[]
    try:
        for offset in (1,7):
            print('MINIMAX_RESIDENT_CHANGED_BEGIN '+json.dumps({'rank':rank,'offset':offset}),flush=True)
            ids.copy_(saved_ids+offset);runtime.prefix.copy_(saved_prefix+offset*13);runtime.prepare();torch.cuda.synchronize()
            if not torch.equal(runtime.prefix.cpu(),saved_prefix+offset*13):raise RuntimeError('changed prefix was not prepared correctly')
            print('MINIMAX_RESIDENT_CHANGED_PREPARED '+json.dumps({'rank':rank,'offset':offset}),flush=True)
            world.barrier();graph.replay()
            torch.cuda.synchronize();world.barrier()
            eager_hidden,eager_logits=forward();torch.cuda.synchronize()
            if hidden.data_ptr()==eager_hidden.data_ptr() or logits.data_ptr()==eager_logits.data_ptr():
                raise RuntimeError('graph/eager outputs must have independent live storage')
            hidden_error=bounded_compare(hidden,eager_hidden,.02,.02)
            logits_error=bounded_compare(logits,eager_logits,.02,.04)
            cases.append({'token_offset':offset,'position_offset':offset*13,
                          'graph_matches_eager':True,'hidden_max_abs':hidden_error,'logits_max_abs':logits_error})
            del eager_hidden,eager_logits
            print('MINIMAX_RESIDENT_CHANGED_PASSED '+json.dumps({'rank':rank,'offset':offset}),flush=True)
    finally:
        ids.copy_(saved_ids);runtime.prefix.copy_(saved_prefix);runtime.prepare();torch.cuda.synchronize()
    world.barrier()
    result={'rank':rank,'scope':'full target VERIFY graph only; excludes DRAFT/accept/commit',
            'shared_zero_host_history':getattr(runtime.backing,'zero_history_layer_alias',False),
            'registered_host_bytes':runtime.backing.registered_bytes,
            'bootstrap_registered_host_bytes':getattr(getattr(runtime,'bootstrap_history_backing',None),'registered_bytes',0),
            'profiler_active_during_timing':False,
            'graph_validation_source_sha256':MODULE_SHA256,'saved_input_template_device':'cpu',
            'trace_export_completed':False,
            'replay_synchronized_each_step':True,'cpu_barrier_outside_timing_per_step':True,
            'verify_graph_ms':gpu,'verify_graph_wall_ms':wall,
            'median_verify_graph_ms':statistics.median(gpu),'iterations':iterations,'trials':trials,
            'changed_input_cases':cases,'finite_hidden_and_logits':True,
            'all_60_full_host_layers_cold_checked':cold['all_60_layers_exercised'],
            'last_timed_step_host_miss_pages_per_layer':misses,
            'last_timed_step_host_swap_bytes':sum(misses)*2*128*128,
            'resident_host_pointer':hex(runtime.backing.storage.data_ptr()),
            'resident_gpu_cache_pointer':hex(runtime.main[0].data_ptr())}
    result['execution_pool_stats']=runtime.execution_pool_owner.check() if hasattr(runtime,'execution_pool_owner') else None
    from hcu_vllm_das.parallel import minimax31_decode
    result['route_check_backend']=getattr(minimax31_decode._executor,'route_check_backend','torch')
    destination.mkdir(parents=True,exist_ok=True)
    (destination/('rank%d.graph.json'%rank)).write_text(json.dumps(result,indent=2)+'\n')
    print('MINIMAX_RESIDENT_GRAPH_QUALIFIED '+json.dumps({'rank':rank,'median_ms':result['median_verify_graph_ms']}),flush=True)
    # Profiling begins only after the complete primary timing and numerical
    # qualification have been persisted. Capture a separate diagnostic graph
    # while the profiler observes its node metadata; late-start graph replay
    # tracing on this runtime omitted most kernel records.
    from minimax31_eager_session import export_trace
    forward();torch.cuda.synchronize();world.barrier()
    profiler=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=False,with_stack=False)
    profiler.start();profile_graph=torch.cuda.CUDAGraph()
    print('MINIMAX_PROFILE_CAPTURE_BEGIN '+json.dumps({'rank':rank}),flush=True)
    with (pool.capture(profile_graph,stream=stream) if pool else torch.cuda.graph(profile_graph,stream=stream)):
        profile_hidden,profile_logits=forward()
    world.barrier()
    print('MINIMAX_PROFILE_CAPTURE_END '+json.dumps({'rank':rank}),flush=True)
    for step in range(2):
        world.barrier()
        with torch.profiler.record_function('VERIFY_GRAPH_STEADY_BS128_T147456_STEP%d'%step):
            profile_graph.replay();torch.cuda.synchronize()
    export_trace(profiler,destination/('rank%d.graph.trace.json.gz'%rank))
    result['trace_export_completed']=True;result['trace_graph_captured_under_profiler']=True
    (destination/('rank%d.graph.json'%rank)).write_text(json.dumps(result,indent=2)+'\n')
    print('MINIMAX_RESIDENT_GRAPH_RESULT '+json.dumps(result),flush=True)
    world.barrier()


def bounded_compare(actual,expected,rtol,atol):
    if actual.shape!=expected.shape or actual.dtype!=expected.dtype:raise RuntimeError('graph/eager output geometry changed')
    maximum=0.
    for a,b in zip(actual.reshape(-1).split(128*6144),expected.reshape(-1).split(128*6144)):
        a=a.float();b=b.float()
        if not bool(torch.isclose(a,b,rtol=rtol,atol=atol).all()):raise RuntimeError('changed-input graph/eager output mismatch')
        maximum=max(maximum,float((a-b).abs().max()))
    return maximum
