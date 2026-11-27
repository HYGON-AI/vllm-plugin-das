"""Run with torchrun: real eight-rank transport, analytic reference and graph."""
import json
import os
import faulthandler
import signal
from pathlib import Path
import torch
import torch.distributed as dist
from hcu_vllm_das.moe.minimax31_ll import LowLatencyMoE


def run_distributed():
    faulthandler.register(signal.SIGUSR1,all_threads=True)
    rank = int(os.environ["LOCAL_RANK"])
    device_index=0 if os.environ.get('M31_SINGLE_VISIBLE_DEVICE')=='1' else rank
    torch.cuda.set_device(device_index)
    # The transport/workspace is deliberately stream-owned. Construct,
    # warm and capture on the same non-default stream, as a serving owner
    # must do; torch.cuda.graph() otherwise silently switches streams.
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    dist.init_process_group("nccl")
    from deepgemm import pack_w4a8_moe_hipc_weight
    h,i,e,n = 6144,3072,16,int(os.environ.get('M31_STRESS_ROWS','16'))
    torch.manual_seed(413+rank)
    w13 = pack_w4a8_moe_hipc_weight(torch.full((e,2*i,h//2),0x11,device="cuda",dtype=torch.int8))
    w2 = pack_w4a8_moe_hipc_weight(torch.full((e,h,i//2),0x11,device="cuda",dtype=torch.int8))
    # Distinct global-expert functions make routing/changed routes observable.
    # Equal expert weights would hide wrong dispatch or stale graph routing.
    expert_ids=torch.arange(rank*e,(rank+1)*e,device='cuda')
    a13=1+(expert_ids%7).float()/8
    a2=.5+(expert_ids%5).float()/4
    s13=(a13[:,None,None].expand(e,2*i,1)/(16*h)).contiguous()
    s2=(a2[:,None,None].expand(e,h,1)/(16*i)).contiguous()
    weights = w13,w2,s13,s2
    executor = LowLatencyMoE(dist.group.WORLD,device=torch.device("cuda",device_index),max_tokens=128)
    x = torch.randn((n,h),device="cuda",dtype=torch.bfloat16)*0.1
    ids = ((torch.arange(n,device="cuda")[:,None]*7 + rank*3 +
            torch.tensor([0,32,64,96],device="cuda")[None,:])%128).long().contiguous()
    rw = torch.tensor([.1,.2,.3,.4],device='cuda').expand(n,4).contiguous()

    def check(actual, source, routes=ids, probabilities=rw):
        scale = source.float().abs().amax(-1,keepdim=True).clamp_min(1e-10)/127
        q = torch.round(source.float()/scale).clamp(-128,127)
        first = (q.sum(-1,keepdim=True)*scale/h*(1+(routes%7).float()/8)).bfloat16().float()
        activated=(first.clamp(max=7)*torch.sigmoid(1.702*first.clamp(max=7))*
                   (first.clamp(-7,7)+1))
        per_expert=(activated*(.5+(routes%5).float()/4)).bfloat16().float()
        reference=(per_expert*probabilities).sum(-1,keepdim=True).bfloat16().expand(-1,h)
        torch.testing.assert_close(actual.float(),reference.float(),rtol=.03,atol=3e-4)

    for _ in range(2): check(executor(x,ids,rw,weights),x)
    dist.barrier()
    if os.environ.get('M31_ASYNC_CHAIN'):
        from contextlib import nullcontext
        length=int(os.environ['M31_ASYNC_CHAIN']);pool=None
        if os.environ.get('M31_EXECUTION_POOL_LIBRARY'):
            from minimax31_execution_pool import ExecutionPool
            pool=ExecutionPool(os.environ['M31_EXECUTION_POOL_LIBRARY'],device_index)
        with pool.context() if pool else nullcontext():
            x.copy_(.2+torch.randn_like(x)*.07)
            check(executor(x,ids,rw,weights),x)
            routes=[]
            for layer in range(length):
                if layer%3==0:hot=[(layer%8)*16+offset for offset in range(4)]
                elif layer%3==1:hot=[(layer*7+rank*3+offset*32)%128 for offset in range(4)]
                else:hot=[(layer*11+offset*32+17)%128 for offset in range(4)]
                routes.append(torch.tensor(hot,device='cuda',dtype=torch.int64).expand(n,4).contiguous())
            def chain(synchronize):
                source=x.clone()
                for route in routes:
                    if os.environ.get("M31_TRANSIENT_ROUTES")=="1":
                        route=route.clone()
                        probability=rw.clone()
                    else:probability=rw
                    combined=executor(source,route,probability,weights)
                    source=(combined*.7+.1).contiguous()
                    if synchronize:torch.cuda.synchronize()
                return source
            expected=chain(True);torch.cuda.synchronize();dist.barrier()
            print('MINIMAX_CHAIN_SERIAL_PASSED '+json.dumps({'rank':rank,'layers':length}),flush=True)
            actual=chain(False)
            print('MINIMAX_CHAIN_ASYNC_ENQUEUED '+json.dumps({'rank':rank,'layers':length}),flush=True)
            torch.cuda.synchronize()
            torch.testing.assert_close(actual,expected,rtol=.01,atol=.001)
            assert float(expected.min())>.11,'positive reference must detect zero or stale outputs'
            result={'rank':rank,'scope':'dependent MoE chain only; excludes attention and real model',
                'layers':length,'rows':n,'varying_skew_and_empty_expert_owners':True,
                'transient_routes':os.environ.get('M31_TRANSIENT_ROUTES')=='1',
                'async_matches_serial':True,'max_abs':float((actual-expected).abs().max()),
                'reference_min':float(expected.min()),'reference_max':float(expected.max()),
                'pool_stats':pool.check() if pool else None}
            if os.environ.get('M31_CHAIN_OUTPUT'):
                root=Path(os.environ['M31_CHAIN_OUTPUT']);root.mkdir(parents=True,exist_ok=True)
                (root/('rank%d.json'%rank)).write_text(json.dumps(result,indent=2)+'\n')
            print('MINIMAX_CHAIN_RESULT '+json.dumps(result),flush=True)
        dist.barrier();dist.destroy_process_group();return
    if os.environ.get('M31_EAGER_SKEW')=='1':
        cases=[]
        for label,hot in [('four_owners',[0,32,64,96]),('one_owner',[0,1,2,3]),('changed_owners',[17,49,81,113])]:
            ids.copy_(torch.tensor(hot,device='cuda').expand(n,4))
            for repeat in range(5):
                # A nonzero mean makes unwritten/stale/zero output rows fail
                # the analytic reference, including saturated expert capacity.
                x.copy_(.2+torch.randn_like(x)*.07)
                check(executor(x,ids,rw,weights),x)
            cases.append({'case':label,'max_received_rows_per_hot_expert':n*8,'repeats':5})
        source=x[:0] if rank==7 else x
        routes=ids[:0] if rank==7 else ids
        probabilities=rw[:0] if rank==7 else rw
        output=executor(source,routes,probabilities,weights)
        if rank==7:assert output.shape==(0,h)
        else:check(output,source,routes,probabilities)
        torch.cuda.synchronize();dist.barrier()
        print(json.dumps({'rank':rank,'rows':n,'eager_only':True,'positive_input_skewed_cases':cases,
            'changed_input_and_experts':True,'empty_rank':True}),flush=True)
        dist.destroy_process_group()
        return
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream): output = executor(x,ids,rw,weights)
    graph.replay(); check(output,x)
    x.copy_(torch.randn_like(x)*.15)
    ids.copy_((ids+17)%128)
    graph.replay(); check(output,x)
    # Adjacent real layers share the same temporary BF16 storage. Verify
    # combine has consumed GEMM2 before the next layer overwrites gateup.
    second_x=(x*.7).contiguous()
    second_ids=((ids+5)%128).contiguous()
    second_rw=rw.flip(-1).contiguous()
    serial=torch.cuda.CUDAGraph()
    with torch.cuda.graph(serial,stream=stream):
        first_output=executor(x,ids,rw,weights)
        second_output=executor(second_x,second_ids,second_rw,weights)
    serial.replay();check(first_output,x);check(second_output,second_x,second_ids,second_rw)
    stress_layers=int(os.environ.get('M31_GRAPH_STRESS_LAYERS','0'))
    if stress_layers:
        inputs=[(x*(1+j*.001)).contiguous() for j in range(stress_layers)]
        routes=[((ids+j*7)%128).contiguous() for j in range(stress_layers)]
        stress=torch.cuda.CUDAGraph()
        with torch.cuda.graph(stress,stream=stream):
            outputs=[executor(source,route,rw,weights) for source,route in zip(inputs,routes)]
        dist.barrier()
        for _ in range(5):stress.replay()
        for output,source,route in zip(outputs,inputs,routes):check(output,source,route,rw)
        for source,route in zip(inputs,routes):
            source.mul_(.8);route.copy_((route+13)%128)
        stress.replay()
        for output,source,route in zip(outputs,inputs,routes):check(output,source,route,rw)
        print(json.dumps({'rank':rank,'stress_layers':stress_layers,'stress_rows':n,
            'all_layers_and_changed_graph':True}),flush=True)
    x.copy_(torch.randn_like(x)*.12);second_x.copy_(torch.randn_like(second_x)*.09)
    second_ids.copy_((second_ids+11)%128)
    serial.replay();check(first_output,x);check(second_output,second_x,second_ids,second_rw)
    # Different input bucket: one DP is empty while its EP peers have work.
    source = x[:0] if rank==7 else x
    routes = ids[:0] if rank==7 else ids
    route_weights = rw[:0] if rank==7 else rw
    result = executor(source,routes,route_weights,weights)
    if rank==7: assert result.shape==(0,h)
    else: check(result,source,routes,route_weights)
    torch.cuda.synchronize(); dist.barrier()
    print(json.dumps({"rank":rank,"numerical":True,"changed_input_graph":True,
        "serial_layers_changed_graph":True,"empty_rank":True,
        "bf16_workspace_bytes_saved":16*8*128*6144*2}),flush=True)
    dist.destroy_process_group()


if __name__ == "__main__": run_distributed()
