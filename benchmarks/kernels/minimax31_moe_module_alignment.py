"""Standalone numerical oracle and matched-fixture EP8 benchmark. No service."""
import argparse
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

import torch
import torch.distributed as dist

from minimax31_moe_sglang_v1 import PackedExperts, ReceivedExperts, AlignedMoE

P = argparse.ArgumentParser()
P.add_argument('--phase', choices=['oracle','guard','fixture','bench'], required=True)
P.add_argument('--root', required=True)
P.add_argument('--reference', action='store_true')
P.add_argument('--rows', type=int, nargs='+', default=[2048,6144,8192])
P.add_argument('--repeats', type=int, default=15)
P.add_argument('--warmup', type=int, default=4)
P.add_argument('--profile', action='store_true')
args = P.parse_args()
root = Path(args.root)
root.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(4)


def raw_weights(e,h,i,seed):
    g=torch.Generator().manual_seed(seed)
    return (torch.randint(-128,128,(e,2*i,h//2),dtype=torch.int8,generator=g),
            torch.randint(-128,128,(e,h,i//2),dtype=torch.int8,generator=g),
            torch.rand((e,2*i,1),generator=g)*.0007+.0002,
            torch.rand((e,h,1),generator=g)*.0003+.0001)


def cpu_quant(x):
    amax=x.float().abs().amax(-1,keepdim=True).clamp_min(1e-10)
    # Independently round the ideal rational value in FP64. CPU FP32 division
    # can turn an exact +/-63.5 into +/-63.499996 and spuriously disagree with
    # the native nearest-even result. Scale remains the declared FP32 scale.
    s=amax/127
    return torch.round(x.double()*127/amax.double()).clamp(-128,127).to(torch.int8),s


def unpack(p):
    u=p.to(torch.int16)&255
    q=torch.stack((u>>4,u&15),-1).flatten(-2)
    return torch.where(q>=8,q-16,q).double()


def oracle(x,w13,w2,s13,s2,ids,rw):
    # Independent integer dot + explicit BF16 rounding; weight after GEMM2 store.
    q,s=cpu_quant(x)
    out=torch.zeros((*ids.shape,x.shape[1]),dtype=torch.float32)
    for e in ids[ids>=0].unique().tolist():
        rows,slots=(ids==e).nonzero(as_tuple=True)
        z=((q[rows].double()@unpack(w13[e]).T).float()*s[rows]*(s13[e].flatten()*16)).bfloat16()
        gate,up=z.float().chunk(2,-1)
        activation=(gate.clamp(max=7)*torch.sigmoid(1.702*gate.clamp(max=7))*(up.clamp(-7,7)+1)).bfloat16()
        aq,asc=cpu_quant(activation)
        y=((aq.double()@unpack(w2[e]).T).float()*asc*(s2[e].flatten()*16)).bfloat16()
        out[rows,slots]=y.float()*rw[rows,slots,None]
    return out.sum(1).bfloat16()


def metrics(got,expected):
    a,b=got.float(),expected.float()
    norm=b.norm().clamp_min(1e-12)
    return {'relative_l2':float((a-b).norm()/norm),
            'cosine':float(torch.nn.functional.cosine_similarity(a.flatten(),b.flatten(),dim=0)),
            'max_abs':float((a-b).abs().max()),'bit_exact':torch.equal(got,expected)}


class SGLangExperts:
    """Call the installed SGLang forward method, with only runner state supplied."""
    def __init__(self,weights):
        from sglang.srt.layers.moe.ep_moe import layer
        self.e,self.h,self.i=weights.validate()
        self.device=weights.w13.device
        if not layer._use_w4a8_contiguous_hipc:
            raise RuntimeError('SGLANG_USE_W4A8_CONTIGUOUS_HIPC must be enabled')
        self.layer=layer
        self.obj=SimpleNamespace(quant_method=object(),w13_weight=weights.w13,w2_weight=weights.w2,
                                w13_weight_scale=weights.s13,w2_weight_scale=weights.s2,
                                moe_runner_config=SimpleNamespace(activation='silu',gemm1_alpha=1.702,
                                    gemm1_beta=1.0,gemm1_clamp_limit=7.0,gate_up_interleaved=False))
        layer.get_offloader=lambda:SimpleNamespace(forbid_copy_engine_usage=False)
    def __call__(self,x,scale,ids,rw,counts):
        return self.layer.DeepEPMoE.forward_deepgemm_w4a8_marlin_contiguous(self.obj,(x,scale,ids,rw,counts))


if args.phase=='guard':
    from minimax31_moe_sglang_v1 import _route_guard
    torch.cuda.set_device(0)
    base=(torch.arange(2052)%136).reshape(513,4)
    weights=torch.ones_like(base,dtype=torch.float32)
    base[10,1]=-1;weights[10,1]=0
    results=[]
    for case in ('valid','negative2','expert136','large_int64','nan_weight','inf_weight','padding_weight'):
        ids,rw=base.clone(),weights.clone()
        if case=='negative2':ids[0,0]=-2
        if case=='expert136':ids[-1,-1]=136
        if case=='large_int64':ids[0,0]=2**32+128
        if case=='nan_weight':rw[0,0]=float('nan')
        if case=='inf_weight':rw[-1,-1]=float('inf')
        if case=='padding_weight':rw[10,1]=1
        expected=bool((((ids>=-1)&(ids<136))&torch.isfinite(rw)&((ids>=0)|(rw==0))).all())
        gi,gw=ids.cuda(),rw.cuda();status=torch.empty(3,device='cuda',dtype=torch.bool)
        _route_guard[(3,)](gi,gw,status,2052,136,1024)
        got=bool(status.all().item());assert got==expected,(case,got,expected)
        assert torch.equal(gi.cpu(),ids)
        torch.testing.assert_close(gw.cpu(),rw,rtol=0,atol=0,equal_nan=True)
        results.append({'case':case,'accepted':got,'matches_independent_CPU':True})
    (root/'guard_validation.json').write_text(json.dumps({'status':'pass','tests':results},indent=2)+'\n')
    print('GUARD_PASS',flush=True)
    raise SystemExit


if args.phase=='oracle':
    torch.cuda.set_device(0)
    torch.manual_seed(20261115)
    raw=raw_weights(17,512,256,311)
    w=PackedExperts.from_checkpoint_bytes(*(t.cuda() for t in raw))
    runner=SGLangExperts(w) if args.reference else ReceivedExperts(w)
    from lightop.quant import per_token_quant_int8
    x=torch.randn(9,512).bfloat16();x[0]=0
    ids=torch.tensor([[16,0,-1,-1],[8,16,0,-1],[-1,-1,-1,-1],[5,1,16,0],
                      [16,-1,-1,-1],[3,7,-1,-1],[0,16,5,1],[3,3,16,-1],[0,1,2,3]],dtype=torch.int64)
    rw=torch.rand(9,4);rw.masked_fill_(ids<0,0);rw[6,0]=0
    rw/=rw.sum(1,keepdim=True).clamp_min(1)
    expected=oracle(x,*raw,ids,rw)
    q,s=per_token_quant_int8(x.cuda())
    cq,cs=cpu_quant(x)
    native_q=q.cpu();different=native_q!=cq
    ideal=x.double()*127/x.float().abs().amax(-1,keepdim=True).clamp_min(1e-10).double()
    half_distance=(ideal.abs()-ideal.abs().floor()-.5).abs()
    # A float32 divider can choose either adjacent integer at an exact half.
    # Reject every non-tie mismatch and any difference larger than one INT8 unit.
    assert (native_q.int()-cq.int()).abs().max()<=1
    assert not different.any() or half_distance[different].max()<1e-5
    torch.testing.assert_close(s.cpu(),cs,rtol=1e-6,atol=0)
    quant_check={'different_INT8_values':int(different.sum()),'only_half_integer_boundaries':True}
    # Use canonical bytes for the independent expert-math oracle on both builds.
    q,s=cq.cuda(),cs.cuda()
    counts=[int((ids==e).sum()) for e in range(17)]
    padded=[(n+255)//256*256 for n in counts]
    q0=q.clone();w0=[t.clone() for t in (w.w13,w.w2,w.s13,w.s2)]
    got=runner(q,s,ids.cuda(),rw.cuda(),padded).cpu()
    m=metrics(got,expected)
    assert got.isfinite().all() and m['relative_l2']<=.0031 and m['cosine']>=.9999967,m
    assert not got[0].any() and not got[2].any()
    assert torch.equal(q,q0) and all(torch.equal(a,b) for a,b in zip(w0,(w.w13,w.w2,w.s13,w.s2)))
    tests=[{'case':'independent_INT64_oracle_post_BF16_weight',**m}, {'case':'input_quant_FP64_reference',**quant_check}]
    if not args.reference:
        for rows in (0,1,9):
            xx=torch.zeros(rows,512,dtype=torch.int8,device='cuda')
            ss=torch.ones(rows,1,device='cuda');ii=torch.full((rows,4),-1,dtype=torch.int64,device='cuda');ww=torch.zeros(rows,4,device='cuda')
            out=runner(xx,ss,ii,ww,[0]*17)
            assert out.shape==(rows,512) and not out.any()
        # Large group, duplicate routes, slot16 and >32768 padded rows.
        rows=8193;xx=torch.randn(rows,512,device='cuda',dtype=torch.bfloat16)
        ii=torch.tensor([0,5,8,16],device='cuda').expand(rows,4).contiguous()
        ww=torch.tensor([.1,.2,.3,.4],device='cuda').expand(rows,4).contiguous()
        qq,ss=per_token_quant_int8(xx);cc=[8448 if e in (0,5,8,16) else 0 for e in range(17)]
        out=runner(qq,ss,ii,ww,cc).clone()
        # Compare changed rows/counts on the same workspace, then recompute original.
        changed=runner(q,s,ids.cuda(),rw.cuda(),padded).clone()
        out2=runner(qq,ss,ii,ww,cc).clone()
        assert torch.equal(out,out2) and torch.equal(changed.cpu(),got)
        tests.append({'case':'workspace_growth_reuse_33792_padded_rows','rows':rows,'bit_exact_repeat':True})
    torch.cuda.synchronize()
    result={'status':'pass','torch':torch.__version__,'tests':tests,'native_reference':args.reference,
            'threshold_source':'existing HIPC independent oracle contract .0031 / .9999967'}
    (root/('oracle_sglang.json' if args.reference else 'oracle_optimized.json')).write_text(json.dumps(result,indent=2)+'\n')
    torch.save({'output':got,'input':x,'ids':ids,'weights':rw},root/('oracle_sglang.pt' if args.reference else 'oracle_optimized.pt'))
    print('ORACLE_PASS',json.dumps(result),flush=True)
    raise SystemExit

if args.phase=='fixture':
    # Generate once; both Torch versions read precisely the same serialized tensors.
    for rank in range(8):
        raw=raw_weights(17,6144,3072,1000+rank)
        cases={}
        for rows in args.rows:
            for pattern in ('balanced','skew'):
                g=torch.Generator().manual_seed(20261115+rank*101+rows)
                x=torch.randn(rows,6144,generator=g).bfloat16()
                scores=torch.rand(rows,136,generator=g)
                if pattern=='skew':scores[:,:17]+=1.0
                ids=scores.topk(4,dim=-1).indices.contiguous()
                rw=torch.rand(rows,4,generator=g);rw/=rw.sum(-1,keepdim=True)
                # Include padding and zero-weight valid routes in every full-size case.
                ids[0,:]=-1;rw[0,:]=0;rw[1,0]=0
                cases[str(rows)+'_'+pattern]={'x':x,'ids':ids,'rw':rw}
        path=root/('fixture_rank%d.pt'%rank)
        torch.save({'weights':raw,'cases':cases},path)
        print('FIXTURE',rank,path.stat().st_size,hashlib.sha256(path.read_bytes()).hexdigest(),flush=True)
    raise SystemExit

rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank)
dist.init_process_group('nccl',timeout=datetime.timedelta(seconds=300))
report={'rank':rank,'torch':torch.__version__,'reference':args.reference,'cases':{}}
fixture_path=root/('fixture_rank%d.pt'%rank)
report['fixture_sha256']=hashlib.sha256(fixture_path.read_bytes()).hexdigest()
fixture=torch.load(fixture_path,map_location='cpu',weights_only=True)
weights=PackedExperts.from_checkpoint_bytes(*(t.cuda() for t in fixture['weights']))
del fixture['weights']


class LegacyMoE:
    def __init__(self):
        import deep_ep
        from hcu_vllm_das.parallel.minimax31_deepep_runner import dispatch_routed
        from triton_tmp_dir.minimax31_hipc_routes_tiled_v1 import run_received_hipc_routes
        self.dispatch,self.local=dispatch_routed,run_received_hipc_routes
        dc,cc=deep_ep.Buffer.get_dispatch_config(8),deep_ep.Buffer.get_combine_config(8)
        nvl=max(c.get_nvl_buffer_size_hint(6144*2,8) for c in (dc,cc))
        self.buffer=deep_ep.Buffer(dist.group.WORLD,num_nvl_bytes=nvl,num_rdma_bytes=0,explicitly_destroy=True)
    def __call__(self,x,ids,rw):
        return self.dispatch(x,torch.empty((x.shape[0],128),device='cuda'),lambda h,l:(rw,ids),
            lambda h,w,i,c:self.local(h,i,w,c,weights.w13,weights.w2,weights.s13,weights.s2),
            self.buffer,ep_rank=rank,recv_id_abi='local_minus_one_v1',physical_experts=136)
    def close(self):
        torch.cuda.synchronize();self.buffer.destroy()


try:
    variants=['sglang64'] if args.reference else ['legacy24','aligned24','aligned64']
    for variant in variants:
        module=LegacyMoE() if variant=='legacy24' else AlignedMoE(dist.group.WORLD,weights,num_sms=24 if variant=='aligned24' else 64,mode=None if args.reference else 1006,expert_factory=SGLangExperts if args.reference else None)
        for case,fixture_case in fixture['cases'].items():
            x,ids,rw=(fixture_case[k].cuda() for k in ('x','ids','rw'))
            before=x.clone()
            quant_hashes=None
            if variant!='legacy24':
                q,s=module.quant(x)
                quant_hashes=[hashlib.sha256(t.cpu().contiguous().numpy().tobytes()).hexdigest() for t in (q,s)]
            for _ in range(args.warmup):out=module(x,ids,rw)
            torch.cuda.synchronize()
            times=[];events=[]
            for iteration in range(args.repeats):
                dist.barrier();torch.cuda.synchronize()
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                t=time.perf_counter();start.record();out=module(x,ids,rw);end.record();torch.cuda.synchronize()
                times.append((time.perf_counter()-t)*1000);events.append(start.elapsed_time(end))
            assert out.isfinite().all() and torch.equal(x,before)
            # One standalone fixture output per variant, outside all timing intervals.
            torch.save(out.cpu(),root/('%s_%s_rank%d.pt'%(variant,case,rank)))
            record={'wall_ms':times,'event_ms':events,'median_wall_ms':statistics.median(times),
                    'median_event_ms':statistics.median(events),'input_unchanged':True,
                    'input_quant_sha256':quant_hashes,
                    'dispatch':getattr(module,'last_dispatch',None)}
            report['cases'][variant+'/'+case]=record
            print('BENCH',rank,variant,case,round(record['median_wall_ms'],4),flush=True)
            if args.profile and case=='8192_balanced':
                dist.barrier();torch.cuda.synchronize()
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                    out=module(x,ids,rw);torch.cuda.synchronize()
                prof.export_chrome_trace(str(root/('%s_rank%d.trace.json'%(variant,rank))))
            if variant=='aligned64' and case=='2048_balanced':
                base=module(x,ids,rw).clone()
                module.shared_fn=lambda h:h*.125
                module.shared_stream=torch.cuda.Stream(device=rank)
                module.routed_scale=.5
                combined=module(x,ids,rw)
                expected=base.mul(.5).add(x*.125)
                assert torch.equal(combined,expected),'shared-stream/outer-scale join differs'
                record['shared_stream_scale_join_bit_exact']=True
                module.shared_fn=None;module.shared_stream=None;module.routed_scale=1.
            (root/('bench_%s_rank%d.json'%('sglang' if args.reference else 'ours',rank))).write_text(json.dumps(report,indent=2)+'\n')
        dist.barrier();module.close();del module;torch.cuda.empty_cache()
    print('BENCH_COMPLETE',rank,flush=True)
finally:
    dist.destroy_process_group()
