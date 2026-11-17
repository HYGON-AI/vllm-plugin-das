# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Independent references for opt-in PCP fusion, including physical cache bytes."""
import pytest
import torch

pytestmark = pytest.mark.hcu

@pytest.fixture(autouse=True)
def require_device():
    if not torch.cuda.is_available():
        pytest.skip("real HCU required")


@pytest.mark.parametrize('pages,queries,local', [(1,1,1),(1152,33,4),(17,187,16),(1024,257,0)])
def test_local_pages_exact(pages, queries, local):
    from hcu_vllm_das.parallel.minimax31_prefill_triton import force_local_pages
    score=torch.randn((4,queries,pages),device='cuda').transpose(1,2)
    end=torch.arange(queries,device='cuda',dtype=torch.int32)*193
    ref=score.clone()
    last=(end-1)//128
    ids=torch.arange(pages,device='cuda').view(1,-1,1)
    if local:ref.masked_fill_((ids<=last.view(1,1,-1)) & (ids>=(last-local+1).clamp_min(0).view(1,1,-1)),1.e30)
    force_local_pages(score,end,local)
    torch.testing.assert_close(score,ref,rtol=0,atol=0)


@pytest.mark.parametrize('scale', [1.,0.3,1.7,0.25])
def test_quant_q_exact_bytes(scale):
    from hcu_vllm_das.parallel.minimax31_prefill_triton import quant_q
    q=torch.randn((187,64,128),device='cuda',dtype=torch.bfloat16)*7
    # Include FP8 tie boundaries, zero, signed zero and finite saturation edges.
    q.view(-1)[:8]=torch.tensor([0.,-0.,1.0625,1.1875,432.,448.,464.,-464.],device='cuda')
    s=torch.tensor(scale,device='cuda',dtype=torch.float32)
    ref=(q.float()/s).to(torch.float8_e4m3fn)
    out=quant_q(q,s)
    assert torch.equal(out.view(torch.uint8),ref.view(torch.uint8))


@pytest.mark.parametrize('main_dtype', [torch.uint8, torch.float8_e4m3fn])
@pytest.mark.parametrize('index_fp8', [False,True])
@pytest.mark.parametrize('rows', [0,1,187,257])
def test_cache_store_exact_physical_bytes(index_fp8,rows,main_dtype):
    from hcu_vllm_das.parallel.minimax31_prefill_triton import store_pcp_cache
    from hcu_vllm_das.parallel.minimax31_msa import bind_fresh_main_cache
    # Nontrivial source row stride, distinct and shuffled main/index physical slots.
    raw=torch.randn((rows,1168),device='cuda',dtype=torch.bfloat16).mul_(1.7)[:,:1152]
    assert raw.stride(0)==1168
    if rows:raw[:,1024:]=torch.linspace(-1024,1024,128,device='cuda',dtype=torch.bfloat16)
    storage=torch.full((4,4,128,256),17,device='cuda',dtype=main_dtype)
    reference=storage.clone()
    bound=bind_fresh_main_cache(storage,fresh=True)
    ref=bind_fresh_main_cache(reference,fresh=True)
    dtype=torch.float8_e4m3fn if index_fp8 else torch.bfloat16
    index=torch.full((5,128,128),1,device='cuda',dtype=dtype);ri=index.clone()
    ms=torch.randperm(512,device='cuda',dtype=torch.int64)[:rows].contiguous()
    ix=torch.randperm(640,device='cuda',dtype=torch.int64)[:rows].contiguous()
    if rows:ms[0]=-1;ix[-1]=-1
    ks=torch.tensor(.3,device='cuda');vs=torch.tensor(1.7,device='cuda')
    # Scalar host loop deliberately avoids using either implementation's scatter.
    for row,(m,j) in enumerate(zip(ms.cpu().tolist(),ix.cpu().tolist())):
        if m>=0:
            ref.key[m].copy_((raw[row,:512].float()/ks).to(torch.float8_e4m3fn).view(4,128))
            ref.value[m].copy_((raw[row,512:1024].float()/vs).to(torch.float8_e4m3fn).view(4,128))
        if j>=0:
            z=raw[row,1024:]
            if index_fp8:z=z.clamp(-448,448).to(dtype)
            ri.view(-1,128)[j].copy_(z)
    store_pcp_cache(raw,ms,ix,bound.key_writer,bound.value_writer,index,ks,vs)
    assert torch.equal(storage.view(torch.uint8),reference.view(torch.uint8))
    assert torch.equal(index.view(torch.uint8),ri.view(torch.uint8))


def test_pack_workspace_reuse_and_scale_guard_native_mutation():
    from types import SimpleNamespace
    from hcu_vllm_das.parallel import minimax31_prefill as opt
    cache=torch.empty((1,4,128,256),device='cuda',dtype=torch.uint8)
    obj=SimpleNamespace(kv_cache=cache,_k_scale=torch.tensor(.3,device='cuda'),_v_scale=torch.tensor(1.7,device='cuda'))
    assert not opt.check_bound_scales(obj)
    opt.bind_validated_scales(obj)
    assert opt.check_bound_scales(obj)
    torch.cuda.synchronize()
    a=torch.randn((187,1024),device='cuda',dtype=torch.bfloat16);b=torch.randn((187,128),device='cuda',dtype=torch.bfloat16)
    opt.reserve_pack_rows(187,a.device)
    out=opt.pack_rows((a,b));ptr=out.data_ptr()
    assert torch.equal(out,torch.cat((a,b),1))
    out=opt.pack_rows((a[:1],b[:1]));assert out.data_ptr()==ptr
    # A different stream is rejected before the guard can race with a writer.
    with torch.cuda.stream(torch.cuda.Stream()):
        with pytest.raises(RuntimeError,match='stream changed'):opt.check_bound_scales(obj)
    obj.kv_cache=torch.empty_like(cache)
    assert not opt.check_bound_scales(obj)


def test_dynamic_selector_matches_independent_uint64_oracle_and_one_variant():
    from hcu_vllm_das.parallel.minimax31_prefill_triton import select_physical_dynamic as kernel
    mask = (1 << 64) - 1
    def mix(x):
        x = (x + 0x9e3779b97f4a7c15) & mask
        x = ((x ^ (x >> 30)) * 0xbf58476d1ce4e5b9) & mask
        x = ((x ^ (x >> 27)) * 0x94d049bb133111eb) & mask
        return x ^ (x >> 31)
    cases = [(0, 0, 0), (1, 0, 1), (1, 1, 3), (187, 1, 59)]
    cases += [(33, 33, layer) for layer in range(3, 60)]
    for extra in [list(range(8)), [0] * 8]:
        physical = list(range(128)) + extra
        copies = [[i for i, e in enumerate(physical) if e == logical] for logical in range(128)]
        counts = torch.tensor([len(x) for x in copies], device='cuda', dtype=torch.int32)
        slots = torch.tensor([x + [-1] * (9-len(x)) for x in copies], device='cuda', dtype=torch.int32)
        for n, real, layer in cases:
            allocated = max(1, n)
            ids = (torch.arange(allocated*4).reshape(allocated, 4) % 129 - 1).long()
            positions = torch.arange(allocated).long() + 131072
            bases = [([0, 1, mask, 0x4d33315043503831][i % 4]) for i in range(allocated)]
            expected = torch.full_like(ids, -1)
            for row in range(real):
                for col in range(4):
                    logical = int(ids[row, col])
                    if logical < 0: continue
                    h = mix(bases[row] ^ int(positions[row]))
                    h = mix(h ^ layer); h = mix(h ^ logical)
                    expected[row, col] = copies[logical][h % len(copies[logical])]
            out = torch.full_like(ids, -9, device='cuda')
            kernel[(max(1, (n*4+255)//256),)](ids.cuda(), positions.cuda(),
                torch.tensor(bases, dtype=torch.uint64, device='cuda'), counts, slots,
                out, n, real, layer, 256)
            assert torch.equal(out[:n].cpu(), expected[:n])
            if n == 0: assert torch.equal(out.cpu(), torch.full_like(ids, -9))
    assert len(kernel.device_caches[torch.cuda.current_device()][0]) == 1


def test_cache_store_reuses_variant_across_rows_and_page_capacity():
    from hcu_vllm_das.parallel.minimax31_prefill_triton import store_pcp_cache, _cache_store
    from hcu_vllm_das.parallel.minimax31_msa import bind_fresh_main_cache
    before = len(_cache_store.device_caches[torch.cuda.current_device()][0])
    for rows, pages in [(1, 1), (17, 4), (33, 4), (1, 4)]:
        raw = torch.randn((rows, 1152), device='cuda', dtype=torch.bfloat16)
        cache = bind_fresh_main_cache(torch.empty((pages, 4, 128, 256),
            device='cuda', dtype=torch.uint8), fresh=True)
        index = torch.empty((pages, 128, 128), device='cuda', dtype=torch.float8_e4m3fn)
        slots = torch.arange(rows, device='cuda', dtype=torch.int64)
        scale = torch.ones((), device='cuda')
        store_pcp_cache(raw, slots, slots, cache.key_writer, cache.value_writer, index, scale, scale)
    torch.cuda.synchronize()
    assert len(_cache_store.device_caches[torch.cuda.current_device()][0]) - before == 1


def test_local_pages_dynamic_geometry_has_at_most_one_variant():
    from hcu_vllm_das.parallel.minimax31_prefill_triton import _local_pages, force_local_pages
    before = len(_local_pages.device_caches[torch.cuda.current_device()][0])
    for heads, queries, pages in [(1,1,1),(2,7,17),(2,9,18),(4,17,129),(4,33,513)]:
        score = torch.randn((heads,queries,pages), device='cuda').transpose(1,2)
        end = torch.arange(queries, device='cuda', dtype=torch.int32) * 193
        ref = score.clone()
        last = (end-1)//128
        ids = torch.arange(pages, device='cuda').view(1,-1,1)
        ref.masked_fill_((ids <= last.view(1,1,-1)) &
                        (ids >= (last-3).clamp_min(0).view(1,1,-1)), 1.e30)
        force_local_pages(score,end,4)
        torch.testing.assert_close(score,ref,rtol=0,atol=0)
    torch.cuda.synchronize()
    assert len(_local_pages.device_caches[torch.cuda.current_device()][0])-before <= 1


def test_quant_q_dynamic_elements_have_at_most_one_variant():
    from hcu_vllm_das.parallel.minimax31_prefill_triton import _quant_q, quant_q
    before = len(_quant_q.device_caches[torch.cuda.current_device()][0])
    scale = torch.tensor(1.3, device='cuda', dtype=torch.float32)
    for elements in [0,1,17,1025,2049,4096]:
        q = torch.linspace(-500,500,elements,device='cuda',dtype=torch.bfloat16)
        out = quant_q(q,scale)
        ref = (q.float()/scale).to(torch.float8_e4m3fn)
        assert torch.equal(out.view(torch.uint8),ref.view(torch.uint8))
    torch.cuda.synchronize()
    assert len(_quant_q.device_caches[torch.cuda.current_device()][0])-before <= 1


@pytest.mark.parametrize('main_dtype',[torch.uint8,torch.float8_e4m3fn])
@pytest.mark.parametrize('scale',[1.,.3,1.7])
def test_cache_store_overflow_bytes_match_stock_writer(main_dtype,scale):
    from hcu_vllm_das.parallel.minimax31_prefill_triton import store_pcp_cache
    from hcu_vllm_das.parallel.minimax31_msa import bind_fresh_main_cache
    from vllm.v1.attention.ops.triton_reshape_and_cache_flash import triton_reshape_and_cache_flash
    from vllm.models.minimax_m3.amd.ops.sparse_pa import minimax_m3_insert_index_cache
    values=torch.tensor([0.,-0.,1.0625,-1.1875,432.,-448.,464.,-500.,512.,-1024.,65504.,-65504.],device='cuda',dtype=torch.bfloat16).repeat(43)[:512]
    raw=torch.empty((3,1168),device='cuda',dtype=torch.bfloat16)[:,:1152]
    raw[:,:512]=values;raw[:,512:1024]=values.flip(0)
    raw[:,1024:]=torch.linspace(-1024,1024,128,device='cuda',dtype=torch.bfloat16)
    a=torch.full((2,4,128,256),17,device='cuda',dtype=main_dtype);b=a.clone()
    bound=bind_fresh_main_cache(a,fresh=True);reference=bind_fresh_main_cache(b,fresh=True)
    index=torch.ones((2,128,128),device='cuda',dtype=torch.float8_e4m3fn);ri=index.clone()
    ms=torch.tensor([0,129,-1],device='cuda',dtype=torch.int64)
    ix=torch.tensor([129,-1,0],device='cuda',dtype=torch.int64)
    ks=torch.tensor(scale,device='cuda');vs=torch.tensor(scale,device='cuda')
    triton_reshape_and_cache_flash(raw[:,:512].view(3,4,128),raw[:,512:1024].view(3,4,128),reference.key_writer,reference.value_writer,ms,'fp8',ks,vs)
    minimax_m3_insert_index_cache(raw[:,1024:].clamp(-448,448),ri,ix)
    store_pcp_cache(raw,ms,ix,bound.key_writer,bound.value_writer,index,ks,vs)
    assert torch.equal(a.view(torch.uint8),b.view(torch.uint8))
    assert torch.equal(index.view(torch.uint8),ri.view(torch.uint8))


def test_pack_profile_budget_and_same_stream_consumption(monkeypatch):
    from hcu_vllm_das.parallel import minimax31_prefill as opt
    monkeypatch.setattr(opt,'_PACK',{})
    a=torch.randn((17,1024),device='cuda',dtype=torch.bfloat16)
    b=torch.randn((17,128),device='cuda',dtype=torch.bfloat16)
    with pytest.raises(RuntimeError,match='profiled'):opt.pack_rows((a,b))
    from types import SimpleNamespace
    from hcu_vllm_das.parallel import minimax31_pcp as pcp
    monkeypatch.setattr(pcp,'_validate_prefill_policy',lambda:True)
    monkeypatch.setattr(pcp,'_prefill_enabled',lambda:True)
    monkeypatch.setattr(pcp.PCPManager,'prepare_inputs_for_capture_or_profile',lambda self,batch:batch)
    manager=pcp.MiniMax31PCPManager.__new__(pcp.MiniMax31PCPManager)
    manager._hcu_native_msa=True;manager._hcu_static=False
    manager.pcp_world_size=8;manager.dcp_world_size=1;manager.shard_decode_requests=True
    def dummy(self,rows,*a,**kw):
        return manager.prepare_inputs_for_capture_or_profile(SimpleNamespace(num_tokens_after_padding=rows,positions=torch.arange(rows,device=self.device)))
    monkeypatch.setattr(pcp.GPUModelRunner,'_dummy_run',dummy)
    runner=pcp.MiniMax31PCPModelRunner.__new__(pcp.MiniMax31PCPModelRunner)
    runner.pcp_manager=manager;runner.max_num_reqs=1;runner.speculator=None;runner.device=a.device
    profiled=runner._dummy_run(257,is_profile=True,skip_attn=True)
    capacity=profiled.num_tokens_after_padding
    assert capacity>=17
    first=opt.pack_rows((a,b));ptr=first.data_ptr()
    assert not first.is_inference()
    # Enqueue consumption before overwriting the borrowed buffer, without a
    # host/device synchronize between these same-stream operations.
    consumed=first.clone()
    second=opt.pack_rows((-a,-b))
    assert second.data_ptr()==ptr
    torch.testing.assert_close(consumed,torch.cat((a,b),1),rtol=0,atol=0)
    torch.testing.assert_close(second,torch.cat((-a,-b),1),rtol=0,atol=0)
    biga=torch.empty((capacity+1,1024),device='cuda',dtype=torch.bfloat16)
    bigb=torch.empty((capacity+1,128),device='cuda',dtype=torch.bfloat16)
    with pytest.raises(RuntimeError,match='profiled'):opt.pack_rows((biga,bigb))
    with torch.cuda.stream(torch.cuda.Stream()):
        with pytest.raises(RuntimeError,match='profiled'):opt.pack_rows((a,b))
