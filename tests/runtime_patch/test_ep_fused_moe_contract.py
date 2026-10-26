# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""CPU/meta contracts for EP FP8 selection; vendor kernels are never launched."""
from __future__ import annotations

import ast
import math
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from vllm.model_executor.layers.fused_moe.activation import MoEActivation

from vllm_hcu.model_executor.layers.fused_moe.ep_fused_moe_support import (
    supports_channel_ep_fused,
)

ROOT = Path(__file__).parents[2]


def load_function(path, name, namespace):
    tree = ast.parse((ROOT / path).read_text())
    function = next(n for n in ast.walk(tree)
                    if isinstance(n, ast.FunctionDef) and n.name == name)
    function.decorator_list = []
    module = ast.Module(body=[
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
        function,
    ], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), path, "exec"), namespace)
    return namespace[name]


def tensors(*, block=False, hidden=4096, intermediate=1536):
    dev = "meta"
    w1 = torch.empty(2, 2 * intermediate, hidden, dtype=torch.float8_e4m3fn, device=dev)
    w2 = torch.empty(2, hidden, intermediate, dtype=w1.dtype, device=dev)
    if block:
        s1 = torch.empty(2, 2 * intermediate // 128, hidden // 128, device=dev)
        s2 = torch.empty(2, hidden // 128, intermediate // 128, device=dev)
    else:
        s1 = torch.empty(2, 2 * intermediate, 1, device=dev)
        s2 = torch.empty(2, hidden, 1, device=dev)
    return w1, w2, s1, s2


def metadata():
    return dict(block_shape=None, activation="swiglustep",
                expert_map=torch.empty(4, dtype=torch.int32, device="meta"),
                apply_router_weight_on_input=False)


@pytest.mark.parametrize("change", [
    {}, {"block_shape": [128, 128]}, {"activation": "silu"},
    {"expert_map": None}, {"apply_router_weight_on_input": True},
])
def test_shared_capability_contract(change):
    args = metadata()
    args.update(change)
    assert supports_channel_ep_fused(*tensors(), **args) == (not change)


@pytest.mark.parametrize("hidden,intermediate", [(2048, 1536), (4096, 2048), (0, 1536)])
def test_unsupported_kernel_dimensions_are_not_kept_unpacked(hidden, intermediate):
    assert not supports_channel_ep_fused(
        *tensors(hidden=hidden, intermediate=intermediate), **metadata()
    )


def test_block_scales_are_rejected_even_without_block_metadata():
    assert not supports_channel_ep_fused(*tensors(block=True), **metadata())


class NativePathReached(Exception):
    pass


def test_block_fp8_uses_original_deepgemm_path_with_opt_in_enabled():
    def native(**kwargs):
        raise NativePathReached

    ns = {
        "torch": torch, "_EP_FUSED_MOE": True,
        "MoEActivation": MoEActivation, "_STEP5_BACKEND_AUDIT": False,
        "current_platform": SimpleNamespace(is_rocm=lambda: True),
        "_HCU_HT_EP_TOKEN_ALIGNMENT": 256,
        "compute_aligned_M_and_alignment": native,
    }
    apply = load_function(
        "vllm_hcu/model_executor/layers/fused_moe/experts/deep_gemm_moe.py", "apply", ns
    )
    w1, w2, s1, s2 = tensors(block=True)
    expert = SimpleNamespace(
        _hcu_step5_owner=True,
        quant_config=SimpleNamespace(use_fp8_w8a8=True, use_int8_w8a8=False),
        block_shape=[128, 128], w1_scale=s1, w2_scale=s2,
        _hcu_logical_n=3072, _hcu_logical_k=4096,
        _apply_ep_fused=lambda *a: pytest.fail("block FP8 entered channel EP GEMM"),
    )
    with pytest.raises(NativePathReached):
        apply(
            expert, torch.empty(4,4096,device="meta"),
            torch.empty(4,4096,dtype=w1.dtype,device="meta"), w1,w2,
            torch.empty(4,2,device="meta"),
            torch.empty(4,2,dtype=torch.int32,device="meta"),
            MoEActivation.SWIGLUSTEP,4,metadata()["expert_map"],
            torch.empty(4,32,device="meta"),None,
            torch.empty(1,device="meta"),torch.empty(1,device="meta"),None,False,
        )


@pytest.mark.parametrize("case", ["channel", "block_scales", "block_activation", "not_per_token"])
def test_ep_execution_validates_channel_scale_semantics(monkeypatch, case):
    fn = load_function(
        "vllm_hcu/model_executor/layers/fused_moe/experts/deep_gemm_moe.py",
        "_apply_ep_fused", {"torch":torch},
    )
    w1,w2,s1,s2 = tensors(block=case == "block_scales")
    cfg = SimpleNamespace(use_fp8_w8a8=True, per_out_ch_quant=True,
                          per_act_token_quant=case != "not_per_token")
    expert = SimpleNamespace(_hcu_step5_owner=True, quant_config=cfg, block_shape=None,
                             w1_scale=s1,w2_scale=s2,gemm1_clamp_limit=7.0)
    seen=[]
    vendor=ModuleType("vllm_hcu.model_executor.layers.fused_moe.ep_fused_moe")
    vendor.ep_fused_experts=lambda *a,**kw:seen.append((a,kw))
    monkeypatch.setitem(sys.modules,vendor.__name__,vendor)
    args = (
        expert,torch.empty(4,4096,device="meta"),
        torch.empty(4,4096,dtype=w1.dtype,device="meta"),w1,w2,
        torch.empty(4,2,device="meta"),torch.empty(4,2,dtype=torch.int32,device="meta"),
        "swiglustep",4,metadata()["expert_map"],
        torch.empty(4,32 if case=="block_activation" else 1,device="meta"),
        None,None,False,
    )
    if case == "channel":
        fn(*args)
        assert len(seen)==1
        assert seen[0][1]["a1_scale"].shape == (4,1)
    else:
        with pytest.raises(ValueError):
            fn(*args)
        assert not seen


@pytest.mark.parametrize("case", [
    "channel", "block", "silu", "no_ep", "router_on_input",
    "bad_dimensions", "block_scales", "static_activation", "batched", "disabled",
    "master_disabled",
])
def test_weight_preparation_agrees_with_execution_capability(monkeypatch, case):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "0" if case=="master_disabled" else "1")
    monkeypatch.setenv("VLLM_HCU_EP_FUSED_MOE", "0" if case=="disabled" else "1")
    if case == "master_disabled":
        # The effective master-off path must decide to pack before importing
        # the optional EP-fused capability provider.
        support = ModuleType(
            "vllm_hcu.model_executor.layers.fused_moe.ep_fused_moe_support"
        )
        monkeypatch.setitem(sys.modules, support.__name__, support)
    packed=[]
    module=ModuleType("deepgemm")
    def pack(weight):
        packed.append(weight)
        return torch.empty(1,1,1,1,1,1,dtype=weight.dtype,device="meta")
    module.marlin_fp8_contiguous_weight=pack
    module.marlin_fp8_masked_weight=pack
    monkeypatch.setitem(sys.modules,"deepgemm",module)
    enum=SimpleNamespace(HCU_DEEPGEMM="hcu",AITER="aiter",DEEPGEMM="deep",
                         BATCHED_DEEPGEMM="batched")
    ns={"hcu_enum":enum,"convert":lambda *a:("native",)}
    convert=load_function(
        "vllm_hcu/patch/worker/op_opt/moe/patch_fp8_oracle.py",
        "hcu_convert_to_fp8_moe_kernel_format",ns,
    )
    layer=SimpleNamespace(
        _hcu_step5_owner=True,
        weight_block_size=[128,128] if case=="block" else None,
        activation="silu" if case=="silu" else "swiglustep",
        expert_map=None if case=="no_ep" else metadata()["expert_map"],
        apply_router_weight_on_input=case=="router_on_input",
    )
    weights=tensors(block=case in ("block","block_scales"),
                    hidden=2048 if case=="bad_dimensions" else 4096)
    result=convert("batched" if case=="batched" else "deep",layer,*weights,
                   torch.ones(1) if case=="static_activation" else None,None)
    if case=="block":
        assert result==("native",) and not packed
    elif case=="channel":
        assert result[0] is weights[0] and result[1] is weights[1]
        assert result[2].shape==(2,3072) and not packed
    else:
        assert len(packed)==2
        assert result[0].ndim==6 and result[1].ndim==6


@pytest.mark.parametrize("output_offset", [0, 4096])
def test_gathered_route_reduction_does_not_alias_output(output_offset):
    m, topk, hidden, intermediate = 8, 8, 4096, 1536
    rows = m * topk
    workspace = torch.empty(rows * max(2 * intermediate, hidden))
    out = workspace[output_offset:output_offset + m * hidden].view(m, hidden)
    observed = {}

    def gemm(a, b, result, *args, **kwargs):
        if result.ndim == 3:
            result.fill_(1)
            observed["routes"] = result

    def reduce(routes, output, ids, expert_map, scale, *add):
        # A legal program schedule that exposes the former cross-token alias.
        for token in [1, 0, *range(2, m)]:
            output[token].copy_(routes[token].sum(0) * scale)

    ns = {
        "torch": torch,
        "tl": SimpleNamespace(bfloat16=None, float16=None, float32=None),
        "FP8": torch.float8_e4m3fn,
        "STEP5_PREFILL_ASM_G1": False,
        "ep_sorting": lambda *a: (None, None, None, None, rows),
        "_gemm": gemm,
        "ep_swiglustep_quant": lambda *a, q: (q, None),
        "DIRECT_MIN": 0,
        "direct_scratch": None,
        "ep_masked_moe_sum": reduce,
        "shared_add_provider": None,
    }
    path = "vllm_hcu/model_executor/layers/fused_moe/ep_fused_moe.py"
    load_function(path, "_carve", ns)
    run = load_function(path, "_ep_fused_experts", ns)
    result = run(
        torch.zeros(m, hidden),
        SimpleNamespace(shape=(44, 2 * intermediate, hidden)),
        SimpleNamespace(shape=(44, hidden, intermediate)),
        torch.ones(m, topk), torch.zeros(m, topk, dtype=torch.int32),
        torch.arange(44), None, None, a1_scale=torch.ones(1),
        configs=({"BLOCK_SIZE_M": 1}, {"BLOCK_SIZE_M": 1}),
        out=out, workspace13=workspace,
    )
    assert result is out
    assert observed["routes"].untyped_storage().data_ptr() != out.untyped_storage().data_ptr()
    torch.testing.assert_close(result, torch.full_like(result, 8))


@pytest.mark.parametrize("chunk,full", [(2, 8), (8, 8)])
def test_modular_chunk_scratch_has_independent_full_output(chunk, full):
    requests = []

    def allocate(*specs):
        requests.extend(specs)
        sizes=[(math.prod(shape)*dtype.itemsize+255)//256*256 for shape,dtype in specs]
        storage=torch.empty(sum(sizes),dtype=torch.uint8)
        offset=0;views=[]
        for (shape,dtype),size in zip(specs,sizes):
            views.append(storage[offset:offset+math.prod(shape)*dtype.itemsize].view(dtype).view(shape))
            offset+=size
        return tuple(views)

    ns = {
        "torch": torch, "prod": math.prod,
        "current_workspace_manager": lambda: SimpleNamespace(get_simultaneous=allocate),
        "_resize_cache": lambda tensor, shape: tensor.flatten()[:math.prod(shape)].view(shape),
        "_step5_ep_workspace_announced": False,
    }
    fn = load_function(
        "vllm_hcu/model_executor/layers/fused_moe/modular_kernel.py",
        "_allocate_buffers", ns,
    )
    expert = SimpleNamespace(
        _hcu_step5_owner=True,
        workspace_dtype=lambda dtype: dtype,
        workspace_shapes=lambda m,n,k,topk,*a: ((m*topk,k),(m*topk,n),(m,k)),
    )
    w13, w2, output = fn(SimpleNamespace(fused_experts=expert),
        torch.bfloat16, "cpu", chunk, full, 16, 8, 2, 352, 22, None, "swiglustep")
    assert w13.shape == (chunk*2, 8) and w2.shape == (chunk*2, 16)
    assert output.shape == (full, 8)
    if chunk < full:
        assert len(requests) == 3
        intervals=sorted((t.data_ptr(),t.data_ptr()+t.numel()*t.element_size()) for t in (w13,w2,output))
        assert all(a[1]<=b[0] for a,b in zip(intervals,intervals[1:]))
    else:
        assert len(requests) == 2
        assert w13.untyped_storage().data_ptr() == output.untyped_storage().data_ptr()


@pytest.mark.parametrize("case", [
    "enabled", "disabled", "master_disabled", "backend_disabled", "not_rocm",
    "block", "not_per_token", "not_per_channel", "other_experts", "other_global",
    "other_topk", "silu", "no_map", "router_on_input", "lora", "bad_scale",
])
@pytest.mark.parametrize("local_experts", [22, 23, 44])
def test_bounded_step5_scratch_requires_the_actual_channel_ep_backend(monkeypatch, case, local_experts):
    w1 = torch.empty(local_experts*2 if case=="other_experts" else local_experts,3072,4096,
                     device="meta",dtype=torch.float8_e4m3fn)
    w2 = torch.empty(local_experts,4096,1536,device="meta",dtype=w1.dtype)
    s1 = torch.empty(local_experts,3072,1,device="meta")
    s2 = torch.empty(local_experts,4096,1,device="meta")
    if case=="bad_scale":s1=torch.empty(1,device="meta")
    expert = SimpleNamespace(
        _hcu_step5_owner=True,
        quant_config=SimpleNamespace(use_fp8_w8a8=True,
            per_out_ch_quant=case!="not_per_channel",per_act_token_quant=case!="not_per_token"),
        block_shape=[128,128] if case=="block" else None,
        w1_scale=s1,w2_scale=s2,_lora_context=object() if case=="lora" else None,
    )
    monkeypatch.setenv("HCU_STEP5_BOUNDED_EP_WORKSPACE", "0" if case=="disabled" else "1")
    vendor=ModuleType("vllm_hcu.model_executor.layers.fused_moe.ep_fused_moe")
    vendor.CHUNK=32768;monkeypatch.setitem(sys.modules,vendor.__name__,vendor)
    fn=load_function(
        "vllm_hcu/model_executor/layers/fused_moe/experts/deep_gemm_moe.py",
        "step5_ep_chunk_size", {
            "os":os,"_EP_FUSED_MOE":case!="backend_disabled",
            "current_platform":SimpleNamespace(is_rocm=lambda:case!="not_rocm"),
            "henvs":SimpleNamespace(optional_custom_op_enabled=lambda enabled:enabled and case!="master_disabled"),
        },
    )
    global_experts = 368 if local_experts == 23 else 352
    result=fn(expert,w1,w2,"silu" if case=="silu" else "swiglustep",
        704 if case=="other_global" else global_experts,
        None if case=="no_map" else torch.empty(global_experts,dtype=torch.int32,device="meta"),
        4 if case=="other_topk" else 8,case=="router_on_input")
    assert result==(32768 if case=="enabled" else None)


def test_route_workspace_is_reused_when_output_has_separate_storage():
    ns = {"torch": torch}
    carve = load_function(
        "vllm_hcu/model_executor/layers/fused_moe/ep_fused_moe.py", "_carve", ns
    )
    workspace = torch.empty(128)
    out = torch.empty(8, 4)
    routes = carve(workspace, (8, 4, 4), torch.float32, "cpu", avoid=out)
    assert routes.untyped_storage().data_ptr() == workspace.untyped_storage().data_ptr()


@pytest.mark.parametrize("tokens", [8, 9, 25, 32])
@pytest.mark.parametrize("output_offset", [0, 4096])
def test_later_chunk_gate_up_preserves_earlier_chunk_output(tokens, output_offset):
    """Exercise the real chunk loop with output in the modular scratch storage."""
    topk, hidden, intermediate = 8, 4096, 1536
    workspace = torch.empty(tokens * topk * hidden + output_offset)
    out = workspace[output_offset:output_offset + tokens * hidden].view(tokens, hidden)

    def gemm(a, b, result, *args, **kwargs):
        # GEMM1 reuses the beginning of workspace13 on every chunk.
        result.fill_(13 if result.ndim == 2 else 1)

    ns = {
        "torch": torch,
        "tl": SimpleNamespace(bfloat16=None, float16=None, float32=None),
        "FP8": torch.float8_e4m3fn, "CHUNK": 2, "_announced": True,
        "STEP5_PREFILL_ASM_G1": False,
        "ep_sorting": lambda ids, *a: (None, None, None, None, ids.numel()),
        "_gemm": gemm,
        "ep_swiglustep_quant": lambda *a, q: (q, None),
        "DIRECT_MIN": 0, "shared_add_provider": None,
        "ep_masked_moe_sum": lambda routes, output, *a: output.copy_(routes.sum(1)),
    }
    path = "vllm_hcu/model_executor/layers/fused_moe/ep_fused_moe.py"
    load_function(path, "_carve", ns)
    load_function(path, "_ep_fused_experts", ns)
    run = load_function(path, "ep_fused_experts", ns)
    result = run(
        torch.zeros(tokens, hidden),
        SimpleNamespace(shape=(22, 2 * intermediate, hidden)),
        SimpleNamespace(shape=(22, hidden, intermediate)),
        torch.ones(tokens, topk), torch.zeros(tokens, topk, dtype=torch.int32),
        torch.arange(352), None, None,
        a1_scale=torch.ones(tokens, 1), out=out, workspace13=workspace,
        configs=({"BLOCK_SIZE_M": 1}, {"BLOCK_SIZE_M": 1}),
    )
    assert result is out
    torch.testing.assert_close(result, torch.full_like(result, 8))


def test_carve_reuses_disjoint_views_of_one_backing_allocation():
    ns={"torch":torch}
    carve=load_function("vllm_hcu/model_executor/layers/fused_moe/ep_fused_moe.py","_carve",ns)
    backing=torch.empty(256)
    scratch=backing[:128];output=backing[128:].view(8,16)
    result=carve(scratch,(8,4,4),torch.float32,"cpu",avoid=output)
    assert result.data_ptr()==scratch.data_ptr()


@pytest.mark.parametrize("mode", ["main", "fallback", "unknown", "flag_off", "sp", "dbo"])
def test_runtime_selected_wrapper_uses_bounded_workspace(monkeypatch, mode):
    """Exercise the same outer expert class returned by the installed FP8 oracle."""
    from vllm.model_executor.layers.fused_moe.experts import triton_deep_gemm_moe as selector
    from vllm.model_executor.layers.fused_moe.prepare_finalize import naive_dp_ep
    from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
        Fp8MoeBackend, backend_to_kernel_cls,
    )
    cls=backend_to_kernel_cls(Fp8MoeBackend.DEEPGEMM)[0]
    assert cls is selector.TritonOrDeepGemmExperts
    if mode=="unknown":cls=type("UnreviewedSelector",(cls,),{})
    outer=object.__new__(cls)
    outer.experts=SimpleNamespace(_hcu_step5_owner=True,step5_ep_chunk_size=lambda *a:32768)
    outer.fallback_experts=SimpleNamespace()
    monkeypatch.setattr(selector,"_valid_deep_gemm",lambda *a:mode!="fallback")
    monkeypatch.setattr(selector,"is_deep_gemm_e8m0_used",lambda:False)
    monkeypatch.setenv("HCU_STEP5_BOUNDED_EP_WORKSPACE","0" if mode=="flag_off" else "1")
    allocations=[]
    class Allocated(Exception):pass
    def allocate(dtype,device,chunk,full,*args):
        allocations.append((chunk,full));raise Allocated
    pf=naive_dp_ep.MoEPrepareAndFinalizeNaiveDPEPModular(is_sequence_parallel=mode=="sp")
    runner=SimpleNamespace(fused_experts=outer,prepare_finalize=pf,N=-1,K=-1,_allocate_buffers=allocate)
    ns={"torch":torch,"os":os,"dbo_enabled":lambda:mode=="dbo"}
    path="vllm_hcu/model_executor/layers/fused_moe/modular_kernel.py"
    tree=ast.parse((ROOT/path).read_text())
    if any(isinstance(n,ast.FunctionDef) and n.name=="_step5_ep_chunk_provider" for n in tree.body):
        load_function(path,"_step5_ep_chunk_provider",ns)
    fn=load_function(path,"_fused_experts",ns)
    with pytest.raises(Allocated):
        fn(runner,torch.bfloat16,torch.empty(65536,4096,device="meta",dtype=torch.float8_e4m3fn),
           torch.empty(65536,1,device="meta"),
           torch.empty(22,3072,4096,device="meta",dtype=torch.float8_e4m3fn),
           torch.empty(22,4096,1536,device="meta",dtype=torch.float8_e4m3fn),
           torch.empty(65536,8,device="meta"),torch.empty(65536,8,device="meta",dtype=torch.int32),
           "swiglustep",352,22,torch.empty(352,device="meta",dtype=torch.int32),False,None)
    assert allocations==[(32768 if mode=="main" else 65536,65536)]
