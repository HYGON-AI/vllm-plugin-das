# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""CPU/meta contracts for EP FP8 selection; vendor kernels are never launched."""
from __future__ import annotations

import ast
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

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
        "current_platform": SimpleNamespace(is_rocm=lambda: True),
        "_HCU_HT_EP_TOKEN_ALIGNMENT": 256,
        "compute_aligned_M_and_alignment": native,
    }
    apply = load_function(
        "vllm_hcu/model_executor/layers/fused_moe/experts/deep_gemm_moe.py", "apply", ns
    )
    w1, w2, s1, s2 = tensors(block=True)
    expert = SimpleNamespace(
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
            "swiglustep",4,metadata()["expert_map"],
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
    expert = SimpleNamespace(quant_config=cfg, block_shape=None,
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
        "ep_sorting": lambda *a: (None, None, None, None, rows),
        "_gemm": gemm,
        "ep_swiglustep_quant": lambda *a, q: (q, None),
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


def test_route_workspace_is_reused_when_output_has_separate_storage():
    ns = {"torch": torch}
    carve = load_function(
        "vllm_hcu/model_executor/layers/fused_moe/ep_fused_moe.py", "_carve", ns
    )
    workspace = torch.empty(128)
    out = torch.empty(8, 4)
    routes = carve(workspace, (8, 4, 4), torch.float32, "cpu", avoid=out)
    assert routes.untyped_storage().data_ptr() == workspace.untyped_storage().data_ptr()
