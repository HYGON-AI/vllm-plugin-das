"""Exercise real Python admission functions without Torch, Triton, or a GPU."""
from __future__ import annotations

import ast
import math
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
G2 = "vllm_hcu/ops/step5/prefill_g2.py"
DEEP = "vllm_hcu/model_executor/layers/fused_moe/experts/deep_gemm_moe.py"


def load_function(path, name, namespace):
    tree = ast.parse((ROOT/path).read_text(encoding="utf-8"))
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    fn.decorator_list = []
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, fn], type_ignores=[])), path, "exec"), namespace)
    return namespace[name]


class Tensor:
    def __init__(self, shape, dtype="fp8", device=None, contiguous=True):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = device or SimpleNamespace(type="cuda", index=0)
        self.contiguous = contiguous

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def is_cuda(self):
        return self.device.type == "cuda"

    def numel(self):
        return math.prod(self.shape)

    def is_contiguous(self):
        return self.contiguous


CONFIG = dict(BLOCK_SIZE_M=256, BLOCK_SIZE_N=256, BLOCK_SIZE_K=64,
              GROUP_SIZE_M=1, COMBINE_SCALE_LOAD=False, USE_MLS_LOAD=False,
              schedule_hint="local-prefetch", sched_latency="none", kpack=1,
              use_block_pingpong=True, buffer_cache_swizzle=False,
              num_warps=8, num_stages=2)


def g2_case(experts=23):
    dev = SimpleNamespace(type="cuda", index=0)
    def t(shape, dtype="fp8"):
        return Tensor(shape, dtype, dev)
    tensors = [t((256, 1536)), t((experts, 4096, 1536)),
               t((65536, 4096), "bf16"), t((65536, 8, 4096), "bf16"),
               t((256, 1), "fp32"), t((experts, 4096, 1), "fp32"),
               t((256,), "int32"), t((256,), "int32"),
               t((1,), "int32"), t((3,), "int32")]
    return tensors, dict(num_tokens=65536, top_k=8, sort_block_m=256, config=dict(CONFIG))


def g2_function(architecture=True):
    launches = []
    class Kernel:
        def __getitem__(self, grid):
            return lambda *a, **k: launches.append((grid, a, k))
    ns = dict(torch=SimpleNamespace(float8_e4m3fn="fp8", bfloat16="bf16",
                                   float32="fp32", int32="int32"),
              triton=SimpleNamespace(cdiv=lambda a, b: (a+b-1)//b),
              _is_gfx938=lambda _: architecture, _prefill_g2_kernel=Kernel(),
              _announced=True)
    return load_function(G2, "try_step5_prefill_g2", ns), launches


@pytest.mark.parametrize("experts,accepted", [(22, True), (23, True), (24, False), (44, False)])
def test_g2_accepts_only_validated_expert_layouts(experts, accepted):
    tensors, kwargs = g2_case(experts)
    fn, launches = g2_function()
    assert fn(*tensors, **kwargs) is accepted
    assert len(launches) == int(accepted)
    if accepted:
        grid, args, launch = launches[0]
        assert grid == (16,) and args[10:] == (65536, 4096, 1536, 8, 256, 256, 64)
        assert launch["num_warps"] == 8 and launch["num_stages"] == 2


@pytest.mark.parametrize("experts,scale_experts", [(22, 23), (23, 22), (23, 24)])
def test_g2_channel_scale_count_matches_actual_experts(experts, scale_experts):
    tensors, kwargs = g2_case(experts)
    tensors[5].shape = (scale_experts, 4096, 1)
    fn, launches = g2_function()
    assert fn(*tensors, **kwargs) is False and not launches


@pytest.mark.parametrize("key,value", [
    ("num_tokens", 128), ("num_tokens", 129), ("num_tokens", 4096),
    ("num_tokens", 65535), ("num_tokens", 65537),
    ("top_k", 4), ("top_k", 16), ("sort_block_m", 128),
])
def test_private_smoke_shapes_are_not_production_eligible(key, value):
    tensors, kwargs = g2_case()
    kwargs[key] = value
    fn, launches = g2_function()
    assert fn(*tensors, **kwargs) is False and not launches


@pytest.mark.parametrize("index", range(10))
@pytest.mark.parametrize("bad", ["noncontiguous", "other_device"])
def test_every_tensor_must_share_contiguous_device_storage(index, bad):
    tensors, kwargs = g2_case()
    if bad == "noncontiguous":
        tensors[index].contiguous = False
    else:
        tensors[index].device = SimpleNamespace(type="cuda", index=1)
    fn, launches = g2_function()
    assert fn(*tensors, **kwargs) is False and not launches


@pytest.mark.parametrize("index,attribute,value", [
    (0, "shape", (256, 1537)), (0, "dtype", "fp8_fnuz"),
    (0, "device", SimpleNamespace(type="cpu", index=None)),
    (2, "shape", (65535, 4096)), (3, "shape", (65536, 7, 4096)),
    (4, "dtype", "bf16"), (5, "dtype", "bf16"),
    (6, "shape", (255,)), (6, "shape", (1, 256)),
    (7, "shape", (255,)), (8, "shape", (0,)),
    (8, "dtype", "int64"), (9, "shape", (2,)), (9, "dtype", "int64"),
])
def test_g2_metadata_rejection_never_launches(index, attribute, value):
    tensors, kwargs = g2_case()
    setattr(tensors[index], attribute, value)
    fn, launches = g2_function()
    assert fn(*tensors, **kwargs) is False and not launches


@pytest.mark.parametrize("key,value", [
    ("BLOCK_SIZE_M", 128), ("BLOCK_SIZE_N", 128), ("BLOCK_SIZE_K", 128),
    ("GROUP_SIZE_M", 2), ("num_warps", 4), ("num_stages", 1),
    ("USE_MLS_LOAD", True), ("COMBINE_SCALE_LOAD", True),
    ("schedule_hint", "none"), ("sched_latency", "2"), ("kpack", 2),
    ("use_block_pingpong", False), ("buffer_cache_swizzle", True),
    ("PERSISTENT_PROGRAMS", 256),
])
def test_unmeasured_launch_recipes_still_use_native_fallback(key, value):
    tensors, kwargs = g2_case()
    kwargs["config"][key] = value
    fn, launches = g2_function()
    assert fn(*tensors, **kwargs) is False and not launches


def test_other_architecture_never_launches():
    tensors, kwargs = g2_case()
    fn, launches = g2_function(architecture=False)
    assert fn(*tensors, **kwargs) is False and not launches


@pytest.mark.parametrize("local_e,global_e,map_shape,accepted", [
    (22, 352, (352,), True), (44, 352, (352,), True),
    (23, 368, (368,), True), (23, 352, (368,), False),
    (23, 368, (352,), False), (23, 368, (16, 23), False),
    (23, 368, None, False), (22, 368, (368,), False),
    (44, 368, (368,), False), (24, 368, (368,), False),
])
def test_bounded_workspace_replica_namespace_is_explicit(monkeypatch, local_e, global_e, map_shape, accepted):
    support_name = "vllm_hcu.model_executor.layers.fused_moe.ep_fused_moe_support"
    vendor_name = "vllm_hcu.model_executor.layers.fused_moe.ep_fused_moe"
    support, vendor = ModuleType(support_name), ModuleType(vendor_name)
    support.supports_channel_ep_fused = lambda *a, **k: True
    vendor.CHUNK = 65536
    monkeypatch.setitem(sys.modules, support_name, support)
    monkeypatch.setitem(sys.modules, vendor_name, vendor)
    monkeypatch.setenv("HCU_STEP5_BOUNDED_EP_WORKSPACE", "1")
    fn = load_function(DEEP, "_hcu_step5_ep_chunk_size", dict(
        os=os, _EP_FUSED_MOE=True, current_platform=SimpleNamespace(is_rocm=lambda: True),
        henvs=SimpleNamespace(optional_custom_op_enabled=lambda value: value)))
    expert = SimpleNamespace(_hcu_step5_owner=True, block_shape=None, _lora_context=None,
        quant_config=SimpleNamespace(use_fp8_w8a8=True, per_out_ch_quant=True, per_act_token_quant=True),
        w1_scale=None, w2_scale=None)
    emap = Tensor(map_shape, "int32") if map_shape is not None else None
    result = fn(expert, Tensor((local_e, 3072, 4096)), Tensor((local_e, 4096, 1536)),
                "swiglustep", global_e, emap, 8, False)
    assert result == (65536 if accepted else None)
