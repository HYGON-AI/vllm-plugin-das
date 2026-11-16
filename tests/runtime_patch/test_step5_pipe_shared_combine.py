# SPDX-License-Identifier: Apache-2.0
"""CPU protocol regressions for pipeline-owned combine and shared output.

Run the real pipeline, EP expert provider and runner combine bodies. Only
GPU kernels, NCCL transport and stream fences are replaced at their boundaries.
"""
from __future__ import annotations

import ast
from contextlib import nullcontext
from contextvars import Context
import ctypes
import importlib.util
from pathlib import Path
import sys
from types import MethodType, ModuleType, SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).parents[2]
MOE = "vllm_hcu/model_executor/layers/fused_moe/"


def _source(path, names, namespace, *, class_name=None):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    if class_name is not None:
        tree = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                    and node.name == class_name)
    nodes = [node for node in ast.walk(tree)
             if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    assert len(nodes) == len(names)
    for node in nodes:
        node.decorator_list = []
    module = ast.Module(body=[
        ast.ImportFrom("__future__", [ast.alias("annotations")], 0), *nodes,
    ], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), path, "exec"), namespace)
    return namespace


def _module(monkeypatch, name, **values):
    if "." in name:
        parent, child = name.rsplit(".", 1)
        if parent not in sys.modules:
            _module(monkeypatch, parent)
    module = ModuleType(name)
    module.__path__ = []
    module.__dict__.update(values)
    monkeypatch.setitem(sys.modules, name, module)
    if "." in name:
        monkeypatch.setattr(sys.modules[parent], child, module, raising=False)
    return module


def _light_module(path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _runtime(monkeypatch, *, dbo=True, sizes=(64, 64, 64, 64), rank=0,
             ep_combine=True, inline=True, uniform=False):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    policy = _light_module(MOE + "combine_runtime.py", "pipe_shared_policy")
    henvs = _light_module("vllm_hcu/platforms/envs.py", "pipe_shared_envs")
    # The old source lacks these helpers. Let the same regression reach its
    # original finish assertion, rather than fail due to a missing test helper.
    allowed = getattr(policy, "ep_shared_add_allowed", lambda: True)
    suspend = getattr(policy, "suspend_ep_shared_add", nullcontext)
    modes = SimpleNamespace(NO_OVERLAP="inline", MULTI_STREAM_OVERLAPPED="overlap")
    ctx = SimpleNamespace(dp_metadata=SimpleNamespace(
        get_chunk_sizes_across_dp_rank=lambda: list(sizes)))
    events = []
    ca = SimpleNamespace(
        disabled=False,
        should_custom_ar=lambda value: value.numel() * value.element_size() <= 8 * 1024**2,
        # Peers contribute the same routed partial (2); only our own shared
        # addition is needed in the returned DP row slice.
        custom_all_reduce=lambda value: value + 2 * (len(sizes) - 1),
    )
    dp = SimpleNamespace(rank_in_group=rank, world_size=len(sizes))
    ep = SimpleNamespace(world_size=len(sizes), device_communicator=SimpleNamespace(ca_comm=ca))
    runner_ns = _source(MOE + "moe_runner.py", {
        "_EpFusedCombineState", "_ep_fused_rows", "_ep_fused_all_reduce",
        "_maybe_combine", "_ep_fused_combine_output",
    }, dict(torch=torch, ep_shared_add_allowed=allowed, SharedExpertsOrder=modes,
            get_ep_group=lambda: ep, get_dp_group=lambda: dp,
            get_tensor_model_parallel_world_size=lambda: 1,
            get_forward_context=lambda: ctx))
    state = runner_ns["_EpFusedCombineState"]()
    runner_ns["_EP_COMBINE"] = state

    class Stream:
        cuda_stream = 0

        def wait_stream(self, other):
            pass

        def wait_event(self, event):
            pass

    class Event:
        def record(self, stream):
            pass

    stream = Stream()
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: stream)
    monkeypatch.setattr(torch.cuda, "Stream", lambda **kwargs: Stream())
    monkeypatch.setattr(torch.cuda, "Event", Event)

    def view(pointer, count, dtype):
        if not count:
            return torch.empty(0, dtype=dtype)
        size = torch.empty((), dtype=dtype).element_size()
        buf = (ctypes.c_ubyte * (count * size)).from_address(pointer)
        return torch.frombuffer(buf, dtype=dtype, count=count)

    class NCCL:
        def ncclGroupStart(self):
            pass

        def ncclGroupEnd(self):
            pass

        def ncclBroadcast(self, src, dst, count, dtype, root, comm, stream):
            ctypes.memmove(dst, src, count * torch.empty((), dtype=dtype).element_size())

        def ncclAllGather(self, src, dst, count, dtype, comm, stream):
            for peer in range(len(sizes)):
                view(dst, count * len(sizes), dtype).view(len(sizes), count)[peer].copy_(
                    view(src, count, dtype))

        def ncclReduce(self, src, dst, count, dtype, op, root, comm, stream):
            if root == rank and count:
                view(dst, count, dtype).copy_(view(src, count, dtype) * len(sizes))

        def ncclReduceScatter(self, src, dst, count, dtype, op, comm, stream):
            value = view(src, count * len(sizes), dtype).view(len(sizes), count)[rank]
            view(dst, count, dtype).copy_(value * len(sizes))

    dp.device_communicator = SimpleNamespace(pynccl_comm=SimpleNamespace(
        disabled=False, nccl=NCCL(), comm=object()))
    _module(monkeypatch, "vllm.distributed", get_dp_group=lambda: dp)
    _module(monkeypatch, "vllm.distributed.device_communicators.pynccl_wrapper",
            buffer_type=lambda p: p, cudaStream_t=lambda s: s,
            ncclRedOpTypeEnum=SimpleNamespace(from_torch=lambda op: op))
    _module(monkeypatch, "vllm.forward_context", get_forward_context=lambda: ctx)

    class NoOP:
        pass

    class Naive:
        is_sequence_parallel = False
        _lora_context = None

    _module(monkeypatch, "vllm.model_executor.layers.fused_moe.topk_weight_and_reduce",
            TopKWeightAndReduceNoOP=NoOP)
    _module(monkeypatch, "vllm.model_executor.layers.fused_moe.prepare_finalize.naive_dp_ep",
            MoEPrepareAndFinalizeNaiveDPEPModular=Naive,
            _quantize_and_setup_dispatch=lambda value, config: (
                value, [torch.ones((value.shape[0], 1))], None))

    def masked_sum(cache, out, ids, expert_map, factor, *add):
        out.fill_(2)
        if add:
            shared, start = add
            out[start:start + shared.shape[0]].add_(shared)
        events.append(("shared-add", bool(add), out.shape[0]))

    def activation(cache, weights, valid, rows, limit, q):
        return q, torch.ones((rows, 1))

    ep_ns = _source(MOE + "ep_fused_moe.py", {"_ep_fused_experts"}, dict(
        torch=torch, tl=SimpleNamespace(bfloat16=0, float16=1, float32=2),
        FP8=torch.float8_e4m3fn,
        pick_configs=lambda *args: ({"BLOCK_SIZE_M": 16}, {"BLOCK_SIZE_M": 16}),
        per_token_quant_triton=lambda value, **kwargs: (value, torch.ones((value.shape[0], 1))),
        STEP5_EXPERT_GRID=False, STEP5_PREFILL_ASM_G1=False, DIRECT_MIN=0,
        ep_sorting=lambda ids, *args: (None, None, None, None, ids.shape[0]),
        _carve=lambda workspace, shape, dtype, device, **kwargs: torch.empty(shape, dtype=dtype),
        _gemm=lambda *args, **kwargs: None,
        ep_swiglustep_quant=activation, ep_masked_moe_sum=masked_sum,
        shared_add_provider=state.shared_add if ep_combine else None,
    ))
    support = _light_module(MOE + "ep_fused_moe_support.py", "pipe_ep_support")
    _module(monkeypatch, "vllm_hcu.model_executor.layers.fused_moe.ep_fused_moe_support",
            supports_channel_ep_fused=support.supports_channel_ep_fused)
    _module(monkeypatch, "vllm_hcu.model_executor.layers.fused_moe.ep_fused_moe",
            ep_fused_experts=ep_ns["_ep_fused_experts"])
    deep = _source(MOE + "experts/deep_gemm_moe.py", {"apply", "_apply_ep_fused"}, dict(
        torch=torch, henvs=henvs, _EP_FUSED_MOE=True,
        current_platform=SimpleNamespace(is_rocm=lambda: True),
        MoEActivation=SimpleNamespace(SWIGLUSTEP="swiglustep")), class_name="DeepGemmExperts")
    expert = SimpleNamespace(
        _hcu_step5_owner=True, block_shape=None,
        quant_config=SimpleNamespace(use_fp8_w8a8=True, use_int8_w8a8=False,
                                     per_out_ch_quant=True, per_act_token_quant=True),
        w1_scale=torch.empty((1, 6), device="meta"),
        w2_scale=torch.empty((1, 4096), device="meta"),
        gemm1_clamp_limit=7.0, a2_scale=None, expects_unquantized_inputs=False,
        activation_format=lambda: "standard", finalize_weight_and_reduce_impl=NoOP,
        moe_problem_size=lambda a, w1, w2, ids: (1, a.shape[0], 6, 4096, 1),
    )
    expert._apply_ep_fused = MethodType(deep["_apply_ep_fused"], expert)
    expert.apply = MethodType(deep["apply"], expert)
    handoff = SimpleNamespace(callback=None)

    def yield_to_comm():
        if handoff.callback is not None:
            callback, handoff.callback = handoff.callback, None
            callback()

    pipe_ns = _source(MOE + "modular_kernel.py", {"_dp_pipe_apply"}, dict(
        torch=torch, henvs=henvs, suspend_ep_shared_add=suspend,
        dbo_enabled=lambda: dbo, join_dp_moe_pipe=lambda: None,
        FusedMoEActivationFormat=SimpleNamespace(Standard="standard"),
        _DP_MOE_PIPE_MIN_TOKENS=1, _EP_DIRECT_SUM=0, _DP_PIPE_UNIFORM=uniform,
        _dp_pipe_stream=None, _dp_pipe_pending=None, _dp_pipe_announced=True,
        _nccl_dtype=lambda value: value.dtype,
        dbo_yield_and_switch_from_compute_to_comm=yield_to_comm,
        dbo_yield_and_switch_from_comm_to_compute=lambda: None,
    ))
    pipeline = SimpleNamespace(
        prepare_finalize=Naive(), fused_experts=expert, N=0,
        _allocate_buffers=lambda *args: (torch.empty(1), torch.empty(1), None))
    runner = SimpleNamespace(
        _ep_fused_combine=ep_combine, do_naive_dispatch_combine=False,
        moe_config=SimpleNamespace(pcp_size=1),
        shared_experts=object(), _wait_late_shared_experts=lambda: None,
        _maybe_reduce_shared_expert_output=lambda value: value,
        _maybe_reduce_final_output=lambda value, _: value)
    runner._ep_fused_combine_output = MethodType(runner_ns["_ep_fused_combine_output"], runner)

    def arm(value):
        shared = torch.full((sizes[rank], 4096), value, dtype=torch.bfloat16)
        state.arm(SimpleNamespace(
            _determine_shared_experts_order=lambda _: modes.NO_OVERLAP if inline else modes.MULTI_STREAM_OVERLAPPED,
            _output=[shared], _output_idx=0), None)
        return shared

    def call(shared):
        value = pipe_ns["_dp_pipe_apply"](
            pipeline, torch.ones((sizes[rank], 4096), dtype=torch.bfloat16),
            torch.empty((1, 6, 4096), dtype=torch.float8_e4m3fn, device="meta"),
            torch.empty((1, 4096, 3), dtype=torch.float8_e4m3fn, device="meta"),
            torch.zeros((sizes[rank], 1), dtype=torch.int32),
            torch.ones((sizes[rank], 1)), "swiglustep", 1, 1, torch.tensor([0]))
        result = runner_ns["_maybe_combine"](runner, shared, value)
        return result

    return SimpleNamespace(call=call, arm=arm, state=state, events=events,
                           handoff=handoff, policy=policy, expert=expert,
                           runner_ns=runner_ns, sizes=sizes, rank=rank)


def test_dbo_pipe_keeps_shared_output_for_runner(monkeypatch):
    runtime = _runtime(monkeypatch)
    shared = runtime.arm(3)
    original, result = runtime.call(shared)
    assert original is shared
    torch.testing.assert_close(result, torch.full_like(shared, 11))
    assert runtime.events == [("shared-add", False, 256)]
    assert runtime.state.added_to is None and not runtime.state.done


@pytest.mark.parametrize("dbo,sizes,rank,uniform", [
    (False, (64, 64, 64, 64), 0, False),
    (True, (64, 64, 64, 64), 0, True),
    (False, (0, 0, 64, 64), 2, False),
])
def test_pipe_also_preserves_shared_add_for_other_collective_layouts(
    monkeypatch, dbo, sizes, rank, uniform,
):
    runtime = _runtime(monkeypatch, dbo=dbo, sizes=sizes, rank=rank, uniform=uniform)
    shared = runtime.arm(5)
    _, result = runtime.call(shared)
    torch.testing.assert_close(result, torch.full_like(shared, 13))
    assert all(not event[1] for event in runtime.events)


@pytest.mark.parametrize("ep_combine,inline", [(False, True), (True, False)])
def test_disabled_or_aux_stream_shared_combine_remains_separate(monkeypatch, ep_combine, inline):
    runtime = _runtime(monkeypatch, ep_combine=ep_combine, inline=inline)
    shared = runtime.arm(3)
    _, result = runtime.call(shared)
    torch.testing.assert_close(result, torch.full_like(shared, 11 if ep_combine else 8))
    assert all(not event[1] for event in runtime.events)


def test_microbatch_handoff_does_not_add_the_other_microbatch_shared_value(monkeypatch):
    runtime = _runtime(monkeypatch)
    first = runtime.arm(3)
    second = []
    runtime.handoff.callback = lambda: second.append(runtime.arm(5))
    _, first_result = runtime.call(first)
    _, second_result = runtime.call(second[0])
    torch.testing.assert_close(first_result, torch.full_like(first, 11))
    torch.testing.assert_close(second_result, torch.full_like(second[0], 13))
    assert all(not event[1] for event in runtime.events)


def test_normal_finalize_still_owns_the_early_shared_add(monkeypatch):
    runtime = _runtime(monkeypatch)
    shared = runtime.arm(3)
    out = torch.full((256, 4096), 2, dtype=torch.bfloat16)
    add, start = runtime.state.shared_add(out)
    out[start:start + shared.shape[0]].add_(add)
    result = runtime.state.finalize_hook(out)
    assert runtime.state.finish()
    torch.testing.assert_close(result, torch.full_like(shared, 11))


def test_unfinished_early_add_assertion_is_preserved(monkeypatch):
    runtime = _runtime(monkeypatch)
    runtime.arm(3)
    assert runtime.state.shared_add(torch.empty((256, 4096), dtype=torch.bfloat16)) is not None
    with pytest.raises(AssertionError):
        runtime.state.finish()


def test_suppression_is_context_local_nested_and_exception_safe(monkeypatch):
    runtime = _runtime(monkeypatch)
    policy = runtime.policy
    other = Context()
    assert policy.ep_shared_add_allowed()
    with pytest.raises(RuntimeError, match="kernel failed"):
        with policy.suspend_ep_shared_add():
            assert not policy.ep_shared_add_allowed()
            assert other.run(policy.ep_shared_add_allowed)
            with policy.suspend_ep_shared_add():
                assert not policy.ep_shared_add_allowed()
            assert not policy.ep_shared_add_allowed()
            raise RuntimeError("kernel failed")
    assert policy.ep_shared_add_allowed()


def test_failed_pipeline_expert_restores_shared_add_policy(monkeypatch):
    runtime = _runtime(monkeypatch)
    runtime.arm(3)

    def fail(**kwargs):
        assert runtime.state.shared_add(kwargs["output"]) is None
        raise RuntimeError("kernel failed")

    runtime.expert.apply = fail
    with pytest.raises(RuntimeError, match="kernel failed"):
        runtime.call(torch.empty((64, 4096), dtype=torch.bfloat16))
    assert runtime.policy.ep_shared_add_allowed()
    assert runtime.state.added_to is None
    assert not runtime.state.finish()
