"""Real target MoERunner padding/routing contracts; CPU communication/kernel doubles.

These tests do not execute a vendor kernel or establish HCU/Graph numerics.
"""
# ruff: noqa: E402
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
import os
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, os.environ.get("QNT009_WHEEL_ROOT", str(ROOT / "adapter_extension/src")))

from hcu_vllm_das.w4a8 import config as api, runtime
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig, FusedMoEParallelConfig, RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.runner import moe_runner as official
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.model_executor.layers.fused_moe.runner.shared_experts import SharedExperts
from vllm.model_executor.layers.linear import UnquantizedLinearMethod


@pytest.fixture(autouse=True)
def no_gpu(monkeypatch):
    def forbidden(*a, **kw):
        pytest.fail("CPU runner contract attempted a GPU query")

    for name in ("get_device_properties", "current_device", "init", "_lazy_init"):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    yield
    assert not torch.cuda.is_initialized()


def config(hidden):
    return FusedMoEConfig(
        num_experts=512, num_local_experts=512, num_logical_experts=512,
        experts_per_token=10, hidden_dim=hidden, intermediate_size=640,
        moe_parallel_config=replace(FusedMoEParallelConfig.make_no_parallel(), tp_size=4),
        activation=MoEActivation.SILU, device="cpu", in_dtype=torch.bfloat16,
        routing_method=RoutingMethodType.Default,
    )


def runner(monkeypatch, logical, physical, shared=True):
    observations = dict(native=[], gate=[], shared=[], reduced=[], fake=[])

    def kernel(x, w1, w2, weights, ids, cfg, **kw):
        assert x.shape[1] == physical and x.is_contiguous()
        assert torch.count_nonzero(x[:, logical:]) == 0
        observations["native"].append(x.clone())
        # Deliberately poison the physical tail: it must never escape the method.
        out = x * 3
        out[:, logical:] = 99
        return out

    plan = runtime.NativePlan(
        runtime.Shape(512, physical, 192, 10, 512, torch.bfloat16),
        (NS(need_shuffle=False),) * 512, NS(aiter_moe=kernel),
    )
    moe = config(logical)
    method = api.SlimQuantW4A8MoEMethod(moe, plan)
    moe.hidden_dim, moe.intermediate_size_per_partition = physical, 192
    method._processed = True
    # The fake CPU kernel uses no actual weights; the real loader is covered
    # by test_w4a8_provider_contract.py.
    layer = RoutedExperts.__new__(RoutedExperts)
    torch.nn.Module.__init__(layer)
    layer.moe_config, layer.quant_method = moe, method
    layer._expert_map = None
    layer.apply_router_weight_on_input = False
    layer.global_num_experts = 512
    for name in ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale"):
        setattr(layer, name, torch.zeros(1))
    method.moe_quant_config = method.get_fused_moe_quant_config(layer)

    # Exercise actual GateLinear.forward -> ReplicatedLinear.forward ->
    # UnquantizedLinearMethod.apply, including the exact failing router call.
    # Construct CPU-only instance state instead of querying HCU eligibility.
    gate = GateLinear.__new__(GateLinear)
    torch.nn.Module.__init__(gate)
    gate.weight = torch.nn.Parameter(torch.zeros(512, logical, dtype=torch.bfloat16))
    gate.bias = None
    gate.skip_bias_add = False
    gate.return_bias = True
    gate.out_dtype = None
    gate.is_unquantized = True
    for flag in ("allow_ll_bf16_gemm", "allow_rocm_bf16x3_router_gemm",
                 "allow_fp32_router_gemm", "allow_bf16x3_router_gemm",
                 "allow_cublas_router_gemm"):
        setattr(gate, flag, False)
    gemm = UnquantizedLinearMethod.__new__(UnquantizedLinearMethod)
    gemm._gemm_impl = lambda layer, x, weight, bias: torch.nn.functional.linear(x, weight, bias)
    gate.quant_method = gemm
    gate.register_forward_pre_hook(lambda module, args: observations["gate"].append(args[0]))

    class Shared(torch.nn.Module):
        def forward(self, x):
            observations["shared"].append(x)
            return x * 2

    def select(*, hidden_states, router_logits, topk_indices_dtype, input_ids):
        assert hidden_states.shape[1] == logical
        assert router_logits.shape == (hidden_states.shape[0], 512)
        assert topk_indices_dtype == torch.int32
        return (torch.ones(hidden_states.shape[0], 10),
                torch.zeros(hidden_states.shape[0], 10, dtype=topk_indices_dtype))

    value = official.MoERunner.__new__(official.MoERunner)
    torch.nn.Module.__init__(value)
    value.moe_config, value.routed_experts = moe, layer
    value.gate = gate
    value._fse_fuse_gate = False
    # Run the real TP-local SharedExperts wrapper on CPU, without a GPU stream.
    monkeypatch.setenv("VLLM_DISABLE_SHARED_EXPERTS_STREAM", "1")
    value._shared_experts = SharedExperts(Shared(), moe, False, lambda: False) if shared else None
    value.routed_input_transform = value.routed_output_transform = None
    value.routed_scaling_factor = 1.0
    value.router = NS(select_experts=select)
    value.layer_name = "cpu.w4.padding"
    # No distributed process group or GPU is allocated. Real runner routing,
    # expert dispatch/apply, shared combination and final reduction decision run.
    value._sequence_parallel_context = nullcontext

    def reduce(x):
        observations["reduced"].append(x)
        return x

    monkeypatch.setattr(official, "tensor_model_parallel_all_reduce", reduce)

    def entry(x, logits, shared_x, ids, name, logical_hint):
        fake = official._moe_forward_shared_fake if shared else official._moe_forward_fake
        expected = fake(x, logits, shared_x, ids, name, logical_hint)
        result = value._forward_impl(x, logits, shared_x, ids)
        expected = expected if shared else (expected,)
        actual = result if shared else (result,)
        for a, b in zip(actual, expected):
            assert a.shape == b.shape and a.stride() == b.stride()
        observations["fake"].append(logical_hint)
        return result

    value._forward_entry = entry
    return value, method, observations


@pytest.mark.parametrize("tokens", [0, 1, 17, 65, 512])
@pytest.mark.parametrize("logical,physical", [(2560, 3072), (3072, 3072)])
@pytest.mark.parametrize("shared", [False, True])
def test_real_runner_gate_shared_and_fake_shape(monkeypatch, tokens, logical, physical, shared):
    value, method, seen = runner(monkeypatch, logical, physical, shared)
    x = torch.ones(tokens, logical, dtype=torch.bfloat16)
    result = value(x, torch.empty(tokens, 512))
    assert result.shape == x.shape and result.is_contiguous()
    assert torch.equal(result, x * (5 if shared else 3))
    assert seen["gate"][0] is x
    if shared:
        assert seen["shared"][0] is x
    assert all(t.shape[-1] == logical for t in seen["reduced"])
    assert seen["fake"] == [logical]
    assert method.skip_forward_padding and method.has_unpadded_output
    assert method.moe.hidden_dim_unpadded == logical
    assert method.moe.hidden_dim == physical
    assert isinstance(value.gate, GateLinear)
    assert value.gate.weight.shape == (512, logical)
    if shared:
        assert isinstance(value.shared_experts, SharedExperts)
    assert len(seen["native"]) == (1 if tokens else 0)
    if tokens:
        assert torch.equal(seen["native"][0][:, :logical], x)


@pytest.mark.parametrize("width", [2559, 3072])
def test_native_rejects_nonlogical_input_before_padding(monkeypatch, width):
    value, method, seen = runner(monkeypatch, 2560, 3072)
    with pytest.raises(runtime.W4A8Error, match="logical"):
        method.apply(value.routed_experts, torch.ones(1, width, dtype=torch.bfloat16),
                     torch.ones(1, 10), torch.zeros(1, 10, dtype=torch.int32), None, None)
    assert not seen["native"]


def test_noncontiguous_logical_input_is_not_silently_copied(monkeypatch):
    value, method, seen = runner(monkeypatch, 2560, 3072)
    x = torch.ones(2, 5120, dtype=torch.bfloat16)[:, ::2]
    with pytest.raises(runtime.W4A8Error, match="contiguous"):
        method.apply(value.routed_experts, x, torch.ones(2, 10),
                     torch.zeros(2, 10, dtype=torch.int32), None, None)
    assert not seen["native"]


@pytest.mark.parametrize("shared", [False, True])
def test_official_custom_op_compile_fake_shape(monkeypatch, shared):
    from torch.fx.experimental.proxy_tensor import make_fx

    value, method, _ = runner(monkeypatch, 2560, 3072, shared)
    op = (torch.ops.vllm.moe_forward_shared.default if shared
          else torch.ops.vllm.moe_forward.default)

    def operation(x, logits):
        return op(x, logits, x if shared else None, None,
                  value._encode_layer_name(), method.moe.hidden_dim_unpadded)

    graph = make_fx(operation, tracing_mode="fake")(
        torch.empty(17, 2560, dtype=torch.bfloat16), torch.empty(17, 512)
    )
    nodes = [node for node in graph.graph.nodes if node.target == op]
    assert len(nodes) == 1
    outputs = nodes[0].meta["val"]
    outputs = outputs if shared else (outputs,)
    assert all(tensor.shape == (17, 2560) and tensor.stride() == (2560, 1)
               for tensor in outputs)


def test_official_int8_fallback_retains_default_padding_protocol(monkeypatch):
    from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
    from vllm.model_executor.layers.fused_moe.oracle.int8 import Int8MoeBackend
    import vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w8a8_int8 as fallback

    monkeypatch.setattr(fallback, "select_int8_moe_backend", lambda **kw: (Int8MoeBackend.TRITON, TritonExperts))
    method = api.SlimQuantW4A8MoEMethod(config(2560), None)
    assert not method.skip_forward_padding and not method.has_unpadded_output
    assert method.maybe_roundup_sizes(2560, 160, torch.bfloat16, method.moe.moe_parallel_config) == (2560, 160)
    method._processed = True
    sentinel = object()
    calls = []
    method.delegate.apply = lambda *a: (calls.append(a), sentinel)[1]
    values = (object(), object(), object(), object(), object(), object())
    assert method.apply(*values) is sentinel and calls == [values]
