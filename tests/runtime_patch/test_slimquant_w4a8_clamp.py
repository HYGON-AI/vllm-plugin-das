# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""CPU control-flow regression; GPU quant/GEMM/alignment ops are CPU doubles.

Keep TritonExperts.apply, quantization dispatch, activation and the functional
None/0 path real. This does not validate HCU kernel accuracy or performance.
"""

from __future__ import annotations

import inspect
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from vllm.model_executor.layers.fused_moe import fused_moe, utils
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.experts import triton_moe

from vllm_hcu.model_executor.layers.quantization import (
    compressed_tensors_moe_runtime as runtime,
)


@pytest.fixture
def cpu_ops(monkeypatch):
    quantized = []
    gemms = []

    def per_token_quant(x):
        scale = x.float().abs().amax(dim=-1, keepdim=True).clamp_min(1e-10) / 127
        q = (x.float() / scale).round().clamp(-127, 127).to(torch.int8)
        quantized.append((x.clone(), q, scale))
        return q, scale

    def gemm(
        a,
        b,
        c,
        a_scale,
        b_scale,
        topk_weights,
        sorted_ids,
        expert_ids,
        num_padded,
        mul_weights,
        top_k,
        config,
        compute_type,
        use_fp8_w8a8,
        use_int8_w8a8,
        use_int8_w8a16,
        use_int4_w4a16,
        per_channel_quant,
        block_shape=None,
        B_bias=None,
    ):
        # A one-expert fixture: compute from the actual quantized tensors and
        # scales received from production apply, rather than returning a sentinel.
        assert a.dtype == b.dtype == torch.int8
        assert use_int8_w8a8 and per_channel_quant
        assert not use_fp8_w8a8 and not use_int8_w8a16 and not use_int4_w4a16
        gemms.append((a, a_scale, b_scale, c.dtype, compute_type, mul_weights))
        values = (a.float() * a_scale) @ (b[0].float() * b_scale[0]).T
        if mul_weights:
            values *= topk_weights
        c.copy_(values.reshape_as(c))

    def assignment(
        topk_ids, config, num_tokens, top_k, global_experts, expert_map, **kwargs
    ):
        return None, topk_ids.flatten(), torch.tensor([topk_ids.numel()])

    def silu_and_mul(output, x):
        gate, up = x.chunk(2, dim=-1)
        output.copy_(torch.nn.functional.silu(gate) * up)

    monkeypatch.setattr(utils, "per_token_quant_int8", per_token_quant)
    for module in (triton_moe, fused_moe):
        monkeypatch.setattr(module, "invoke_fused_moe_triton_kernel", gemm)
        monkeypatch.setattr(module, "_prepare_expert_assignment", assignment)
        monkeypatch.setattr(
            module, "try_get_optimal_moe_config", lambda *a, **k: {"BLOCK_SIZE_M": 1}
        )
    monkeypatch.setattr(
        triton_moe.ops, "moe_sum", lambda x, out: out.copy_(x.sum(dim=1))
    )
    monkeypatch.setattr(torch.ops._C, "silu_and_mul", silu_and_mul, raising=False)
    # Execute the original tensor formula without invoking torch.compile on CPU.
    clamp = utils.swiglu_limit_func
    monkeypatch.setattr(
        triton_moe,
        "swiglu_limit_func",
        getattr(clamp, "_torchdynamo_orig_callable", clamp),
    )
    return quantized, gemms


def _inputs(dtype, limit=10.0):
    x = torch.tensor([[127, 0, 127, 0], [254, 0, 254, 0]], dtype=dtype)
    w1 = torch.tensor(
        [[[1, 0, 0, 0], [-1, 0, 0, 0], [1, 0, 0, 0], [-1, 0, 0, 0]]], dtype=torch.int8
    )
    w2 = torch.tensor([[[1, 0], [2, 0], [0, 0], [-1, 0]]], dtype=torch.int8)
    for weight in (w1, w2):
        weight._hcu_vllm_w4a8_unpacked = True
    moe = SimpleNamespace(
        activation=MoEActivation.SILU,
        num_experts=1,
        swiglu_limit=limit,
        moe_parallel_config=SimpleNamespace(use_all2all_kernels=False),
    )
    method = SimpleNamespace(
        moe=moe,
        moe_quant_config=SimpleNamespace(
            w1_scale=torch.full((1, 4, 1), 1 / 16),
            w2_scale=torch.full((1, 4, 1), 1 / 32),
            a1_scale=None,
            a2_scale=None,
        ),
    )
    layer = SimpleNamespace(
        w13_weight=w1,
        w2_weight=w2,
        global_num_experts=1,
        _expert_map=None,
        apply_router_weight_on_input=False,
    )
    weights = torch.tensor([[0.25], [0.5]])
    ids = torch.zeros((2, 1), dtype=torch.int32)
    return method, layer, x, weights, ids


def _experts_apply(dtype, router_on_input):
    method, layer, x, weights, ids = _inputs(dtype)
    config = FusedMoEQuantConfig.make(
        torch.int8,
        per_act_token_quant=True,
        weight_dtype="int4",
        w1_scale=method.moe_quant_config.w1_scale * 16,
        w2_scale=method.moe_quant_config.w2_scale * 16,
        gemm1_clamp_limit=10.0,
    )
    experts = runtime._clamped_w4a8_triton_experts_cls()(method.moe, config)
    # Prepare/finalize owns input weighting; expert.apply receives it applied.
    if router_on_input:
        x = (x * weights).to(dtype)
    kwargs = {
        "output": torch.empty_like(x),
        "hidden_states": x,
        "w1": layer.w13_weight,
        "w2": layer.w2_weight,
        "topk_weights": weights,
        "topk_ids": ids,
        "activation": MoEActivation.SILU,
        "global_num_experts": -1,
        "expert_map": None,
        "a1q_scale": None,
        "a2_scale": None,
        "workspace13": torch.empty((2, 1, 4), dtype=dtype),
        "workspace2": torch.empty((2, 1, 4), dtype=dtype),
        "expert_tokens_meta": None,
        "apply_router_weight_on_input": router_on_input,
    }
    return experts, kwargs


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("router_on_input", [False, True])
def test_clamped_w4a8_real_expert_apply_keeps_float_compute_and_int8_data(
    cpu_ops,
    dtype,
    router_on_input,
):
    experts, kwargs = _experts_apply(dtype, router_on_input)
    original = kwargs["hidden_states"].clone()
    experts.apply(**kwargs)
    output = kwargs["output"]
    expected = torch.tensor([[50, 100, 0, -50], [50, 100, 0, -50]], dtype=dtype)
    if not router_on_input:
        expected *= kwargs["topk_weights"]
    torch.testing.assert_close(output, expected, atol=0.03, rtol=0.003)
    assert output.dtype == dtype
    torch.testing.assert_close(kwargs["hidden_states"], original)
    quantized, gemms = cpu_ops
    assert len(quantized) == len(gemms) == 2
    expected_scale = [[0.25], [1.0]] if router_on_input else [[1.0], [2.0]]
    torch.testing.assert_close(quantized[0][2], torch.tensor(expected_scale))
    torch.testing.assert_close(
        quantized[0][1], torch.tensor([[127, 0, 127, 0]] * 2, dtype=torch.int8)
    )
    # Clamp must happen before quantization of GEMM2's input.
    torch.testing.assert_close(
        quantized[1][0],
        torch.tensor([[100, 0]] * 2, dtype=dtype),
        atol=0.03,
        rtol=0.003,
    )
    for index, (a, scale, b_scale, out_dtype, compute_type, mul_weights) in enumerate(
        gemms
    ):
        assert a is quantized[index][1]
        assert scale is quantized[index][2]
        assert out_dtype == dtype
        assert (
            compute_type
            == {
                torch.bfloat16: triton_moe.tl.bfloat16,
                torch.float16: triton_moe.tl.float16,
                torch.float32: triton_moe.tl.float32,
            }[dtype]
        )
        torch.testing.assert_close(
            b_scale, torch.full((1, 4, 1), 1.0 if index == 0 else 0.5)
        )
        assert mul_weights == (index == 1 and not router_on_input)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("limit", [None, 0.0])
def test_w4a8_none_or_zero_limit_retains_real_functional_path(cpu_ops, dtype, limit):
    method, layer, x, weights, ids = _inputs(dtype, limit)
    result = runtime.apply_vllm_w4a8_moe(method, layer, x, weights, ids)
    # Without clamp: silu(127)*127=16129; silu(254)*254=64516.
    expected = torch.tensor(
        [[2016.125, 4032.25, 0, -2016.125], [16129, 32258, 0, -16129]], dtype=dtype
    )
    torch.testing.assert_close(result, expected, atol=1, rtol=0.015)
    assert result.dtype == dtype
    assert not hasattr(method, "_hcu_clamped_w4a8_triton_kernel")
    assert len(cpu_ops[0]) == len(cpu_ops[1]) == 2


def test_clamped_w4a8_apply_matches_upstream_parameter_contract():
    custom = inspect.signature(runtime._clamped_w4a8_triton_experts_cls().apply)
    upstream = inspect.signature(triton_moe.TritonExperts.apply)
    assert tuple(custom.parameters) == tuple(upstream.parameters)


def test_clamped_w4a8_rejects_upstream_apply_signature_drift(monkeypatch):
    monkeypatch.setattr(runtime, "_CLAMPED_W4A8_TRITON_EXPERTS_CLS", None)
    monkeypatch.setattr(triton_moe.TritonExperts, "apply", lambda self, new_arg: None)
    with pytest.raises(
        runtime.HcuCompressedTensorsMoeError, match="TritonExperts.apply contract"
    ):
        runtime._clamped_w4a8_triton_experts_cls()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("allow_aiter", [False, True])
def test_positive_limit_dispatch_reaches_real_expert_apply(
    cpu_ops, monkeypatch, dtype, allow_aiter
):
    import vllm.model_executor.layers.fused_moe.modular_kernel as mk
    from vllm.model_executor.layers.fused_moe import all2all_utils

    class CpuKernel:
        """Replace communication/workspace orchestration, never expert.apply."""

        def __init__(self, prepare_finalize, experts):
            self.fused_experts = experts

        def apply(self, **kwargs):
            x, w1, ids = kwargs["hidden_states"], kwargs["w1"], kwargs["topk_ids"]
            shapes = self.fused_experts.workspace_shapes(
                x.shape[0],
                w1.shape[1],
                x.shape[1],
                ids.shape[1],
                1,
                1,
                None,
                kwargs["activation"],
            )
            output = torch.empty(shapes[2], dtype=x.dtype)
            kwargs.pop("shared_experts")
            kwargs.pop("shared_experts_input")
            self.fused_experts.apply(
                output=output,
                a1q_scale=None,
                a2_scale=None,
                workspace13=torch.empty(shapes[0], dtype=x.dtype),
                workspace2=torch.empty(shapes[1], dtype=x.dtype),
                expert_tokens_meta=None,
                **kwargs,
            )
            return output

    monkeypatch.setattr(mk, "FusedMoEKernel", CpuKernel)
    monkeypatch.setattr(
        all2all_utils, "maybe_make_prepare_finalize", lambda **k: object()
    )
    aiter_moe = ModuleType("aiter.moe")
    aiter_moe.MoeQuantType = SimpleNamespace(W4A8="W4A8")
    aiter = ModuleType("aiter")
    aiter.__path__ = []
    aiter.moe = aiter_moe
    monkeypatch.setitem(sys.modules, "aiter", aiter)
    monkeypatch.setitem(sys.modules, "aiter.moe", aiter_moe)
    monkeypatch.setattr(runtime, "select_aiter_moe_config", lambda *a, **k: None)
    method, layer, x, weights, ids = _inputs(dtype)
    result = runtime.apply_aiter_w4a8_moe(
        method, layer, x, weights, ids, allow_aiter=allow_aiter
    )
    expected = torch.tensor([[12.5, 25, 0, -12.5], [25, 50, 0, -25]], dtype=dtype)
    torch.testing.assert_close(result, expected, atol=0.03, rtol=0.003)
    assert len(cpu_ops[0]) == len(cpu_ops[1]) == 2


@pytest.mark.parametrize("unsupported", ["lora", "fp8", "wna16", "emulation"])
def test_clamped_w4a8_rejects_unsupported_expert_modes(cpu_ops, unsupported):
    experts, kwargs = _experts_apply(torch.float16, False)
    if unsupported == "lora":
        assert experts.supports_lora() is False
        experts._lora_context = object()
    elif unsupported == "emulation":
        experts.quantization_emulation = True
    else:
        experts.quant_config._a1.dtype = (
            torch.float8_e4m3fn if unsupported == "fp8" else None
        )
    with pytest.raises(runtime.HcuCompressedTensorsMoeError, match="clamped W4A8"):
        experts.apply(**kwargs)
    assert not cpu_ops[0] and not cpu_ops[1]
