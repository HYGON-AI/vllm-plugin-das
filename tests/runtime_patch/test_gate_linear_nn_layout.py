# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from types import ModuleType

import pytest
import torch

from vllm_hcu.patch.worker.op_opt import patch_gate_linear
from vllm_hcu.platforms import envs as henvs


@pytest.fixture(autouse=True)
def _no_model_config(monkeypatch):
    monkeypatch.setattr(patch_gate_linear, "_get_model_type", lambda: None)


def _gate_module():
    class GateLinear:
        def forward(self, x: torch.Tensor):
            return "official", x

    module = ModuleType(patch_gate_linear.TARGET_MODULE)
    module.GateLinear = GateLinear
    return module, GateLinear


def test_gate_linear_bf16_fp32_uses_hcu_nn_weight_layout(monkeypatch):
    module, gate_class = _gate_module()
    monkeypatch.setattr(henvs, "VLLM_USE_NN", True)

    calls = []

    def fake_mm(x, weight, *, out_dtype):
        calls.append((x, weight, out_dtype))
        return torch.empty((x.shape[0], weight.shape[1]), dtype=out_dtype)

    monkeypatch.setattr(torch, "mm", fake_mm)
    assert patch_gate_linear.apply_to_module(module) is True
    assert patch_gate_linear.apply_to_module(module) is False

    gate = gate_class()
    gate.allow_cublas_router_gemm = True
    gate.weight = torch.empty((6, 4), dtype=torch.bfloat16)
    output, output_bias = gate.forward(torch.empty((3, 6), dtype=torch.bfloat16))

    assert output.shape == (3, 4)
    assert output.dtype == torch.float32
    assert output_bias is None
    assert calls[0][1] is gate.weight
    assert calls[0][2] == torch.float32


def test_gate_linear_preserves_official_path_without_hcu_nn_layout(monkeypatch):
    module, gate_class = _gate_module()
    monkeypatch.setattr(henvs, "VLLM_USE_NN", False)
    assert patch_gate_linear.apply_to_module(module) is True

    gate = gate_class()
    gate.allow_cublas_router_gemm = True
    gate.weight = torch.empty((4, 6), dtype=torch.bfloat16)
    x = torch.empty((3, 6), dtype=torch.bfloat16)

    assert gate.forward(x) == ("official", x)


@pytest.mark.parametrize("nn_layout", [True, False])
@pytest.mark.parametrize("tokens", [1, 17])
def test_deepseek_v4_gate_matches_bf16_output_then_fp32(
    monkeypatch, nn_layout, tokens
):
    module, gate_class = _gate_module()
    monkeypatch.setattr(patch_gate_linear, "_get_model_type", lambda: "deepseek_v4")
    monkeypatch.setattr(henvs, "VLLM_USE_NN", nn_layout)
    assert patch_gate_linear.apply_to_module(module)
    assert not patch_gate_linear.apply_to_module(module)
    gate = gate_class()
    # Forward must not need the construction-time model configuration.
    monkeypatch.setattr(patch_gate_linear, "_get_model_type", lambda: None)
    gate.allow_cublas_router_gemm = True
    torch.manual_seed(42)
    weight = torch.randn(6, 4, dtype=torch.bfloat16)
    gate.weight = weight if nn_layout else weight.T.contiguous()
    x = torch.randn(tokens, 6, dtype=torch.bfloat16)
    expected = (x @ weight).float()
    output, bias = gate.forward(x)
    assert output.dtype == torch.float32
    assert bias is None
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


@pytest.mark.parametrize("model_type", ["deepseek_v3", None])
def test_gate_preserves_direct_fp32_path(monkeypatch, model_type):
    module, gate_class = _gate_module()
    monkeypatch.setattr(patch_gate_linear, "_get_model_type", lambda: model_type)
    monkeypatch.setattr(henvs, "VLLM_USE_NN", True)
    patch_gate_linear.apply_to_module(module)
    gate = gate_class()
    gate.allow_cublas_router_gemm = True
    gate.weight = torch.empty(6, 4, dtype=torch.bfloat16)
    calls = []

    def fake_mm(x, weight, *, out_dtype):
        calls.append(out_dtype)
        return torch.empty(x.shape[0], weight.shape[1], dtype=out_dtype)

    monkeypatch.setattr(torch, "mm", fake_mm)
    output, _ = gate.forward(torch.empty(3, 6, dtype=torch.bfloat16))
    assert output.dtype == torch.float32
    assert calls == [torch.float32]


def test_deepseek_v4_gate_preserves_ineligible_dispatch(monkeypatch):
    module, gate_class = _gate_module()
    monkeypatch.setattr(patch_gate_linear, "_get_model_type", lambda: "deepseek_v4")
    monkeypatch.setattr(henvs, "VLLM_USE_NN", True)
    patch_gate_linear.apply_to_module(module)
    gate = gate_class()
    gate.allow_cublas_router_gemm = False
    gate.weight = torch.empty(6, 4, dtype=torch.bfloat16)
    x = torch.empty(3, 6, dtype=torch.bfloat16)
    assert gate.forward(x) == ("official", x)
