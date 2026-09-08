# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from types import ModuleType

import torch

from vllm_hcu.patch.worker.op_opt import patch_gate_linear
from vllm_hcu.platforms import envs as henvs


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
