# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""GLM-5.3 router fast path is bound after model weights are loaded."""

import pytest
import torch

from vllm_hcu.model_executor.layers.fused_moe import glm53_router_gemv as runtime
from vllm_hcu.platforms import envs as henvs


def test_router_binding_keeps_fallback_and_mtp(monkeypatch):
    from vllm_hcu.platforms.hcu import on_gfx938

    if not torch.cuda.is_available() or not on_gfx938():
        pytest.skip("requires gfx938")
    monkeypatch.setattr(henvs, "VLLM_USE_NN", True)
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.delenv(runtime._ROUTER_GEMV_ENV, raising=False)
    calls = []

    def fast_path(x, weight):
        calls.append((tuple(x.shape), tuple(weight.shape)))
        return torch.full((x.shape[0], 288), 7, dtype=torch.float32, device=x.device)

    monkeypatch.setattr(runtime, "glm53_router_gemv", fast_path)

    class Gate(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(
                torch.randn(4096, 288, dtype=torch.bfloat16, device="cuda")
            )
            self.bias = None
            self.out_dtype = torch.float32
            self.allow_cublas_router_gemm = True

        def forward(self, x):
            return torch.full((x.shape[0], 288), -1, device=x.device), None

    class Glm5NextDecoderLayer(torch.nn.Module):
        def __init__(self, mtp=False):
            super().__init__()
            self.is_mtp_layer = mtp
            self.mlp = torch.nn.Module()
            self.mlp.gate = Gate()

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList(
                [Glm5NextDecoderLayer(), Glm5NextDecoderLayer(mtp=True)]
            )

    model = Model()
    assert runtime.bind_glm53_router_gates(model) == 1
    assert runtime.bind_glm53_router_gates(model) == 0
    monkeypatch.setenv(runtime._ROUTER_GEMV_ENV, "0")
    assert runtime.bind_glm53_router_gates(Model()) == 0
    monkeypatch.setenv(runtime._ROUTER_GEMV_ENV, "1")
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "0")
    assert runtime.bind_glm53_router_gates(Model()) == 0
    torch.testing.assert_close(
        model.layers[0].mlp.gate._hcu_router_gemv_weight,
        model.layers[0].mlp.gate.weight.T.contiguous(),
    )
    x = torch.ones(1, 4096, dtype=torch.bfloat16, device="cuda")
    actual, bias = model.layers[0].mlp.gate(x)
    assert bias is None
    assert torch.all(actual == 7)
    fallback, _ = model.layers[0].mlp.gate(x.expand(17, -1).contiguous())
    assert torch.all(fallback == -1)
    mtp, _ = model.layers[1].mlp.gate(x)
    assert torch.all(mtp == -1)
    assert calls == [((1, 4096), (288, 4096))]

    disabled = Glm5NextDecoderLayer()
    disabled.mlp.gate.allow_cublas_router_gemm = False
    assert runtime._bind_gate(disabled.mlp.gate) is False
    monkeypatch.setattr(henvs, "VLLM_USE_NN", False)
    assert runtime._bind_gate(Glm5NextDecoderLayer().mlp.gate) is False
