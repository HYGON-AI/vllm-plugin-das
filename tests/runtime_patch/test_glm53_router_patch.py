# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""GLM-5.3 router fast path is bound after model weights are loaded."""

import pytest
import torch
from vllm.model_executor.offloader import (
    PrefetchOffloader,
    get_offloader,
    set_offloader,
)

from vllm_hcu.model_executor.layers.fused_moe import glm53_router_gemv as runtime
from vllm_hcu.platforms import envs as henvs


@pytest.mark.parametrize(
    ("master", "child", "expected"),
    [
        (None, None, True),
        ("1", "0", False),
        ("0", "1", False),
        ("true", "true", True),
        ("false", None, False),
    ],
)
def test_router_enable_policy(monkeypatch, master, child, expected):
    if master is None:
        monkeypatch.delenv("VLLM_HCU_USE_CUSTOM_OPS", raising=False)
    else:
        monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", master)
    if child is None:
        monkeypatch.delenv(runtime._ROUTER_GEMV_ENV, raising=False)
    else:
        monkeypatch.setenv(runtime._ROUTER_GEMV_ENV, child)
    assert henvs.glm53_router_gemv_enabled() is expected


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

    child_disabled = Model()
    monkeypatch.setenv(runtime._ROUTER_GEMV_ENV, "0")
    assert runtime.bind_glm53_router_gates(child_disabled) == 0

    master_disabled = Model()
    monkeypatch.setenv(runtime._ROUTER_GEMV_ENV, "1")
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "0")
    assert runtime.bind_glm53_router_gates(master_disabled) == 0

    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    disabled = Glm5NextDecoderLayer()
    disabled.mlp.gate.allow_cublas_router_gemm = False
    assert runtime._bind_gate(disabled.mlp.gate) is False
    monkeypatch.setattr(henvs, "VLLM_USE_NN", False)
    assert runtime._bind_gate(Glm5NextDecoderLayer().mlp.gate) is False


def test_prefetch_offloaded_router_is_not_bound(monkeypatch):
    from vllm_hcu.platforms.hcu import on_gfx938

    if not torch.cuda.is_available() or not on_gfx938():
        pytest.skip("requires gfx938 prefetch offload")
    monkeypatch.setattr(henvs, "VLLM_USE_NN", True)
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.delenv(runtime._ROUTER_GEMV_ENV, raising=False)

    class Gate(torch.nn.Module):
        def __init__(self, first_expert):
            super().__init__()
            weight = torch.zeros(
                (4096, 288), dtype=torch.bfloat16, device="cuda"
            )
            weight[:, first_expert : first_expert + 8] = 1
            self.weight = torch.nn.Parameter(weight, requires_grad=False)
            self.bias = None
            self.out_dtype = torch.float32
            self.allow_cublas_router_gemm = True

        def forward(self, x):
            return torch.mm(x, self.weight, out_dtype=torch.float32), None

    class Glm5NextDecoderLayer(torch.nn.Module):
        def __init__(self, first_expert):
            super().__init__()
            self.is_mtp_layer = False
            self.mlp = torch.nn.Module()
            self.mlp.gate = Gate(first_expert)

        def forward(self, x):
            return self.mlp.gate(x)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList(
                [
                    Glm5NextDecoderLayer(0),
                    Glm5NextDecoderLayer(8),
                    Glm5NextDecoderLayer(16),
                    Glm5NextDecoderLayer(24),
                ]
            )

    previous_offloader = get_offloader()
    offloader = PrefetchOffloader(
        group_size=2,
        num_in_group=1,
        prefetch_step=1,
        offload_params={"gate.weight"},
    )
    set_offloader(offloader)
    try:
        model = Model()
        assert offloader.wrap_modules(iter(model.layers)) == list(model.layers)
        offloader.post_init()

        # Both offloaded layers reuse one GPU slot for gate.weight. A packed
        # copy made after post_init would capture whichever layer owns it.
        assert (
            model.layers[1].mlp.gate.weight.data_ptr()
            == model.layers[3].mlp.gate.weight.data_ptr()
        )
        assert runtime.bind_glm53_router_gates(model) == 2
        assert hasattr(model.layers[0].mlp.gate, "_hcu_router_gemv_weight")
        assert not hasattr(model.layers[1].mlp.gate, "_hcu_router_gemv_weight")
        assert hasattr(model.layers[2].mlp.gate, "_hcu_router_gemv_weight")
        assert not hasattr(model.layers[3].mlp.gate, "_hcu_router_gemv_weight")

        for _ in range(2):
            for num_tokens in (1, 8, 16, 17):
                x = torch.ones(
                    (num_tokens, 4096),
                    dtype=torch.bfloat16,
                    device="cuda",
                )
                for layer_index, first_expert in enumerate((0, 8, 16, 24)):
                    logits, bias = model.layers[layer_index](x)
                    assert bias is None
                    assert torch.all(
                        logits[:, first_expert : first_expert + 8] == 4096
                    )
                    assert torch.count_nonzero(logits).item() == num_tokens * 8

        x = torch.ones((8, 4096), dtype=torch.bfloat16, device="cuda")
        offloader.sync_prev_onload()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_logits = [model.layers[index](x)[0] for index in range(4)]
            offloader.join_after_forward()
        graph.replay()
        torch.cuda.synchronize()
        for logits, first_expert in zip(graph_logits, (0, 8, 16, 24)):
            assert torch.all(logits[:, first_expert : first_expert + 8] == 4096)
            assert torch.count_nonzero(logits).item() == 8 * 8
    finally:
        torch.cuda.synchronize()
        set_offloader(previous_offloader)
