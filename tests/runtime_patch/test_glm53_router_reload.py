# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Weight updates must reach the router's already captured packed buffer."""

from types import SimpleNamespace

import pytest
import torch
from vllm.v1.worker.gpu_worker import Worker

from vllm_hcu.model_executor.layers.fused_moe import glm53_router_gemv as runtime
from vllm_hcu.platforms import envs as henvs
from vllm_hcu.platforms.hcu import on_gfx938
from vllm_hcu.v1.worker import HcuGPUWorker


@pytest.fixture
def bound_worker(monkeypatch):
    if not torch.cuda.is_available() or not on_gfx938():
        pytest.skip("requires gfx938 router kernel")
    monkeypatch.setattr(henvs, "VLLM_USE_NN", True)

    class Gate(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(
                torch.zeros((4096, 288), dtype=torch.bfloat16, device="cuda"),
                requires_grad=False,
            )
            self.bias = None
            self.out_dtype = torch.float32
            self.allow_cublas_router_gemm = True

        def forward(self, x):
            return torch.mm(x, self.weight, out_dtype=torch.float32), None

    gate = Gate()
    model = torch.nn.Module()
    model.gate = gate
    assert runtime._bind_gate(gate)
    worker = object.__new__(HcuGPUWorker)
    worker.model_runner = SimpleNamespace(model=model)
    worker._weight_update_is_draft = False
    return worker, gate


@pytest.mark.parametrize("entry", ["reload_weights", "finish_weight_update"])
@pytest.mark.parametrize("replace_parameter", [False, True])
def test_weight_updates_refresh_router_and_existing_graph(
    monkeypatch, bound_worker, entry, replace_parameter
):
    worker, gate = bound_worker

    # Mock only checkpoint/transport IO. The HCU worker callback, packed
    # weight refresh, actual Triton GEMV and graph replay are all real.
    def update_parameter(self, *args, **kwargs):
        if replace_parameter:
            gate.weight = torch.nn.Parameter(
                torch.ones_like(gate.weight), requires_grad=False
            )
        else:
            gate.weight.fill_(1)

    monkeypatch.setattr(Worker, entry, update_parameter)
    x = torch.ones((1, 4096), dtype=torch.bfloat16, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            gate(x)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured, _ = gate(x)
    pointer = gate._hcu_router_gemv_weight.data_ptr()
    assert torch.all(captured == 0)

    getattr(worker, entry)()
    assert gate._hcu_router_gemv_weight.data_ptr() == pointer
    graph.replay()
    torch.cuda.synchronize()
    eager, _ = gate(x)
    fallback, _ = gate(x.expand(17, -1).contiguous())
    assert torch.all(captured == 4096)
    assert torch.all(eager == 4096)
    assert torch.all(fallback == 4096)


def test_draft_weight_update_does_not_refresh_target_router(monkeypatch, bound_worker):
    worker, gate = bound_worker
    worker._weight_update_is_draft = True
    gate.weight.fill_(1)
    monkeypatch.setattr(Worker, "finish_weight_update", lambda self: None)
    worker.finish_weight_update()
    assert torch.all(gate._hcu_router_gemv_weight == 0)
