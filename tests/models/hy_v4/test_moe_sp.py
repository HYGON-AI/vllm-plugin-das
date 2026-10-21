# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

"""CPU checks for HY4's token split/gather and shared-expert weight ownership.

Execute the patched HY4 classes with CPU linear/expert and collective stand-ins.
This does not import vLLM or execute DeepEP, FP8 kernels, or GPU collectives.
"""
import ast
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

MOE_PATH = Path(__file__).resolve().parents[3] / "vllm_hcu/models/hy_v4/moe.py"


class GatherCapture(Exception):
    pass


def load_model():
    state = SimpleNamespace(tp=1, rank=0, outputs=None, chunked=False)

    class Gate(nn.Module):
        def __init__(self, input_size, output_size, **kwargs):
            super().__init__()
            self.weight = torch.arange(
                output_size * input_size, dtype=torch.float32, device="cpu"
            ).reshape(output_size, input_size) / 1000

        def forward(self, x):
            return nn.functional.linear(x, self.weight), None

    class MergedColumn(Gate):
        def __init__(self, input_size, output_sizes, disable_tp=False, **kwargs):
            super().__init__(input_size, sum(output_sizes))
            self.disable_tp = disable_tp
            if not disable_tp and state.tp > 1:
                self.weight = torch.cat([
                    part.chunk(state.tp, 0)[state.rank]
                    for part in self.weight.split(output_sizes, 0)
                ])

    class Row(Gate):
        def __init__(self, input_size, output_size, disable_tp=False, **kwargs):
            super().__init__(input_size, output_size)
            self.disable_tp = disable_tp
            if not disable_tp and state.tp > 1:
                self.weight = self.weight.chunk(state.tp, 1)[state.rank]

    class Activation(nn.Module):
        def forward(self, x):
            gate, up = x.chunk(2, -1)
            return nn.functional.silu(gate) * up

    class Experts(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.options = kwargs
            self.shared = kwargs.get("shared_experts")
            self.inputs = []

        def forward(self, hidden_states, router_logits):
            self.inputs.append(hidden_states.detach().clone())
            result = hidden_states * 1.3 + router_logits[:, :4] * 0.2
            return result if self.shared is None else result + self.shared(hidden_states)

    def split(x):
        extra = (-x.shape[0]) % state.tp
        padded = nn.functional.pad(x, (0, 0, 0, extra))
        return padded.chunk(state.tp, 0)[state.rank].clone()

    def gather(x, dim):
        assert dim == 0 and x.is_contiguous()
        if state.outputs is None:
            raise GatherCapture(x.detach().clone())
        return torch.cat(state.outputs, dim)

    tree = ast.parse(MOE_PATH.read_text())
    # Keep the actual model classes, chunk helper, and its environment constant.
    tree.body = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
                 or isinstance(node, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "_HYV4_MOE_CHUNK_SIZE_ENV"
                         for t in node.targets)]
    namespace = {
        "__name__": "hy4_cpu_check", "os": os, "torch": torch, "nn": nn,
        "PretrainedConfig": object, "VllmConfig": object, "QuantizationConfig": object,
        "logger": SimpleNamespace(debug_once=lambda *args: None),
        "get_tensor_model_parallel_world_size": lambda: state.tp,
        "get_ep_group": lambda: SimpleNamespace(
            rank_in_group=state.rank, device_group=SimpleNamespace(size=lambda: 32)),
        "GateLinear": Gate, "MergedColumnParallelLinear": MergedColumn,
        "RowParallelLinear": Row, "SiluAndMul": Activation, "FusedMoE": Experts,
        "sequence_parallel_chunk": split, "tensor_model_parallel_all_gather": gather,
        "linear_gate_pcp_shard_enabled": lambda: state.chunked,
        "in_replicated_mtp_batch": lambda: state.chunked,
    }
    exec(compile(tree, str(MOE_PATH), "exec"), namespace)
    return namespace, state


def config(sp, shared):
    model = SimpleNamespace(
        hidden_size=4, num_experts=256, num_experts_per_tok=8,
        num_shared_experts=shared, expert_hidden_dim=4,
        hidden_act="silu", route_norm=True,
    )
    parallel = SimpleNamespace(
        use_sequence_parallel_moe=sp, enable_expert_parallel=True,
        eplb_config=SimpleNamespace(num_redundant_experts=0),
    )
    return model, SimpleNamespace(parallel_config=parallel)


@pytest.mark.parametrize("shape", [(64, 4), (32, 4), (16, 4), (5, 4), (1, 4), (2, 3, 4)])
@pytest.mark.parametrize("shared", [0, 1])
@pytest.mark.parametrize("chunked", [False, True])
def test_sp_preserves_values_and_token_order(shape, shared, chunked, monkeypatch):
    namespace, state = load_model()
    monkeypatch.delenv("VLLM_HCU_HYV4_MOE_CHUNK_SIZE", raising=False)
    x = torch.arange(torch.tensor(shape).prod().item(), dtype=torch.float32,
                     device="cpu").reshape(shape) / 100
    cls = namespace["HYV4MoEFused"]
    model, runtime = config(False, shared)
    expected = cls(model, None, vllm_config=runtime)(x)

    state.tp = 2
    state.chunked = chunked
    if chunked:
        monkeypatch.setenv("VLLM_HCU_HYV4_MOE_CHUNK_SIZE", "3")
    model, runtime = config(True, shared)
    workers = []
    outputs = []
    tokens = x.numel() // x.shape[-1]
    padded = nn.functional.pad(x.reshape(-1, 4), (0, 0, 0, tokens % 2))
    for rank in range(2):
        state.rank = rank
        worker = cls(model, None, vllm_config=runtime)
        assert worker.experts.options["is_sequence_parallel"]
        if shared:
            assert worker.shared_experts.gate_up_proj.disable_tp
            assert worker.shared_experts.down_proj.disable_tp
        with pytest.raises(GatherCapture) as captured:
            worker(x)
        outputs.append(captured.value.args[0])
        received = torch.cat(worker.experts.inputs, 0)
        torch.testing.assert_close(received, padded.chunk(2, 0)[rank])
        assert received.shape[0] == (tokens + 1) // 2
        workers.append(worker)
    state.outputs = outputs
    for rank, worker in enumerate(workers):
        state.rank = rank
        actual = worker(x)
        assert actual.shape == x.shape
        torch.testing.assert_close(actual, expected)


def test_no_sp_keeps_full_batch_and_dense_tp(monkeypatch):
    namespace, state = load_model()
    monkeypatch.delenv("VLLM_HCU_HYV4_MOE_CHUNK_SIZE", raising=False)
    state.tp = 2
    model, runtime = config(False, 0)
    worker = namespace["HYV4MoEFused"](model, None, vllm_config=runtime)
    x = torch.ones((64, 4), device="cpu")
    assert worker(x).shape == x.shape
    assert worker.experts.inputs[0].shape == x.shape
    assert not worker.experts.options["is_sequence_parallel"]
    dense = namespace["HYV4FeedForward"](4, 4, "silu")
    assert not dense.gate_up_proj.disable_tp
    assert not dense.down_proj.disable_tp
