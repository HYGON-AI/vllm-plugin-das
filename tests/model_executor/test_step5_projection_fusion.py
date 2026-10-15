# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Projection fusion must preserve the linear layers' supported behavior."""

import os
import subprocess
import sys


def test_qkvg_fusion_bias_layout_and_loaded_weights():
    # Keep model registration and lazy import callbacks out of other unit tests.
    script = r"""
import torch
from vllm_hcu import hcu_platform_register_model
hcu_platform_register_model()
from vllm_hcu.models import step5_text as m
from vllm_hcu.platforms import envs
envs.VLLM_USE_NN = True
m.FUSE_QKVG = True

def projection(shape, dtype, bias):
    p = torch.nn.Module()
    p.weight = torch.nn.Parameter(torch.ones(shape, dtype=dtype))
    p.bias = torch.nn.Parameter(torch.ones(shape[-1], dtype=dtype)) if bias else None
    return p

def attention(qbias=False, gbias=False, qdtype=torch.bfloat16, gdtype=None, gshape=(4,2)):
    a = m.Step5Attention.__new__(m.Step5Attention)
    torch.nn.Module.__init__(a)
    a.qkv_proj = projection((4,6), qdtype, qbias)
    a.g_proj = projection(gshape, gdtype or qdtype, gbias)
    a.register_buffer("qkvg_weight", None, persistent=False)
    return a

for qbias, gbias in [(True,False),(False,True),(True,True)]:
    a = attention(qbias, gbias)
    old = a.qkv_proj.weight.data_ptr()
    a._maybe_fuse_qkvg()
    assert a.qkvg_weight is None
    assert a.qkv_proj.weight.data_ptr() == old
    x = torch.ones(2,4,dtype=torch.bfloat16)
    q = x @ a.qkv_proj.weight
    if qbias:
        q += a.qkv_proj.bias
    assert (q == (5 if qbias else 4)).all()

for dtype in [torch.bfloat16, torch.float16]:
    a = attention(qdtype=dtype)
    a._maybe_fuse_qkvg()
    assert a.qkvg_weight is not None
    # Weight loaders populate the shared views after the constructor.
    with torch.no_grad():
        a.qkv_proj.weight.fill_(1)
        a.g_proj.weight.fill_(2)
    x = torch.ones(2,4,dtype=dtype)
    actual = x @ a.qkvg_weight
    expected = torch.cat([x @ a.qkv_proj.weight, x @ a.g_proj.weight],dim=-1)
    torch.testing.assert_close(actual, expected)

for kwargs in [dict(gshape=(8,)),dict(gshape=(3,2)),dict(gdtype=torch.float16),
               dict(qdtype=torch.float32)]:
    a = attention(**kwargs)
    a._maybe_fuse_qkvg()
    assert a.qkvg_weight is None
print("projection regressions passed")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=os.environ.copy(),
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_router_shared_activation_is_an_explicit_gate_result():
    # Keep model registration and module-level Step5 feature flags isolated.
    script = r"""
from types import SimpleNamespace
import torch
from vllm_hcu import hcu_platform_register_model
hcu_platform_register_model()
from vllm_hcu.models import step5_text as m

class Shared(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fused_act = "stale"
        self.limit = 7.0
        self.gate_up_proj = SimpleNamespace(
            weight=torch.ones(4, 6, dtype=torch.bfloat16)
        )

shared = Shared()
gate = m.BF16FP32RouterLinear.__new__(m.BF16FP32RouterLinear)
torch.nn.Module.__init__(gate)
gate.weight = torch.nn.Parameter(torch.ones(4, 5, dtype=torch.bfloat16))
gate._fused_shared = [shared]
gate._rsg_workspace = (object(), object())
expected_logits = torch.arange(10, dtype=torch.float32).view(2, 5)
expected_act = torch.arange(6, dtype=torch.bfloat16).view(2, 3)
def rsg(*args, **kwargs):
    assert kwargs["workspace"] is gate._rsg_workspace
    return expected_logits, expected_act
gate._rsg = rsg
gate._gemm = lambda *args: (_ for _ in ()).throw(AssertionError("fallback"))

logits, act = gate(torch.ones(2, 4, dtype=torch.bfloat16))
torch.testing.assert_close(logits, expected_logits)
assert logits.data_ptr() == expected_logits.data_ptr()
assert act is expected_act
assert shared.fused_act is None

shared.fused_act = "stale"
fallback, act = gate(torch.ones(2, 4, dtype=torch.float32))
torch.testing.assert_close(fallback, torch.full((2, 5), 4.0))
assert act is None
assert shared.fused_act is None

class Experts(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.is_internal_router = False
        self.moe_config = SimpleNamespace(is_sequence_parallel=False)
        self.seen = None
    def forward(self, hidden_states, router_logits):
        self.seen = shared.fused_act
        return hidden_states

block = m.FusedMoEBlock.__new__(m.FusedMoEBlock)
torch.nn.Module.__init__(block)
block.gate = gate
block.share_expert = shared
block.experts = Experts()
block.fold_shared_expert = True
block.fuse_all_reduce = False
torch.cuda.current_stream = lambda: SimpleNamespace(cuda_stream=123)

hidden = torch.ones(2, 4, dtype=torch.bfloat16)
shared_out, routed_out = block(hidden)
assert shared_out is None
assert routed_out.data_ptr() == hidden.data_ptr()
assert block.experts.seen[0] == 123
assert block.experts.seen[1] == hidden.shape
assert block.experts.seen[2] is expected_act

class StaticStream:
    cuda_stream = 123

class DataExperts(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.is_internal_router = False
        self.moe_config = SimpleNamespace(is_sequence_parallel=False)
    def forward(self, hidden_states, router_logits):
        act = shared.fused_act[2]
        return hidden_states + act.sum(dim=1, keepdim=True)

stream = StaticStream()
torch.cuda.current_stream = lambda: stream
gate._rsg = lambda x, w, wgu, limit, workspace: (
    x.float() @ w.float(), x[:, :3]
)
block.experts = DataExperts()
compiled = torch.compile(block, backend="eager", fullgraph=True)
eager = block(hidden)[1]
torch.testing.assert_close(compiled(hidden)[1], eager)
hidden2 = hidden + 1
torch.testing.assert_close(compiled(hidden2)[1], block(hidden2)[1])
print("explicit fused shared activation handoff passed")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=os.environ.copy(),
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_router_shared_workspaces_are_not_shared_between_layers():
    script = r"""
import torch
from vllm_hcu.ops.step5.router_gemm import make_router_shared_workspace

first = make_router_shared_workspace(torch.device("cpu"), 352, 384)
second = make_router_shared_workspace(torch.device("cpu"), 352, 384)
assert first[0].shape == second[0].shape == (786432,)
assert first[1].shape == second[1].shape == (24,)
assert first[0].data_ptr() != second[0].data_ptr()
assert first[1].data_ptr() != second[1].data_ptr()
assert not first[1].any() and not second[1].any()
print("per-layer router/shared workspaces passed")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=os.environ.copy(),
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
