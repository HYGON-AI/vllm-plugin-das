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
        [sys.executable, "-c", script], capture_output=True, text=True,
        env=os.environ.copy(), timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
