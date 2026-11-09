# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""D192 dispatch and real FP8 attention over padded NHD/HND cache views."""
from __future__ import annotations

import ast
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

SOURCE = Path(__file__).resolve().parents[2] / "vllm_hcu/v1/attention/backends/fa_utils.py"


def _guard(layout):
    tree = ast.parse(SOURCE.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == "can_use_paged_fp8_d192_fwd")
    namespace = {"torch": torch, "Tensor": torch.Tensor,
                 "get_kv_cache_layout": layout if callable(layout) else lambda: layout}
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, node], type_ignores=[]))
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace[node.name]


@pytest.mark.parametrize("layout", ["NHD", "HND"])
@pytest.mark.parametrize("page", [64, 128])
def test_d192_dispatch_accepts_both_cache_layouts(layout, page):
    k = torch.empty((2, page, 4, 192), dtype=torch.float8_e4m3fn)
    out = torch.empty((3, 64, 192), dtype=torch.bfloat16)
    assert _guard(layout)(k, k, out, None, 0.0, None, allow_hnd=True)


@pytest.mark.parametrize("layout,expected", [("NHD", True), ("HND", False)])
def test_non_step5_pd_dispatch_preserves_original_layout_guard(layout, expected):
    k = torch.empty((2, 64, 4, 192), dtype=torch.float8_e4m3fn)
    out = torch.empty((3, 64, 192), dtype=torch.bfloat16)
    assert _guard(layout)(k, k, out, None, 0.0, None) is expected


def test_unsupported_dtype_does_not_query_layout():
    k = torch.empty((2, 64, 4, 192), dtype=torch.bfloat16)
    out = torch.empty((3, 64, 192), dtype=torch.bfloat16)
    assert not _guard(lambda: pytest.fail("ineligible dtype queried layout"))(
        k, k, out, None, 0.0, None)


@pytest.mark.parametrize("layout,head,page,dtype,out_dtype,softcap", [
    ("unknown", 192, 64, torch.float8_e4m3fn, torch.bfloat16, 0.0),
    ("HND", 128, 64, torch.float8_e4m3fn, torch.bfloat16, 0.0),
    ("HND", 192, 32, torch.float8_e4m3fn, torch.bfloat16, 0.0),
    ("HND", 192, 64, torch.bfloat16, torch.bfloat16, 0.0),
    ("HND", 192, 64, torch.float8_e4m3fn, torch.float16, 0.0),
    ("HND", 192, 64, torch.float8_e4m3fn, torch.bfloat16, 1.0),
])
def test_d192_dispatch_retains_unsupported_call_guards(layout, head, page, dtype, out_dtype, softcap):
    k = torch.empty((2, page, 4, head), dtype=dtype)
    out = torch.empty((3, 64, head), dtype=out_dtype)
    assert not _guard(layout)(k, k, out, None, softcap, None, allow_hnd=True)


@pytest.fixture(scope="module")
def d192_gpu_kernels():
    if not torch.cuda.is_available():
        pytest.skip("HCU GPU is required")
    arch = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")
    if not arch.startswith("gfx938"):
        pytest.skip("gfx938 FP8 D192 kernel is required")
    from vllm_hcu.v1.attention.backends.fa_utils import paged_fp8_d192_fwd
    from vllm_hcu.v1.attention.backends.swa_decode import (
        swa_decode_fp8_d192, swa_decode_fp8_d192_gate_quant,
    )
    return SimpleNamespace(prefix=paged_fp8_d192_fwd, decode=swa_decode_fp8_d192,
                           gated=swa_decode_fp8_d192_gate_quant)


@pytest.mark.parametrize("layout", ["NHD", "HND"])
@pytest.mark.parametrize("page", [64, 128])
@pytest.mark.parametrize("heads", [1, 4])
@pytest.mark.parametrize("query_len", [1, 4, 5, 17])
def test_fp8_d192_padded_shared_kv_storage(d192_gpu_kernels, layout, page, heads, query_len):
    torch.manual_seed(17)
    torch.backends.cuda.matmul.allow_tf32 = False
    dtype = torch.float8_e4m3fn
    blocks, dim, query_heads = 9, 192, heads * 16
    seq_len = 5 * page + 17
    block_stride = 2 * heads * page * dim + 256
    raw = torch.zeros((128 + blocks * block_stride,), device="cuda", dtype=torch.bfloat16).to(dtype)
    if layout == "HND":
        physical = torch.as_strided(raw, (blocks, 2, heads, page, dim),
                                    (block_stride, heads * page * dim, page * dim, dim, 1), 128)
        k, v = physical[:, 0].transpose(1, 2), physical[:, 1].transpose(1, 2)
    else:
        physical = torch.as_strided(raw, (blocks, 2, page, heads, dim),
                                    (block_stride, page * heads * dim, heads * dim, dim, 1), 128)
        k, v = physical[:, 0], physical[:, 1]
    k.copy_((torch.randn(k.shape, device="cuda", dtype=torch.bfloat16) * 0.5).to(dtype))
    v.copy_((torch.randn(v.shape, device="cuda", dtype=torch.bfloat16) * 0.5).to(dtype))
    q = (torch.randn((query_len, query_heads, dim), device="cuda", dtype=torch.bfloat16) * 0.5).to(dtype)
    out = torch.empty(q.shape, device="cuda", dtype=torch.bfloat16)
    table = torch.tensor([[7, 2, 4, 1, 8, 0]], device="cuda", dtype=torch.int32)
    cu = torch.tensor([0, query_len], device="cuda", dtype=torch.int32)
    lengths = torch.tensor([seq_len], device="cuda", dtype=torch.int32)
    scale = dim ** -0.5
    d192_gpu_kernels.prefix(q, k, v, out, cu, lengths, table,
                            query_len, seq_len, scale, True, [511, 0])
    selected = table[0].long()
    keys = k[selected].flatten(0, 1)[:seq_len].float().repeat_interleave(16, dim=1).transpose(0, 1)
    values = v[selected].flatten(0, 1)[:seq_len].float().repeat_interleave(16, dim=1).transpose(0, 1)
    scores = (q.float().transpose(0, 1) @ keys.transpose(1, 2)) * scale
    key_pos = torch.arange(seq_len, device="cuda")
    query_pos = seq_len - query_len + torch.arange(query_len, device="cuda")
    visible = (key_pos[None, :] <= query_pos[:, None]) & (key_pos[None, :] >= query_pos[:, None] - 511)
    reference = (scores.masked_fill(~visible[None, :, :], float("-inf")).softmax(-1) @ values).transpose(0, 1)

    def check(actual, expected):
        diff = actual.float() - expected
        nrmse = (diff.square().mean() / expected.square().mean().clamp_min(1e-12)).sqrt().item()
        assert math.isfinite(nrmse) and nrmse < 0.06
        assert diff.abs().max().item() < 0.12

    check(out, reference)
    if query_len <= 4:
        d192_gpu_kernels.decode(q, k, v, out, cu, lengths, table, scale, 511)
        check(out, reference)
        gate = torch.randn((query_len, query_heads), device="cuda", dtype=torch.bfloat16) * 0.25
        quantized = torch.empty((query_len, query_heads * dim), device="cuda", dtype=dtype)
        scales = torch.empty((query_len, 1), device="cuda", dtype=torch.float32)
        d192_gpu_kernels.gated(q, k, v, cu, lengths, table, scale, 511, gate, quantized, scales)
        restored = quantized.float().reshape(query_len, query_heads, dim) * scales.reshape(query_len, 1, 1)
        check(restored, reference * gate.float().sigmoid()[:, :, None])
    assert k.dtype == v.dtype == dtype
    assert k.untyped_storage().data_ptr() == v.untyped_storage().data_ptr() == raw.data_ptr()
