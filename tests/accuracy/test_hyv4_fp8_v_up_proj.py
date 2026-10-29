# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

_spec = importlib.util.spec_from_file_location(
    "_hy4_v_up_proj",
    Path(__file__).resolve().parents[2] / "vllm_hcu/models/hy_v4/v_up_proj.py",
)
v_up_proj = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v_up_proj)


@pytest.fixture
def quant_ops(monkeypatch):
    calls = []
    monkeypatch.setenv("VLLM_HCU_HY4_FP8_V_UP_PROJ", "1")
    monkeypatch.setattr(torch.version, "hip", "test")

    def quantize(x):
        calls.append(x.clone())
        scale = x.float().abs().amax().clamp_min(1e-12).reshape(1) / 448
        return (x.float() / scale).to(torch.float8_e4m3fn), scale

    vllm = ModuleType("vllm")
    vllm._custom_ops = SimpleNamespace(scaled_fp8_quant=quantize)
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm_hcu.hcu_ops", ModuleType("hcu_ops"))
    return calls


@pytest.mark.parametrize("flatten_input", [False, True])
@pytest.mark.parametrize("tokens", [1, 3, 16])
def test_fp8_projection_preserves_head_layout_and_output_buffer(
    monkeypatch, quant_ops, flatten_input, tokens
):
    torch.manual_seed(7)
    heads, latent, value = 2, 32, 16
    weight = torch.randn(heads, latent, value).bfloat16()
    layer = SimpleNamespace(
        W_UV=weight,
        num_heads=heads,
        kv_lora_rank=latent,
        v_head_dim=value,
        is_aiter_triton_fp4_bmm_enabled=False,
        is_aiter_triton_fp8_bmm_enabled=False,
    )
    assert v_up_proj.prepare_fp8_v_up_proj(layer, torch.bfloat16)
    torch.testing.assert_close(quant_ops[0], weight.transpose(1, 2).reshape(-1, latent))
    assert layer.W_UV_fp8.shape == (heads, value, latent)
    assert layer.W_UV_fp8.is_contiguous()
    assert layer.W_UV_fp8_scale.numel() == 1

    x = torch.randn(tokens, heads, latent).bfloat16()
    # The MLA caller passes a slice of a larger preallocated output.
    output = torch.full((tokens + 2, heads * value), 123, dtype=torch.bfloat16)
    native_calls = []

    def native(q, w, xs, ws, out):
        native_calls.append(out.data_ptr())
        assert q.shape == (tokens, heads, latent) and q.is_contiguous()
        assert out.shape == (tokens, heads, value) and out.is_contiguous()
        result = torch.einsum("thk,hvk->thv", q.float() * xs, w.float() * ws)
        out.copy_(result)

    monkeypatch.setattr(torch.ops.hcu_ops, "hipblaslt_fp8_bmm", native, raising=False)
    v_up_proj.fp8_v_up_proj(
        layer, x.reshape(tokens, -1) if flatten_input else x, output[:tokens]
    )
    assert native_calls == [output.data_ptr()]
    reference = torch.einsum("thk,hkv->thv", x.float(), weight.float())
    error = (
        (output[:tokens].view_as(reference).float() - reference).square().mean().sqrt()
    )
    assert error / reference.square().mean().sqrt() < 0.06
    assert torch.all(output[tokens:] == 123)


def test_empty_projection_does_not_quantize_or_launch(quant_ops):
    layer = SimpleNamespace(num_heads=2, kv_lora_rank=32, v_head_dim=16)
    v_up_proj.fp8_v_up_proj(layer, torch.empty(0, 2, 32), torch.empty(0, 32))
    assert not quant_ops


def test_disabled_path_requires_no_weights_or_native_extension(monkeypatch):
    monkeypatch.delenv("VLLM_HCU_HY4_FP8_V_UP_PROJ", raising=False)
    assert not v_up_proj.prepare_fp8_v_up_proj(object(), torch.bfloat16)


@pytest.mark.parametrize("unsupported", ["cuda", "fp16", "fp4_bmm", "fp8_bmm"])
def test_enabled_path_rejects_incompatible_modes(monkeypatch, quant_ops, unsupported):
    layer = SimpleNamespace(
        is_aiter_triton_fp4_bmm_enabled=unsupported == "fp4_bmm",
        is_aiter_triton_fp8_bmm_enabled=unsupported == "fp8_bmm",
    )
    if unsupported == "cuda":
        monkeypatch.setattr(torch.version, "hip", None)
    dtype = torch.float16 if unsupported == "fp16" else torch.bfloat16
    with pytest.raises(ValueError, match="requires ROCm BF16 activations"):
        v_up_proj.prepare_fp8_v_up_proj(layer, dtype)
    assert not quant_ops


@pytest.mark.hcu
@pytest.mark.parametrize("tokens", [1, 3, 16, 129])
@pytest.mark.parametrize("heads", [1, 2, 16])
def test_native_hipblaslt_fp8_bmm_and_graph_replay(tokens, heads):
    if not torch.cuda.is_available() or torch.version.hip is None:
        pytest.skip("requires a live HCU with the HIPBLASLt extension built")
    import vllm_hcu.hcu_ops  # noqa: F401
    from vllm import _custom_ops as ops

    torch.manual_seed(19)
    latent, value = 512, 128
    x = torch.randn(tokens, heads, latent, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(heads, value, latent, device="cuda", dtype=torch.bfloat16)
    q, xs = ops.scaled_fp8_quant(x.reshape(-1, latent))
    qw, ws = ops.scaled_fp8_quant(w.reshape(-1, latent))
    output = torch.full((tokens + 2, heads, value), 123, device="cuda", dtype=x.dtype)

    def run():
        torch.ops.hcu_ops.hipblaslt_fp8_bmm(
            q.view_as(x), qw.view_as(w), xs, ws, output[:tokens]
        )

    run()  # cache the BLASLt plan before capture
    expected = torch.einsum(
        "thk,hvk->thv", q.view_as(x).float() * xs, qw.view_as(w).float() * ws
    )
    torch.testing.assert_close(
        output[:tokens], expected.to(x.dtype), atol=0.125, rtol=0.02
    )
    assert torch.all(output[tokens:] == 123)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    # Scales must be device inputs to the captured GEMM, not cached host values.
    xs.mul_(0.5)
    output[:tokens].zero_()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(
        output[:tokens], (expected * 0.5).to(x.dtype), atol=0.125, rtol=0.02
    )
    assert torch.all(output[tokens:] == 123)
