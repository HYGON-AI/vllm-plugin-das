"""Protect gate strides, source numerical parity and graph output ownership."""
import ast
import os
from pathlib import Path
import sys
import types

import pytest
import torch

from vllm_hcu.models.kimi_k3.amd.ops import kda_gated_norm as norm


def test_invalid_shapes_fail_before_device_launch():
    with pytest.raises(ValueError, match="matching"):
        norm.rms_norm_gated(torch.empty(2, 4), torch.empty(3, 4), None, None)
    with pytest.raises(ValueError, match="Residual"):
        norm.rms_norm_gated(torch.empty(2, 4), torch.empty(2, 4), None, None,
                            residual=torch.empty(1, 4))


def test_kimi_import_keeps_base_class_and_registry_unchanged():
    from vllm.model_executor.layers.fla.ops.kda import FusedRMSNormGated
    from vllm_hcu.models.kimi_k3.amd.kimi_gdn_linear_attn import (
        FusedRMSNormGated as KimiNorm,
    )
    assert KimiNorm is norm.KimiFusedRMSNormGated
    assert KimiNorm is not FusedRMSNormGated
    assert issubclass(KimiNorm, FusedRMSNormGated)
    assert FusedRMSNormGated.forward_cuda.__module__ == "vllm.model_executor.layers.fla.ops.kda"


@pytest.fixture(scope="module")
def source_norm():
    """Load only norm definitions for an optional frozen-source parity test.

    No source imports or production path substitution: keep the original file
    location so Triton can inspect these functions' unmodified source text.
    """
    root = os.environ.get("KIMI_KDA_SOURCE_ROOT")
    if not root:
        pytest.skip("Set KIMI_KDA_SOURCE_ROOT for frozen-source device parity")
    path = Path(root) / "vllm/third_party/flash_linear_attention/ops/kda.py"
    from vllm.triton_utils import triton, tl
    from vllm.utils.math_utils import cdiv, next_power_of_2
    module = types.ModuleType("_kda_frozen_source_norm")
    module.__file__ = str(path)
    module.__dict__.update(torch=torch, triton=triton, tl=tl,
                           cdiv=cdiv, next_power_of_2=next_power_of_2)
    sys.modules[module.__name__] = module
    names = {"layer_norm_gated_fwd_kernel", "layer_norm_gated_fwd_kernel1",
             "layer_norm_gated_fwd", "rms_norm_gated"}
    tree = ast.parse(path.read_text(), filename=str(path))
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in names]
    assert {node.name for node in tree.body} == names
    exec(compile(tree, str(path), "exec"), module.__dict__)
    yield module.rms_norm_gated
    del sys.modules[module.__name__]


def gate_view(tokens, heads, width, layout, dtype):
    if layout == "row_strided":
        storage = torch.randn(tokens, heads * width * 2, device="cuda", dtype=dtype)
        return storage[:, heads * width:].view(tokens, heads, width)
    if layout == "inner_strided":
        return torch.randn(tokens, heads, width * 2, device="cuda", dtype=dtype)[..., ::2]
    return torch.randn(tokens, heads, width, device="cuda", dtype=dtype)


@pytest.mark.parametrize("tokens,heads,width,layout,activation,residual", [
    (1, 24, 128, "row_strided", "sigmoid", False),
    (32, 24, 128, "row_strided", "sigmoid", False),
    (17, 3, 96, "row_strided", "swish", True),
    (4, 2, 512, "contiguous", "silu", False),
    (4, 2, 768, "contiguous", "sigmoid", True),
])
def test_device_matches_frozen_source_bitwise(
    source_norm, tokens, heads, width, layout, activation, residual,
):
    if not torch.cuda.is_available():
        pytest.skip("HCU required")
    torch.manual_seed(42)
    x = torch.randn(1, tokens, heads, width, device="cuda", dtype=torch.bfloat16)
    g = gate_view(tokens, heads, width, layout, x.dtype)
    w = torch.randn(width, device="cuda", dtype=torch.float32)
    r = torch.randn_like(x) if residual else None
    expected = source_norm(x.clone(), g, w, None, activation,
                           residual=r, prenorm=True, residual_in_fp32=True)
    actual_x = x.clone()
    actual = norm.rms_norm_gated(actual_x, g, w, None, activation,
                                 residual=r, prenorm=True, residual_in_fp32=True)
    assert actual[0].data_ptr() == actual_x.data_ptr()
    for a, b in zip(actual, expected):
        assert torch.isfinite(a).all()
        assert torch.equal(a, b), (a.float() - b.float()).abs().max().item()


@pytest.mark.parametrize("layout", ["row_strided", "inner_strided"])
def test_device_matches_math_and_keeps_gate_unchanged(layout):
    if not torch.cuda.is_available():
        pytest.skip("HCU required")
    x = torch.randn(17, 3, 128, device="cuda", dtype=torch.bfloat16)
    g = gate_view(17, 3, 128, layout, x.dtype)
    saved_g = g.clone()
    expected = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
    expected = (expected * torch.sigmoid(g.float())).to(x.dtype)
    actual = norm.rms_norm_gated(x, g, None, None, activation="sigmoid")
    torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.01)
    assert torch.equal(g, saved_g)


def test_device_row_strided_gate_never_materialized(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("HCU required")
    x = torch.randn(1, 32, 24, 128, device="cuda", dtype=torch.bfloat16)
    g = gate_view(32, 24, 128, "row_strided", x.dtype)
    gate_ptr = g.data_ptr()
    original = torch.Tensor.contiguous

    def guarded(tensor, *args, **kwargs):
        if tensor.data_ptr() == gate_ptr:
            raise AssertionError("Row-strided gate must be consumed directly")
        return original(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "contiguous", guarded)
    norm.rms_norm_gated(x, g, None, None, activation="sigmoid")
    torch.cuda.synchronize()


def test_device_graph_replay_uses_new_gate_and_input():
    if not torch.cuda.is_available():
        pytest.skip("HCU required")
    x = torch.randn(1, 1, 24, 128, device="cuda", dtype=torch.bfloat16)
    g = gate_view(1, 24, 128, "row_strided", x.dtype)
    for _ in range(2):
        norm.rms_norm_gated(x, g, None, None, activation="sigmoid")
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = norm.rms_norm_gated(x, g, None, None, activation="sigmoid")
    for seed in (101, 202):
        torch.manual_seed(seed)
        x.copy_(torch.randn_like(x))
        g.copy_(torch.randn_like(g))
        expected = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
        expected = (expected * torch.sigmoid(g.float())).to(x.dtype)
        graph.replay()
        torch.cuda.synchronize()
        assert out.data_ptr() == x.data_ptr()
        torch.testing.assert_close(out, expected, rtol=0.01, atol=0.01)
