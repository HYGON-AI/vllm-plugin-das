# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Portable contracts for the V4.1-only, prefill-only FlashMLA adapter."""

from __future__ import annotations

import ast
from pathlib import Path
from types import ModuleType

import pytest
import torch

from vllm_hcu.patch.worker.op_opt import patch_deepseek_v41_rocm_flashmla_sparse as patch
from vllm_hcu.patch.worker.op_opt._common import PatchCompatibilityError
from vllm_hcu.platforms import envs
from vllm_hcu.v1.attention.ops import deepseek_v41_flashmla as runtime


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setenv("VLLM_HCU_DSV41_FLASHMLA_PREFILL", "1")
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.setattr(runtime, "_supported_device", lambda q: True)


def _inputs(heads=16, width=128):
    q = torch.zeros(3, heads, 512, dtype=torch.bfloat16)
    return dict(
        q=q,
        kv=torch.zeros(256, 1, 512, dtype=torch.bfloat16),
        indices=torch.arange(width, dtype=torch.int32).repeat(3, 1),
        topk_length=torch.tensor([0, 1, width], dtype=torch.int32),
        scale=512**-0.5,
        head_dim=512,
        nope_head_dim=448,
        rope_head_dim=64,
        attn_sink=torch.zeros(heads + 2, dtype=torch.float32),
        output=torch.full_like(q, 7),
    )


@pytest.mark.parametrize("child,master,expected", [
    (None, None, True), (None, "1", True), (None, "0", False),
    ("0", None, False), ("0", "1", False), ("1", "0", False),
    ("true", "false", False), ("1", "1", True), ("True", None, True),
])
def test_master_and_default_on(monkeypatch, child, master, expected):
    for name, value in (
        ("VLLM_HCU_DSV41_FLASHMLA_PREFILL", child),
        ("VLLM_HCU_USE_CUSTOM_OPS", master),
    ):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    assert envs.dsv41_flashmla_prefill_enabled() is expected
    assert envs.hcu_vllm_environment_variables["VLLM_HCU_DSV41_FLASHMLA_PREFILL"]() is expected


def test_disabled_does_not_resolve_kernel_or_probe_device(monkeypatch):
    monkeypatch.setenv("VLLM_HCU_DSV41_FLASHMLA_PREFILL", "0")
    def forbidden(*args):
        pytest.fail("disabled experiment must not probe device or load FlashMLA")
    monkeypatch.setattr(runtime, "_supported_device", forbidden)
    monkeypatch.setattr(runtime, "_flashmla_prefill", forbidden)
    inputs = _inputs()
    assert not runtime.try_flashmla_prefill(**inputs)
    assert torch.all(inputs["output"] == 7)


@pytest.mark.parametrize("heads", [4, 8, 16, 64, 128])
@pytest.mark.parametrize("three_dimensional", [False, True])
def test_dispatch_masks_lengths_slices_sink_and_preserves_inputs(
    enabled, monkeypatch, heads, three_dimensional,
):
    inputs = _inputs(heads=heads)
    if three_dimensional:
        inputs["indices"] = inputs["indices"].unsqueeze(1)
    indices_before = inputs["indices"].clone()
    calls = []
    def kernel(**kwargs):
        calls.append(kwargs)
        assert kwargs["indices"].shape == (3, 1, 128)
        assert kwargs["indices"].dtype == torch.int32
        assert torch.all(kwargs["indices"][0] == -1)
        assert kwargs["indices"][1, 0, 0] == 0
        assert torch.all(kwargs["indices"][1, 0, 1:] == -1)
        assert kwargs["attn_sink"].shape == (heads,)
        assert kwargs["d_v"] == 512
        kwargs["indices"].fill_(-1)  # Model buffers must survive native sorting.
        return torch.full_like(kwargs["q"], 2), None, None
    monkeypatch.setattr(runtime, "_flashmla_prefill", lambda: kernel)
    # The model's output may be a narrowed view in a mixed prefill/decode batch.
    storage = torch.full((6, heads, 512), 7, dtype=torch.bfloat16)
    inputs["output"] = storage[::2]
    assert runtime.try_flashmla_prefill(**inputs)
    assert len(calls) == 1
    assert torch.all(storage[::2] == 2)
    assert torch.all(storage[1::2] == 7)
    torch.testing.assert_close(inputs["indices"], indices_before)


def test_optional_sink_and_lengths(enabled, monkeypatch):
    inputs = _inputs()
    inputs.update(topk_length=None, attn_sink=None)
    before = inputs["indices"].clone()
    def kernel(**kwargs):
        assert kwargs["topk_length"] is None and kwargs["attn_sink"] is None
        kwargs["indices"].fill_(-1)
        return torch.zeros_like(kwargs["q"]), None, None
    monkeypatch.setattr(runtime, "_flashmla_prefill", lambda: kernel)
    assert runtime.try_flashmla_prefill(**inputs)
    torch.testing.assert_close(inputs["indices"], before)


@pytest.mark.parametrize("change", [
    {"head_dim": 576}, {"nope_head_dim": 512}, {"rope_head_dim": 0},
    {"q": torch.zeros(3, 32, 512, dtype=torch.bfloat16)},
    {"q": torch.zeros(3, 16, 512)},
    {"kv": torch.zeros(256, 2, 512, dtype=torch.bfloat16)},
    {"indices": None}, {"indices": torch.zeros(3, 64, dtype=torch.int32)},
    {"indices": torch.zeros(3, 128, dtype=torch.int64)},
    {"indices": torch.zeros(3, 2, 128, dtype=torch.int32)},
    {"topk_length": torch.zeros(3, dtype=torch.int64)},
    {"topk_length": torch.zeros(2, dtype=torch.int32)},
    {"attn_sink": torch.zeros(15)}, {"attn_sink": torch.zeros(16, dtype=torch.bfloat16)},
    {"output": torch.zeros(3, 16, 512)},
    {"ragged_indices": torch.zeros(3, dtype=torch.int32)},
    {"ragged_indptr": torch.zeros(4, dtype=torch.int32)},
])
def test_unsupported_inputs_keep_output_untouched(enabled, monkeypatch, change):
    inputs = _inputs()
    inputs.update(change)
    before = inputs["output"].clone()
    def forbidden():
        pytest.fail("unsupported input must not resolve FlashMLA")
    monkeypatch.setattr(runtime, "_flashmla_prefill", forbidden)
    assert not runtime.try_flashmla_prefill(**inputs)
    torch.testing.assert_close(inputs["output"], before)


def test_unavailable_library_and_device_fall_back(enabled, monkeypatch):
    monkeypatch.setattr(runtime, "_flashmla_prefill", lambda: None)
    inputs = _inputs()
    assert not runtime.try_flashmla_prefill(**inputs)
    monkeypatch.setattr(runtime, "_supported_device", lambda q: False)
    assert not runtime.try_flashmla_prefill(**inputs)
    assert torch.all(inputs["output"] == 7)


def test_kernel_failure_is_not_swallowed(enabled, monkeypatch):
    def kernel(**kwargs):
        raise RuntimeError("kernel failure")
    monkeypatch.setattr(runtime, "_flashmla_prefill", lambda: kernel)
    with pytest.raises(RuntimeError, match="kernel failure"):
        runtime.try_flashmla_prefill(**_inputs())


def _model_module():
    calls = []
    def prefill(q, kv, indices, topk_length, scale, head_dim, nope_head_dim,
                rope_head_dim, attn_sink, output, ragged_indices=None, ragged_indptr=None):
        calls.append((q, ragged_indices, ragged_indptr))
        return "original"
    module = ModuleType(patch.TARGET_MODULE)
    module.rocm_sparse_attn_prefill = prefill
    module.rocm_sparse_attn_decode = object()
    return module, calls


def test_local_patch_idempotency_and_exact_fallback(monkeypatch):
    module, calls = _model_module()
    original = module.rocm_sparse_attn_prefill
    decode = module.rocm_sparse_attn_decode
    shared = ModuleType("vllm.v1.attention.ops.rocm_aiter_mla_sparse")
    shared.rocm_sparse_attn_prefill = original
    assert patch.apply_to_module(module)
    assert not patch.apply_to_module(module)
    monkeypatch.setattr(patch, "try_flashmla_prefill", lambda **kwargs: False)
    inputs = _inputs()
    assert module.rocm_sparse_attn_prefill(**inputs) == "original"
    assert calls[0][0] is inputs["q"]
    assert module.rocm_sparse_attn_decode is decode
    assert shared.rocm_sparse_attn_prefill is original
    monkeypatch.setattr(patch, "try_flashmla_prefill", lambda **kwargs: True)
    assert module.rocm_sparse_attn_prefill(**inputs) is None
    assert len(calls) == 1
    module.rocm_sparse_attn_prefill = original
    with pytest.raises(PatchCompatibilityError, match="stale"):
        patch.apply_to_module(module)


def test_signature_drift_and_wrong_target_fail_before_mutation():
    module, _ = _model_module()
    module.rocm_sparse_attn_prefill = lambda q: None
    with pytest.raises(PatchCompatibilityError, match="signature"):
        patch.apply_to_module(module)
    assert not hasattr(module, patch._MARKER)
    with pytest.raises(PatchCompatibilityError, match="expected module"):
        patch.apply_to_module(ModuleType("vllm.models.deepseek_v4.amd.rocm"))


def test_adapter_is_registered_as_a_model_local_callback():
    source = Path(__file__).resolve().parents[2] / "vllm_hcu/patch/worker/__init__.py"
    tree = ast.parse(source.read_text())
    assert any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id == "_adapter"
        and [arg.value for arg in node.args if isinstance(arg, ast.Constant)]
        == ["op_opt", "patch_deepseek_v41_rocm_flashmla_sparse"]
        for node in ast.walk(tree)
    )
