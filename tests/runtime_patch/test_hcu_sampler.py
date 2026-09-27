# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

os.environ.setdefault("VLLM_PLUGINS", "__disabled__")

import pytest
import torch
from vllm.v1.sample.ops.topk_topp_sampler import TopKTopPSampler


def _load_topk_topp_sample() -> ModuleType:
    """Load the sampler without importing unrelated lightop-backed ops."""
    module_path = (
        Path(__file__).resolve().parents[2]
        / "vllm_hcu"
        / "ops"
        / "topk_topp_sample.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_vllm_hcu_test_topk_topp_sample",
        module_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


topk_topp_sample = _load_topk_topp_sample()
HcuSampler = topk_topp_sample.HcuSampler
HcuTopKTopPSampler = topk_topp_sample.HcuTopKTopPSampler


def test_hcu_sampler_keeps_official_topk_topp_when_custom_sampler_disabled(
    monkeypatch,
):
    monkeypatch.setattr(topk_topp_sample.henvs, "VLLM_HCU_USE_CUSTOM_OPS", True)
    monkeypatch.setattr(
        topk_topp_sample.henvs,
        "VLLM_HCU_USE_CUSTOM_TOPK_TOPP_SAMPLER",
        False,
    )

    sampler = HcuSampler()

    assert type(sampler.topk_topp_sampler) is TopKTopPSampler


def test_hcu_sampler_uses_hcu_topk_topp_only_when_explicitly_enabled(
    monkeypatch,
):
    monkeypatch.setattr(topk_topp_sample.henvs, "VLLM_HCU_USE_CUSTOM_OPS", True)
    monkeypatch.setattr(
        topk_topp_sample.henvs,
        "VLLM_HCU_USE_CUSTOM_TOPK_TOPP_SAMPLER",
        True,
    )

    sampler = HcuSampler()

    assert type(sampler.topk_topp_sampler) is HcuTopKTopPSampler


def test_hcu_topk_topp_forward_is_feature_gated_and_lazy(monkeypatch):
    sampler = HcuTopKTopPSampler()
    native_calls = []
    monkeypatch.setattr(
        sampler,
        "forward_native",
        lambda logits, generators, k, p: (
            native_calls.append((logits, generators, k, p)) or torch.tensor([7]),
            None,
        ),
    )
    logits = torch.zeros(1, 8)
    k = torch.tensor([2])

    monkeypatch.delitem(sys.modules, "lightop", raising=False)
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "0")
    monkeypatch.setattr(
        topk_topp_sample.henvs,
        "VLLM_HCU_USE_CUSTOM_TOPK_TOPP_SAMPLER",
        True,
    )
    token_ids, _ = sampler(logits, {}, k, None)
    assert token_ids.tolist() == [7]
    assert len(native_calls) == 1
    assert "lightop" not in sys.modules

    lightop = ModuleType("lightop")
    custom_calls = []
    lightop.sampling = SimpleNamespace(
        top_k_top_p_sampling_from_probs=lambda probs, top_k, top_p, deterministic: (
            custom_calls.append((probs, top_k, top_p, deterministic))
            or torch.tensor([[3]])
        )
    )
    monkeypatch.setitem(sys.modules, "lightop", lightop)
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    token_ids, _ = sampler(logits, {}, k, None)
    assert token_ids.tolist() == [3]
    assert len(custom_calls) == 1
    assert custom_calls[0][3] is True

    monkeypatch.setitem(sys.modules, "lightop", ModuleType("lightop"))
    with pytest.raises(RuntimeError, match="lightop is unavailable"):
        sampler(logits, {}, k, None)

    token_ids, _ = sampler(logits, {0: torch.Generator()}, k, None)
    assert token_ids.tolist() == [7]
    assert len(native_calls) == 2


def test_hcu_topk_topp_custom_path_receives_softmax_probs_and_filters(
    monkeypatch,
):
    sampler = HcuTopKTopPSampler()
    calls = []

    def custom_sampling(probs, top_k, top_p, deterministic):
        calls.append((probs, top_k, top_p, deterministic))
        return torch.tensor([[2], [0]], dtype=torch.int64)

    lightop = ModuleType("lightop")
    lightop.sampling = SimpleNamespace(
        top_k_top_p_sampling_from_probs=custom_sampling,
    )
    monkeypatch.setitem(sys.modules, "lightop", lightop)
    monkeypatch.setattr(topk_topp_sample.henvs, "VLLM_HCU_USE_CUSTOM_OPS", True)
    monkeypatch.setattr(
        topk_topp_sample.henvs,
        "VLLM_HCU_USE_CUSTOM_TOPK_TOPP_SAMPLER",
        True,
    )
    logits = torch.tensor(
        [[0.0, 1.0, 2.0], [4.0, 0.0, -4.0]],
        dtype=torch.float32,
    )
    top_k = torch.tensor([2, 1], dtype=torch.int32)
    top_p = torch.tensor([0.8, 1.0], dtype=torch.float32)

    token_ids, logprobs = sampler(logits, {}, top_k, top_p)

    assert token_ids.tolist() == [2, 0]
    assert logprobs is None
    assert len(calls) == 1
    probs, observed_top_k, observed_top_p, deterministic = calls[0]
    torch.testing.assert_close(probs, logits.softmax(dim=-1, dtype=torch.float32))
    assert probs.is_contiguous()
    assert observed_top_k is top_k
    assert observed_top_p is top_p
    assert deterministic is True
