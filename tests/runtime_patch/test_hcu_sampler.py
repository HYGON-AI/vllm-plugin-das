# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

os.environ.setdefault("VLLM_PLUGINS", "__disabled__")

import pytest
import torch
import vllm.v1.attention.backend as target_attention_backend
from vllm.v1.sample.ops.topk_topp_sampler import TopKTopPSampler
from vllm.v1.sample.metadata import SamplingMetadata


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


@pytest.fixture(scope="module")
def runner_module():
    patch = pytest.MonkeyPatch()
    # The installed target wheel predates this HCU-only metadata alias.
    patch.setattr(
        target_attention_backend,
        "CpCommonAttentionMetadata",
        object,
        raising=False,
    )
    module = importlib.import_module("vllm_hcu.v1.hcu_model_runner")
    yield module
    patch.undo()


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
    monkeypatch.setattr(topk_topp_sample.henvs, "VLLM_HCU_USE_CUSTOM_OPS", False)
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
    monkeypatch.setattr(topk_topp_sample.henvs, "VLLM_HCU_USE_CUSTOM_OPS", True)
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


@pytest.mark.parametrize(
    "logprobs_mode",
    ["raw_logprobs", "processed_logits", "processed_logprobs"],
)
def test_dummy_profile_uses_safe_greedy_then_native_random_warmup(
    runner_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    logprobs_mode: str,
) -> None:
    """Dummy profiling must avoid full-vocabulary top-k and warm native RNG."""

    class _Model:
        @staticmethod
        def compute_logits(hidden_states: torch.Tensor) -> torch.Tensor:
            return torch.tensor(
                [[0.0, 0.5, 1.0, 1.5], [1.5, 1.0, 0.5, 0.0]],
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )

    runner = object.__new__(runner_module.GPUModelRunner)
    runner.device = torch.device("cpu")
    runner.model = _Model()
    runner.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(multimodal_config=None),
    )
    runner.speculative_config = None

    sampler = HcuSampler(logprobs_mode=logprobs_mode)
    sampler.topk_topp_sampler = HcuTopKTopPSampler(logprobs_mode=logprobs_mode)
    runner.sampler = sampler

    metadata_calls: list[SamplingMetadata] = []
    sampler_forward = sampler.forward

    def record_sampler_call(
        *, logits: torch.Tensor, sampling_metadata: SamplingMetadata
    ) -> Any:
        metadata_calls.append(sampling_metadata)
        return sampler_forward(logits=logits, sampling_metadata=sampling_metadata)

    monkeypatch.setattr(sampler, "forward", record_sampler_call)

    native_calls: list[dict[int, torch.Generator]] = []
    native_forward = sampler.topk_topp_sampler.forward_native

    def record_native_random_call(
        logits: torch.Tensor,
        generators: dict[int, torch.Generator],
        top_k: torch.Tensor | None,
        top_p: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        native_calls.append(generators)
        return native_forward(logits, generators, top_k, top_p)

    monkeypatch.setattr(
        sampler.topk_topp_sampler,
        "forward_native",
        record_native_random_call,
    )

    runner._dummy_sampler_run(torch.ones(2, 3))

    assert len(metadata_calls) == 2
    assert all(isinstance(metadata, SamplingMetadata) for metadata in metadata_calls)
    greedy_metadata, random_metadata = metadata_calls
    assert greedy_metadata.all_greedy is True
    assert greedy_metadata.all_random is False
    assert greedy_metadata.top_k.tolist() == [0, 0]
    assert random_metadata.all_greedy is False
    assert random_metadata.all_random is True
    # The native sampler uses vocab_size to represent disabled top-k; zero
    # would be indexed as a real top-k value and fail before RNG warmup.
    assert random_metadata.top_k.tolist() == [4, 4]
    assert random_metadata.top_k.tolist() != [3, 3]
    assert random_metadata.generators
    assert native_calls == [random_metadata.generators]
