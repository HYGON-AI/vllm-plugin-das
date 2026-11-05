# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

import os
import sys
from types import ModuleType

os.environ.setdefault("VLLM_PLUGINS", "__disabled__")

import pytest
import torch
from vllm.v1.sample.ops.topk_topp_sampler import TopKTopPSampler

from vllm_hcu.ops import aiter_sampling
from vllm_hcu.patch.worker.op_opt import patch_topk_topp_sampler
from vllm_hcu.patch.worker.op_opt._common import PatchCompatibilityError


def _das_sampling_module(calls: list) -> ModuleType:
    sampling = ModuleType("aiter.ops.sampling")

    def top_k_top_p_sampling_from_probs(
        probs, top_k, top_p, indices=None, filter_apply_order="joint",
        deterministic=True, generator=None, check_nan=False, seed=None,
        offset=None,
    ):
        calls.append(("kp", top_k, top_p, indices, deterministic))
        return torch.zeros(probs.shape[0], dtype=torch.int32)

    def top_p_sampling_from_probs(
        probs, top_p, indices=None, deterministic=True, generator=None,
        check_nan=False, seed=None, offset=None,
    ):
        calls.append(("p", top_p, indices, deterministic))
        return torch.ones(probs.shape[0], dtype=torch.int32)

    sampling.top_k_top_p_sampling_from_probs = top_k_top_p_sampling_from_probs
    sampling.top_p_sampling_from_probs = top_p_sampling_from_probs
    return sampling


def _sampler_module(base: type = TopKTopPSampler) -> ModuleType:
    # Subclass so patching never mutates the real vLLM class.
    module = ModuleType(patch_topk_topp_sampler.TARGET_MODULE)
    module.TopKTopPSampler = type("TopKTopPSampler", (base,), {})
    return module


def test_facade_translates_split_arguments():
    calls = []
    ops = aiter_sampling.DasAiterSamplingOps(_das_sampling_module(calls))
    probs = torch.full((2, 4), 0.25)
    k = torch.tensor([2, 3])
    p = torch.tensor([0.5, 0.9])

    ops.top_k_top_p_sampling_from_probs(probs, None, k, 0, p, 0.0, deterministic=True)
    ops.top_k_top_p_sampling_from_probs(probs, None, None, 5, None, 0.7)
    ops.top_p_sampling_from_probs(probs, None, p, 0.0, deterministic=True)

    assert calls[0] == ("kp", k, p, None, True)
    assert calls[1] == ("kp", 5, 0.7, None, False)
    assert calls[2] == ("p", p, None, True)


@pytest.mark.parametrize("top_k", [torch.tensor([2, 1, 9]), 2])
def test_top_k_renorm_probs_matches_reference(top_k):
    probs = torch.tensor(
        [
            [0.1, 0.4, 0.3, 0.2],
            [0.7, 0.1, 0.1, 0.1],
            [0.25, 0.25, 0.25, 0.25],
        ]
    )
    renorm = aiter_sampling.top_k_renorm_probs(probs, top_k)

    ks = top_k.tolist() if isinstance(top_k, torch.Tensor) else [top_k] * 3
    for row, k in zip(range(probs.shape[0]), ks):
        k = min(k, probs.shape[1])
        threshold = probs[row].topk(k).values[-1]
        expected = torch.where(probs[row] >= threshold, probs[row], 0.0)
        torch.testing.assert_close(renorm[row], expected / expected.sum())


def test_resolve_prefers_registered_torch_ops(monkeypatch):
    monkeypatch.setattr(aiter_sampling, "_has_torch_ops", lambda: True)
    assert aiter_sampling.resolve_aiter_sampling_ops() is torch.ops.aiter


def test_resolve_adapts_das_wrappers_and_rejects_unknown_api(monkeypatch):
    monkeypatch.setattr(aiter_sampling, "_has_torch_ops", lambda: False)
    das = _das_sampling_module([])
    aiter_pkg = ModuleType("aiter")
    aiter_ops = ModuleType("aiter.ops")
    aiter_pkg.ops = aiter_ops
    aiter_ops.sampling = das
    monkeypatch.setitem(sys.modules, "aiter", aiter_pkg)
    monkeypatch.setitem(sys.modules, "aiter.ops", aiter_ops)
    monkeypatch.setitem(sys.modules, "aiter.ops.sampling", das)

    ops = aiter_sampling.resolve_aiter_sampling_ops()
    assert isinstance(ops, aiter_sampling.DasAiterSamplingOps)

    # Upstream-style Python signature (probs, indices, ...) is not adapted.
    das.top_p_sampling_from_probs = lambda probs, indices, top_p: None
    assert aiter_sampling.resolve_aiter_sampling_ops() is None


def test_patch_accepts_installed_vllm_sampler_and_is_idempotent():
    module = _sampler_module()
    sampler_class = module.TopKTopPSampler

    assert patch_topk_topp_sampler.apply_to_module(module) is True
    patched = sampler_class._init_aiter_ops
    assert patch_topk_topp_sampler.apply_to_module(module) is False
    assert sampler_class._init_aiter_ops is patched
    assert TopKTopPSampler._init_aiter_ops is not patched


def _make_patched(original_result: bool):
    calls = []

    class Base:
        def _init_aiter_ops(self):
            calls.append("original")
            if original_result:
                self.aiter_ops = torch.ops.aiter
            return original_result

        def aiter_sample(self, logits, k, p, generators):
            raise AssertionError("not called")

    module = ModuleType(patch_topk_topp_sampler.TARGET_MODULE)
    module.TopKTopPSampler = Base
    patch_topk_topp_sampler.apply_to_module(module)
    return Base, calls


def test_patched_init_uses_resolved_facade(monkeypatch):
    sampler_class, calls = _make_patched(True)
    facade = object()
    monkeypatch.setattr(aiter_sampling, "resolve_aiter_sampling_ops", lambda: facade)

    sampler = sampler_class()
    assert sampler._init_aiter_ops() is True
    assert sampler.aiter_ops is facade
    assert calls == ["original"]


def test_patched_init_falls_back_to_native_when_unsupported(monkeypatch):
    sampler_class, _ = _make_patched(True)
    monkeypatch.setattr(aiter_sampling, "resolve_aiter_sampling_ops", lambda: None)
    sampler = sampler_class()
    sampler.forward_native = lambda *args: "native"

    assert sampler._init_aiter_ops() is False
    assert sampler.aiter_ops is None
    assert sampler._aiter_ops_import_failed is True
    assert sampler.forward() == "native"


def test_patched_init_keeps_original_import_failure(monkeypatch):
    sampler_class, calls = _make_patched(False)
    monkeypatch.setattr(
        aiter_sampling,
        "resolve_aiter_sampling_ops",
        lambda: pytest.fail("must not resolve after import failure"),
    )
    assert sampler_class()._init_aiter_ops() is False
    assert calls == ["original"]


def test_patch_rejects_target_drift():
    class Drifted:
        def _init_aiter_ops(self, extra):
            return True

        def aiter_sample(self, logits, k, p, generators):
            return None

    module = ModuleType(patch_topk_topp_sampler.TARGET_MODULE)
    module.TopKTopPSampler = Drifted
    with pytest.raises(PatchCompatibilityError, match="incompatible signature"):
        patch_topk_topp_sampler.apply_to_module(module)

