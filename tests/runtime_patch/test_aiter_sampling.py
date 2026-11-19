# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

import os
import sys
from types import ModuleType

os.environ.setdefault("VLLM_PLUGINS", "__disabled__")

import pytest
import torch

# Skip, rather than fail collection, on vLLM versions where the sampler moved.
sampler_ops = pytest.importorskip("vllm.v1.sample.ops.topk_topp_sampler")
TopKTopPSampler = sampler_ops.TopKTopPSampler

from vllm_hcu.ops import aiter_sampling
from vllm_hcu.patch.worker.op_opt import patch_topk_topp_sampler
from vllm_hcu.patch.worker.op_opt._common import PatchCompatibilityError


def _das_sampling_module(calls: list) -> ModuleType:
    """Fake ``aiter.ops.sampling`` with the DAS 0.1.8 wrapper signatures."""
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


@pytest.fixture
def das_aiter(monkeypatch):
    """Install a fake DAS ``aiter.ops.sampling`` and hide torch.ops.aiter."""
    calls: list = []
    das = _das_sampling_module(calls)
    aiter_pkg = ModuleType("aiter")
    aiter_ops = ModuleType("aiter.ops")
    aiter_pkg.ops = aiter_ops
    aiter_ops.sampling = das
    monkeypatch.setitem(sys.modules, "aiter", aiter_pkg)
    monkeypatch.setitem(sys.modules, "aiter.ops", aiter_ops)
    monkeypatch.setitem(sys.modules, "aiter.ops.sampling", das)
    monkeypatch.setattr(aiter_sampling, "_has_torch_ops", lambda: False)
    monkeypatch.delenv("VLLM_HCU_USE_AITER_SAMPLER", raising=False)
    monkeypatch.delenv("VLLM_HCU_USE_CUSTOM_OPS", raising=False)
    return das, calls


@pytest.fixture
def device():
    # vLLM's native fallback runs a Triton kernel when Triton is available, so
    # tests that can reach forward_native use the accelerator when present.
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _reference_renorm(probs: torch.Tensor, ks: list[int]) -> torch.Tensor:
    rows = []
    for row, k in zip(probs, ks):
        if k <= 0 or k >= row.numel():
            rows.append(row / row.sum())
            continue
        threshold = row.topk(k).values[-1]
        kept = torch.where(row >= threshold, row, 0.0)
        rows.append(kept / kept.sum())
    return torch.stack(rows)


# --- backends -------------------------------------------------------------


def test_das_backend_passes_merged_arguments():
    calls = []
    ops = aiter_sampling.DasAiterSamplingOps(_das_sampling_module(calls))
    probs = torch.full((2, 4), 0.25)
    k = torch.tensor([2, 3])
    p = torch.tensor([0.5, 0.9])

    ops.top_k_top_p_sampling_from_probs(probs, k, p)
    ops.top_p_sampling_from_probs(probs, p)

    assert calls == [("kp", k, p, None, True), ("p", p, None, True)]


def test_torch_backend_splits_tensor_and_scalar_arguments():
    calls = []

    class Namespace:
        def top_k_top_p_sampling_from_probs(self, *args, **kwargs):
            calls.append(("kp", args[1:], kwargs))

        def top_p_sampling_from_probs(self, *args, **kwargs):
            calls.append(("p", args[1:], kwargs))

        def top_k_renorm_probs(self, *args):
            calls.append(("k", args[1:], {}))

    ops = aiter_sampling.TorchAiterSamplingOps(Namespace())
    probs = torch.full((2, 4), 0.25)
    k = torch.tensor([2, 3])

    ops.top_k_top_p_sampling_from_probs(probs, k, 0.7)
    ops.top_p_sampling_from_probs(probs, 0.7)
    ops.top_k_renorm_probs(probs, k)

    assert calls == [
        ("kp", (None, k, 0, None, 0.7), {"deterministic": True}),
        ("p", (None, None, 0.7), {"deterministic": True}),
        ("k", (k, 0), {}),
    ]


# --- top_k_renorm_probs -------------------------------------------------------

_PROBS = torch.tensor(
    [
        [0.1, 0.4, 0.3, 0.2],
        [0.7, 0.1, 0.1, 0.1],
        [0.25, 0.25, 0.25, 0.25],
        [0.05, 0.15, 0.5, 0.3],
    ]
)


@pytest.mark.parametrize(
    "ks",
    [
        [2, 1, 9, 3],
        # vLLM disables top-k with k == vocab_size; k <= 0 sentinels must also
        # mean "disabled", never greedy.
        [0, -1, 4, 2],
        [0, 0, 0, 0],
    ],
)
def test_top_k_renorm_probs_tensor_matches_reference(ks):
    renorm = aiter_sampling.top_k_renorm_probs(_PROBS, torch.tensor(ks))
    torch.testing.assert_close(renorm, _reference_renorm(_PROBS, ks))


@pytest.mark.parametrize("k", [2, 0, -1, 4, 9])
def test_top_k_renorm_probs_scalar_matches_reference(k):
    renorm = aiter_sampling.top_k_renorm_probs(_PROBS, k)
    torch.testing.assert_close(renorm, _reference_renorm(_PROBS, [k] * 4))


def test_top_k_renorm_probs_does_not_sort_full_vocab(monkeypatch):
    def fail_sort(*args, **kwargs):
        raise AssertionError("full-vocab sort")

    monkeypatch.setattr(torch.Tensor, "sort", fail_sort)
    probs = torch.rand(3, 4096).softmax(dim=-1)
    ks = [5, 0, 40]
    renorm = aiter_sampling.top_k_renorm_probs(probs, torch.tensor(ks))
    torch.testing.assert_close(renorm, _reference_renorm(probs, ks))


# --- backend resolution -------------------------------------------------------


def test_resolve_prefers_compatible_torch_ops(monkeypatch):
    monkeypatch.setattr(aiter_sampling, "_has_torch_ops", lambda: True)
    ops = aiter_sampling.resolve_aiter_sampling_ops()
    assert isinstance(ops, aiter_sampling.TorchAiterSamplingOps)


def test_resolve_adapts_das_wrappers(das_aiter):
    ops = aiter_sampling.resolve_aiter_sampling_ops()
    assert isinstance(ops, aiter_sampling.DasAiterSamplingOps)


@pytest.mark.parametrize(
    "name, replacement",
    [
        # Upstream-style Python signature (probs, indices, ...).
        ("top_p_sampling_from_probs", lambda probs, indices, top_p: None),
        # Matching prefix but no ``indices``/``deterministic`` keywords.
        ("top_p_sampling_from_probs", lambda probs, top_p, deterministic=True: None),
        ("top_k_top_p_sampling_from_probs", lambda probs, top_k, top_p, indices=None: None),
        ("top_k_top_p_sampling_from_probs", None),
    ],
)
def test_resolve_rejects_incompatible_das_wrappers(das_aiter, name, replacement):
    das, _ = das_aiter
    setattr(das, name, replacement)
    assert aiter_sampling.resolve_aiter_sampling_ops() is None


_UPSTREAM_SCHEMAS = {
    "top_k_top_p_sampling_from_probs": (
        "(Tensor probs, Tensor? indices, Tensor? maybe_top_k_arr, int top_k_val, "
        "Tensor? maybe_top_p_arr, float top_p_val, bool deterministic=False) -> Tensor"
    ),
    "top_p_sampling_from_probs": (
        "(Tensor probs, Tensor? indices, Tensor? maybe_top_p_arr, float top_p_val, "
        "bool deterministic=False) -> Tensor"
    ),
    "top_k_renorm_probs": (
        "(Tensor probs, Tensor? maybe_top_k_arr, int top_k_val) -> Tensor"
    ),
}


def _define_namespace(name: str, schemas: dict[str, str]):
    library = torch.library.Library(name, "DEF")
    for op, schema in schemas.items():
        library.define(op + schema)
    return library, getattr(torch.ops, name)


def test_torch_op_schema_check_accepts_upstream_schemas():
    library, namespace = _define_namespace("hcu_test_aiter_ok", _UPSTREAM_SCHEMAS)
    assert all(
        aiter_sampling._torch_op_compatible(namespace, spec)
        for spec in aiter_sampling._OP_SPECS
    )
    del library


@pytest.mark.parametrize(
    "op, schema",
    [
        # Same name, merged (DAS-style) arguments.
        ("top_p_sampling_from_probs", "(Tensor probs, Tensor top_p) -> Tensor"),
        # Missing the keyword-passed ``deterministic``.
        (
            "top_p_sampling_from_probs",
            "(Tensor probs, Tensor? indices, Tensor? maybe_top_p_arr, "
            "float top_p_val) -> Tensor",
        ),
        # Extra required trailing argument.
        (
            "top_k_renorm_probs",
            "(Tensor probs, Tensor? maybe_top_k_arr, int top_k_val, int extra) -> Tensor",
        ),
    ],
)
def test_torch_op_schema_check_rejects_drift(op, schema):
    library, namespace = _define_namespace(
        f"hcu_test_aiter_bad_{abs(hash(schema)) % 10**8}", {op: schema}
    )
    spec = next(s for s in aiter_sampling._OP_SPECS if s.name == op)
    assert not aiter_sampling._torch_op_compatible(namespace, spec)
    del library


# --- patch, exercised through the real vLLM call chain ------------------------


def _patched_sampler_class() -> type:
    # Subclass so patching never mutates the real vLLM class.
    module = ModuleType(patch_topk_topp_sampler.TARGET_MODULE)
    module.TopKTopPSampler = type("TopKTopPSampler", (TopKTopPSampler,), {})
    assert patch_topk_topp_sampler.apply_to_module(module) is True
    return module.TopKTopPSampler


def _rocm_aiter_sampler(sampler_class: type):
    """State TopKTopPSampler.__init__ sets when it selects forward_hip."""
    sampler = sampler_class.__new__(sampler_class)
    torch.nn.Module.__init__(sampler)
    sampler.logprobs_mode = "raw_logprobs"
    sampler.use_fp64_gumbel = False
    sampler.aiter_ops = None
    sampler._aiter_ops_import_failed = False
    sampler.forward = sampler.forward_hip
    return sampler


def test_forward_hip_uses_das_wrappers_end_to_end(das_aiter):
    _, calls = das_aiter
    sampler = _rocm_aiter_sampler(_patched_sampler_class())
    logits = torch.randn(2, 8)
    k = torch.tensor([3, 8])
    p = torch.tensor([0.9, 1.0])

    ids, logprobs = sampler.forward(logits, {}, k, p)
    assert logprobs is None
    assert ids.tolist() == [0, 0]
    assert isinstance(sampler.aiter_ops, aiter_sampling.DasAiterSamplingOps)

    ids, _ = sampler.forward(logits, {}, None, p)
    assert ids.tolist() == [1, 1]

    assert [call[0] for call in calls] == ["kp", "p"]
    kind, top_k, top_p, indices, deterministic = calls[0]
    assert torch.equal(top_k, k) and top_p is p
    assert indices is None and deterministic is True
    assert calls[1] == ("p", p, None, True)


def test_forward_hip_top_k_only_respects_disabled_rows(das_aiter):
    sampler = _rocm_aiter_sampler(_patched_sampler_class())
    logits = torch.full((2, 6), -10.0)
    logits[:, 4] = 10.0
    logits[0, 1] = 9.0
    # Row 0: k=1 keeps only token 4.  Row 1: k=0 disables top-k, so the
    # full distribution survives (renormalized) and sampling stays valid.
    k = torch.tensor([1, 0])
    for _ in range(8):
        ids, _ = sampler.forward(logits.clone(), {}, k, None)
        assert ids[0].item() == 4
        assert 0 <= ids[1].item() < 6


def test_forward_hip_maps_disabled_top_k_before_joint_kernel(das_aiter):
    _, calls = das_aiter
    sampler = _rocm_aiter_sampler(_patched_sampler_class())
    k = torch.tensor([0, -1, 3, 8])
    sampler.forward(torch.randn(4, 8), {}, k, torch.full((4,), 0.9))
    assert calls[0][1].tolist() == [8, 8, 3, 8]
    # The caller's tensor is not mutated.
    assert k.tolist() == [0, -1, 3, 8]


def test_forward_hip_generators_bypass_aiter(das_aiter, device):
    _, calls = das_aiter
    sampler = _rocm_aiter_sampler(_patched_sampler_class())
    generator = torch.Generator(device=device).manual_seed(0)
    ids, _ = sampler.forward(
        torch.randn(1, 8, device=device),
        {0: generator},
        torch.tensor([2], device=device),
        torch.tensor([0.9], device=device),
    )
    assert ids.shape == (1,)
    assert calls == []


@pytest.mark.parametrize(
    "master, feature",
    [
        ("1", "0"), ("1", "false"), ("1", "False"),
        # VLLM_HCU_USE_CUSTOM_OPS is the master switch for this custom op.
        ("0", "1"), ("false", "1"), ("0", "0"),
    ],
)
def test_switch_off_keeps_native_sampling(
    das_aiter, monkeypatch, device, master, feature
):
    _, calls = das_aiter
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", master)
    monkeypatch.setenv("VLLM_HCU_USE_AITER_SAMPLER", feature)
    monkeypatch.setattr(
        aiter_sampling,
        "resolve_aiter_sampling_ops",
        lambda: pytest.fail("must not resolve when the switch is off"),
    )
    sampler = _rocm_aiter_sampler(_patched_sampler_class())

    ids, _ = sampler.forward(
        torch.randn(2, 8, device=device),
        {},
        torch.tensor([3, 8], device=device),
        torch.tensor([0.9, 1.0], device=device),
    )
    assert ids.shape == (2,)
    assert calls == []
    assert sampler.aiter_ops is None
    assert sampler._aiter_ops_import_failed is True
    assert sampler.forward == sampler.forward_native


def test_switch_defaults_on(das_aiter):
    import vllm_hcu.platforms.envs as henvs

    assert henvs.VLLM_HCU_USE_AITER_SAMPLER is True
    assert patch_topk_topp_sampler._use_aiter_sampler() is True


@pytest.mark.parametrize("master", ["1", "0"])
@pytest.mark.parametrize("feature", ["1", "0"])
def test_aiter_sampler_requires_custom_ops_master(monkeypatch, master, feature):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", master)
    monkeypatch.setenv("VLLM_HCU_USE_AITER_SAMPLER", feature)
    expected = master == "1" and feature == "1"
    assert patch_topk_topp_sampler._use_aiter_sampler() is expected


def test_unsupported_aiter_falls_back_to_native(das_aiter, monkeypatch, device):
    _, calls = das_aiter
    monkeypatch.setattr(aiter_sampling, "resolve_aiter_sampling_ops", lambda: None)
    sampler = _rocm_aiter_sampler(_patched_sampler_class())

    ids, _ = sampler.forward(
        torch.randn(2, 8, device=device), {}, None,
        torch.tensor([0.9, 1.0], device=device),
    )
    assert ids.shape == (2,)
    assert calls == []
    assert sampler._aiter_ops_import_failed is True
    assert sampler.forward == sampler.forward_native


def test_missing_aiter_keeps_original_import_failure(monkeypatch, device):
    monkeypatch.delenv("VLLM_HCU_USE_AITER_SAMPLER", raising=False)
    monkeypatch.setitem(sys.modules, "aiter", None)
    monkeypatch.setitem(sys.modules, "aiter.ops.sampling", None)
    monkeypatch.setattr(
        aiter_sampling,
        "resolve_aiter_sampling_ops",
        lambda: pytest.fail("must not resolve after import failure"),
    )
    sampler = _rocm_aiter_sampler(_patched_sampler_class())

    ids, _ = sampler.forward(
        torch.randn(1, 8, device=device), {}, None,
        torch.tensor([0.9], device=device),
    )
    assert ids.shape == (1,)
    assert sampler._aiter_ops_import_failed is True


def test_patch_is_idempotent_and_leaves_vllm_class_untouched():
    module = ModuleType(patch_topk_topp_sampler.TARGET_MODULE)
    module.TopKTopPSampler = type("TopKTopPSampler", (TopKTopPSampler,), {})
    sampler_class = module.TopKTopPSampler

    assert patch_topk_topp_sampler.apply_to_module(module) is True
    patched = (sampler_class._init_aiter_ops, sampler_class.aiter_sample)
    assert patch_topk_topp_sampler.apply_to_module(module) is False
    assert (sampler_class._init_aiter_ops, sampler_class.aiter_sample) == patched
    assert TopKTopPSampler._init_aiter_ops is not patched[0]
    assert TopKTopPSampler.aiter_sample is not patched[1]
    assert not hasattr(sampler_class, "_vllm_hcu_original_init_aiter_ops")


@pytest.mark.parametrize("drifted", ["_init_aiter_ops", "aiter_sample"])
def test_patch_rejects_target_drift(drifted):
    members = {
        "_init_aiter_ops": lambda self: True,
        "aiter_sample": lambda self, logits, k, p, generators: None,
    }
    members[drifted] = lambda self, extra, *rest: None
    module = ModuleType(patch_topk_topp_sampler.TARGET_MODULE)
    module.TopKTopPSampler = type("Drifted", (), members)
    with pytest.raises(PatchCompatibilityError, match="incompatible signature"):
        patch_topk_topp_sampler.apply_to_module(module)
