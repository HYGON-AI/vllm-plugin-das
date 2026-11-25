# SPDX-License-Identifier: Apache-2.0
"""Shared config/layout/ownership contracts; native vendor numerics are separate."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from hcu_vllm_das.moe import aiter_fp8 as api


@pytest.fixture
def vendor(monkeypatch):
    def lookup(m, e, n1, n2, k, topk, block, dtype, quant, **kwargs):
        return True, SimpleNamespace(
            quant_type=quant,
            solution_type="asm",
            dtype=dtype,
            config={"SOL_ID1": m + 100, "SOL_ID2": 200, "BLOCK_SIZE_M": 32},
            need_shuffle=bool(kwargs["use_shuffle"]),
            need_shuffle_scale=False,
        )

    def pack(w1, w2, config):
        # Marker outputs expose whether the boundary uses the provider's result;
        # this is not a simulated numerical/packing qualification.
        return (
            torch.full(w1.shape, 2.0).to(w1.dtype),
            torch.full(w2.shape, 3.0).to(w2.dtype),
        )

    def forbidden(*args, **kwargs):
        raise AssertionError("An invalid request reached the native vendor")

    implementation = SimpleNamespace(
        get_aiter_moe_config=lookup,
        AiterMoeConfig=SimpleNamespace,
        MoeQuantType=SimpleNamespace(FP8_W8A8="fp8_w8a8"),
        aiter_moe_shfl_weight=pack,
        aiter_moe=forbidden,
    )
    monkeypatch.setattr(api, "_vendor", lambda: implementation)
    return implementation


def _problem(tokens=6):
    # Deliberately a different model geometry; the shared API has no HY4 constants.
    return api.Problem(
        tokens,
        3,
        8,
        12,
        4,
        2,
        torch.bfloat16,
        gemm1_limit=10.0,
        architecture="unit-test-only",
    )


def _weights():
    return (
        torch.ones(3, 8, 4).to(torch.float8_e4m3fn),
        torch.ones(3, 12, 4).to(torch.float8_e4m3fn),
        torch.ones(3, 8, 1),
        torch.ones(3, 12, 1),
    )


def _inputs(problem):
    return (
        torch.zeros(problem.tokens, problem.k, dtype=problem.dtype),
        torch.ones(problem.tokens, problem.topk, dtype=torch.float32),
        torch.zeros(problem.tokens, problem.topk, dtype=torch.int32),
    )


@pytest.mark.parametrize("shuffle", [False, True])
def test_actual_problem_is_forwarded_to_vendor_before_profile(shuffle, vendor):
    problem = _problem(17)
    selected = api.get_config(problem, use_shuffle=shuffle)
    assert selected.provenance == "vendor_query"
    assert selected.vendor_config.config["SOL_ID1"] == 117
    assert selected.vendor_config.need_shuffle == shuffle
    assert selected.evidence is None


def test_missing_query_requires_explicit_matching_evidence(vendor):
    vendor.get_aiter_moe_config = lambda *a, **k: (False, None)
    problem = _problem()
    with pytest.raises(NotImplementedError, match="No AITER"):
        api.get_config(problem, use_shuffle=False)
    profile = api.QualifiedProfile(
        problem.weight_key,
        False,
        (("SOL_ID1", 10), ("SOL_ID2", 20), ("BLOCK_SIZE_M", 32)),
        evidence="test-contract-only-not-native",
        max_tokens=16,
    )
    selected = api.get_config(problem, use_shuffle=False, profiles=(profile,))
    assert selected.provenance == "qualified_profile"
    assert selected.evidence == profile.evidence
    assert selected.vendor_config.config["SOL_ID1"] == 10
    with pytest.raises(NotImplementedError):
        api.get_config(
            replace(problem, tokens=17), use_shuffle=False, profiles=(profile,)
        )
    with pytest.raises(NotImplementedError):
        api.get_config(replace(problem, k=8), use_shuffle=False, profiles=(profile,))


@pytest.mark.parametrize("invalid", ["dtype", "backend"])
def test_incompatible_vendor_answers_refuse(invalid, vendor):
    original = vendor.get_aiter_moe_config

    def answer(*args, **kwargs):
        found, config = original(*args, **kwargs)
        if invalid == "dtype":
            config.dtype = torch.float16
        else:
            config.solution_type = "triton"
        return found, config

    vendor.get_aiter_moe_config = answer
    with pytest.raises(ValueError, match="incompatible"):
        api.get_config(_problem(), use_shuffle=False)


def test_returned_config_controls_prepared_layout(vendor):
    original = vendor.get_aiter_moe_config

    def answer(*args, **kwargs):
        found, config = original(*args, **kwargs)
        config.need_shuffle = True
        return found, config

    vendor.get_aiter_moe_config = answer
    selected = api.get_config(_problem(), use_shuffle=False)
    prepared = api.prepare_weights(*_weights(), selected)
    assert selected.vendor_config.need_shuffle is True
    assert torch.equal(prepared.w1.float(), torch.full((3, 8, 4), 2.0))


@pytest.mark.parametrize("shuffle", [False, True])
def test_prepare_uses_public_pack_output_and_preserves_raw_inputs(shuffle, vendor):
    weights = _weights()
    selected = api.get_config(_problem(), use_shuffle=shuffle)
    prepared = api.prepare_weights(*weights, selected)
    assert torch.equal(weights[0].float(), torch.ones(3, 8, 4))
    assert torch.equal(weights[1].float(), torch.ones(3, 12, 4))
    if shuffle:
        assert torch.equal(prepared.w1.float(), torch.full((3, 8, 4), 2.0))
        assert torch.equal(prepared.w2.float(), torch.full((3, 12, 4), 3.0))
    else:
        assert prepared.w1 is weights[0] and prepared.w2 is weights[1]
    assert prepared.s1 is weights[2] and prepared.s2 is weights[3]
    rebound = prepared.rebind(
        tuple(torch.nn.Parameter(t, requires_grad=False) for t in prepared.tensors)
    )
    rebound.validate_bindings(rebound.tensors)
    assert rebound.is_current(rebound.tensors)
    with pytest.raises(ValueError, match="storage"):
        prepared.rebind(tuple(t.clone() for t in prepared.tensors))


def test_mutated_config_refuses_before_native(vendor):
    selected = api.get_config(_problem(), use_shuffle=False)
    prepared = api.prepare_weights(*_weights(), selected)
    selected.vendor_config.config["SOL_ID1"] = 999
    with pytest.raises(ValueError, match="mutated"):
        api.run(*_inputs(selected.problem), prepared, selected)


@pytest.mark.parametrize("inference", [False, True])
def test_packed_weights_copy_into_existing_parameter_storage(inference, vendor):
    with torch.inference_mode(inference):
        owners = tuple(torch.nn.Parameter(t, requires_grad=False) for t in _weights())
        loaders = tuple(object() for _ in owners)
        for tensor, loader in zip(owners, loaders):
            tensor.weight_loader = loader
        pointers = tuple(t.data_ptr() for t in owners)
        selected = api.get_config(_problem(), use_shuffle=True)
        prepared = api.prepare_weights(*owners, selected)
        installed = prepared.copy_into(owners)
        assert installed.tensors == owners
        assert tuple(t.data_ptr() for t in owners) == pointers
        assert tuple(t.weight_loader for t in owners) == loaders
        assert torch.equal(owners[0].float(), torch.full((3, 8, 4), 2.0))
        assert torch.equal(owners[1].float(), torch.full((3, 12, 4), 3.0))
        assert installed.is_current(owners)
        installed.validate_bindings(owners)
        with pytest.raises(ValueError, match="storage"):
            installed.rebind(tuple(t.clone() for t in owners))


def test_packed_layout_cannot_silently_switch(vendor):
    selected = api.get_config(_problem(), use_shuffle=True)
    prepared = api.prepare_weights(*_weights(), selected)
    other = api.get_config(_problem(7), use_shuffle=True)
    with pytest.raises(ValueError, match="layout"):
        api.run(*_inputs(other.problem), prepared, other)


def test_changed_prepared_storage_refuses_before_native(vendor):
    selected = api.get_config(_problem(), use_shuffle=False)
    prepared = api.prepare_weights(*_weights(), selected)
    prepared.s1.add_(1)
    with pytest.raises(ValueError, match="changed"):
        api.run(*_inputs(selected.problem), prepared, selected)


def test_empty_problem_has_empty_owned_output_without_vendor(vendor):
    selected = api.get_config(_problem(0), use_shuffle=False)
    prepared = api.prepare_weights(*_weights(), selected)
    output = api.run(*_inputs(selected.problem), prepared, selected)
    assert output.shape == (0, 12) and output.dtype == torch.bfloat16


def test_router_dtype_normalization_and_output_owner(vendor):
    problem = _problem()
    selected = api.get_config(problem, use_shuffle=False)
    prepared = api.prepare_weights(*_weights(), selected)

    def provider_result(x, w1, w2, weights, ids, config, **kwargs):
        value = 7 if ids.dtype == torch.int32 and weights.dtype == torch.float32 else -3
        return x.new_full((x.shape[0], problem.n2), value)

    vendor.aiter_moe = provider_result
    x, weights, ids = _inputs(problem)
    ids = ids.to(torch.int64)
    weights = weights.to(torch.bfloat16)
    output = torch.full((problem.tokens, problem.n2), -9, dtype=problem.dtype)
    result = api.run(x, weights, ids, prepared, selected, output=output)
    assert result is output and torch.equal(output, torch.full_like(output, 7))
    assert ids.dtype == torch.int64 and weights.dtype == torch.bfloat16


@pytest.mark.parametrize("inference", [False, True])
@pytest.mark.parametrize("installed", [False, True])
@pytest.mark.parametrize("index", range(4))
def test_same_object_data_replacement_invalidates_original_storage(
    inference, installed, index, vendor
):
    with torch.inference_mode(inference):
        weights = _weights()
        selected = api.get_config(_problem(), use_shuffle=installed)
        prepared = api.prepare_weights(*weights, selected)
        if installed:
            weights = tuple(torch.nn.Parameter(t, requires_grad=False) for t in weights)
            prepared = prepared.copy_into(weights)
        originals = tuple(t.detach() for t in prepared.tensors)
        tensor = prepared.tensors[index]
        pointer, version = tensor.data_ptr(), api._version(tensor)
        tensor.data = tensor.clone()
        assert tensor.data_ptr() != pointer
        assert api._version(tensor) == version
        assert not prepared.is_current(prepared.tensors)
        with pytest.raises(ValueError, match="changed"):
            prepared.validate_bindings(prepared.tensors)
        # Neither the new allocation nor an alias of the original allocation
        # may launder an already invalid prepared source through rebind.
        for targets in (prepared.tensors, originals):
            with pytest.raises(ValueError, match="changed"):
                prepared.rebind(targets)
        destinations = tuple(torch.empty_like(t) for t in prepared.tensors)
        with pytest.raises(ValueError, match="changed"):
            prepared.copy_into(destinations)


def test_version_changed_source_cannot_be_rebound(vendor):
    selected = api.get_config(_problem(), use_shuffle=False)
    prepared = api.prepare_weights(*_weights(), selected)
    prepared.s1.add_(1)
    with pytest.raises(ValueError, match="changed"):
        prepared.rebind(tuple(t.detach() for t in prepared.tensors))
