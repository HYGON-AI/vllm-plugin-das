# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Static EPLB plans must fail closed before checkpoint writes."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest
import torch

from tests.models.static_eplb_test_utils import GenericMoE, config_and_map

from vllm_hcu.model_executor.layers.fused_moe.static_eplb import (
    StaticEplbPlan,
    bind_static_eplb_plan,
    load_static_logical_expert,
    load_static_eplb_plan,
    resolve_offline_eplb_model_key,
    verify_static_plan_across_ep_ranks,
)


def _write_plan(path, rows=None, *, model_key="HYV4ForCausalLM", **metadata):
    rows = [[0, 1, 2, 1], [2, 0, 1, 2]] if rows is None else rows
    model_map = {
        "record_kind": "proposal",
        "model_name": "hy4-test",
        "model_class": model_key,
        "num_moe_layers": 2,
        "num_logical_experts": 3,
        "num_physical_experts": 4,
        "num_redundant_experts": 1,
        "physical_to_logical_map": rows,
    }
    model_map.update(metadata)
    path.write_text(
        json.dumps(
            {
                "version": 2,
                "format": "vllm_offline_eplb_physical_to_logical_by_model",
                "model_maps": {model_key: model_map},
            }
        )
    )
    return path


def _load(path, **kwargs):
    request = {
        "model_key": "HYV4ForCausalLM",
        "expected_shape": (2, 4),
        "num_logical_experts": 3,
        "num_redundant_experts": 1,
    }
    request.update(kwargs)
    return load_static_eplb_plan(path, **request)


def test_plan_is_immutable_and_fingerprints_exact_file_bytes(tmp_path) -> None:
    path = _write_plan(tmp_path / "map.json")
    plan = _load(path)

    assert plan.source_path == str(path.resolve())
    assert plan.source_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert plan.physical_to_logical_map.tolist() == [
        [0, 1, 2, 1],
        [2, 0, 1, 2],
    ]
    assert plan.layer_map(0) == (0, 1, 2, 1)
    assert plan.fingerprint() == (
        "HYV4ForCausalLM",
        plan.source_sha256,
        (2, 4),
        3,
        4,
        1,
    )

    with pytest.raises(FrozenInstanceError):
        plan.num_logical_experts = 7  # type: ignore[misc]
    tensor_copy = plan.physical_to_logical_map
    tensor_copy.fill_(9)
    assert plan.layer_map(0) == (0, 1, 2, 1)


def test_cache_invalidates_same_size_same_mtime_replacement(tmp_path) -> None:
    path = _write_plan(tmp_path / "map.json")
    stat = path.stat()
    first = _load(path)

    _write_plan(path, [[2, 0, 1, 2], [0, 1, 2, 1]])
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    second = _load(path)

    assert second.source_sha256 != first.source_sha256
    assert second.layer_map(0) == (2, 0, 1, 2)


def test_target_and_mtp_model_keys_use_runtime_class_name() -> None:
    target = type("HYV4ForCausalLM", (), {})()
    mtp = type("HYV4MTP", (), {})()
    parallel_config = SimpleNamespace(pipeline_parallel_size=1)

    assert (
        resolve_offline_eplb_model_key(target, parallel_config)
        == "HYV4ForCausalLM"
    )
    assert resolve_offline_eplb_model_key(mtp, parallel_config) == "HYV4MTP"


def test_loader_selects_mtp_entry_from_multi_model_file(tmp_path) -> None:
    path = _write_plan(tmp_path / "map.json")
    payload = json.loads(path.read_text())
    mtp_entry = dict(payload["model_maps"]["HYV4ForCausalLM"])
    mtp_entry["model_class"] = "HYV4MTP"
    mtp_entry["physical_to_logical_map"] = [
        [2, 0, 1, 2],
        [0, 1, 2, 1],
    ]
    payload["model_maps"]["HYV4MTP"] = mtp_entry
    path.write_text(json.dumps(payload))

    plan = _load(path, model_key="HYV4MTP")

    assert plan.model_key == "HYV4MTP"
    assert plan.layer_map(0) == (2, 0, 1, 2)


def test_initial_map_requires_explicit_calibration_opt_out(tmp_path) -> None:
    path = _write_plan(tmp_path / "map.json", record_kind="initial")

    with pytest.raises(ValueError, match="proposal"):
        _load(path)
    assert _load(path, require_proposal=False).record_kind == "initial"


@pytest.mark.parametrize(
    "rows",
    [
        [],
        [[0, 1, 2, 1]],
        [[0, 1, 2, 1], [0]],
        [[True, 1, 2, 1]] * 2,
        [[0.0, 1, 2, 1]] * 2,
        [[-1, 1, 2, 1]] * 2,
        [[0, 1, 3, 1]] * 2,
        [[0, 1, 1, 1]] * 2,
    ],
)
def test_invalid_rows_are_rejected(tmp_path, rows) -> None:
    with pytest.raises(ValueError):
        _load(_write_plan(tmp_path / "map.json", rows))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"model_key": "Missing"},
        {"model_key": ""},
        {"expected_shape": (1, 4)},
        {"expected_shape": (True, 4)},
        {"num_logical_experts": True},
        {"num_logical_experts": 0},
        {"num_redundant_experts": True},
        {"num_redundant_experts": -1},
        {"num_redundant_experts": 2},
    ],
)
def test_invalid_load_request_is_rejected(tmp_path, kwargs) -> None:
    with pytest.raises(ValueError):
        _load(_write_plan(tmp_path / "map.json"), **kwargs)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("record_kind", "profile"),
        ("record_kind", 1),
        ("model_name", ""),
        ("model_class", "HYV4MTP"),
        ("num_moe_layers", 1),
        ("num_moe_layers", True),
        ("num_logical_experts", 4),
        ("num_logical_experts", True),
        ("num_physical_experts", 3),
        ("num_physical_experts", True),
        ("num_redundant_experts", 0),
        ("num_redundant_experts", True),
    ],
)
def test_declared_metadata_must_match(tmp_path, field, value) -> None:
    with pytest.raises(ValueError):
        _load(_write_plan(tmp_path / "map.json", **{field: value}))


@pytest.mark.parametrize(
    "field",
    [
        "record_kind",
        "model_name",
        "model_class",
        "num_moe_layers",
        "num_logical_experts",
        "num_physical_experts",
        "num_redundant_experts",
        "physical_to_logical_map",
    ],
)
def test_required_model_metadata_cannot_be_omitted(tmp_path, field) -> None:
    path = _write_plan(tmp_path / "map.json")
    payload = json.loads(path.read_text())
    payload["model_maps"]["HYV4ForCausalLM"].pop(field)
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match=field):
        _load(path)


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"model_maps": {}},
        {"version": True, "model_maps": {}},
        {"version": 1, "model_maps": {}},
        {"version": 2, "model_maps": []},
        {
            "version": 2,
            "format": "unknown",
            "model_maps": {},
        },
    ],
)
def test_version_two_container_is_required(tmp_path, payload) -> None:
    path = tmp_path / "map.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError):
        _load(path)


def test_constructor_rejects_mutable_map_rows() -> None:
    with pytest.raises(ValueError):
        StaticEplbPlan(
            model_key="HYV4ForCausalLM",
            source_path="/tmp/map.json",
            source_sha256="a" * 64,
            record_kind="proposal",
            _map_values=[[0, 1]],  # type: ignore[arg-type]
            num_logical_experts=2,
            num_physical_experts=2,
            num_redundant_experts=0,
        )


def test_pipeline_parallelism_is_rejected() -> None:
    model = type("HYV4ForCausalLM", (), {})()
    with pytest.raises(ValueError, match="pipeline parallel"):
        resolve_offline_eplb_model_key(
            model,
            SimpleNamespace(pipeline_parallel_size=2),
        )


def test_matching_plan_fingerprints_are_gathered_on_ep_cpu_group(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    plan = _load(_write_plan(tmp_path / "map.json"))
    cpu_group = object()
    ep_group = SimpleNamespace(world_size=2, cpu_group=cpu_group)
    monkeypatch.setattr("torch.distributed.is_initialized", lambda: True)
    monkeypatch.setattr("vllm.distributed.get_ep_group", lambda: ep_group)

    def gather(output, value, *, group):
        assert group is cpu_group
        output[:] = [value, value]

    monkeypatch.setattr("torch.distributed.all_gather_object", gather)
    verify_static_plan_across_ep_ranks(plan)


def test_unequal_ep_plan_fingerprints_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    plan = _load(_write_plan(tmp_path / "map.json"))
    ep_group = SimpleNamespace(world_size=2, cpu_group=object())
    monkeypatch.setattr("torch.distributed.is_initialized", lambda: True)
    monkeypatch.setattr("vllm.distributed.get_ep_group", lambda: ep_group)

    def gather(output, value, *, group):
        del group
        output[:] = [value, (*value[:-1], value[-1] + 1)]

    monkeypatch.setattr("torch.distributed.all_gather_object", gather)
    with pytest.raises(RuntimeError, match="fingerprint mismatch"):
        verify_static_plan_across_ep_ranks(plan)


def test_direct_loader_duplicates_local_slots_and_skips_nonlocal_slots() -> None:
    owner = SimpleNamespace(_vllm_hcu_static_eplb_row=(2, 0, 1, 2))
    values = torch.zeros(2, 2)

    def original(
        self,
        *,
        param,
        loaded_weight,
        weight_name,
        shard_id,
        expert_id,
        return_success,
    ):
        assert self is owner
        assert weight_name == "w2_weight" and shard_id == "w2"
        if expert_id < 2:
            return False
        param[expert_id - 2].copy_(loaded_weight)
        return True

    result = load_static_logical_expert(
        owner,
        original,
        param=values,
        loaded_weight=torch.tensor([7.0, 9.0]),
        weight_name="w2_weight",
        shard_id="w2",
        logical_expert_id=2,
        return_success=True,
    )

    assert result is True
    assert values.tolist() == [[0.0, 0.0], [7.0, 9.0]]


@pytest.mark.parametrize("rank", [0, 1])
def test_binding_loads_each_logical_checkpoint_into_static_local_replicas(
    tmp_path,
    rank: int,
) -> None:
    model = GenericMoE(rank)
    config, _ = config_and_map(tmp_path)

    plan = bind_static_eplb_plan(config, model)
    for layer in model.moe_layers:
        owner = layer.routed_experts
        for logical_expert_id in range(3):
            owner.w2_weight.weight_loader(
                owner.w2_weight,
                torch.full((2, 2), logical_expert_id + 1.0),
                "w2_weight",
                "w2",
                logical_expert_id,
                return_success=True,
            )

    expected = (
        [[1.0, 2.0], [3.0, 1.0]]
        if rank == 0
        else [[3.0, 2.0], [2.0, 3.0]]
    )
    assert plan is model._vllm_hcu_static_eplb_plan
    for index, layer in enumerate(model.moe_layers):
        assert layer.routed_experts.w2_weight[:, 0, 0].tolist() == expected[index]
        assert (
            layer.routed_experts._vllm_hcu_static_eplb_row
            == plan.layer_map(index)
        )


def test_binding_preserves_shared_router_and_global_state(tmp_path) -> None:
    model = GenericMoE()
    config, _ = config_and_map(tmp_path)
    shared = torch.nn.Parameter(torch.tensor([5.0]), requires_grad=False)
    shared_loader = object()
    shared.weight_loader = shared_loader
    model.register_parameter("shared_expert_weight", shared)

    owners = []
    for layer in model.moe_layers:
        owner = layer.routed_experts
        owner.e_score_correction_bias = torch.nn.Parameter(
            torch.full((3,), 6.0), requires_grad=False
        )
        owner.w2_input_scale = torch.nn.Parameter(
            torch.full((3,), 7.0), requires_grad=False
        )
        owner.w2_input_scale.weight_loader = owner.weight_loader
        owners.append(
            (
                owner,
                owner.e_score_correction_bias,
                owner.w2_input_scale.weight_loader,
            )
        )

    bind_static_eplb_plan(config, model)

    assert model.shared_expert_weight is shared
    assert model.shared_expert_weight.weight_loader is shared_loader
    for owner, router_bias, scale_loader in owners:
        assert owner.e_score_correction_bias is router_bias
        assert torch.equal(router_bias, torch.full((3,), 6.0))
        assert owner.w2_input_scale.weight_loader is scale_loader
        assert torch.equal(owner.w2_input_scale, torch.full((3,), 7.0))


def test_fused_shared_expert_keeps_current_physical_tail_loader(tmp_path) -> None:
    model = GenericMoE()
    config, _ = config_and_map(tmp_path)
    for layer in model.moe_layers:
        owner = layer.routed_experts
        owner.expert_map_manager.num_fused_shared_experts = 1
        owner.expert_map_manager._expert_map = torch.tensor([0, 1, -1, -1, 2])
        for name, parameter in list(owner.named_parameters()):
            replacement = torch.nn.Parameter(
                torch.zeros((3, *parameter.shape[1:])),
                requires_grad=False,
            )
            replacement.weight_loader = owner.weight_loader
            if "scale" in name:
                replacement.quant_method = "channel"
            setattr(owner, name, replacement)

    bind_static_eplb_plan(config, model)

    for layer in model.moe_layers:
        owner = layer.routed_experts
        owner.w2_weight.weight_loader(
            owner.w2_weight,
            torch.full((2, 2), 9.0),
            "w2_weight",
            "w2",
            3,
            return_success=True,
        )
        assert owner.w2_weight[:, 0, 0].tolist() == [0.0, 0.0, 9.0]


def test_direct_loader_rejects_unsplit_fused_expert_tensor(tmp_path) -> None:
    model = GenericMoE()
    config, _ = config_and_map(tmp_path)
    bind_static_eplb_plan(config, model)
    owner = model.moe_layers[0].routed_experts

    with pytest.raises(ValueError, match="fused|split"):
        owner.w2_weight.weight_loader(
            owner.w2_weight,
            torch.ones(3, 2, 2),
            "w2_weight",
            "w2",
            0,
            return_success=True,
        )
    assert torch.count_nonzero(owner.w2_weight) == 0


def test_ep_weight_filter_rejected_before_any_binding_mutation(tmp_path) -> None:
    model = GenericMoE()
    config, _ = config_and_map(tmp_path)
    config.parallel_config.enable_ep_weight_filter = True
    original_loaders = [
        layer.routed_experts.w2_weight.weight_loader
        for layer in model.moe_layers
    ]

    with pytest.raises(ValueError, match="weight filter"):
        bind_static_eplb_plan(config, model)

    assert not hasattr(model, "_vllm_hcu_static_eplb_plan")
    for layer, original_loader in zip(model.moe_layers, original_loaders):
        assert not hasattr(
            layer.routed_experts, "_vllm_hcu_static_eplb_row"
        )
        assert layer.routed_experts.w2_weight.weight_loader is original_loader


def test_binding_is_atomic_when_a_later_layer_is_not_current_routed_experts(
    tmp_path,
) -> None:
    model = GenericMoE()
    config, _ = config_and_map(tmp_path)
    del model.moe_layers[1].routed_experts

    with pytest.raises(ValueError, match="RoutedExperts"):
        bind_static_eplb_plan(config, model)

    assert not hasattr(model, "_vllm_hcu_static_eplb_plan")
    assert not hasattr(
        model.moe_layers[0].routed_experts,
        "_vllm_hcu_static_eplb_row",
    )


@pytest.mark.parametrize("count_name", ["num_logical_experts", "num_experts"])
def test_binding_rejects_routed_expert_count_mismatch(
    tmp_path,
    count_name: str,
) -> None:
    model = GenericMoE()
    config, _ = config_and_map(tmp_path)
    setattr(model.moe_layers[1].routed_experts.moe_config, count_name, 99)

    with pytest.raises(ValueError, match="count"):
        bind_static_eplb_plan(config, model)
    assert not hasattr(model, "_vllm_hcu_static_eplb_plan")


def test_binding_rejects_unsupported_quantization_owner(tmp_path) -> None:
    model = GenericMoE()
    config, _ = config_and_map(tmp_path)

    class UnsupportedMethod(torch.nn.Module):
        supports_eplb = False

    model.moe_layers[1].routed_experts.quant_method = UnsupportedMethod()

    with pytest.raises(ValueError, match="EPLB unsupported"):
        bind_static_eplb_plan(config, model)
    assert not hasattr(model, "_vllm_hcu_static_eplb_plan")


def test_binding_rejects_foreign_parameter_loader_before_publication(
    tmp_path,
) -> None:
    model = GenericMoE()
    config, _ = config_and_map(tmp_path)
    owner = model.moe_layers[1].routed_experts
    owner.w2_weight.weight_loader = (
        model.moe_layers[0].routed_experts.weight_loader
    )

    with pytest.raises(ValueError, match="w2_weight.*unsupported loader owner"):
        bind_static_eplb_plan(config, model)
    assert not hasattr(model, "_vllm_hcu_static_eplb_plan")


def test_binding_rejects_missing_local_moe_layers(tmp_path) -> None:
    model = GenericMoE()
    config, _ = config_and_map(tmp_path)
    model.moe_layers = torch.nn.ModuleList()

    with pytest.raises(ValueError, match="no local MoE layers"):
        bind_static_eplb_plan(config, model)


def test_identical_rebind_is_idempotent_but_changed_file_fails(tmp_path) -> None:
    model = GenericMoE()
    config, path = config_and_map(tmp_path)
    first = bind_static_eplb_plan(config, model)
    loaders = [
        layer.routed_experts.w2_weight.weight_loader
        for layer in model.moe_layers
    ]

    assert bind_static_eplb_plan(config, model) is first
    assert [
        layer.routed_experts.w2_weight.weight_loader
        for layer in model.moe_layers
    ] == loaders

    payload = json.loads(path.read_text())
    payload["model_maps"]["GenericMoE"]["physical_to_logical_map"] = [
        [2, 0, 1, 2],
        [0, 1, 2, 1],
    ]
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="changed|rebind"):
        bind_static_eplb_plan(config, model)
    assert model._vllm_hcu_static_eplb_plan is first


def test_binding_filters_model_and_inner_checkpoint_mappings(tmp_path) -> None:
    from types import MethodType

    model = GenericMoE()
    inner = torch.nn.Module()
    inner.moe_layers = model.moe_layers
    model.model = inner

    def mapping(self):
        del self
        return [("param", "checkpoint", expert_id, "w2") for expert_id in range(4)]

    model.get_expert_mapping = MethodType(mapping, model)
    inner.get_expert_mapping = MethodType(mapping, inner)
    config, _ = config_and_map(tmp_path)

    plan = bind_static_eplb_plan(config, model)

    assert model._vllm_hcu_static_eplb_plan is plan
    assert inner._vllm_hcu_static_eplb_plan is plan
    assert {entry[2] for entry in model.get_expert_mapping()} == {0, 1, 2}
    assert {entry[2] for entry in inner.get_expert_mapping()} == {0, 1, 2}
