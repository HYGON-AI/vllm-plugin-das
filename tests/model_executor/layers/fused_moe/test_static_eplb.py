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

from vllm_hcu.model_executor.layers.fused_moe.static_eplb import (
    StaticEplbPlan,
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
