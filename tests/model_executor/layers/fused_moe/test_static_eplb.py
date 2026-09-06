# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
import torch

from vllm_hcu.model_executor.layers.fused_moe.static_eplb import (
    bind_static_eplb_plan,
    load_static_eplb_plan,
    maybe_load_static_eplb_plan,
)


def _write_map(path: Path, rows: list[list[int]], key: str = "HYV4ForCausalLM") -> bytes:
    raw = json.dumps(
        {"version": 2, "model_maps": {key: {"physical_to_logical_map": rows}}}
    ).encode()
    path.write_bytes(raw)
    return raw


def test_load_plan_selects_model_and_builds_stable_fingerprint(tmp_path: Path) -> None:
    path = tmp_path / "maps.json"
    raw = json.dumps(
        {
            "version": 2,
            "model_maps": {
                "HYV4ForCausalLM": {
                    "physical_to_logical_map": [[0, 1, 2, 3, 0, 1]]
                },
                "HYV4MTP": {
                    "physical_to_logical_map": [[3, 2, 1, 0, 3, 2]]
                },
            },
        }
    ).encode()
    path.write_bytes(raw)

    plan = load_static_eplb_plan(
        path,
        model_key="HYV4MTP",
        expected_shape=(1, 6),
        num_logical_experts=4,
        num_redundant_experts=2,
    )

    assert plan.model_key == "HYV4MTP"
    assert plan.source_path == str(path.resolve())
    assert plan.source_sha256 == hashlib.sha256(raw).hexdigest()
    assert plan.physical_to_logical_map.dtype == torch.int64
    assert plan.physical_to_logical_map.device.type == "cpu"
    assert plan.layer_map(0) == (3, 2, 1, 0, 3, 2)
    assert plan.num_logical_experts == 4
    assert plan.num_physical_experts == 6
    assert plan.num_redundant_experts == 2
    assert plan.fingerprint() == (
        "HYV4MTP",
        hashlib.sha256(raw).hexdigest(),
        (1, 6),
        4,
        6,
        2,
    )

    leaked = plan.physical_to_logical_map
    leaked[0, 0] = 0
    assert plan.layer_map(0) == (3, 2, 1, 0, 3, 2)


def test_plan_cache_invalidates_when_file_identity_changes(tmp_path: Path) -> None:
    path = tmp_path / "maps.json"
    first_raw = _write_map(path, [[0, 1, 2, 3, 0, 1]])
    first = load_static_eplb_plan(
        path,
        model_key="HYV4ForCausalLM",
        expected_shape=(1, 6),
        num_logical_experts=4,
        num_redundant_experts=2,
    )
    second_raw = _write_map(path, [[3, 2, 1, 0, 3, 2]])
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    second = load_static_eplb_plan(
        path,
        model_key="HYV4ForCausalLM",
        expected_shape=(1, 6),
        num_logical_experts=4,
        num_redundant_experts=2,
    )

    assert first.source_sha256 == hashlib.sha256(first_raw).hexdigest()
    assert second.source_sha256 == hashlib.sha256(second_raw).hexdigest()
    assert first.source_sha256 != second.source_sha256
    assert second.layer_map(0) == (3, 2, 1, 0, 3, 2)


@pytest.mark.parametrize(
    ("payload", "expected_shape", "redundant", "message"),
    [
        (
            {"model_maps": {"HYV4MTP": {"physical_to_logical_map": [[0, 1, 2, 3, 0, 1]]}}},
            (1, 6),
            2,
            "does not contain key",
        ),
        ({"physical_to_logical_map": [[0, 1, 2, 3, 0, True]]}, (1, 6), 2, "integer expert ids"),
        ({"physical_to_logical_map": [[0, 1, 2, 3, 0, 1.0]]}, (1, 6), 2, "integer expert ids"),
        ({"physical_to_logical_map": [[0, 1, 2, 3, 0]]}, (1, 6), 2, "has shape"),
        ({"physical_to_logical_map": [[0, 1, 2, 4, 0, 1]]}, (1, 6), 2, ">= 4"),
        ({"physical_to_logical_map": [[0, 1, 2, 0, 1, 2]]}, (1, 6), 2, "misses logical experts"),
        ({"physical_to_logical_map": [[0, 1, 2, 3, 0, 1]]}, (1, 6), 1, "redundant"),
    ],
)
def test_load_plan_rejects_invalid_contracts(
    tmp_path: Path,
    payload: dict,
    expected_shape: tuple[int, int],
    redundant: int,
    message: str,
) -> None:
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_static_eplb_plan(
            path,
            model_key="HYV4ForCausalLM",
            expected_shape=expected_shape,
            num_logical_experts=4,
            num_redundant_experts=redundant,
        )


def test_legacy_mtp_plan_selects_final_rows(tmp_path: Path) -> None:
    path = tmp_path / "legacy.json"
    path.write_text(
        json.dumps(
            {
                "physical_to_logical_map": [
                    [0, 1, 2, 3, 0, 1],
                    [1, 0, 2, 3, 1, 0],
                    [3, 2, 1, 0, 3, 2],
                ]
            }
        ),
        encoding="utf-8",
    )

    plan = load_static_eplb_plan(
        path,
        model_key="HYV4MTP",
        expected_shape=(1, 6),
        num_logical_experts=4,
        num_redundant_experts=2,
    )

    assert plan.layer_map(0) == (3, 2, 1, 0, 3, 2)


def _config_for_path(path: Path | None, **overrides):
    values = {
        "_vllm_hcu_expert_map_path": str(path) if path else None,
        "enable_expert_parallel": True,
        "enable_eplb": True,
        "enable_ep_weight_filter": False,
    }
    values.update(overrides)
    return type(
        "Config",
        (),
        {"parallel_config": type("Parallel", (), values)()},
    )()


def test_maybe_load_plan_uses_parallel_config_before_weight_load(tmp_path: Path) -> None:
    path = tmp_path / "maps.json"
    _write_map(path, [[3, 2, 1, 0, 3, 2]])

    plan = maybe_load_static_eplb_plan(
        _config_for_path(path),
        model_key="HYV4ForCausalLM",
        num_moe_layers=1,
        num_logical_experts=4,
        num_physical_experts=6,
        num_redundant_experts=2,
    )

    assert plan is not None
    assert plan.layer_map(0) == (3, 2, 1, 0, 3, 2)
    assert (
        maybe_load_static_eplb_plan(
            _config_for_path(None),
            model_key="HYV4ForCausalLM",
            num_moe_layers=1,
            num_logical_experts=4,
            num_physical_experts=6,
            num_redundant_experts=2,
        )
        is None
    )


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"enable_expert_parallel": False}, "expert parallel"),
        ({"enable_eplb": False}, "EPLB"),
        ({"enable_ep_weight_filter": True}, "EP weight filtering"),
    ],
)
def test_maybe_load_plan_rejects_unsupported_parallel_modes(
    tmp_path: Path,
    override: dict,
    message: str,
) -> None:
    path = tmp_path / "maps.json"
    _write_map(path, [[0, 1, 2, 3, 0, 1]])

    with pytest.raises(ValueError, match=message):
        maybe_load_static_eplb_plan(
            _config_for_path(path, **override),
            model_key="HYV4ForCausalLM",
            num_moe_layers=1,
            num_logical_experts=4,
            num_physical_experts=6,
            num_redundant_experts=2,
        )


class _FakeMoERunner:
    def __init__(self) -> None:
        self.routed_experts = type("RoutedExperts", (), {})()


class _FakeMoEModel:
    def __init__(self, *, runners: list[object] | None = None) -> None:
        self.expert_weights = []
        self.num_moe_layers = 2
        self.num_expert_groups = 1
        self.num_logical_experts = 3
        self.num_physical_experts = 4
        self.num_local_physical_experts = 4
        self.num_routed_experts = 3
        self.num_shared_experts = 0
        self.num_redundant_experts = 1
        self.moe_layers = (
            [_FakeMoERunner(), _FakeMoERunner()] if runners is None else runners
        )

    def set_eplb_state(self, *args: object) -> None:
        del args

    def update_physical_experts_metadata(self, *args: object) -> None:
        del args


def _write_binder_map(path: Path) -> None:
    _write_map(
        path,
        [[2, 1, 0, 2], [1, 0, 2, 1]],
        key="_FakeMoEModel",
    )


def test_bind_plan_publishes_each_row_to_routed_experts(tmp_path: Path) -> None:
    path = tmp_path / "maps.json"
    _write_binder_map(path)
    config = _config_for_path(path)
    model = _FakeMoEModel()

    plan = bind_static_eplb_plan(config, model)

    assert model._vllm_hcu_static_eplb_plan is plan
    assert model.moe_layers[0].routed_experts._vllm_hcu_static_eplb_row == (
        2,
        1,
        0,
        2,
    )
    assert model.moe_layers[1].routed_experts._vllm_hcu_static_eplb_row == (
        1,
        0,
        2,
        1,
    )


def test_bind_plan_returns_none_without_a_configured_path() -> None:
    assert bind_static_eplb_plan(_config_for_path(None), object()) is None


def test_bind_plan_rejects_a_non_moe_model_without_hyv4_prefix(tmp_path: Path) -> None:
    path = tmp_path / "maps.json"
    _write_binder_map(path)

    with pytest.raises(ValueError, match="MixtureOfExperts") as error:
        bind_static_eplb_plan(_config_for_path(path), object())

    assert "HY V4" not in str(error.value)


def test_bind_plan_rejects_layer_count_mismatch_before_publishing(
    tmp_path: Path,
) -> None:
    path = tmp_path / "maps.json"
    _write_binder_map(path)
    model = _FakeMoEModel(runners=[_FakeMoERunner()])

    with pytest.raises(ValueError, match="moe_layers"):
        bind_static_eplb_plan(_config_for_path(path), model)

    assert not hasattr(model, "_vllm_hcu_static_eplb_plan")
    assert not hasattr(
        model.moe_layers[0].routed_experts,
        "_vllm_hcu_static_eplb_row",
    )


def test_bind_plan_rejects_missing_routed_experts_before_publishing(
    tmp_path: Path,
) -> None:
    path = tmp_path / "maps.json"
    _write_binder_map(path)
    first = _FakeMoERunner()
    model = _FakeMoEModel(runners=[first, object()])

    with pytest.raises(ValueError, match="routed_experts"):
        bind_static_eplb_plan(_config_for_path(path), model)

    assert not hasattr(model, "_vllm_hcu_static_eplb_plan")
    assert not hasattr(first.routed_experts, "_vllm_hcu_static_eplb_row")


def test_maybe_load_plan_uses_generic_validation_errors(tmp_path: Path) -> None:
    path = tmp_path / "maps.json"
    _write_binder_map(path)

    with pytest.raises(ValueError, match="expert parallel") as error:
        maybe_load_static_eplb_plan(
            _config_for_path(path, enable_expert_parallel=False),
            model_key="_FakeMoEModel",
            num_moe_layers=2,
            num_logical_experts=3,
            num_physical_experts=4,
            num_redundant_experts=1,
        )

    assert "HY V4" not in str(error.value)
