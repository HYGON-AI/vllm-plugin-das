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
    load_static_eplb_plan,
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
