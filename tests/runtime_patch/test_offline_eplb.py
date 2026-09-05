# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

import json
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from vllm_hcu.patch.worker.framework_opt.patch_offline_eplb import (
    PATCH_ID,
    TARGET_MODULE,
    apply_to_module,
    load_offline_expert_map,
    record_offline_expert_map,
)
from vllm_hcu.patch.worker import worker_callback_names


def test_record_merges_main_and_mtp_maps_atomically(tmp_path: Path) -> None:
    output = tmp_path / "hy4-eplb.json"
    main_map = torch.tensor([[0, 1, 2, 3, 0, 1], [0, 2, 1, 3, 2, 3]])
    mtp_map = torch.tensor([[3, 2, 1, 0, 3, 2]])

    record_offline_expert_map(
        output,
        model_key="HYV4ForCausalLM",
        model_name="/models/Hy4-preview-Channel-FP8-w8a8-v2",
        model_class="HYV4ForCausalLM",
        physical_to_logical_map=main_map,
        num_logical_experts=4,
        num_redundant_experts=2,
    )
    record_offline_expert_map(
        output,
        model_key="HYV4MTP",
        model_name="/models/Hy4-preview-Channel-FP8-w8a8-v2",
        model_class="HYV4MTP",
        physical_to_logical_map=mtp_map,
        num_logical_experts=4,
        num_redundant_experts=2,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["version"] == 2
    assert payload["format"] == "vllm_offline_eplb_physical_to_logical_by_model"
    assert sorted(payload["model_maps"]) == ["HYV4ForCausalLM", "HYV4MTP"]
    assert payload["model_maps"]["HYV4ForCausalLM"][
        "physical_to_logical_map"
    ] == main_map.tolist()
    assert payload["model_maps"]["HYV4MTP"]["physical_to_logical_map"] == mtp_map.tolist()
    assert not output.with_suffix(".json.tmp").exists()


def test_load_selects_requested_model_and_preserves_dtype(tmp_path: Path) -> None:
    path = tmp_path / "hy4-eplb.json"
    path.write_text(
        json.dumps(
            {
                "version": 2,
                "format": "vllm_offline_eplb_physical_to_logical_by_model",
                "model_maps": {
                    "HYV4ForCausalLM": {
                        "physical_to_logical_map": [[0, 1, 2, 3, 0, 1]],
                    },
                    "HYV4MTP": {
                        "physical_to_logical_map": [[3, 2, 1, 0, 3, 2]],
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    loaded = load_offline_expert_map(
        path,
        model_key="HYV4MTP",
        expected_shape=(1, 6),
        num_logical_experts=4,
        dtype=torch.int64,
        device=torch.device("cpu"),
    )

    assert loaded.dtype == torch.int64
    assert loaded.tolist() == [[3, 2, 1, 0, 3, 2]]


def test_load_legacy_map_uses_last_layers_for_mtp(tmp_path: Path) -> None:
    path = tmp_path / "legacy.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "format": "vllm_offline_eplb_physical_to_logical",
                "physical_to_logical_map": [
                    [0, 1, 2, 3, 0, 1],
                    [1, 0, 2, 3, 1, 0],
                    [3, 2, 1, 0, 3, 2],
                ],
            }
        ),
        encoding="utf-8",
    )

    loaded = load_offline_expert_map(
        path,
        model_key="HYV4MTP",
        expected_shape=(1, 6),
        num_logical_experts=4,
        dtype=torch.int64,
        device=torch.device("cpu"),
    )

    assert loaded.tolist() == [[3, 2, 1, 0, 3, 2]]


@pytest.mark.parametrize(
    ("raw_map", "message"),
    [
        ([[0, 1, 2, 0, 1, 2]], "misses logical experts"),
        ([[0, 1, 2, 3, 0]], "has shape"),
        ([[0, 1, 2, 4, 0, 1]], "logical expert id >= 4"),
        ([[0, 1, 2, -1, 0, 1]], "negative expert ids"),
        ([[0, 1, 2, 3, 0, 1.5]], "integer expert ids"),
    ],
)
def test_load_rejects_invalid_maps(
    tmp_path: Path,
    raw_map: list[list[int | float]],
    message: str,
) -> None:
    path = tmp_path / "invalid.json"
    path.write_text(
        json.dumps({"physical_to_logical_map": raw_map}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match=message):
        load_offline_expert_map(
            path,
            model_key="HYV4ForCausalLM",
            expected_shape=(1, 6),
            num_logical_experts=4,
            dtype=torch.int64,
            device=torch.device("cpu"),
        )


class _FakeDeviceGroup:
    def rank(self) -> int:
        return 0


class _FakeModelConfig:
    model = "/models/Hy4-preview-Channel-FP8-w8a8-v2"

    def compute_hash(self) -> str:
        return "hy4-hash"


class HYV4ForCausalLM:
    num_logical_experts = 4
    num_redundant_experts = 2
    num_physical_experts = 6
    num_moe_layers = 1
    expert_weights = [[torch.zeros(1)]]


def test_offline_eplb_patch_is_registered_in_worker_inventory() -> None:
    assert (PATCH_ID, TARGET_MODULE) in worker_callback_names()


def _make_eplb_module(
    *,
    record_path: Path | None = None,
    load_path: Path | None = None,
) -> tuple[ModuleType, list[tuple[torch.Tensor, torch.Tensor]]]:
    module = ModuleType(
        "vllm.distributed.eplb.eplb_state"
    )
    rearrangements: list[tuple[torch.Tensor, torch.Tensor]] = []

    class EplbModelState:
        pass

    class EplbState:
        def __init__(self) -> None:
            self.parallel_config = SimpleNamespace(
                _vllm_hcu_expert_map_record_path=(
                    str(record_path) if record_path is not None else None
                ),
                _vllm_hcu_expert_map_path=(
                    str(load_path) if load_path is not None else None
                ),
            )
            self.device = torch.device("cpu")
            self.model_states: dict[str, EplbModelState] = {}
            self.should_record_tensor = torch.tensor(True)
            self.is_async = True
            self.official_steps = 0
            self.official_profile_steps = 0

        def add_model(self, model, model_config) -> None:
            state = EplbModelState()
            state.physical_to_logical_map = torch.tensor(
                [[0, 1, 2, 3, 0, 1]], dtype=torch.int64
            )
            state.logical_to_physical_map = torch.empty(0)
            state.logical_replica_count = torch.empty(0)
            state.model_name = model_config.model
            state.model = model
            state.expert_buffer = [torch.zeros(1)]
            state.communicator = object()
            state.pending_result = None
            self.model_states[model_config.compute_hash()] = state

        def step(self, is_dummy=False, is_profile=False, log_stats=False):
            del is_dummy, log_stats
            self.official_steps += 1
            if is_profile:
                self.official_profile_steps += 1
            return "official-step"

    def rearrange_expert_weights_inplace(
        source_map,
        target_map,
        expert_weights,
        expert_buffer,
        ep_group,
        communicator,
        is_profile=False,
        rank_mapping=None,
    ) -> None:
        del expert_weights, expert_buffer, ep_group, communicator, is_profile, rank_mapping
        rearrangements.append((source_map.clone(), target_map.clone()))

    def commit(model_state, new_physical_to_logical_map) -> None:
        model_state.physical_to_logical_map.copy_(new_physical_to_logical_map)

    def commit_layer(model_state, new_physical_to_logical_map, layer) -> None:
        model_state.physical_to_logical_map[layer].copy_(new_physical_to_logical_map)

    def move_to_workspace(model_state, ep_rank) -> None:
        del model_state, ep_rank

    module.EplbModelState = EplbModelState
    module.EplbState = EplbState
    module.rearrange_expert_weights_inplace = rearrange_expert_weights_inplace
    module._commit_eplb_maps = commit
    module._commit_eplb_maps_for_layer = commit_layer
    module._move_to_workspace = move_to_workspace
    module.get_ep_group = lambda: SimpleNamespace(device_group=_FakeDeviceGroup())
    module.logger = SimpleNamespace(info=lambda *args, **kwargs: None)
    return module, rearrangements


def test_runtime_patch_records_initial_and_committed_maps(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    module, _ = _make_eplb_module(record_path=output)
    assert apply_to_module(module)
    state = module.EplbState()
    model = HYV4ForCausalLM()
    state.add_model(model, _FakeModelConfig())

    initial = json.loads(output.read_text(encoding="utf-8"))
    assert initial["model_maps"]["HYV4ForCausalLM"][
        "physical_to_logical_map"
    ] == [[0, 1, 2, 3, 0, 1]]

    committed = torch.tensor([[3, 2, 1, 0, 3, 2]], dtype=torch.int64)
    model_state = state.model_states["hy4-hash"]
    module._commit_eplb_maps(model_state, committed)
    recorded = json.loads(output.read_text(encoding="utf-8"))
    assert recorded["model_maps"]["HYV4ForCausalLM"][
        "physical_to_logical_map"
    ] == committed.tolist()


def test_runtime_patch_loads_static_map_and_freezes_dynamic_eplb(
    tmp_path: Path,
) -> None:
    source = tmp_path / "load.json"
    source.write_text(
        json.dumps(
            {
                "model_maps": {
                    "HYV4ForCausalLM": {
                        "physical_to_logical_map": [[3, 2, 1, 0, 3, 2]],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    module, rearrangements = _make_eplb_module(load_path=source)
    assert apply_to_module(module)
    state = module.EplbState()
    state.add_model(HYV4ForCausalLM(), _FakeModelConfig())

    assert len(rearrangements) == 1
    assert rearrangements[0][0].tolist() == [[0, 1, 2, 3, 0, 1]]
    assert rearrangements[0][1].tolist() == [[3, 2, 1, 0, 3, 2]]
    assert state.model_states["hy4-hash"].physical_to_logical_map.tolist() == [
        [3, 2, 1, 0, 3, 2]
    ]
    assert state.should_record_tensor.item() is False
    assert state.is_async is False

    assert state.step(is_profile=True) is None
    assert state.official_profile_steps == 0
    assert state.step(is_profile=False) is None
    assert state.official_steps == 0


def test_runtime_patch_record_mode_preserves_dynamic_eplb_steps(
    tmp_path: Path,
) -> None:
    output = tmp_path / "record.json"
    module, _ = _make_eplb_module(record_path=output)
    assert apply_to_module(module)
    state = module.EplbState()
    state.add_model(HYV4ForCausalLM(), _FakeModelConfig())

    assert state.step(is_profile=False) == "official-step"
    assert state.official_steps == 1
    assert state.should_record_tensor.item() is True
    assert state.is_async is True
