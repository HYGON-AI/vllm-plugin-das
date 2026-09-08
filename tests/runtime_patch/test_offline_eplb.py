# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

import json
import multiprocessing
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from vllm_hcu.model_executor.layers.fused_moe.static_eplb import (
    bind_static_eplb_plan,
    load_static_eplb_plan,
)
from vllm_hcu.patch.worker import worker_callback_names
from vllm_hcu.patch.worker.framework_opt._common import PatchCompatibilityError
from vllm_hcu.patch.worker.framework_opt.patch_offline_eplb import (
    PATCH_ID,
    TARGET_MODULE,
    apply_to_module,
    load_offline_expert_map,
    record_offline_expert_map,
)


def _record_map_in_process(output: str, model_index: int, start_event) -> None:
    start_event.wait()
    record_offline_expert_map(
        output,
        model_key=f"model-{model_index}",
        model_name=f"/models/model-{model_index}",
        model_class=f"Model{model_index}",
        physical_to_logical_map=torch.tensor([[0, 1, 2, 3]]),
        num_logical_experts=4,
        num_redundant_experts=0,
    )


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


def test_record_serializes_writers_across_processes(tmp_path: Path) -> None:
    output = tmp_path / "shared-eplb.json"
    context = multiprocessing.get_context("fork")
    start_event = context.Event()
    processes = [
        context.Process(
            target=_record_map_in_process,
            args=(str(output), model_index, start_event),
        )
        for model_index in range(8)
    ]
    for process in processes:
        process.start()
    start_event.set()
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert sorted(payload["model_maps"]) == [f"model-{index}" for index in range(8)]


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
    world_size = 1
    rank_in_group = 0

    def __init__(self) -> None:
        self.barrier_calls = 0

    def rank(self) -> int:
        return 0

    def barrier(self) -> None:
        self.barrier_calls += 1


class _FakeModelConfig:
    model = "/models/Hy4-preview-Channel-FP8-w8a8-v2"

    def compute_hash(self) -> str:
        return "hy4-hash"


class HYV4ForCausalLM:
    num_logical_experts = 4
    num_redundant_experts = 2
    num_physical_experts = 6
    num_moe_layers = 1
    moe_layers = [SimpleNamespace()]
    expert_weights = [[torch.zeros(1)]]


class GenericMoEForCausalLM(HYV4ForCausalLM):
    pass


class _StageMoERunner:
    def __init__(self) -> None:
        self.routed_experts = SimpleNamespace()

    def get_expert_weights(self) -> list[torch.Tensor]:
        return [torch.zeros(1)]

    def set_eplb_state(self, **kwargs: object) -> None:
        del kwargs


class PipelineMoEForCausalLM(HYV4ForCausalLM):
    def __init__(self, local_moe_layers: int) -> None:
        self.num_moe_layers = 3
        self.num_expert_groups = 1
        self.num_local_physical_experts = 6
        self.num_routed_experts = 4
        self.num_shared_experts = 0
        self.moe_layers = [_StageMoERunner() for _ in range(local_moe_layers)]
        self.expert_weights = [[torch.zeros(1)] for _ in self.moe_layers]

    def set_eplb_state(self, *args: object) -> None:
        del args

    def update_physical_experts_metadata(self, *args: object) -> None:
        del args


class Llama4ForConditionalGeneration(PipelineMoEForCausalLM):
    def __init__(self) -> None:
        super().__init__(local_moe_layers=1)
        self.num_moe_layers = 1
        self.language_model = PipelineMoEForCausalLM(local_moe_layers=1)
        self.language_model.num_moe_layers = 1
        self.language_model.moe_layers = self.moe_layers


def test_offline_eplb_patch_is_registered_in_worker_inventory() -> None:
    assert (PATCH_ID, TARGET_MODULE) in worker_callback_names()


def _make_eplb_module(
    *,
    record_path: Path | None = None,
    load_path: Path | None = None,
    pipeline_parallel_size: int = 1,
    pipeline_parallel_rank: int = 0,
) -> tuple[ModuleType, list[tuple[torch.Tensor, torch.Tensor]], _FakeDeviceGroup]:
    module = ModuleType(
        "vllm.distributed.eplb.eplb_state"
    )
    rearrangements: list[tuple[torch.Tensor, torch.Tensor]] = []
    ep_group = _FakeDeviceGroup()

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
                pipeline_parallel_size=pipeline_parallel_size,
                pipeline_parallel_rank=pipeline_parallel_rank,
            )
            self.device = torch.device("cpu")
            self.model_states: dict[str, EplbModelState] = {}
            self.should_record_tensor = torch.tensor(True)
            self.is_async = True
            self.official_steps = 0
            self.official_profile_steps = 0

        def add_model(self, model, model_config) -> None:
            state = EplbModelState()
            state.physical_to_logical_map = (
                torch.tensor([0, 1, 2, 3, 0, 1], dtype=torch.int64)
                .unsqueeze(0)
                .expand(model.num_moe_layers, -1)
                .clone()
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
            self.official_steps += 1
            if is_profile:
                self.official_profile_steps += 1
            if getattr(self, "simulate_official_window_step", False):
                if not is_dummy and self._should_record_current_step(
                    log_stats=log_stats
                ):
                    for model_state in self.model_states.values():
                        model_state.expert_load_window[
                            self.expert_load_window_step
                        ].copy_(model_state.expert_load_pass)
                        model_state.expert_load_pass.zero_()
                    self.expert_load_window_step = (
                        self.expert_load_window_step + 1
                    ) % self.expert_load_window_size
                self._update_layer_should_record(log_stats=log_stats)
            if getattr(self, "simulate_official_rearrange_step", False):
                self.expert_rearrangement_step += 1
                if (
                    self.expert_rearrangement_step
                    >= self.expert_rearrangement_step_interval
                ):
                    self.expert_rearrangement_step = 0
                    self.rearrange()
            return "official-step"

        def _sync_load_pass(self):
            return [
                model_state.expert_load_pass.clone()
                for model_state in self.model_states.values()
            ]

        def _should_record_current_step(self, log_stats=False):
            return log_stats

        def _update_layer_should_record(self, log_stats=False):
            self.should_record_tensor.fill_(
                self._should_record_current_step(log_stats=log_stats)
            )

        def rearrange(self, is_profile=False, rank_mapping=None):
            model_state = self.model_states["hy4-hash"]
            candidate = (
                torch.tensor([3, 2, 1, 0, 3, 2], dtype=torch.int64)
                .unsqueeze(0)
                .expand_as(model_state.physical_to_logical_map)
                .clone()
            )
            module.rearrange_expert_weights_inplace(
                model_state.physical_to_logical_map,
                candidate,
                model_state.model.expert_weights,
                model_state.expert_buffer,
                ep_group,
                model_state.communicator,
                is_profile,
                rank_mapping,
            )
            if not is_profile:
                module._commit_eplb_maps(model_state, candidate)
            return "official-rearrange"

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
    module.get_ep_group = lambda: SimpleNamespace(
        device_group=ep_group,
        world_size=ep_group.world_size,
        rank_in_group=ep_group.rank_in_group,
        barrier=ep_group.barrier,
    )
    module.logger = SimpleNamespace(info=lambda *args, **kwargs: None)
    return module, rearrangements, ep_group


def test_runtime_patch_records_initial_and_committed_maps(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    module, _, _ = _make_eplb_module(record_path=output)
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


def test_two_pp_stages_record_and_load_independent_local_maps(
    tmp_path: Path,
) -> None:
    output = tmp_path / "pipeline.json"
    stage_maps = (
        torch.tensor(
            [
                [3, 2, 1, 0, 3, 2],
                [2, 3, 0, 1, 2, 3],
            ],
            dtype=torch.int64,
        ),
        torch.tensor([[1, 0, 3, 2, 1, 0]], dtype=torch.int64),
    )

    for pp_rank, stage_map in enumerate(stage_maps):
        module, rearrangements, _ = _make_eplb_module(
            record_path=output,
            pipeline_parallel_size=2,
            pipeline_parallel_rank=pp_rank,
        )
        assert apply_to_module(module)
        state = module.EplbState()
        model = PipelineMoEForCausalLM(local_moe_layers=stage_map.shape[0])
        state.add_model(model, _FakeModelConfig())
        model_state = state.model_states["hy4-hash"]

        assert model.num_moe_layers == stage_map.shape[0]
        assert model_state.physical_to_logical_map.shape == stage_map.shape
        module._commit_eplb_maps(model_state, stage_map)
        assert rearrangements == []

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert set(payload["model_maps"]) == {
        "PipelineMoEForCausalLM#pp_rank=0",
        "PipelineMoEForCausalLM#pp_rank=1",
    }
    for pp_rank, stage_map in enumerate(stage_maps):
        key = f"PipelineMoEForCausalLM#pp_rank={pp_rank}"
        assert payload["model_maps"][key]["physical_to_logical_map"] == (
            stage_map.tolist()
        )

        model = PipelineMoEForCausalLM(local_moe_layers=stage_map.shape[0])
        config = SimpleNamespace(
            parallel_config=SimpleNamespace(
                _vllm_hcu_expert_map_path=str(output),
                enable_expert_parallel=True,
                enable_eplb=True,
                enable_ep_weight_filter=False,
                pipeline_parallel_size=2,
                pipeline_parallel_rank=pp_rank,
            )
        )
        plan = bind_static_eplb_plan(config, model)
        assert plan is not None
        assert plan.model_key == key
        assert plan.physical_to_logical_map.tolist() == stage_map.tolist()

        load_module, rearrangements, _ = _make_eplb_module(
            load_path=output,
            pipeline_parallel_size=2,
            pipeline_parallel_rank=pp_rank,
        )
        assert apply_to_module(load_module)
        load_state = load_module.EplbState()
        load_state.add_model(model, _FakeModelConfig())
        assert rearrangements == []
        assert load_state.model_states[
            "hy4-hash"
        ].physical_to_logical_map.tolist() == stage_map.tolist()


def test_dynamic_eplb_keeps_the_models_declared_global_layer_count() -> None:
    module, _, _ = _make_eplb_module()
    assert apply_to_module(module)
    state = module.EplbState()
    model = PipelineMoEForCausalLM(local_moe_layers=1)

    state.add_model(model, _FakeModelConfig())

    assert model.num_moe_layers == 3
    assert state.model_states["hy4-hash"].physical_to_logical_map.shape == (3, 6)


def test_nested_mllama_record_is_loadable_under_the_same_public_key(
    tmp_path: Path,
) -> None:
    output = tmp_path / "mllama.json"
    record_module, _, _ = _make_eplb_module(record_path=output)
    assert apply_to_module(record_module)
    record_state = record_module.EplbState()
    record_state.add_model(Llama4ForConditionalGeneration(), _FakeModelConfig())

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert set(payload["model_maps"]) == {"Llama4ForConditionalGeneration"}

    model = Llama4ForConditionalGeneration()
    plan = bind_static_eplb_plan(
        SimpleNamespace(
            parallel_config=SimpleNamespace(
                _vllm_hcu_expert_map_path=str(output),
                enable_expert_parallel=True,
                enable_eplb=True,
                enable_ep_weight_filter=False,
                pipeline_parallel_size=1,
            )
        ),
        model,
    )
    assert plan is not None
    assert plan.model_key == "Llama4ForConditionalGeneration"
    assert model.language_model._vllm_hcu_static_eplb_plan is plan

    load_module, rearrangements, _ = _make_eplb_module(load_path=output)
    assert apply_to_module(load_module)
    load_state = load_module.EplbState()
    load_state.add_model(model, _FakeModelConfig())

    assert rearrangements == []
    assert load_state.model_states[
        "hy4-hash"
    ].physical_to_logical_map.tolist() == [[0, 1, 2, 3, 0, 1]]


def test_runtime_patch_commits_generic_static_plan_without_rearrangement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "load.json"
    source.write_text(
        json.dumps(
            {
                "model_maps": {
                    "GenericMoEForCausalLM": {
                        "physical_to_logical_map": [[3, 2, 1, 0, 3, 2]],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    model = GenericMoEForCausalLM()
    model._vllm_hcu_static_eplb_plan = load_static_eplb_plan(
        source,
        model_key="GenericMoEForCausalLM",
        expected_shape=(1, 6),
        num_logical_experts=4,
        num_redundant_experts=2,
    )
    module, rearrangements, ep_group = _make_eplb_module(load_path=source)
    ep_group.world_size = 2
    fingerprint_checks: list[tuple[object, object]] = []

    def gather_fingerprints(output, value, group=None) -> None:
        fingerprint_checks.append((value, group))
        output[:] = [value, value]

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather_fingerprints)
    cpu_group = object()
    module.get_ep_group = lambda: SimpleNamespace(
        device_group=ep_group,
        cpu_group=cpu_group,
        world_size=2,
        rank_in_group=0,
        barrier=ep_group.barrier,
    )
    assert apply_to_module(module)
    state = module.EplbState()
    state.add_model(model, _FakeModelConfig())

    assert rearrangements == []
    expected_fingerprint = model._vllm_hcu_static_eplb_plan.fingerprint()
    assert fingerprint_checks == [(expected_fingerprint, cpu_group)]
    assert ep_group.barrier_calls == 1
    assert state.model_states["hy4-hash"].physical_to_logical_map.tolist() == [
        [3, 2, 1, 0, 3, 2]
    ]
    assert state.should_record_tensor.item() is False
    assert state.is_async is False

    assert state.step(is_profile=True) is None
    assert state.official_profile_steps == 0
    assert state.step(is_profile=False) is None
    assert state.official_steps == 0


def test_runtime_patch_record_mode_plans_without_live_rearrangement(
    tmp_path: Path,
) -> None:
    output = tmp_path / "record.json"
    module, rearrangements, _ = _make_eplb_module(record_path=output)
    assert apply_to_module(module)
    state = module.EplbState()
    state.add_model(HYV4ForCausalLM(), _FakeModelConfig())

    assert state.step(is_profile=False) == "official-step"
    assert state.official_steps == 1
    assert state.should_record_tensor.item() is True
    assert state.is_async is False

    model_state = state.model_states["hy4-hash"]
    initial_map = model_state.physical_to_logical_map.clone()
    assert state.rearrange() == "official-rearrange"

    assert rearrangements == []
    assert torch.equal(model_state.physical_to_logical_map, initial_map)
    recorded = json.loads(output.read_text(encoding="utf-8"))
    assert recorded["model_maps"]["HYV4ForCausalLM"][
        "physical_to_logical_map"
    ] == [[3, 2, 1, 0, 3, 2]]


def test_runtime_patch_logs_balancedness_across_ep_ranks_per_layer() -> None:
    module, _, _ = _make_eplb_module()
    log_calls: list[tuple[object, ...]] = []
    module.logger.info = lambda _message, *args: log_calls.append(args)
    module.get_ep_group = lambda: SimpleNamespace(
        device_group=SimpleNamespace(size=lambda: 2, rank=lambda: 0),
    )
    assert apply_to_module(module)

    state = module.EplbState()
    state.add_model(HYV4ForCausalLM(), _FakeModelConfig())
    model_state = state.model_states["hy4-hash"]
    # Both layers put 8 tokens on rank 0 and 2 on rank 1. The correct
    # aggregate is avg=(5 + 5)=10, max=(8 + 8)=16, ratio=0.625.
    model_state.expert_load_pass = torch.tensor([[8, 2], [8, 2]])
    model_state.expert_load_window = torch.zeros((1, 2, 2), dtype=torch.int64)
    state.expert_rearrangement_step = 0
    state.expert_rearrangement_step_interval = 100
    state.expert_load_window_step = 0
    state.expert_load_window_size = 1
    state.parallel_config.eplb_config = SimpleNamespace(
        log_balancedness_interval=1,
    )
    state._sync_load_pass = lambda: [model_state.expert_load_pass.clone()]
    state._should_record_current_step = lambda log_stats=False: log_stats
    state._update_layer_should_record = lambda log_stats=False: None

    assert state.step(log_stats=True) == "official-step"

    assert len(log_calls) == 1
    _, model_name, avg_tokens, max_tokens, balancedness, _ = log_calls[0]
    assert model_name == "/models/Hy4-preview-Channel-FP8-w8a8-v2"
    assert avg_tokens == 10.0
    assert max_tokens == 16.0
    assert balancedness == 0.625
    assert model_state.expert_load_window.tolist() == [[[8, 2], [8, 2]]]
    assert model_state.expert_load_pass.tolist() == [[0, 0], [0, 0]]


def test_runtime_patch_does_not_double_record_near_rearrangement() -> None:
    module, _, _ = _make_eplb_module()
    module.get_ep_group = lambda: SimpleNamespace(
        device_group=SimpleNamespace(size=lambda: 2, rank=lambda: 0),
    )
    assert apply_to_module(module)

    state = module.EplbState()
    state.add_model(HYV4ForCausalLM(), _FakeModelConfig())
    model_state = state.model_states["hy4-hash"]
    model_state.expert_load_pass = torch.tensor([[8, 2]])
    model_state.expert_load_window = torch.zeros((2, 1, 2), dtype=torch.int64)
    state.expert_rearrangement_step = 0
    state.expert_rearrangement_step_interval = 1
    state.expert_load_window_step = 0
    state.expert_load_window_size = 2
    state.parallel_config.eplb_config = SimpleNamespace(
        log_balancedness_interval=1,
    )
    state.simulate_official_window_step = True

    sync_calls = 0
    original_sync = state._sync_load_pass

    def sync_load_pass():
        nonlocal sync_calls
        sync_calls += 1
        return original_sync()

    should_record_calls: list[bool] = []
    update_calls: list[bool] = []
    state._sync_load_pass = sync_load_pass
    state._should_record_current_step = lambda log_stats=False: (
        should_record_calls.append(log_stats) or True
    )

    def update_layer_should_record(log_stats=False):
        update_calls.append(log_stats)
        state.should_record_tensor.fill_(log_stats)

    state._update_layer_should_record = update_layer_should_record

    assert state.step(log_stats=True) == "official-step"

    assert sync_calls == 1
    assert model_state.expert_load_window.tolist() == [
        [[8, 2]],
        [[0, 0]],
    ]
    assert model_state.expert_load_pass.tolist() == [[0, 0]]
    assert state.expert_load_window_step == 1
    assert should_record_calls == [False, False]
    assert update_calls == [False, True]
    assert state.should_record_tensor.item() is True


def test_runtime_patch_record_mode_keeps_profile_rearrangement(
    tmp_path: Path,
) -> None:
    module, rearrangements, _ = _make_eplb_module(
        record_path=tmp_path / "record.json"
    )
    assert apply_to_module(module)
    state = module.EplbState()
    state.add_model(HYV4ForCausalLM(), _FakeModelConfig())

    assert state.rearrange(is_profile=True) == "official-rearrange"
    assert len(rearrangements) == 1


def test_runtime_patch_keeps_online_rearrangement_without_offline_paths() -> None:
    module, rearrangements, _ = _make_eplb_module()
    assert apply_to_module(module)
    state = module.EplbState()
    state.add_model(HYV4ForCausalLM(), _FakeModelConfig())

    assert state.is_async is True
    assert state.rearrange() == "official-rearrange"
    assert len(rearrangements) == 1
    assert state.model_states["hy4-hash"].physical_to_logical_map.tolist() == [
        [3, 2, 1, 0, 3, 2]
    ]


def test_runtime_patch_logs_load_but_skips_disabled_dynamic_rearrangement() -> None:
    module, rearrangements, _ = _make_eplb_module()
    log_calls: list[tuple[object, ...]] = []
    module.logger.info = lambda _message, *args: log_calls.append(args)
    module.get_ep_group = lambda: SimpleNamespace(
        device_group=SimpleNamespace(size=lambda: 2, rank=lambda: 0),
    )
    assert apply_to_module(module)

    state = module.EplbState()
    state.parallel_config._vllm_hcu_eplb_disable_rearrange = True
    state.add_model(HYV4ForCausalLM(), _FakeModelConfig())
    assert state.is_async is False
    model_state = state.model_states["hy4-hash"]
    model_state.expert_load_pass = torch.tensor([[8, 2]])
    model_state.expert_load_window = torch.zeros((1, 1, 2), dtype=torch.int64)
    state.expert_rearrangement_step = 0
    state.expert_rearrangement_step_interval = 1
    state.expert_load_window_step = 0
    state.expert_load_window_size = 1
    state.parallel_config.eplb_config = SimpleNamespace(
        log_balancedness_interval=1,
    )
    state.simulate_official_rearrange_step = True

    assert state.step(log_stats=True) == "official-step"

    assert state.official_steps == 1
    assert state.expert_rearrangement_step == 0
    assert rearrangements == []
    assert any(args[4] == 0.625 for args in log_calls if len(args) == 6)

    assert state.rearrange(is_profile=True) == "official-rearrange"
    assert len(rearrangements) == 1


def test_runtime_patch_rejects_configured_static_map_without_plan(
    tmp_path: Path,
) -> None:
    source = tmp_path / "load.json"
    source.write_text(
        json.dumps({"physical_to_logical_map": [[3, 2, 1, 0, 3, 2]]}),
        encoding="utf-8",
    )
    module, rearrangements, ep_group = _make_eplb_module(load_path=source)
    assert apply_to_module(module)
    state = module.EplbState()
    model = GenericMoEForCausalLM()

    with pytest.raises(PatchCompatibilityError, match="before checkpoint loading"):
        state.add_model(model, _FakeModelConfig())

    assert rearrangements == []
    assert ep_group.barrier_calls == 0


def test_runtime_patch_rejects_cross_rank_plan_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "load.json"
    source.write_text(
        json.dumps({"physical_to_logical_map": [[3, 2, 1, 0, 3, 2]]}),
        encoding="utf-8",
    )
    model = HYV4ForCausalLM()
    model._vllm_hcu_static_eplb_plan = load_static_eplb_plan(
        source,
        model_key="HYV4ForCausalLM",
        expected_shape=(1, 6),
        num_logical_experts=4,
        num_redundant_experts=2,
    )
    module, rearrangements, ep_group = _make_eplb_module(load_path=source)
    ep_group.world_size = 2
    monkeypatch.setattr(
        torch.distributed,
        "all_gather_object",
        lambda output, value, group=None: output.__setitem__(slice(None), [value, ("bad",)]),
    )
    module.get_ep_group = lambda: SimpleNamespace(
        device_group=ep_group,
        cpu_group=object(),
        world_size=2,
        rank_in_group=0,
        barrier=ep_group.barrier,
    )
    assert apply_to_module(module)
    state = module.EplbState()

    with pytest.raises(RuntimeError, match="fingerprints differ"):
        state.add_model(model, _FakeModelConfig())

    assert rearrangements == []
    assert ep_group.barrier_calls == 0
