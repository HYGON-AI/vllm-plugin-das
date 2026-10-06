# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

import hashlib
import inspect
import json
import os
import stat
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType, SimpleNamespace

import pytest
import torch

from vllm_hcu.patch.worker.framework_opt.patch_offline_eplb import (
    record_offline_expert_map,
    validate_offline_eplb_record_complete,
)

from tests.models.static_eplb_test_utils import GenericMoE, config_and_map
from vllm_hcu.model_executor.layers.fused_moe.static_eplb import (
    bind_static_eplb_plan,
)
from vllm_hcu.patch.config import HcuFeatureConfig, bind_hcu_eplb_config


def _record(
    path,
    *,
    model_key="HYV4ForCausalLM",
    record_kind="proposal",
    rows=None,
    num_logical_experts=3,
    num_redundant_experts=1,
):
    rows = [[0, 1, 2, 1], [2, 0, 1, 2]] if rows is None else rows
    return record_offline_expert_map(
        path,
        model_key=model_key,
        model_name="hy4-test",
        model_class=model_key,
        physical_to_logical_map=torch.tensor(rows),
        num_logical_experts=num_logical_experts,
        num_redundant_experts=num_redundant_experts,
        record_kind=record_kind,
    )


def test_record_is_rank_zero_only(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    import vllm_hcu.patch.worker.framework_opt.patch_offline_eplb as api

    path = tmp_path / "record.json"
    monkeypatch.setattr(api, "_ep_rank", lambda: 1)
    _record(path)
    assert not path.exists()

    monkeypatch.setattr(api, "_ep_rank", lambda: 0)
    _record(path)
    assert path.exists()


def test_record_merges_target_and_mtp_with_exact_version_two_metadata(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm_hcu.patch.worker.framework_opt.patch_offline_eplb as api

    monkeypatch.setattr(api, "_ep_rank", lambda: 0)
    path = tmp_path / "record.json"
    _record(path, model_key="HYV4ForCausalLM", record_kind="initial")
    _record(
        path,
        model_key="HYV4MTP",
        rows=[[0, 1, 2, 1]],
        record_kind="proposal",
    )
    _record(path, model_key="HYV4ForCausalLM", record_kind="proposal")

    payload = json.loads(path.read_text())
    assert payload["version"] == 2
    assert payload["format"] == (
        "vllm_offline_eplb_physical_to_logical_by_model"
    )
    assert set(payload["model_maps"]) == {"HYV4ForCausalLM", "HYV4MTP"}
    target = payload["model_maps"]["HYV4ForCausalLM"]
    assert target == {
        "record_kind": "proposal",
        "model_name": "hy4-test",
        "model_class": "HYV4ForCausalLM",
        "num_moe_layers": 2,
        "num_logical_experts": 3,
        "num_physical_experts": 4,
        "num_redundant_experts": 1,
        "physical_to_logical_map": [[0, 1, 2, 1], [2, 0, 1, 2]],
    }


def test_proposal_record_cannot_be_downgraded_to_initial(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm_hcu.patch.worker.framework_opt.patch_offline_eplb as api

    monkeypatch.setattr(api, "_ep_rank", lambda: 0)
    path = tmp_path / "record.json"
    _record(path, record_kind="proposal")
    original = path.read_bytes()
    _record(path, record_kind="initial")

    assert path.read_bytes() == original


def test_record_uses_lock_and_preserves_concurrent_model_entries(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm_hcu.patch.worker.framework_opt.patch_offline_eplb as api

    monkeypatch.setattr(api, "_ep_rank", lambda: 0)
    path = tmp_path / "record.json"
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(_record, path, model_key="HYV4ForCausalLM"),
            pool.submit(
                _record,
                path,
                model_key="HYV4MTP",
                rows=[[0, 1, 2, 1]],
            ),
        ]
        for future in futures:
            future.result()

    assert set(json.loads(path.read_text())["model_maps"]) == {
        "HYV4ForCausalLM",
        "HYV4MTP",
    }
    assert path.with_suffix(path.suffix + ".lock").exists()


def test_record_fsyncs_file_and_parent_and_cleans_temporary_file(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm_hcu.patch.worker.framework_opt.patch_offline_eplb as api

    monkeypatch.setattr(api, "_ep_rank", lambda: 0)
    path = tmp_path / "record.json"
    original_fsync = os.fsync
    fsynced_modes = []

    def fsync(fd):
        fsynced_modes.append(os.fstat(fd).st_mode)
        return original_fsync(fd)

    monkeypatch.setattr(api.os, "fsync", fsync)
    _record(path)

    assert any(stat.S_ISREG(mode) for mode in fsynced_modes)
    assert any(stat.S_ISDIR(mode) for mode in fsynced_modes)
    assert not list(tmp_path.glob(f".{path.name}.*.tmp"))


def test_malformed_existing_record_is_preserved_on_failure(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm_hcu.patch.worker.framework_opt.patch_offline_eplb as api

    monkeypatch.setattr(api, "_ep_rank", lambda: 0)
    path = tmp_path / "record.json"
    path.write_text("malformed")
    with pytest.raises(ValueError, match="malformed|JSON"):
        _record(path)
    assert path.read_text() == "malformed"
    assert not list(tmp_path.glob(f".{path.name}.*.tmp"))


def _complete_record(path):
    return validate_offline_eplb_record_complete(
        path,
        required_models={"HYV4ForCausalLM": 2, "HYV4MTP": 1},
        num_logical_experts=3,
        num_redundant_experts=1,
    )


def test_complete_record_requires_target_and_mtp_proposals(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm_hcu.patch.worker.framework_opt.patch_offline_eplb as api

    monkeypatch.setattr(api, "_ep_rank", lambda: 0)
    path = tmp_path / "record.json"
    _record(path, model_key="HYV4ForCausalLM")
    with pytest.raises(ValueError, match="HYV4MTP"):
        _complete_record(path)

    _record(
        path,
        model_key="HYV4MTP",
        rows=[[0, 1, 2, 1]],
        record_kind="initial",
    )
    with pytest.raises(ValueError, match="proposal"):
        _complete_record(path)

    _record(
        path,
        model_key="HYV4MTP",
        rows=[[0, 1, 2, 1]],
        record_kind="proposal",
    )
    assert _complete_record(path) == hashlib.sha256(path.read_bytes()).hexdigest()


def test_complete_record_rejects_wrong_counts(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm_hcu.patch.worker.framework_opt.patch_offline_eplb as api

    monkeypatch.setattr(api, "_ep_rank", lambda: 0)
    path = tmp_path / "record.json"
    _record(path, model_key="HYV4ForCausalLM")
    _record(
        path,
        model_key="HYV4MTP",
        rows=[[0, 1, 2, 1]],
    )
    payload = json.loads(path.read_text())
    payload["model_maps"]["HYV4MTP"]["num_redundant_experts"] = 0
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="count|redundant"):
        _complete_record(path)


def _set_mode(config, tmp_path, mode: str) -> None:
    if mode == "static":
        return
    updates = {}
    if mode == "record":
        updates["expert_map_record_path"] = str(tmp_path / "record.json")
    config.additional_config["hcu"] = HcuFeatureConfig(**updates).to_dict()
    bind_hcu_eplb_config(config)


def _setup_state(tmp_path, monkeypatch: pytest.MonkeyPatch, mode="static"):
    import vllm.distributed.eplb.eplb_state as upstream
    import vllm_hcu.patch.worker.framework_opt.patch_offline_eplb as api

    for owner, names in (
        (upstream.EplbState, ("add_model", "step", "rearrange")),
        (upstream, ("_commit_eplb_maps", "rearrange_expert_weights_inplace")),
    ):
        for name in names:
            monkeypatch.setattr(owner, name, inspect.unwrap(getattr(owner, name)))
    monkeypatch.delattr(upstream, api._MARKER, raising=False)

    model = GenericMoE()
    config, path = config_and_map(tmp_path)
    _set_mode(config, tmp_path, mode)
    parallel = config.parallel_config
    parallel.num_ubatches = 0
    parallel.eplb_config = SimpleNamespace(
        use_async=True,
        window_size=2,
        step_interval=1,
        policy="default",
        communicator="nixl",
        log_balancedness_interval=1,
    )
    device_group = SimpleNamespace(rank=lambda: 0, size=lambda: 2)
    group = SimpleNamespace(
        world_size=2,
        rank_in_group=0,
        cpu_group=None,
        device_group=device_group,
        barrier=lambda: None,
    )
    monkeypatch.setattr(upstream, "get_ep_group", lambda: group)
    monkeypatch.setattr(upstream, "get_eplb_group", lambda: group)
    monkeypatch.setattr(upstream, "get_node_count", lambda: 1)
    monkeypatch.setattr(upstream.current_platform, "is_rocm", lambda: False)
    communicator_backends = []

    def create_communicator(**kwargs):
        communicator_backends.append(kwargs["backend"])
        return SimpleNamespace()

    monkeypatch.setattr(upstream, "create_eplb_communicator", create_communicator)
    transfer_calls = []

    def transfer(
        old_global_expert_indices,
        new_global_expert_indices,
        expert_weights,
        expert_buffer,
        ep_group,
        communicator,
        is_profile=False,
        rank_mapping=None,
    ):
        transfer_calls.append(
            (
                (
                    old_global_expert_indices,
                    new_global_expert_indices,
                    expert_weights,
                    expert_buffer,
                    ep_group,
                    communicator,
                ),
                {"is_profile": is_profile, "rank_mapping": rank_mapping},
            )
        )

    monkeypatch.setattr(upstream, "rearrange_expert_weights_inplace", transfer)
    api.apply_to_module(upstream)

    for layer in model.moe_layers:
        layer.eplb_state = upstream.EplbLayerState()
        layer.get_expert_weights = (
            lambda layer=layer: list(layer.routed_experts.parameters())
        )
        layer.set_eplb_state = layer.eplb_state.set_layer_state
    if mode == "static":
        bind_static_eplb_plan(config, model)

    state = upstream.EplbState(parallel, torch.device("cpu"))
    model_config = SimpleNamespace(
        model="hy4-test",
        compute_hash=lambda: f"{mode}-model",
    )
    return SimpleNamespace(
        api=api,
        upstream=upstream,
        state=state,
        model=model,
        model_config=model_config,
        config=config,
        path=path,
        group=group,
        communicator_backends=communicator_backends,
        transfer_calls=transfer_calls,
    )


def _install_cpu_events(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        torch.cuda,
        "Event",
        lambda **kwargs: SimpleNamespace(
            record=lambda: None,
            synchronize=lambda: None,
            elapsed_time=lambda other: 0,
        ),
    )


def test_record_add_publishes_initial_map_and_disables_async(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _setup_state(tmp_path, monkeypatch, mode="record")
    fixture.state.add_model(fixture.model, fixture.model_config)

    payload = json.loads((tmp_path / "record.json").read_text())
    entry = payload["model_maps"]["GenericMoE"]
    assert entry["record_kind"] == "initial"
    assert entry["physical_to_logical_map"] == [[0, 1, 2, 0]] * 2
    assert fixture.state.is_async is False
    assert fixture.state._vllm_hcu_offline_proposal_events == 0
    assert fixture.state._vllm_hcu_offline_transfer_events == 0
    assert fixture.state._vllm_hcu_offline_rearrangement_events == 0


def test_record_proposal_uses_policy_without_transfer_or_live_commit(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _setup_state(tmp_path, monkeypatch, mode="record")
    fixture.state.add_model(fixture.model, fixture.model_config)
    candidate = torch.tensor([[2, 0, 1, 2], [0, 1, 2, 1]])
    policy_calls = []
    fixture.state.policy = SimpleNamespace(
        rebalance_experts=lambda *args: policy_calls.append(args) or candidate
    )
    fixture.state._allreduce_list = lambda values: values
    _install_cpu_events(monkeypatch)

    fixture.state.rearrange()

    entry = json.loads((tmp_path / "record.json").read_text())["model_maps"][
        "GenericMoE"
    ]
    assert entry["record_kind"] == "proposal"
    assert entry["physical_to_logical_map"] == candidate.tolist()
    assert policy_calls
    assert fixture.transfer_calls == []
    live = fixture.state.model_states["record-model"]
    assert live.physical_to_logical_map.tolist() == [[0, 1, 2, 0]] * 2
    assert fixture.state._vllm_hcu_offline_proposal_events == 1
    assert fixture.state._vllm_hcu_offline_transfer_events == 0
    assert fixture.state._vllm_hcu_offline_rearrangement_events == 0


def test_record_dummy_and_profile_work_cannot_publish_proposal(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _setup_state(tmp_path, monkeypatch, mode="record")
    fixture.state.add_model(fixture.model, fixture.model_config)
    candidate = torch.tensor([[2, 0, 1, 2], [0, 1, 2, 1]])
    fixture.state.policy = SimpleNamespace(rebalance_experts=lambda *args: candidate)
    fixture.state._allreduce_list = lambda values: values
    _install_cpu_events(monkeypatch)

    fixture.state.step(is_dummy=True)
    fixture.state.rearrange(is_profile=True)

    entry = json.loads((tmp_path / "record.json").read_text())["model_maps"][
        "GenericMoE"
    ]
    assert entry["record_kind"] == "initial"
    assert fixture.state._vllm_hcu_offline_proposal_events == 0


def test_record_rejects_elastic_rank_mapping(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _setup_state(tmp_path, monkeypatch, mode="record")
    fixture.state.add_model(fixture.model, fixture.model_config)
    with pytest.raises(ValueError, match="elastic"):
        fixture.state.rearrange(rank_mapping={0: 0, 1: 1})


def test_static_add_commits_plan_with_gloo_and_zero_mutation_counters(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _setup_state(tmp_path, monkeypatch, mode="static")
    fixture.state.parallel_config._vllm_hcu_eplb_static_dispatch_policy = (
        "locality_fair"
    )
    fixture.group.rank_in_group = 1
    fixture.group.device_group = SimpleNamespace(rank=lambda: 1, size=lambda: 2)

    fixture.state.add_model(fixture.model, fixture.model_config)

    live = fixture.state.model_states["static-model"]
    assert live.physical_to_logical_map.tolist() == [
        [0, 1, 2, 1],
        [2, 0, 1, 2],
    ]
    assert live.logical_to_physical_map[0, 1, :2].tolist() == [3, 1]
    assert fixture.communicator_backends == ["torch_gloo"]
    assert fixture.state.parallel_config.eplb_config.communicator == "nixl"
    assert fixture.state.is_async is False
    assert fixture.transfer_calls == []
    assert fixture.state._vllm_hcu_offline_proposal_events == 0
    assert fixture.state._vllm_hcu_offline_transfer_events == 0
    assert fixture.state._vllm_hcu_offline_rearrangement_events == 0


def test_static_add_logs_committed_runtime_map_fingerprint(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm_hcu.patch.worker.framework_opt.patch_offline_eplb as api

    fixture = _setup_state(tmp_path, monkeypatch, mode="static")
    fixture.state.parallel_config._vllm_hcu_eplb_static_dispatch_policy = (
        "locality_fair"
    )
    fixture.group.device_group = SimpleNamespace(rank=lambda: 1, size=lambda: 2)
    info_calls = []
    monkeypatch.setattr(
        api,
        "logger",
        SimpleNamespace(info=lambda *args: info_calls.append(args)),
    )

    fixture.state.add_model(fixture.model, fixture.model_config)

    assert info_calls == [
        (
            "Committed static EPLB runtime map: model=%s source_sha256=%s "
            "shape=%s dispatch_policy=%s ep_rank=%d/%d "
            "transfer_events=%d rearrangement_events=%d",
            "GenericMoE",
            fixture.model._vllm_hcu_static_eplb_plan.source_sha256,
            (2, 4),
            "locality_fair",
            1,
            2,
            0,
            0,
        )
    ]


def test_static_committed_map_reaches_installed_deepep_and_masked_deepgemm(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
    from vllm_hcu.patch.worker.op_opt.moe import (
        patch_base_router,
        patch_deepep_ll,
    )
    from vllm_hcu.platforms import envs as henvs

    patch_base_router.apply()
    patch_deepep_ll.apply()
    from vllm.model_executor.layers.fused_moe.prepare_finalize import (
        deepep_ll as deepep_ll_module,
    )
    from vllm.model_executor.layers.fused_moe.router import (
        base_router as base_router_module,
    )
    from vllm_hcu.model_executor.layers.fused_moe.experts import (
        dpsk_v4_deep_gemm_moe as deepgemm_module,
    )

    fixture = _setup_state(tmp_path, monkeypatch, mode="static")
    fixture.state.parallel_config._vllm_hcu_eplb_static_dispatch_policy = (
        "locality_fair"
    )
    fixture.group.rank_in_group = 1
    fixture.group.device_group = SimpleNamespace(rank=lambda: 1, size=lambda: 2)
    fixture.state.add_model(fixture.model, fixture.model_config)
    layer_state = fixture.model.moe_layers[0].eplb_state
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_TORCH_EPLB_MAP_RECORD", True)
    monkeypatch.setattr(base_router_module, "dbo_current_ubatch_id", lambda: 0)

    class StaticRouter(base_router_module.BaseRouter):
        @property
        def routing_method_type(self):
            return RoutingMethodType.Unspecified

        def _compute_routing(
            self,
            hidden_states,
            router_logits,
            indices_type,
            *,
            input_ids=None,
        ):
            del hidden_states, router_logits, indices_type, input_ids
            return (
                torch.ones((2, 1)),
                torch.tensor([[1], [1]], dtype=torch.int64),
            )

    router = StaticRouter(
        top_k=1,
        global_num_experts=fixture.model.num_logical_experts,
        eplb_state=layer_state,
    )
    _, physical_ids = router._select_experts(
        torch.empty((2, 2048)),
        torch.empty((2, fixture.model.num_logical_experts)),
        torch.int64,
    )

    dispatch = {}
    expert_x = torch.ones(
        (2, 1, 2048),
        dtype=torch.float8_e4m3fn,
    )
    expert_x_scale = torch.ones((2, 1, 1))
    local_token_counts = torch.tensor([0, 1], dtype=torch.int32)

    class Buffer:
        def low_latency_dispatch(
            self,
            x,
            topk_idx,
            topk_weight,
            num_max_dispatch_tokens_per_rank,
            num_experts,
            quant_type=1,
            quant_group_size=0,
            fp8_round_scale=False,
            async_finish=False,
            return_recv_hook=False,
        ):
            dispatch.update(
                topk_idx=topk_idx.clone(),
                num_experts=num_experts,
                quant_type=quant_type,
            )
            del (
                x,
                topk_weight,
                num_max_dispatch_tokens_per_rank,
                quant_group_size,
                fp8_round_scale,
                async_finish,
                return_recv_hook,
            )
            return (
                (expert_x, expert_x_scale),
                local_token_counts,
                "static-handle",
                None,
                lambda: None,
            )

    quant_config = SimpleNamespace(
        quant_dtype=torch.float8_e4m3fn,
        block_shape=[1, 128],
        per_act_token_quant=True,
        a1_scale=None,
        a2_scale=None,
        a1_gscale=None,
        w1_scale=torch.ones((2, 8)),
        w2_scale=torch.ones((2, 2048)),
        gemm1_clamp_limit=10.0,
        use_int8_w8a8=False,
    )
    prepare_finalize = deepep_ll_module.DeepEPLLPrepareAndFinalize(
        Buffer(),
        max_tokens_per_rank=8,
        num_dispatchers=1,
        use_fp8_dispatch=True,
    )
    hook, receiver = prepare_finalize.prepare_async(
        torch.ones((2, 2048), dtype=torch.bfloat16),
        torch.ones((2, 1)),
        physical_ids,
        fixture.model.num_physical_experts,
        torch.tensor([-1, -1, 0, 1], dtype=torch.int32),
        False,
        quant_config,
    )
    hook()
    routed_x, routed_scale, metadata, routed_ids, routed_weights = receiver()

    experts = object.__new__(deepgemm_module.DeepEPDeepGemmMaskedExperts)
    experts._deepgemm_w13 = torch.empty((1, 1, 1, 1, 1, 1))
    experts._deepgemm_w2 = torch.empty((1, 1, 1, 1, 1, 1))
    experts.quant_config = quant_config
    experts.moe_problem_size = lambda *_args: (2, 1, 8, 2048, 1)
    gemm_calls = []

    def masked_gemm(_a, _b, destination, masked_m, expected_m):
        gemm_calls.append((masked_m.clone(), expected_m))
        destination.fill_(len(gemm_calls))
        return destination

    monkeypatch.setattr(
        deepgemm_module,
        "m_grouped_fp8_gemm_nt_masked",
        masked_gemm,
    )
    monkeypatch.setattr(
        deepgemm_module,
        "fuse_silu_mul_fp8_quant_ep",
        lambda output, **_kwargs: (
            output[..., :4].to(torch.float8_e4m3fn),
            torch.ones(output.shape[:2]),
        ),
    )
    output = torch.empty((2, 1, 2048))
    experts.apply(
        output=output,
        hidden_states=routed_x,
        w1=experts._deepgemm_w13,
        w2=experts._deepgemm_w2,
        topk_weights=torch.ones((2, 1)),
        topk_ids=physical_ids,
        activation=MoEActivation.SILU,
        global_num_experts=fixture.model.num_physical_experts,
        expert_map=torch.tensor([-1, -1, 0, 1], dtype=torch.int32),
        a1q_scale=routed_scale,
        a2_scale=None,
        workspace13=torch.empty(0),
        workspace2=torch.empty((2, 1, 2048)),
        expert_tokens_meta=metadata,
        apply_router_weight_on_input=False,
    )

    assert layer_state.logical_to_physical_map[1, :2].tolist() == [3, 1]
    assert physical_ids.tolist() == [[3], [1]]
    assert torch.equal(dispatch["topk_idx"], physical_ids)
    assert dispatch["num_experts"] == 4
    assert dispatch["quant_type"] == 2
    assert routed_ids is None and routed_weights is None
    assert [counts.tolist() for counts, _ in gemm_calls] == [[0, 1], [0, 1]]
    assert [expected for _, expected in gemm_calls] == [1, 1]
    assert torch.equal(output, torch.full_like(output, 2))
    assert fixture.transfer_calls == []
    assert fixture.state._vllm_hcu_offline_transfer_events == 0
    assert fixture.state._vllm_hcu_offline_rearrangement_events == 0


def test_static_add_restores_requested_communicator_when_factory_fails(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _setup_state(tmp_path, monkeypatch, mode="static")
    fixture.state.parallel_config.eplb_config.communicator = "nixl"

    def fail_factory(**kwargs):
        assert kwargs["backend"] == "torch_gloo"
        raise RuntimeError("controlled communicator failure")

    monkeypatch.setattr(
        fixture.upstream,
        "create_eplb_communicator",
        fail_factory,
    )
    with pytest.raises(RuntimeError, match="controlled communicator failure"):
        fixture.state.add_model(fixture.model, fixture.model_config)
    assert fixture.state.parallel_config.eplb_config.communicator == "nixl"
    assert fixture.state.model_states == {}


def test_static_step_and_rearrange_are_strict_noops(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _setup_state(tmp_path, monkeypatch, mode="static")
    fixture.state.add_model(fixture.model, fixture.model_config)
    before = (
        fixture.state.expert_rearrangement_step,
        fixture.state.expert_load_window_step,
    )
    for kwargs in ({}, {"is_dummy": True}, {"is_profile": True}):
        assert fixture.state.step(**kwargs) is None
    assert fixture.state.rearrange() is None
    assert fixture.state.rearrange(is_profile=True) is None
    assert before == (
        fixture.state.expert_rearrangement_step,
        fixture.state.expert_load_window_step,
    )
    assert fixture.transfer_calls == []


def test_static_state_rejects_models_bound_to_mixed_source_sha(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _setup_state(tmp_path, monkeypatch, mode="static")
    fixture.state.add_model(fixture.model, fixture.model_config)

    payload = json.loads(fixture.path.read_text())
    payload["model_maps"]["GenericMoE"]["physical_to_logical_map"] = [
        [1, 0, 2, 1],
        [0, 2, 1, 0],
    ]
    fixture.path.write_text(json.dumps(payload))
    second = GenericMoE()
    for layer in second.moe_layers:
        layer.eplb_state = fixture.upstream.EplbLayerState()
        layer.get_expert_weights = (
            lambda layer=layer: list(layer.routed_experts.parameters())
        )
        layer.set_eplb_state = layer.eplb_state.set_layer_state
    bind_static_eplb_plan(fixture.config, second)

    second_config = SimpleNamespace(
        model="hy4-test-mtp",
        compute_hash=lambda: "static-mtp-model",
    )
    with pytest.raises(ValueError, match="mixed source SHA-256"):
        fixture.state.add_model(second, second_config)


def test_dynamic_eplb_delegates_to_current_initialization(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _setup_state(tmp_path, monkeypatch, mode="dynamic")
    fixture.state.parallel_config.eplb_config.communicator = "torch_nccl"
    fixture.state.parallel_config.eplb_config.use_async = False

    fixture.state.add_model(fixture.model, fixture.model_config)

    assert fixture.communicator_backends == ["torch_nccl"]
    assert fixture.state.model_states[
        "dynamic-model"
    ].physical_to_logical_map.tolist() == [[0, 1, 2, 0]] * 2
    assert not hasattr(fixture.state, "_vllm_hcu_offline_proposal_events")


def _isolated_offline_target():
    import vllm.distributed.eplb.eplb_state as upstream
    import vllm_hcu.patch.worker.framework_opt.patch_offline_eplb as api

    target = ModuleType(api.TARGET_MODULE)
    target.EplbState = type(
        "EplbState",
        (),
        {
            name: inspect.unwrap(getattr(upstream.EplbState, name))
            for name in ("add_model", "step", "rearrange")
        },
    )
    for name in ("_commit_eplb_maps", "rearrange_expert_weights_inplace"):
        setattr(target, name, inspect.unwrap(getattr(upstream, name)))
    return api, target


def _offline_hooks(target):
    return [
        (target.EplbState, name)
        for name in ("add_model", "step", "rearrange")
    ] + [
        (target, name)
        for name in ("_commit_eplb_maps", "rearrange_expert_weights_inplace")
    ]


@pytest.mark.parametrize(
    "name", ["_commit_eplb_maps", "rearrange_expert_weights_inplace"]
)
def test_offline_adapter_rejects_incompatible_mutation_signature(name) -> None:
    api, target = _isolated_offline_target()

    def incompatible(unexpected):
        del unexpected

    setattr(target, name, incompatible)
    originals = [getattr(owner, hook) for owner, hook in _offline_hooks(target)]
    with pytest.raises(api.PatchCompatibilityError, match="signature"):
        api.apply_to_module(target)
    assert all(
        getattr(owner, hook) is original
        for (owner, hook), original in zip(_offline_hooks(target), originals)
    )


@pytest.mark.parametrize("index", range(5))
def test_offline_adapter_rejects_stale_wrapper_identity(index) -> None:
    api, target = _isolated_offline_target()
    api.apply_to_module(target)
    owner, name = _offline_hooks(target)[index]
    setattr(owner, name, inspect.unwrap(getattr(owner, name)))
    with pytest.raises(api.PatchCompatibilityError, match="stale"):
        api.apply_to_module(target)
