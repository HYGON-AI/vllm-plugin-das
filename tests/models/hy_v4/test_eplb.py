# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

import vllm_hcu.models.hy_v4.moe as hy_v4_moe
from tests.models.static_eplb_test_utils import GenericMoE, config_and_map
from tests.models.hy_v4.test_mtp import _minimal_draft, _mtp
from tests.models.hy_v4.test_weight_loading import checkpoint_model
from vllm_hcu.model_executor.layers.fused_moe.static_eplb import (
    bind_static_eplb_plan,
)
from vllm_hcu.models.hy_v4 import model as hy_v4_model
from vllm_hcu.models.hy_v4 import mtp as hy_v4_mtp
from vllm_hcu.models.hy_v4.model import _HYV4CheckpointAccounting
from vllm_hcu.patch.config import HcuFeatureConfig


class _FakeGate(nn.Module):
    def forward(self, hidden_states):
        return torch.zeros(
            hidden_states.shape[0],
            256,
            dtype=torch.float32,
            device=hidden_states.device,
        ), None


class _FakeExperts(nn.Module):
    def forward(self, *, hidden_states, router_logits):
        del router_logits
        return hidden_states


def _hf_config() -> SimpleNamespace:
    return SimpleNamespace(
        hidden_size=32,
        hidden_act="silu",
        num_experts=256,
        num_experts_per_tok=8,
        expert_hidden_dim=16,
        num_shared_experts=0,
        route_norm=True,
        router_scaling_factor=1.0,
        swiglu_limit=0.0,
    )


def _vllm_config(
    *,
    offline_field: str | None = "expert_map_path",
    redundant: int = 8,
    use_async: bool = False,
    enable_eplb: bool = True,
    enable_expert_parallel: bool = True,
    enable_elastic_ep: bool = False,
    pipeline_parallel_size: int = 1,
) -> SimpleNamespace:
    hcu = {}
    if offline_field is not None:
        hcu[offline_field] = "/tmp/hy4-eplb.json"
    return SimpleNamespace(
        additional_config={"hcu": HcuFeatureConfig(**hcu).to_dict()},
        parallel_config=SimpleNamespace(
            enable_eplb=enable_eplb,
            enable_expert_parallel=enable_expert_parallel,
            enable_elastic_ep=enable_elastic_ep,
            enable_ep_weight_filter=False,
            pipeline_parallel_size=pipeline_parallel_size,
            eplb_config=SimpleNamespace(
                num_redundant_experts=redundant,
                use_async=use_async,
            ),
        ),
    )


def _patch_moe_dependencies(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        hy_v4_moe,
        "get_tensor_model_parallel_world_size",
        lambda: 8,
    )
    monkeypatch.setattr(
        hy_v4_moe,
        "get_ep_group",
        lambda: SimpleNamespace(
            rank_in_group=3,
            device_group=SimpleNamespace(size=lambda: 8),
        ),
    )
    monkeypatch.setattr(hy_v4_moe, "GateLinear", lambda *args, **kwargs: _FakeGate())

    def factory(**kwargs):
        captured.update(kwargs)
        return _FakeExperts()

    monkeypatch.setattr(hy_v4_moe, "FusedMoEFactory", factory)
    return captured


@pytest.mark.parametrize(
    ("offline_field", "prefix"),
    [
        ("expert_map_path", "model.layers.3.mlp"),
        ("expert_map_record_path", "model.layers.61.mtp_block.mlp"),
    ],
)
def test_hy_v4_offline_eplb_allocates_ep8_redundant_topology(
    monkeypatch: pytest.MonkeyPatch,
    offline_field: str,
    prefix: str,
) -> None:
    captured = _patch_moe_dependencies(monkeypatch)

    layer = hy_v4_moe.HYV4MoEFused(
        config=_hf_config(),
        vllm_config=_vllm_config(offline_field=offline_field),
        prefix=prefix,
        enable_eplb=True,
    )

    assert layer.n_logical_experts == 256
    assert layer.n_redundant_experts == 8
    assert layer.n_physical_experts == 264
    assert layer.n_local_physical_experts == 33
    assert layer.physical_expert_start == 99
    assert layer.physical_expert_end == 132
    assert captured["enable_eplb"] is True
    assert captured["num_redundant_experts"] == 8


@pytest.mark.parametrize(
    ("updates", "match"),
    [
        ({"offline_field": None}, "offline.*path"),
        ({"offline_field": "expert_map_path", "redundant": 0}, "redundant"),
        ({"use_async": True}, "async"),
        ({"enable_eplb": False}, "enable_eplb"),
        ({"enable_expert_parallel": False}, "expert parallel"),
        ({"enable_elastic_ep": True}, "elastic"),
        ({"pipeline_parallel_size": 2}, "pipeline parallel"),
        ({"redundant": 7}, "divisible"),
    ],
)
def test_hy_v4_offline_eplb_rejects_unsupported_modes(
    monkeypatch: pytest.MonkeyPatch,
    updates: dict[str, object],
    match: str,
) -> None:
    _patch_moe_dependencies(monkeypatch)
    config = _vllm_config(**updates)

    with pytest.raises((ValueError, NotImplementedError), match=match):
        hy_v4_moe.HYV4MoEFused(
            config=_hf_config(),
            vllm_config=config,
            enable_eplb=True,
        )


def test_hy_v4_offline_eplb_rejects_both_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_moe_dependencies(monkeypatch)
    config = _vllm_config()
    config.additional_config["hcu"]["expert_map_record_path"] = (
        "/tmp/hy4-record.json"
    )

    with pytest.raises(ValueError, match="mutually exclusive"):
        hy_v4_moe.HYV4MoEFused(
            config=_hf_config(),
            vllm_config=config,
            enable_eplb=True,
        )


def _copy_generic_metadata(owner: nn.Module, generic: GenericMoE) -> None:
    for name in (
        "num_moe_layers",
        "num_expert_groups",
        "num_logical_experts",
        "num_physical_experts",
        "num_local_physical_experts",
        "num_routed_experts",
        "num_shared_experts",
        "num_redundant_experts",
    ):
        setattr(owner, name, getattr(generic, name))
    owner.expert_weights = []


def _offline_owner(kind: str, *, mode: str = "static"):
    generic = GenericMoE()
    if kind == "target":
        outer = object.__new__(hy_v4_model.HYV4ForCausalLM)
        inner = object.__new__(hy_v4_model.HYV4Model)
    else:
        outer = object.__new__(hy_v4_mtp.HYV4MTP)
        inner = object.__new__(hy_v4_mtp.HYV4MultiTokenPredictor)
    nn.Module.__init__(outer)
    nn.Module.__init__(inner)
    inner.moe_layers = generic.moe_layers if kind == "target" else nn.ModuleList(
        [generic.moe_layers[1]]
    )
    _copy_generic_metadata(inner, generic)
    inner.num_moe_layers = len(inner.moe_layers)
    outer.model = inner
    outer.moe_layers = list(inner.moe_layers)
    _copy_generic_metadata(outer, generic)
    outer.num_moe_layers = len(outer.moe_layers)
    for owner in (outer, inner):
        owner.config = SimpleNamespace(num_experts=generic.num_logical_experts)
        owner._vllm_hcu_offline_eplb_mode = mode
    return outer, inner


def _install_layer_states(model) -> None:
    from vllm.distributed.eplb.eplb_state import EplbLayerState

    for layer in model.moe_layers:
        layer.eplb_state = EplbLayerState()
        layer.get_expert_weights = (
            lambda layer=layer: list(layer.routed_experts.parameters())
        )
        layer.set_eplb_state = layer.eplb_state.set_layer_state


@pytest.mark.parametrize(
    ("kind", "model_key", "rows"),
    [
        ("target", "HYV4ForCausalLM", [[0, 1, 2, 1], [2, 0, 1, 2]]),
        ("mtp", "HYV4MTP", [[2, 0, 1, 2]]),
    ],
)
def test_hy_v4_static_state_requires_and_uses_distinct_bound_plan(
    tmp_path,
    kind: str,
    model_key: str,
    rows: list[list[int]],
) -> None:
    outer, inner = _offline_owner(kind)
    _install_layer_states(outer)
    layers = len(rows)
    load = torch.zeros((layers, 4), dtype=torch.int32)
    logical = torch.full((layers, 3, 3), -1, dtype=torch.int64)
    counts = torch.zeros((layers, 3), dtype=torch.int64)

    with pytest.raises((ValueError, NotImplementedError), match="plan"):
        outer.set_eplb_state(load, logical, counts)

    config, _ = config_and_map(tmp_path, key=model_key, rows=rows)
    plan = bind_static_eplb_plan(config, outer)
    assert plan is not None
    assert plan.model_key == model_key

    outer.set_eplb_state(load, logical, counts)
    state_owner = inner if kind == "target" else outer
    assert len(state_owner.expert_weights) == layers
    assert (
        state_owner.moe_layers[-1].eplb_state.logical_to_physical_map.data_ptr()
        == logical[-1].data_ptr()
    )
    outer.update_physical_experts_metadata(4, 2)
    with pytest.raises(ValueError, match="cannot change"):
        outer.update_physical_experts_metadata(8, 4)


def test_hy_v4_record_state_uses_runtime_metadata_without_static_plan() -> None:
    outer, inner = _offline_owner("target", mode="record")
    _install_layer_states(outer)
    load = torch.zeros((2, 4), dtype=torch.int32)
    logical = torch.full((2, 3, 3), -1, dtype=torch.int64)
    counts = torch.zeros((2, 3), dtype=torch.int64)

    outer.set_eplb_state(load, logical, counts)

    assert len(inner.expert_weights) == 2
    assert not hasattr(outer, "_vllm_hcu_static_eplb_plan")

    with pytest.raises(ValueError, match="state"):
        outer.set_eplb_state(
            torch.zeros((2, 3), dtype=torch.int32),
            logical,
            counts,
        )


def test_hy_v4_static_state_rejects_changed_bound_layer_row(tmp_path) -> None:
    outer, _ = _offline_owner("target")
    _install_layer_states(outer)
    rows = [[0, 1, 2, 1], [2, 0, 1, 2]]
    config, _ = config_and_map(
        tmp_path,
        key="HYV4ForCausalLM",
        rows=rows,
    )
    bind_static_eplb_plan(config, outer)
    outer.moe_layers[0].routed_experts._vllm_hcu_static_eplb_row = (
        1,
        0,
        2,
        1,
    )

    with pytest.raises(ValueError, match="layer|row"):
        outer.set_eplb_state(
            torch.zeros((2, 4), dtype=torch.int32),
            torch.full((2, 3, 3), -1, dtype=torch.int64),
            torch.zeros((2, 3), dtype=torch.int64),
        )


def test_hy_v4_record_fused_loader_initializes_redundant_experts() -> None:
    calls: list[tuple[int, float]] = []
    parameter = nn.Parameter(torch.empty(1))

    def weight_loader(
        param,
        loaded_weight,
        name,
        shard_id,
        expert_id,
        return_success,
    ) -> bool:
        del param, name, shard_id
        assert return_success is True
        calls.append((expert_id, loaded_weight.item()))
        return True

    parameter.weight_loader = weight_loader
    model = object.__new__(hy_v4_model.HYV4Model)
    nn.Module.__init__(model)
    model._vllm_hcu_offline_eplb_mode = "record"

    assert model.load_fused_expert_weights(
        "experts.w13_weight",
        {"experts.w13_weight": parameter},
        torch.tensor([[10.0], [20.0], [30.0], [40.0]]),
        "w1",
        num_experts=4,
        num_redundant_experts=2,
    )
    assert calls == [
        (0, 10.0),
        (1, 20.0),
        (2, 30.0),
        (3, 40.0),
        (4, 10.0),
        (5, 20.0),
    ]


def test_hy_v4_record_ledger_requires_local_redundant_physical_slot() -> None:
    generic = GenericMoE(rank=1)
    model = nn.Module()
    model.config = SimpleNamespace(num_experts=3)
    model.num_physical_experts = 4
    model.layers = nn.ModuleList()
    layer = nn.Module()
    layer.mlp = nn.Module()
    layer.mlp.experts = generic.moe_layers[0]
    model.layers.append(layer)

    ledger = _HYV4CheckpointAccounting(model)

    name = "layers.0.mlp.experts.routed_experts.w13_weight"
    assert ledger.expected[name] == {
        (2, "w1"),
        (2, "w3"),
        (3, "w1"),
        (3, "w3"),
    }


def _checkpoint_owner(kind, checkpoint_model, monkeypatch, rank):
    generic = GenericMoE(rank)
    if kind == "target":
        model = checkpoint_model
        inner = model.model
        inner.layers = nn.ModuleList()
        for experts in generic.moe_layers:
            layer = nn.Module()
            layer.mlp = nn.Module()
            layer.mlp.experts = experts
            inner.layers.append(layer)
        rows = [[0, 1, 2, 1], [2, 0, 1, 2]]
        del inner.get_expert_mapping
        owners = (model, inner)
        layers = generic.moe_layers
    else:
        model = _minimal_draft(_mtp(), monkeypatch)
        mlp = model.model.layers["2"].mtp_block.mlp
        del mlp.gate_up_proj
        mlp.experts = generic.moe_layers[1]
        del model.get_expert_mapping
        layers = nn.ModuleList([generic.moe_layers[1]])
        rows = [[2, 0, 1, 2]]
        owners = (model, model.model)
    for owner in owners:
        _copy_generic_metadata(owner, generic)
        owner.num_moe_layers = len(layers)
        owner.moe_layers = layers if owner is owners[-1] else list(layers)
        owner._vllm_hcu_offline_eplb_mode = "static"
        if not hasattr(owner, "config"):
            owner.config = model.config
        owner.config.num_experts = 3
    return model, rows


def _checkpoint_weights(model, *, fused: bool):
    weights = [
        (name, torch.zeros_like(param))
        for name, param in model.named_parameters()
        if ".experts." not in name
    ]
    for name, param in model.named_parameters():
        del param
        if ".experts." not in name:
            continue
        base, leaf = name.split(".experts.routed_experts.")
        suffix = "_scale" if "scale" in leaf else ""
        columns = 1 if suffix else 2
        if fused:
            projection = (
                "gate_up_proj" if leaf.startswith("w13") else "down_proj"
            )
            projection_rows = 4 if projection == "gate_up_proj" else 2
            weights.append(
                (
                    base + ".experts." + projection + suffix,
                    torch.stack(
                        [
                            torch.full(
                                (projection_rows, columns),
                                logical + 1.0,
                            )
                            for logical in range(3)
                        ]
                    ),
                )
            )
        else:
            projections = (
                ("gate_proj", "up_proj")
                if leaf.startswith("w13")
                else ("down_proj",)
            )
            for projection in projections:
                for logical in range(3):
                    weights.append(
                        (
                            f"{base}.experts.{logical}.{projection}.weight"
                            f"{suffix}",
                            torch.full((2, columns), logical + 1.0),
                        )
                    )
    if type(model).__name__ == "HYV4MTP":
        weights = [
            (name.replace(".mtp_block.", "."), value)
            for name, value in weights
        ]
    return weights


@pytest.mark.parametrize("kind", ["target", "mtp"])
@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("rank", [0, 1])
def test_hy_v4_static_loader_keeps_complete_replicated_checkpoint_ledger(
    tmp_path,
    checkpoint_model,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    fused: bool,
    rank: int,
) -> None:
    model, rows = _checkpoint_owner(
        kind,
        checkpoint_model,
        monkeypatch,
        rank,
    )
    config, _ = config_and_map(
        tmp_path,
        key=type(model).__name__,
        rows=rows,
    )
    bind_static_eplb_plan(config, model)

    loaded = model.load_weights(iter(_checkpoint_weights(model, fused=fused)))

    assert loaded == set(dict(model.named_parameters()))
    for index, layer in enumerate(model.moe_layers):
        expected = [
            logical + 1.0
            for logical in rows[index][rank * 2 : rank * 2 + 2]
        ]
        expert_owner = layer.routed_experts
        for name in (
            "w13_weight",
            "w2_weight",
            "w13_weight_scale",
            "w2_weight_scale",
        ):
            assert getattr(expert_owner, name)[:, 0, 0].tolist() == expected


@pytest.mark.parametrize("kind", ["target", "mtp"])
def test_hy_v4_static_binding_preserves_router_bias_checkpoint_owner(
    tmp_path,
    checkpoint_model,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    model, rows = _checkpoint_owner(
        kind,
        checkpoint_model,
        monkeypatch,
        rank=0,
    )
    mlps = (
        [layer.mlp for layer in model.model.layers]
        if kind == "target"
        else [
            layer.mtp_block.mlp
            for layer in model.model.layers.values()
        ]
    )
    for mlp in mlps:
        mlp.expert_bias = nn.Parameter(torch.tensor([2.0, 3.0, 5.0]))
        mlp.experts.routed_experts.e_score_correction_bias = mlp.expert_bias
    before = dict(model.named_parameters())
    config, _ = config_and_map(
        tmp_path,
        key=type(model).__name__,
        rows=rows,
    )

    bind_static_eplb_plan(config, model)

    assert dict(model.named_parameters()).keys() == before.keys()
    ledger_owner = model.model if kind == "target" else model
    ledger = _HYV4CheckpointAccounting(ledger_owner)
    bias_names = [
        name for name in ledger.expected if name.endswith("expert_bias")
    ]
    assert len(bias_names) == len(mlps)
    assert all(ledger.expected[name] == {None} for name in bias_names)
    assert not any(
        "e_score_correction_bias" in name for name in ledger.expected
    )
    for mlp in mlps:
        assert (
            mlp.expert_bias
            is mlp.experts.routed_experts.e_score_correction_bias
        )
