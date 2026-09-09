# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

from importlib import import_module
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tests.models.static_eplb_test_utils import apply_real_moe_layer_patch
from vllm_hcu.model_executor.layers.fused_moe import static_eplb
from vllm_hcu.patch.worker.framework_opt import patch_llama4_static_eplb


class _RoutedExperts:
    def __init__(self, local_physical_ids: set[int]) -> None:
        self.local_physical_ids = local_physical_ids
        self.attempted_physical_ids: list[int] = []
        self.loaded: dict[int, torch.Tensor] = {}
        self.loaded_by_shard: dict[tuple[int, object], torch.Tensor] = {}
        # vLLM quant methods capture this standard bound callback while their
        # parameters are created.
        self.parameter_weight_loader = self.weight_loader

    def original_weight_loader(
        self,
        param,
        loaded_weight,
        weight_name,
        shard_id,
        expert_id,
        return_success=False,
    ):
        del param, weight_name
        self.attempted_physical_ids.append(expert_id)
        loaded = expert_id in self.local_physical_ids
        if loaded:
            self.loaded[expert_id] = loaded_weight.clone()
            self.loaded_by_shard[expert_id, shard_id] = loaded_weight.clone()
        return loaded if return_success else None

    def weight_loader(
        self,
        param,
        loaded_weight,
        weight_name,
        shard_id,
        expert_id,
        return_success=False,
    ):
        return static_eplb.load_static_logical_expert(
            self,
            type(self).original_weight_loader,
            param=param,
            loaded_weight=loaded_weight,
            weight_name=weight_name,
            shard_id=shard_id,
            logical_expert_id=expert_id,
            return_success=return_success,
        )


class _MoERunner:
    def __init__(self, routed_experts: object) -> None:
        self.routed_experts = routed_experts

    def get_expert_weights(self):
        return []

    def set_eplb_state(self, **kwargs) -> None:
        del kwargs


class _DeepseekV2Like(nn.Module):
    """Small structural fake for the audited DeepSeek V2 loader contract."""

    def __init__(self, runner: object) -> None:
        super().__init__()
        self.expert_weights = []
        self.num_moe_layers = 1
        self.num_expert_groups = 1
        self.num_logical_experts = 3
        self.num_physical_experts = 4
        self.num_local_physical_experts = 1
        self.num_routed_experts = 3
        self.num_shared_experts = 0
        self.num_redundant_experts = 1
        self.moe_layers = [runner]

    def set_eplb_state(self, *args) -> None:
        del args

    def update_physical_experts_metadata(self, *args) -> None:
        del args


def _plan() -> static_eplb.StaticEplbPlan:
    return static_eplb.StaticEplbPlan(
        model_key="_DeepseekV2Like",
        source_path="/tmp/map.json",
        source_sha256="a" * 64,
        _map_values=((2, 1, 0, 2),),
        num_logical_experts=3,
        num_physical_experts=4,
        num_redundant_experts=1,
    )


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            _vllm_hcu_expert_map_path="fake-map.json",
        )
    )


def test_patched_standard_mapping_feeds_logical_id_to_common_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    routed_experts = _RoutedExperts(local_physical_ids={3})
    model = _DeepseekV2Like(_MoERunner(routed_experts))
    plan = _plan()
    monkeypatch.setattr(
        static_eplb,
        "maybe_load_static_eplb_plan",
        lambda *args, **kwargs: plan,
    )

    assert static_eplb.bind_static_eplb_plan(_config(), model) is plan

    make_expert_mapping = apply_real_moe_layer_patch(monkeypatch)
    mapping = make_expert_mapping(
        model,
        ckpt_gate_proj_name="gate_proj",
        ckpt_down_proj_name="down_proj",
        ckpt_up_proj_name="up_proj",
        num_experts=3,
        num_redundant_experts=1,
    )
    assert sorted({entry[2] for entry in mapping}) == [0, 1, 2]
    parameter_name, checkpoint_prefix, expert_id, shard_id = next(
        entry
        for entry in mapping
        if entry[1] == "experts.2.down_proj."
    )

    # This mapping-to-loader handoff is shared by the audited DeepSeek V2/V4,
    # HY V3, GLM4 MoE, and ordinary MTP split-expert loaders.
    parameter = nn.Parameter(torch.empty(1))
    parameter.weight_loader = routed_experts.parameter_weight_loader
    checkpoint_expert = torch.tensor([20.0, 21.0])
    loaded = parameter.weight_loader(
        parameter,
        checkpoint_expert,
        f"model.layers.1.mlp.{parameter_name}weight",
        shard_id=shard_id,
        expert_id=expert_id,
        return_success=True,
    )

    assert checkpoint_prefix == "experts.2.down_proj."
    assert loaded is True
    assert routed_experts.attempted_physical_ids == [0, 3]
    assert set(routed_experts.loaded) == {3}
    torch.testing.assert_close(routed_experts.loaded[3], checkpoint_expert)


def test_custom_mega_moe_layer_is_rejected_before_checkpoint_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DeepseekV4MegaMoEExperts:
        @staticmethod
        def weight_loader(*args, **kwargs):
            del args, kwargs

    model = _DeepseekV2Like(DeepseekV4MegaMoEExperts())
    plan = _plan()
    monkeypatch.setattr(
        static_eplb,
        "maybe_load_static_eplb_plan",
        lambda *args, **kwargs: plan,
    )

    with pytest.raises(ValueError, match="routed_experts"):
        static_eplb.bind_static_eplb_plan(_config(), model)

    assert not hasattr(model, "_vllm_hcu_static_eplb_plan")


def test_real_llama4_fused_loader_direct_loads_configured_logical_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the audited vLLM all-expert 3-D checkpoint path."""

    llama4 = import_module(patch_llama4_static_eplb.TARGET_MODULE)
    llama4_model = llama4.Llama4Model
    original = llama4_model.load_moe_expert_weights
    monkeypatch.setattr(llama4_model, "load_moe_expert_weights", original)
    for attribute in (
        patch_llama4_static_eplb._MODULE_MARKER,
        "_vllm_hcu_original_load_moe_expert_weights",
    ):
        current = getattr(llama4, attribute, None)
        monkeypatch.setattr(llama4, attribute, current, raising=False)

    assert patch_llama4_static_eplb.apply_to_module(llama4) is True
    assert patch_llama4_static_eplb.apply_to_module(llama4) is False

    routed_experts = _RoutedExperts(local_physical_ids={0, 1, 2, 3})
    setattr(routed_experts, "_vllm_hcu_static_eplb_row", (2, 1, 0, 2))
    runner = SimpleNamespace(
        routed_experts=routed_experts,
        # This is the initial EPLB placement that the audited upstream method
        # uses to slice the all-expert tensor.  Static loading must ignore it.
        expert_map=torch.tensor([0, 1, 2, -1]),
    )
    model = SimpleNamespace(
        layers=[
            SimpleNamespace(
                feed_forward=SimpleNamespace(experts=runner),
            )
        ],
        named_modules=lambda: (),
    )

    parameter = nn.Parameter(torch.empty(1))
    parameter.weight_loader = routed_experts.parameter_weight_loader
    full_param_name = (
        "model.layers.0.feed_forward.experts.routed_experts.w2_weight"
    )
    checkpoint_name = "model.layers.0.feed_forward.experts.down_proj"
    checkpoint_weight = torch.arange(18, dtype=torch.float32).reshape(3, 2, 3)
    loaded_params: set[str] = set()
    mapping = [
        (
            "experts.routed_experts.w2_",
            "experts.0.down_proj.",
            0,
            "w2",
        )
    ]

    assert llama4_model.load_moe_expert_weights(
        model,
        checkpoint_name,
        checkpoint_weight,
        {full_param_name: parameter},
        loaded_params,
        mapping,
        fused=True,
    ) is True

    assert routed_experts.attempted_physical_ids == [2, 1, 0, 3]
    assert set(routed_experts.loaded) == {0, 1, 2, 3}
    expected = checkpoint_weight.transpose(-1, -2)
    for physical_id, logical_id in enumerate((2, 1, 0, 2)):
        torch.testing.assert_close(
            routed_experts.loaded[physical_id],
            expected[logical_id],
        )
    assert loaded_params == {full_param_name}

    routed_experts.attempted_physical_ids.clear()
    routed_experts.loaded_by_shard.clear()
    gate_up_param_name = (
        "model.layers.0.feed_forward.experts.routed_experts.w13_weight"
    )
    gate_up_parameter = nn.Parameter(torch.empty(1))
    gate_up_parameter.weight_loader = routed_experts.parameter_weight_loader
    gate_up_weight = torch.arange(24, dtype=torch.float32).reshape(3, 2, 4)
    gate_up_mapping = [
        (
            "experts.routed_experts.w13_",
            "experts.0.gate_up_proj.",
            0,
            "w1",
        ),
        (
            "experts.routed_experts.w13_",
            "experts.0.gate_up_proj.",
            0,
            "w3",
        ),
    ]
    assert llama4_model.load_moe_expert_weights(
        model,
        "model.layers.0.feed_forward.experts.gate_up_proj",
        gate_up_weight,
        {gate_up_param_name: gate_up_parameter},
        set(),
        gate_up_mapping,
        fused=True,
    ) is True
    gate_up_shards = gate_up_weight.transpose(-1, -2).chunk(2, dim=-2)
    for shard_id, shard in zip(("w1", "w3"), gate_up_shards, strict=True):
        for physical_id, logical_id in enumerate((2, 1, 0, 2)):
            torch.testing.assert_close(
                routed_experts.loaded_by_shard[physical_id, shard_id],
                shard[logical_id],
            )

    routed_experts.attempted_physical_ids.clear()
    with pytest.raises(ValueError, match="expected 3 rows"):
        llama4_model.load_moe_expert_weights(
            model,
            checkpoint_name,
            checkpoint_weight[:2],
            {full_param_name: parameter},
            set(),
            mapping,
            fused=True,
        )
    assert routed_experts.attempted_physical_ids == []
