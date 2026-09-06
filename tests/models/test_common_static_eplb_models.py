# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_hcu.model_executor.layers.fused_moe import static_eplb


class _RoutedExperts:
    def __init__(self, local_physical_ids: set[int]) -> None:
        self.local_physical_ids = local_physical_ids
        self.attempted_physical_ids: list[int] = []
        self.loaded: dict[int, torch.Tensor] = {}
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
        del param, weight_name, shard_id
        self.attempted_physical_ids.append(expert_id)
        loaded = expert_id in self.local_physical_ids
        if loaded:
            self.loaded[expert_id] = loaded_weight.clone()
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


def test_deepseek_style_standard_loader_fans_out_a_logical_mapping(
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

    # This is the terminal call shape shared by the audited DeepSeek V2/V4,
    # HY V3, GLM4 MoE, and ordinary MTP split-expert loaders.
    parameter = nn.Parameter(torch.empty(1))
    parameter.weight_loader = routed_experts.parameter_weight_loader
    checkpoint_expert = torch.tensor([20.0, 21.0])
    loaded = parameter.weight_loader(
        parameter,
        checkpoint_expert,
        "model.layers.1.mlp.experts.routed_experts.w2_weight",
        shard_id="w2",
        expert_id=2,
        return_success=True,
    )

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
