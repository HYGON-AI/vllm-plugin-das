# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.model_executor.models.interfaces import is_mixture_of_experts

from vllm_hcu.models.hy_v4 import model as hy_v4_model
from vllm_hcu.models.hy_v4 import mtp as hy_v4_mtp
from vllm_hcu.model_executor.layers.fused_moe import static_eplb


class _FakeMoELayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.arange(8).reshape(2, 4).float())
        self.eplb_state = None
        self.routed_experts = SimpleNamespace()

    def get_expert_weights(self):
        return [self.weight]

    def set_eplb_state(self, **kwargs) -> None:
        self.eplb_state = kwargs


def _set_moe_metadata(module: nn.Module, layer: _FakeMoELayer) -> None:
    module.expert_weights = []
    module.num_moe_layers = 1
    module.num_expert_groups = 1
    module.num_logical_experts = 4
    module.num_physical_experts = 6
    module.num_local_physical_experts = 2
    module.num_routed_experts = 4
    module.num_shared_experts = 1
    module.num_redundant_experts = 2
    module.moe_layers = [layer]


def test_target_registers_expert_weights_and_routing_state() -> None:
    layer = _FakeMoELayer()
    model = object.__new__(hy_v4_model.HYV4Model)
    nn.Module.__init__(model)
    _set_moe_metadata(model, layer)
    expert_load = torch.zeros((1, 6), dtype=torch.int32)
    logical_to_physical = torch.tensor([[[0, 4], [1, 5], [2, -1], [3, -1]]])
    replica_count = torch.tensor([[2, 2, 1, 1]])

    model.set_eplb_state(
        expert_load,
        logical_to_physical,
        replica_count,
    )

    assert len(model.expert_weights) == 1
    assert model.expert_weights[0][0] is layer.weight
    assert layer.eplb_state == {
        "moe_layer_idx": 0,
        "expert_load_view": expert_load,
        "logical_to_physical_map": logical_to_physical,
        "logical_replica_count": replica_count,
    }


def test_target_wrapper_exposes_mixture_of_experts_contract(monkeypatch) -> None:
    layer = _FakeMoELayer()
    static_plan = _static_plan(((3, 2, 1, 0, 3, 2),))

    class FakeInnerModel(nn.Module):
        def __init__(self, **kwargs) -> None:
            super().__init__()
            self.make_empty_intermediate_tensors = object()
            _set_moe_metadata(self, layer)

        def set_eplb_state(self, *args) -> None:
            hy_v4_model.HYV4Model.set_eplb_state(self, *args)

        def update_physical_experts_metadata(
            self,
            num_physical_experts: int,
            num_local_physical_experts: int,
        ) -> None:
            self.num_physical_experts = num_physical_experts
            self.num_local_physical_experts = num_local_physical_experts
            self.num_redundant_experts = (
                num_physical_experts - self.num_logical_experts
            )

    monkeypatch.setattr(hy_v4_model, "HYV4Model", FakeInnerModel)
    monkeypatch.setattr(
        hy_v4_model,
        "get_pp_group",
        lambda: SimpleNamespace(is_last_rank=False),
    )
    config = SimpleNamespace(
        vocab_size=64,
        hidden_size=32,
        tie_word_embeddings=False,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=config),
        quant_config=None,
        parallel_config=SimpleNamespace(
            eplb_config=SimpleNamespace(num_redundant_experts=2),
            _vllm_hcu_expert_map_path="fake-map.json",
        ),
    )
    monkeypatch.setattr(
        static_eplb,
        "maybe_load_static_eplb_plan",
        lambda *args, **kwargs: static_plan,
    )

    model = hy_v4_model.HYV4ForCausalLM(vllm_config=vllm_config)
    assert not hasattr(model, "_vllm_hcu_static_eplb_plan")

    assert static_eplb.bind_static_eplb_plan(vllm_config, model) is static_plan

    assert is_mixture_of_experts(model)
    model.update_physical_experts_metadata(8, 2)
    assert model.num_physical_experts == 8
    assert model.num_local_physical_experts == 2
    assert model.num_redundant_experts == 4
    assert model._vllm_hcu_static_eplb_plan is static_plan
    assert model.model._vllm_hcu_static_eplb_plan is static_plan


def test_mtp_registers_expert_weights_and_routing_state() -> None:
    layer = _FakeMoELayer()
    predictor = object.__new__(hy_v4_mtp.HYV4MultiTokenPredictor)
    nn.Module.__init__(predictor)
    _set_moe_metadata(predictor, layer)
    expert_load = torch.zeros((1, 6), dtype=torch.int32)
    logical_to_physical = torch.tensor([[[0, 4], [1, 5], [2, -1], [3, -1]]])
    replica_count = torch.tensor([[2, 2, 1, 1]])

    predictor.set_eplb_state(
        expert_load,
        logical_to_physical,
        replica_count,
    )

    assert len(predictor.expert_weights) == 1
    assert predictor.expert_weights[0][0] is layer.weight
    assert layer.eplb_state is not None
    assert layer.eplb_state["moe_layer_idx"] == 0


def test_mtp_wrapper_exposes_mixture_of_experts_contract(monkeypatch) -> None:
    layer = _FakeMoELayer()
    static_plan = _static_plan(((3, 2, 1, 0, 3, 2),))

    class FakePredictor(nn.Module):
        def __init__(self, **kwargs) -> None:
            super().__init__()
            self.quant_config = None
            _set_moe_metadata(self, layer)

        def set_eplb_state(self, *args) -> None:
            hy_v4_mtp.HYV4MultiTokenPredictor.set_eplb_state(self, *args)

        def update_physical_experts_metadata(
            self,
            num_physical_experts: int,
            num_local_physical_experts: int,
        ) -> None:
            self.num_physical_experts = num_physical_experts
            self.num_local_physical_experts = num_local_physical_experts
            self.num_redundant_experts = (
                num_physical_experts - self.num_logical_experts
            )

    monkeypatch.setattr(hy_v4_mtp, "HYV4MultiTokenPredictor", FakePredictor)
    config = SimpleNamespace()
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=config),
        parallel_config=SimpleNamespace(
            _vllm_hcu_expert_map_path="fake-map.json",
        ),
    )
    monkeypatch.setattr(
        static_eplb,
        "maybe_load_static_eplb_plan",
        lambda *args, **kwargs: static_plan,
    )

    model = hy_v4_mtp.HYV4MTP(vllm_config=vllm_config)
    assert not hasattr(model, "_vllm_hcu_static_eplb_plan")

    assert static_eplb.bind_static_eplb_plan(vllm_config, model) is static_plan

    assert is_mixture_of_experts(model)
    model.update_physical_experts_metadata(8, 2)
    assert model.num_physical_experts == 8
    assert model.num_local_physical_experts == 2
    assert model.num_redundant_experts == 4
    assert model._vllm_hcu_static_eplb_plan is static_plan
    assert model.model._vllm_hcu_static_eplb_plan is static_plan


def test_target_fused_loader_copies_logical_weights_to_redundant_experts() -> None:
    calls: list[tuple[int, float]] = []
    param = nn.Parameter(torch.empty(1))

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

    param.weight_loader = weight_loader
    checkpoint = torch.tensor([[10.0], [20.0], [30.0], [40.0]])
    model = object.__new__(hy_v4_model.HYV4Model)
    nn.Module.__init__(model)

    loaded = model.load_fused_expert_weights(
        "experts.w13_weight",
        {"experts.w13_weight": param},
        checkpoint,
        "w1",
        num_experts=4,
        num_redundant_experts=2,
    )

    assert loaded is True
    assert calls == [
        (0, 10.0),
        (1, 20.0),
        (2, 30.0),
        (3, 40.0),
        (4, 10.0),
        (5, 20.0),
    ]


def test_mtp_fused_loader_copies_logical_weights_to_redundant_experts() -> None:
    calls: list[tuple[int, float]] = []
    param = nn.Parameter(torch.empty(1))

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

    param.weight_loader = weight_loader
    checkpoint = torch.tensor([[10.0], [20.0], [30.0], [40.0]])
    model = object.__new__(hy_v4_mtp.HYV4MTP)
    nn.Module.__init__(model)

    loaded = model._load_fused_expert_weights(
        "experts.w13_weight",
        {"experts.w13_weight": param},
        checkpoint,
        "w1",
        num_experts=4,
        num_redundant_experts=2,
    )

    assert loaded is True
    assert calls == [
        (0, 10.0),
        (1, 20.0),
        (2, 30.0),
        (3, 40.0),
        (4, 10.0),
        (5, 20.0),
    ]


def _static_plan(rows: tuple[tuple[int, ...], ...]) -> static_eplb.StaticEplbPlan:
    return static_eplb.StaticEplbPlan(
        model_key="HYV4ForCausalLM",
        source_path="/tmp/map.json",
        source_sha256="a" * 64,
        _map_values=rows,
        num_logical_experts=4,
        num_physical_experts=6,
        num_redundant_experts=2,
    )


@pytest.mark.parametrize("owner", ["target", "mtp"])
@pytest.mark.parametrize(
    ("parameter_name", "checkpoint"),
    [
        (
            "experts.w13_weight",
            torch.arange(4 * 2 * 3, dtype=torch.float32).reshape(4, 2, 3),
        ),
        (
            "experts.w13_weight_scale",
            torch.arange(4 * 2, dtype=torch.float32).reshape(4, 2, 1),
        ),
        ("experts.w13_input_scale", torch.arange(4, dtype=torch.float32)),
    ],
)
def test_static_fused_loader_delegates_logical_rows_to_common_fan_out(
    owner: str,
    parameter_name: str,
    checkpoint: torch.Tensor,
) -> None:
    row = (3, 2, 1, 0, 3, 2)
    destination = torch.full(
        (len(row), *checkpoint.shape[1:]),
        float("nan"),
        dtype=checkpoint.dtype,
    )

    class RoutedExperts:
        _vllm_hcu_static_eplb_row = row

        @staticmethod
        def original_weight_loader(
            routed_experts,
            param,
            loaded_weight,
            weight_name,
            shard_id,
            expert_id,
            return_success=False,
        ):
            del routed_experts, param, weight_name, shard_id
            destination[expert_id].copy_(loaded_weight)
            return True if return_success else None

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
                self.original_weight_loader,
                param=param,
                loaded_weight=loaded_weight,
                weight_name=weight_name,
                shard_id=shard_id,
                logical_expert_id=expert_id,
                return_success=return_success,
            )

    routed_experts = RoutedExperts()
    param = nn.Parameter(torch.empty(1))
    param.weight_loader = routed_experts.weight_loader
    model_class = (
        hy_v4_model.HYV4Model if owner == "target" else hy_v4_mtp.HYV4MTP
    )
    model = object.__new__(model_class)
    nn.Module.__init__(model)
    loader = (
        model.load_fused_expert_weights
        if owner == "target"
        else model._load_fused_expert_weights
    )

    loaded = loader(
        parameter_name,
        {parameter_name: param},
        checkpoint,
        "w1",
        num_experts=4,
        num_redundant_experts=2,
    )

    assert loaded is True
    for physical_expert_id, logical_expert_id in enumerate(row):
        torch.testing.assert_close(
            destination[physical_expert_id],
            checkpoint[logical_expert_id],
            rtol=0,
            atol=0,
        )


def test_direct_load_is_tensor_equivalent_to_legacy_rearrangement() -> None:
    checkpoint = torch.arange(4 * 3, dtype=torch.float32).reshape(4, 3)
    source_map = (0, 1, 2, 3, 0, 1)
    target_map = (3, 2, 1, 0, 3, 2)
    legacy_loaded = torch.stack([checkpoint[logical] for logical in source_map])
    legacy_rearranged = torch.stack(
        [
            legacy_loaded[source_map.index(logical)]
            for logical in target_map
        ]
    )
    direct_loaded = torch.stack([checkpoint[logical] for logical in target_map])

    torch.testing.assert_close(direct_loaded, legacy_rearranged, rtol=0, atol=0)
