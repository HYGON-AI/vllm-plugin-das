# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from vllm.model_executor.models.interfaces import is_mixture_of_experts

from vllm_hcu.models.hy_v4 import model as hy_v4_model
from vllm_hcu.models.hy_v4 import mtp as hy_v4_mtp


class _FakeMoELayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.arange(8).reshape(2, 4).float())
        self.eplb_state = None

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
            eplb_config=SimpleNamespace(num_redundant_experts=2)
        ),
    )

    model = hy_v4_model.HYV4ForCausalLM(vllm_config=vllm_config)

    assert is_mixture_of_experts(model)
    model.update_physical_experts_metadata(8, 2)
    assert model.num_physical_experts == 8
    assert model.num_local_physical_experts == 2
    assert model.num_redundant_experts == 4


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
    )

    model = hy_v4_mtp.HYV4MTP(vllm_config=vllm_config)

    assert is_mixture_of_experts(model)
    model.update_physical_experts_metadata(8, 2)
    assert model.num_physical_experts == 8
    assert model.num_local_physical_experts == 2
    assert model.num_redundant_experts == 4


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
