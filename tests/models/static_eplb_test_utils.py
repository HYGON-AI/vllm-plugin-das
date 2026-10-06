# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Small CPU-only owners using the current RoutedExperts loader contract."""

from __future__ import annotations

import json
from types import SimpleNamespace

import torch
from vllm.model_executor.layers.fused_moe.expert_map_manager import (
    ExpertMapManager,
)
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.models.interfaces import MixtureOfExperts

from vllm_hcu.patch.config import HcuFeatureConfig


class GenericMoE(torch.nn.Module, MixtureOfExperts):
    def __init__(self, rank: int = 0) -> None:
        super().__init__()
        self.num_moe_layers = 2
        self.num_expert_groups = 1
        self.num_logical_experts = self.num_routed_experts = 3
        self.num_physical_experts = 4
        self.num_local_physical_experts = 2
        self.num_redundant_experts = 1
        self.num_shared_experts = 0
        self.expert_weights: list[torch.Tensor] = []
        self.moe_layers = torch.nn.ModuleList()
        for index in range(2):
            layer = torch.nn.Module()
            owner = RoutedExperts.__new__(RoutedExperts)
            torch.nn.Module.__init__(owner)
            owner.layer_name = f"layers.{index}.experts"
            owner.local_num_experts = 2
            owner.quant_config = None
            owner.quant_method = UnquantizedFusedMoEMethod.__new__(
                UnquantizedFusedMoEMethod
            )
            torch.nn.Module.__init__(owner.quant_method)
            owner.moe_config = SimpleNamespace(
                num_logical_experts=3,
                num_experts=4,
                tp_rank=0,
                tp_size=1,
                is_act_and_mul=True,
                hidden_dim_unpadded=2,
                moe_parallel_config=SimpleNamespace(tp_size=1),
            )
            owner.expert_map_manager = ExpertMapManager.__new__(ExpertMapManager)
            owner.expert_map_manager._expert_map = torch.tensor(
                [0, 1, -1, -1] if rank == 0 else [-1, -1, 0, 1]
            )
            owner.expert_map_manager.num_fused_shared_experts = 0
            owner.ckpt_gate_proj_name = "gate_proj"
            owner.ckpt_up_proj_name = "up_proj"
            owner.ckpt_down_proj_name = "down_proj"
            owner.lora_base_layer_prefix = ""
            shapes = {
                "w13_weight": (2, 4, 2),
                "w2_weight": (2, 2, 2),
                "w13_weight_scale": (2, 4, 1),
                "w2_weight_scale": (2, 2, 1),
            }
            for name, shape in shapes.items():
                parameter = torch.nn.Parameter(
                    torch.zeros(shape),
                    requires_grad=False,
                )
                parameter.weight_loader = owner.weight_loader
                if "scale" in name:
                    parameter.quant_method = "channel"
                owner.register_parameter(name, parameter)
            layer.routed_experts = owner
            self.moe_layers.append(layer)


def config_and_map(
    tmp_path,
    *,
    key: str = "GenericMoE",
    rows: list[list[int]] | None = None,
):
    path = tmp_path / "map.json"
    rows = [[0, 1, 2, 1], [2, 0, 1, 2]] if rows is None else rows
    path.write_text(
        json.dumps(
            {
                "version": 2,
                "format": "vllm_offline_eplb_physical_to_logical_by_model",
                "model_maps": {
                    key: {
                        "record_kind": "proposal",
                        "model_name": "generic-test",
                        "model_class": key,
                        "num_moe_layers": len(rows),
                        "num_logical_experts": 3,
                        "num_physical_experts": 4,
                        "num_redundant_experts": 1,
                        "physical_to_logical_map": rows,
                    }
                },
            }
        )
    )
    feature_config = HcuFeatureConfig(expert_map_path=str(path))
    parallel_config = SimpleNamespace(
        enable_expert_parallel=True,
        enable_eplb=True,
        enable_ep_weight_filter=False,
        pipeline_parallel_size=1,
        _vllm_hcu_expert_map_path=str(path),
    )
    return (
        SimpleNamespace(
            additional_config={"hcu": feature_config.to_dict()},
            parallel_config=parallel_config,
            load_config=SimpleNamespace(load_format="safetensors"),
        ),
        path,
    )
