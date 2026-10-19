# SPDX-License-Identifier: Apache-2.0

from types import ModuleType, SimpleNamespace

import pytest
import torch

from vllm_hcu.patch.worker.op_opt import patch_kimi_k3_eplb_model as patch


def test_kimi_model_exposes_moe_protocol_and_current_weight_views(monkeypatch):
    class Runner:
        def __init__(self):
            self.routed_experts = SimpleNamespace(
                moe_config=SimpleNamespace(num_experts=10),
                local_num_experts=5,
            )
            self.weights = [torch.arange(6).view(2, 3)]
            self.state = None
            self.updated = 0

        def get_expert_weights(self):
            return self.weights

        def set_eplb_state(self, **kwargs):
            self.state = kwargs

        def update_expert_map(self):
            self.updated += 1

    class KimiMoE:
        def __init__(self, config):
            self.config = config

    class Model:
        def __init__(self):
            self.start_layer = 0
            self.end_layer = 2
            self.layers = [
                SimpleNamespace(mlp=SimpleNamespace(experts=Runner())),
                SimpleNamespace(mlp=KimiMoE(SimpleNamespace())),
            ]
            self.layers[1].mlp.experts = Runner()

        def load_weights(self, weights):
            return {entry[0] for entry in weights}

    class KimiLinearForCausalLM:
        def __init__(self):
            self.model = Model()
            self.config = SimpleNamespace(
                num_experts=10,
                num_expert_group=2,
                num_shared_experts=1,
            )

    target = ModuleType(patch.TARGET_MODULE)
    target.KimiLinearForCausalLM = KimiLinearForCausalLM
    target.KimiMoE = KimiMoE
    target.KimiLinearModel = Model
    monkeypatch.setitem(__import__("sys").modules, patch.TARGET_MODULE, target)

    assert patch.apply_to_module(target)
    model = KimiLinearForCausalLM()
    assert patch.apply_to_module(target) is False
    assert model.num_moe_layers == 1
    assert model.num_logical_experts == 10
    assert model.num_physical_experts == 10
    assert model.num_local_physical_experts == 5
    assert model.num_redundant_experts == 0
    assert model.expert_weights == []
    from vllm.model_executor.models.interfaces import MixtureOfExperts

    assert isinstance(model, MixtureOfExperts)

    load = torch.zeros((1, 1, 1))
    mapping = torch.zeros((1, 1), dtype=torch.int32)
    replicas = torch.ones((1, 1), dtype=torch.int32)
    model.set_eplb_state(load, mapping, replicas)
    runner = model.moe_layers[0]
    assert model.expert_weights == [runner.weights]
    assert runner.state["moe_layer_idx"] == 0
    assert runner.state["expert_load_view"] is load

    model.update_physical_experts_metadata(10, 5)
    assert runner.updated == 1


def test_kimi_eplb_protocol_patch_is_in_worker_callback_inventory():
    from vllm_hcu.patch.worker import worker_callback_names

    callbacks = dict(worker_callback_names())
    assert callbacks[patch.PATCH_ID] == patch.TARGET_MODULE


# These fixtures bypass accelerator allocation only. Mapping construction,
# RoutedExperts.load_weights/weight_loader, TP slicing, and the Kimi inline
# loader below are the installed vLLM implementations.
def _cpu_expert_model(ep_size, ep_rank, *, packed=False, redundant=2, static=False):
    from types import MethodType

    from torch import nn
    from vllm.model_executor.layers.fused_moe.expert_map_manager import (
        determine_expert_map,
    )
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
    from vllm.models.kimi_k3.amd.linear import KimiLinearModel
    from vllm_hcu.model_executor.layers.fused_moe import static_eplb

    logical = max(4, ep_size)
    physical = logical + redundant
    local, expert_map, _ = determine_expert_map(ep_size, ep_rank, physical)
    owner = RoutedExperts.__new__(RoutedExperts)
    nn.Module.__init__(owner)
    owner.layer_name = "model.layers.0.mlp.experts"
    # KimiMoE does not override the factory's default checkpoint names.
    owner.ckpt_gate_proj_name = "gate_proj"
    owner.ckpt_down_proj_name = "down_proj"
    owner.ckpt_up_proj_name = "up_proj"
    owner.lora_base_layer_prefix = ""
    owner.quant_config = None
    owner.quant_method = SimpleNamespace()
    owner.is_fused_checkpoint_transposed = False
    owner.moe_config = SimpleNamespace(
        num_experts=physical,
        num_logical_experts=logical,
        tp_rank=0,
        tp_size=1,
        is_act_and_mul=True,
        moe_parallel_config=SimpleNamespace(tp_size=1),
    )
    owner.expert_map_manager = SimpleNamespace(
        num_fused_shared_experts=0,
        map_global_to_local=lambda physical_id: (
            physical_id if expert_map is None else int(expert_map[physical_id])
        ),
    )
    weight_loader = owner.weight_loader
    row = tuple(range(logical)) + tuple(i % logical for i in range(redundant))
    if static:
        row = tuple(reversed(row))
        owner._vllm_hcu_static_eplb_row = row
        owner.get_expert_mapping = MethodType(
            static_eplb._static_expert_mapping, owner
        )
        weight_loader = MethodType(
            static_eplb._static_loader(RoutedExperts.weight_loader), owner
        )

    dtype = torch.int8 if packed else torch.float32
    for name, shape in (
        ("w13_weight", (local, 4, 2)), ("w2_weight", (local, 2, 2)),
        ("w13_weight_scale", (local, 4, 1)), ("w2_weight_scale", (local, 2, 1)),
    ):
        scale = "scale" in name
        param = nn.Parameter(
            torch.full(shape, -77, dtype=torch.float32 if scale else dtype),
            requires_grad=False,
        )
        param.weight_loader = weight_loader
        if scale:
            param.quant_method = "channel"
        owner.register_parameter(name, param)

    class CPUModel(KimiLinearModel):
        pass

    model = CPUModel.__new__(CPUModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        linear_attn_config=None,
        q_lora_rank=None,
        is_moe=True,
        num_experts=logical,
        num_hidden_layers=1,
        num_nextn_predict_layers=0,
    )
    model.layers = nn.ModuleList([nn.Module()])
    model.layers[0].mlp = nn.Module()
    model.layers[0].mlp.experts = nn.Module()
    model.layers[0].mlp.experts.routed_experts = owner
    weights = []
    expected = {}
    for logical_id in range(logical):
        for shard_index, shard in enumerate(("w1", "w2", "w3"), start=1):
            value = logical_id * 3 + shard_index
            for suffix, shape, tensor_dtype in (
                ("weight_packed" if packed else "weight", (2, 2), dtype),
                ("weight_scale", (2, 1), torch.float32),
            ):
                tensor = torch.full(shape, value, dtype=tensor_dtype)
                weights.append(
                    (f"layers.0.mlp.experts.{logical_id}.{shard}.{suffix}", tensor)
                )
                kind = "scale" if suffix == "weight_scale" else "weight"
                expected[logical_id, shard, kind] = tensor
    return model, owner, weights, expected, row, expert_map


def test_native_kimi_loader_leaves_redundant_experts_uninitialized():
    # The failing pre-fix behavior is retained as a reproducible CPU probe.
    model, owner, weights, _, _, _ = _cpu_expert_model(2, 1)
    model.load_weights(iter(weights))
    assert torch.all(owner.w13_weight[1:] == -77)
    assert torch.all(owner.w2_weight[1:] == -77)
    assert torch.all(owner.w13_weight_scale[1:] == -77)
    assert torch.all(owner.w2_weight_scale[1:] == -77)


@pytest.mark.parametrize("ep_size", [1, 2, 16])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("static", [False, True])
def test_kimi_redundant_experts_load_all_weights_and_scales(ep_size, packed, static):
    for ep_rank in range(ep_size):
        model, owner, weights, expected, row, expert_map = _cpu_expert_model(
            ep_size,
            ep_rank,
            packed=packed,
            redundant=ep_size,
            static=static,
        )
        patch._wrap_model_load_weights(type(model))
        loaded = model.load_weights(iter(weights))
        assert loaded == {
            f"layers.0.mlp.experts.routed_experts.{name}"
            for name in (
                "w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale"
            )
        }
        for physical_id, logical_id in enumerate(row):
            local_id = (
                physical_id if expert_map is None else int(expert_map[physical_id])
            )
            if local_id < 0:
                continue
            for kind in ("weight", "scale"):
                suffix = "_scale" if kind == "scale" else ""
                torch.testing.assert_close(
                    getattr(owner, "w13_weight" + suffix)[local_id],
                    torch.cat(
                        [expected[logical_id, shard, kind] for shard in ("w1", "w3")]
                    ),
                )
                torch.testing.assert_close(
                    getattr(owner, "w2_weight" + suffix)[local_id],
                    expected[logical_id, "w2", kind],
                )
            if not packed:
                # Explicitly route through each physical slot on CPU and compare
                # its output with the corresponding checkpoint logical expert.
                x = torch.tensor([0.01, -0.02])
                gate, up = owner.w13_weight[local_id].chunk(2)
                actual = owner.w2_weight[local_id] @ (
                    torch.nn.functional.silu(gate @ x) * (up @ x)
                )
                reference = expected[logical_id, "w2", "weight"] @ (
                    torch.nn.functional.silu(expected[logical_id, "w1", "weight"] @ x)
                    * (expected[logical_id, "w3", "weight"] @ x)
                )
                torch.testing.assert_close(actual, reference)


def test_kimi_without_redundancy_preserves_native_loader():
    model, owner, weights, _, _, _ = _cpu_expert_model(2, 1, redundant=0)
    expected_loaded = model.load_weights(iter(weights))
    expected = {name: value.detach().clone() for name, value in owner.named_parameters()}
    patch._wrap_model_load_weights(type(model))
    assert model.load_weights(iter(weights)) == expected_loaded
    for name, value in owner.named_parameters():
        torch.testing.assert_close(value, expected[name])


def test_kimi_redundant_loader_preserves_other_weights_and_pp_missing_layer_skip():
    from torch import nn
    from vllm.model_executor.models.utils import PPMissingLayer

    model, _, weights, _, _, _ = _cpu_expert_model(2, 1)
    # A PP stage retains the full layer index space and marks nonlocal layers.
    # Merely omitting a module is not a supported native loader skip contract.
    model.config.num_hidden_layers = 4
    model.start_layer, model.end_layer = 0, 1
    model.layers.extend(PPMissingLayer() for _ in range(3))
    model.extra_weight = nn.Parameter(torch.full((2, 2), -77.0))
    patch._wrap_model_load_weights(type(model))
    weights += [
        ("extra_weight", torch.ones(2, 2), {}),
        ("layers.3.mlp.experts.0.w1.weight", torch.zeros(2, 2)),
    ]
    loaded = model.load_weights(iter(weights))
    assert "extra_weight" in loaded
    assert not any(name.startswith("layers.3.") for name in loaded)
    torch.testing.assert_close(model.extra_weight, torch.ones(2, 2))
