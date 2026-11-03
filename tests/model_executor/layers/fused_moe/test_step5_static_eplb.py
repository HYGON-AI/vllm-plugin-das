# SPDX-License-Identifier: Apache-2.0
"""CPU owner/loader contracts; no model construction or device kernels."""
from __future__ import annotations

import ast
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from tests.models.static_eplb_test_utils import GenericMoE, config_and_map
from vllm_hcu.model_executor.layers.fused_moe.static_eplb import (
    _resolve_static_eplb_owner,
    bind_static_eplb_plan,
    load_static_eplb_plan,
)


def _step5_block_type():
    # Exercise the actual small forwarding properties without importing the
    # complete Step5 model, which owns accelerator-specific implementations.
    source = Path(__file__).resolve().parents[4] / "vllm_hcu/models/step5_text.py"
    module = ast.parse(source.read_text())
    block = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == "FusedMoEBlock")
    properties = [n for n in block.body if isinstance(n, ast.FunctionDef) and n.name in {"routed_experts", "eplb_state"}]
    assert len(properties) == 2 and all(len(n.decorator_list) == 1 and isinstance(n.decorator_list[0], ast.Name) and n.decorator_list[0].id == "property" for n in properties)
    wrapper = ast.ClassDef(name="Step5Block", bases=[ast.Name(id="Module", ctx=ast.Load())], keywords=[], body=properties, decorator_list=[])
    if "type_params" in ast.ClassDef._fields:
        wrapper.type_params = []
    namespace = {"Module": torch.nn.Module}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[])), str(source), "exec"), namespace)
    return namespace["Step5Block"]


def test_step5_block_forwarding_is_read_only_and_preserves_identity():
    block = _step5_block_type()()
    owner, state = object(), object()
    block.experts = SimpleNamespace(routed_experts=owner, eplb_state=state)
    assert block.routed_experts is owner and block.eplb_state is state
    with pytest.raises(AttributeError):
        block.routed_experts = object()


@pytest.fixture
def step5_owner_types(monkeypatch):
    # Nominal imported identities are controlled here, not just class names.
    class Step5ForCausalLM(GenericMoE):
        pass

    class Step5ForConditionalGeneration(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.language_model = inner

    class Step5MTP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.is_dense_mtp = True
            self.num_moe_layers = self.num_logical_experts = 0
            self.num_physical_experts = self.num_redundant_experts = 0
            self.num_local_physical_experts = self.num_routed_experts = 0
            self.num_shared_experts = 0
            self.moe_layers = []

    for name, cls in [("step5_text", Step5ForCausalLM), ("step5_vl", Step5ForConditionalGeneration), ("step5_mtp", Step5MTP)]:
        module = ModuleType("vllm_hcu.models." + name)
        setattr(module, cls.__name__, cls)
        monkeypatch.setitem(sys.modules, module.__name__, module)
    import vllm.model_executor.models.interfaces as interfaces
    monkeypatch.setattr(interfaces, "get_mixture_of_experts_model", lambda model: getattr(model, "language_model", None))
    return Step5ForCausalLM, Step5ForConditionalGeneration, Step5MTP


def test_vl_static_preload_uses_canonical_inner_key(tmp_path, step5_owner_types):
    core, wrapper, _ = step5_owner_types
    model = core()
    config, _ = config_and_map(tmp_path, key="Step5ForCausalLM")
    plan = bind_static_eplb_plan(config, wrapper(model))
    assert plan.model_key == "Step5ForCausalLM"
    assert model._vllm_hcu_static_eplb_plan is plan
    assert _resolve_static_eplb_owner(model) is model


def test_only_actual_dense_step5_mtp_can_skip_map_binding(tmp_path, step5_owner_types):
    _, _, mtp = step5_owner_types
    config, path = config_and_map(tmp_path)
    config.parallel_config.enable_expert_parallel = False
    config.parallel_config.enable_eplb = False
    path.unlink()  # A dense drafter needs no map entry or map-file read.
    assert bind_static_eplb_plan(config, mtp()) is None
    impostor = type("Step5MTP", (torch.nn.Module,), {})()
    impostor.is_dense_mtp = True
    impostor.num_moe_layers = 0
    impostor.moe_layers = []
    with pytest.raises(ValueError, match="MixtureOfExperts"):
        _resolve_static_eplb_owner(impostor)


@pytest.mark.parametrize("flag", ["enable_eplb", "enable_expert_parallel"])
def test_nonempty_moe_still_requires_both_parallel_flags(tmp_path, flag):
    config, _ = config_and_map(tmp_path)
    setattr(config.parallel_config, flag, False)
    with pytest.raises(ValueError, match="requires EPLB and expert parallel"):
        bind_static_eplb_plan(config, GenericMoE())


@pytest.mark.parametrize("field,value", [("is_dense_mtp", False), ("num_moe_layers", False), ("num_physical_experts", 1), ("num_routed_experts", 1), ("moe_layers", [object()])])
def test_malformed_dense_step5_mtp_fails_closed(step5_owner_types, field, value):
    model = step5_owner_types[2]()
    setattr(model, field, value)
    with pytest.raises(ValueError, match="MixtureOfExperts"):
        _resolve_static_eplb_owner(model)


def test_foreign_vl_or_wrong_inner_is_not_silently_accepted(step5_owner_types):
    _, wrapper, _ = step5_owner_types
    with pytest.raises(ValueError, match="MixtureOfExperts"):
        _resolve_static_eplb_owner(wrapper(GenericMoE()))
    impostor = type("Step5ForConditionalGeneration", (torch.nn.Module,), {})()
    with pytest.raises(ValueError, match="MixtureOfExperts"):
        _resolve_static_eplb_owner(impostor)


def _e23_rows():
    # Post-placement logical IDs. Keep each rank's old22 and append one slot.
    return [[logical for rank in range(16) for logical in [*range(rank * 22, rank * 22 + 22), 22 if rank == 0 else 0]],
            [logical for rank in range(16) for logical in [*range(rank * 22, rank * 22 + 22), ((rank + 1) % 16) * 22]]]


def _write_e23_plan(path, rows, key):
    path.write_text(json.dumps({"version": 2, "format": "vllm_offline_eplb_physical_to_logical_by_model", "model_maps": {key: dict(record_kind="proposal", model_name="generic-test", model_class=key, num_moe_layers=len(rows), num_logical_experts=352, num_physical_experts=368, num_redundant_experts=16, physical_to_logical_map=rows)}}))


@pytest.mark.parametrize("rank", range(16))
@pytest.mark.parametrize("weight_dtype", [torch.float32, torch.int8])
def test_packed_logical_callbacks_fill_every_e23_slot_and_channel_scale(tmp_path, rank, weight_dtype):
    model = GenericMoE()
    model.num_logical_experts = model.num_routed_experts = 352
    model.num_physical_experts = 368
    model.num_local_physical_experts = 23
    model.num_redundant_experts = 16
    block_type = _step5_block_type()
    for index, layer in enumerate(tuple(model.moe_layers)):
        owner = layer.routed_experts
        owner.local_num_experts = 23
        owner.moe_config.num_experts = 368
        owner.moe_config.num_logical_experts = 352
        mapping = torch.full((368,), -1, dtype=torch.long)
        mapping[rank * 23:(rank + 1) * 23] = torch.arange(23)
        owner.expert_map_manager._expert_map = mapping
        for name, shape in {"w13_weight": (23, 4, 2), "w2_weight": (23, 2, 2), "w13_weight_scale": (23, 4, 1), "w2_weight_scale": (23, 2, 1)}.items():
            parameter = torch.nn.Parameter(torch.full(shape, -113, dtype=torch.float32 if "scale" in name else weight_dtype), requires_grad=False)
            parameter.weight_loader = owner.weight_loader
            if "scale" in name:
                parameter.quant_method = "channel"
            setattr(owner, name, parameter)
        block = block_type()
        block.experts = SimpleNamespace(routed_experts=owner, eplb_state=None)
        model.moe_layers[index] = block
    config, path = config_and_map(tmp_path)
    rows = _e23_rows()
    _write_e23_plan(path, rows, "GenericMoE")
    bind_static_eplb_plan(config, model)
    for layer_index, block in enumerate(model.moe_layers):
        owner = block.routed_experts
        gold = {}
        for logical in range(352):
            for projection, shard in enumerate(("w1", "w3", "w2")):
                values = torch.tensor([[logical % 127, logical // 127], [projection + 3, layer_index + 7]], dtype=weight_dtype)
                scale = torch.tensor([[logical + 0.25 + projection], [logical + 0.5 + layer_index]], dtype=torch.float32)
                weight_name = "w2_weight" if shard == "w2" else "w13_weight"
                scale_name = weight_name + "_scale"
                for name, value in [(weight_name, values), (scale_name, scale)]:
                    parameter = getattr(owner, name)
                    parameter.weight_loader(parameter, value, name, shard_id=shard, expert_id=logical)
                    gold[logical, shard, name] = value
        for local, logical in enumerate(rows[layer_index][rank * 23:(rank + 1) * 23]):
            for shard in ("w1", "w3", "w2"):
                for name in (("w2_weight", "w2_weight_scale") if shard == "w2" else ("w13_weight", "w13_weight_scale")):
                    value = getattr(owner, name)[local]
                    if shard != "w2":
                        value = value[:2] if shard == "w1" else value[2:]
                    assert torch.equal(value, gold[logical, shard, name]), (rank, layer_index, local, logical, shard, name)


def test_e23_plan_validates_all88_rows_and_rejects_missing_logical(tmp_path):
    path = tmp_path / "e23.json"
    rows = [_e23_rows()[index % 2] for index in range(88)]
    _write_e23_plan(path, rows, "Step5ForCausalLM")
    request = dict(model_key="Step5ForCausalLM", expected_model_name="generic-test", expected_shape=(88, 368), num_logical_experts=352, num_redundant_experts=16)
    plan = load_static_eplb_plan(path, **request)
    assert plan.physical_to_logical_map.shape == (88, 368)
    rows[0] = [0 if value == 351 else value for value in rows[0]]
    _write_e23_plan(path, rows, "Step5ForCausalLM")
    with pytest.raises(ValueError, match="misses logical"):
        load_static_eplb_plan(path, **request)
