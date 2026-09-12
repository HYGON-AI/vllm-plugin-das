from types import SimpleNamespace

import pytest
import torch

from tests.models.static_eplb_test_utils import GenericMoE, config_and_map
from tests.models.hy_v4.test_weight_loading import checkpoint_model
from tests.models.hy_v4.test_mtp import _minimal_draft, _mtp
from vllm_hcu.models.hy_v4.model import HYV4Model, _HYV4CheckpointAccounting
from vllm_hcu.model_executor.layers.fused_moe.static_eplb import bind_static_eplb_plan


def build_model(kind, checkpoint_model, monkeypatch, rank):
    generic = GenericMoE(rank)
    if kind == "target":
        model = checkpoint_model
        inner = model.model
        inner.layers = torch.nn.ModuleList()
        for experts in generic.moe_layers:
            layer = torch.nn.Module()
            layer.mlp = torch.nn.Module()
            layer.mlp.experts = experts
            inner.layers.append(layer)
        rows = [[0, 1, 2, 1], [2, 0, 1, 2]]
        del inner.get_expert_mapping
        owners = [model, inner]
        layers = generic.moe_layers
    else:
        model = _minimal_draft(_mtp(), monkeypatch)
        mlp = model.model.layers["2"].mtp_block.mlp
        del mlp.gate_up_proj
        mlp.experts = generic.moe_layers[1]
        del model.get_expert_mapping
        layers = torch.nn.ModuleList([generic.moe_layers[1]])
        rows = [[2, 0, 1, 2]]
        owners = [model, model.model]
    for owner in owners:
        for name in ("num_logical_experts", "num_routed_experts", "num_physical_experts",
                     "num_local_physical_experts", "num_redundant_experts", "num_expert_groups", "num_shared_experts"):
            setattr(owner, name, getattr(generic, name))
        owner.num_moe_layers = len(layers)
        owner.moe_layers = layers
        owner.expert_weights = []
        if kind == "mtp":
            owner.config = model.config
        owner.config.num_experts = 3
    return model, rows


def weights_for(model, fused):
    weights = [(name, torch.zeros_like(param)) for name, param in model.named_parameters()
               if ".experts." not in name]
    for name, param in model.named_parameters():
        if ".experts." not in name:
            continue
        base, leaf = name.split(".experts.routed_experts.")
        suffix = "_scale" if "scale" in leaf else ""
        columns = 1 if suffix else 2
        if fused:
            projection = "gate_up_proj" if leaf.startswith("w13") else "down_proj"
            rows = 4 if projection == "gate_up_proj" else 2
            weights.append((base + ".experts." + projection + suffix,
                            torch.stack([torch.full((rows, columns), i + 1.) for i in range(3)])))
        else:
            projections = ("gate_proj", "up_proj") if leaf.startswith("w13") else ("down_proj",)
            for projection in projections:
                for logical in range(3):
                    weights.append((f"{base}.experts.{logical}.{projection}.weight{suffix}",
                                    torch.full((2, columns), logical + 1.)))
    if type(model).__name__ == "HYV4MTP":
        weights = [(name.replace(".mtp_block.", "."), value) for name, value in weights]
    return weights


@pytest.mark.parametrize("kind", ["target", "mtp"])
@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("rank", [0, 1])
def test_hyv4_static_load_keeps_complete_checkpoint_ledger(tmp_path, checkpoint_model, monkeypatch, kind, fused, rank):
    model, rows = build_model(kind, checkpoint_model, monkeypatch, rank)
    config, _ = config_and_map(tmp_path, key=type(model).__name__, rows=rows)
    bind_static_eplb_plan(config, model)
    loaded = model.load_weights(iter(weights_for(model, fused)))
    assert loaded == set(dict(model.named_parameters()))
    for index, layer in enumerate(model.moe_layers):
        values = [i + 1. for i in rows[index][rank * 2:rank * 2 + 2]]
        owner = layer.routed_experts
        for name in ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale"):
            assert getattr(owner, name)[:, 0, 0].tolist() == values


def test_hyv4_ledger_requires_replicated_logical_expert(tmp_path, checkpoint_model, monkeypatch):
    model, rows = build_model("target", checkpoint_model, monkeypatch, 1)
    config, _ = config_and_map(tmp_path, key=type(model).__name__, rows=rows)
    bind_static_eplb_plan(config, model)
    ledger = _HYV4CheckpointAccounting(model.model)
    name = "layers.1.mlp.experts.routed_experts.w13_weight"
    assert ledger.expected[name] == {(1, "w1"), (1, "w3"), (2, "w1"), (2, "w3")}


def test_hyv4_bound_state_delegates_to_current_protocol(tmp_path, checkpoint_model, monkeypatch):
    from vllm.distributed.eplb.eplb_state import EplbLayerState
    model, rows = build_model("target", checkpoint_model, monkeypatch, 0)
    config, _ = config_and_map(tmp_path, key=type(model).__name__, rows=rows)
    bind_static_eplb_plan(config, model)
    for layer in model.moe_layers:
        layer.eplb_state = EplbLayerState()
        layer.get_expert_weights = lambda layer=layer: list(layer.routed_experts.parameters())
        layer.set_eplb_state = layer.eplb_state.set_layer_state
    load = torch.zeros(2, 4, dtype=torch.int32)
    logical = torch.tensor([[[0, -1], [1, 3], [2, -1]], [[1, -1], [2, -1], [0, 3]]])
    counts = torch.tensor([[1, 2, 1], [1, 1, 2]])
    model.set_eplb_state(load, logical, counts)
    assert model.model.moe_layers[1].eplb_state.logical_to_physical_map.data_ptr() == logical[1].data_ptr()
    assert len(model.model.expert_weights) == 2
    model.update_physical_experts_metadata(4, 2)
    with pytest.raises(ValueError, match="static|Static"):
        model.update_physical_experts_metadata(8, 4)


def test_hyv4_static_constructor_reaches_current_moe_owner(monkeypatch):
    import vllm_hcu.models.hy_v4.moe as moe
    from tests.models.hy_v4.test_moe import _hf_config, _vllm_config, _FakeGate, _FakeExperts, _FakeSharedExperts
    config = _vllm_config("triton")
    config.parallel_config._vllm_hcu_expert_map_path = "/tmp/map.json"
    monkeypatch.setattr(moe, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(moe, "get_ep_group", lambda: SimpleNamespace(rank_in_group=0,
        device_group=SimpleNamespace(size=lambda: 2)))
    monkeypatch.setattr(moe, "GateLinear", lambda *args, **kwargs: _FakeGate())
    monkeypatch.setattr(moe, "HYV4FeedForward", lambda *args, **kwargs: _FakeSharedExperts())
    experts = _FakeExperts()
    monkeypatch.setattr(moe, "FusedMoE", lambda **kwargs: experts)
    layer = moe.HYV4MoEFused(_hf_config(), vllm_config=config, enable_eplb=True)
    assert layer.experts is experts and layer.enable_eplb
