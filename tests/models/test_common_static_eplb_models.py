import json
from types import SimpleNamespace

import pytest
import torch

from tests.models.static_eplb_test_utils import GenericMoE, config_and_map
from vllm_hcu.model_executor.layers.fused_moe import static_eplb as api


def bind(config, model):
    assert hasattr(api, "bind_static_eplb_plan"), "common static binding missing"
    return api.bind_static_eplb_plan(config, model)


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("kind", ["split", "fused"])
@pytest.mark.parametrize("scale", [False, True])
def test_two_layer_current_loader_loads_static_physical_rows(tmp_path, rank, kind, scale):
    model = GenericMoE(rank)
    config, path = config_and_map(tmp_path)
    methods = [layer.routed_experts.quant_method for layer in model.moe_layers]
    bind(config, model)
    for layer in model.moe_layers:
        owner = layer.routed_experts
        if kind == "fused" and scale:
            # HCU owns the channel-scale split; the same captured parameter
            # loaders consume each logical row, without transposing scales.
            for logical in range(3):
                for shard in ("w1", "w3"):
                    owner.w13_weight_scale.weight_loader(owner.w13_weight_scale,
                        torch.full((2, 1), logical + 1.), "w13_weight_scale", shard,
                        logical, return_success=True)
        elif kind == "fused":
            values = torch.stack([torch.full((4, 2), i + 1.) for i in range(3)])
            list(owner.load_weights(iter([("gate_up_proj", values)])))
        else:
            suffix = "weight_scale" if scale else "weight"
            values = [(f"{logical}.{projection}.{suffix}",
                       torch.full((2, 1 if scale else 2), logical + 1.))
                      for logical in range(3) for projection in ("gate_proj", "up_proj")]
            list(owner.load_weights(iter(values)))
    expected = ([[1, 2], [3, 1]] if rank == 0 else [[3, 2], [2, 3]])
    for index, layer in enumerate(model.moe_layers):
        owner = layer.routed_experts
        param = owner.w13_weight_scale if scale else owner.w13_weight
        assert param[:, 0, 0].tolist() == expected[index]
        assert param[:, 2, 0].tolist() == expected[index]
        assert owner.quant_method is methods[index]


def test_binding_rejects_entire_model_before_publishing_rows(tmp_path):
    model = GenericMoE()
    config, path = config_and_map(tmp_path)
    del model.moe_layers[1].routed_experts
    with pytest.raises(ValueError):
        bind(config, model)
    assert not hasattr(model.moe_layers[0].routed_experts, "_vllm_hcu_static_eplb_row")
    assert not hasattr(model, "_vllm_hcu_static_eplb_plan")


@pytest.mark.parametrize("field", ["enable_eplb", "enable_expert_parallel", "enable_ep_weight_filter"])
def test_binding_rejects_unsupported_config(tmp_path, field):
    model = GenericMoE()
    config, _ = config_and_map(tmp_path)
    setattr(config.parallel_config, field, field == "enable_ep_weight_filter")
    with pytest.raises(ValueError):
        bind(config, model)


def test_binding_cannot_rebind_changed_plan(tmp_path):
    model = GenericMoE()
    config, path = config_and_map(tmp_path)
    first = bind(config, model)
    path.write_text(json.dumps({"model_maps": {"GenericMoE": {
        "physical_to_logical_map": [[2, 0, 1, 2], [0, 1, 2, 1]]}}}))
    with pytest.raises(ValueError, match="changed|rebind"):
        bind(config, model)
    assert model._vllm_hcu_static_eplb_plan is first


def test_pp_stage_key_and_inner_owner_are_bound(tmp_path):
    model = GenericMoE()
    inner = torch.nn.Module()
    inner.moe_layers = model.moe_layers
    model.model = inner
    config, _ = config_and_map(tmp_path, key="GenericMoE#pp_rank=1")
    config.parallel_config.pipeline_parallel_size = 2
    config.parallel_config.pipeline_parallel_rank = 1
    plan = bind(config, model)
    assert plan.model_key == "GenericMoE#pp_rank=1"
    assert inner._vllm_hcu_static_eplb_plan is plan


def test_inactive_binding_preserves_every_loader(tmp_path):
    model = GenericMoE()
    config, _ = config_and_map(tmp_path)
    config.parallel_config._vllm_hcu_expert_map_path = None
    original = model.moe_layers[0].routed_experts.w13_weight.weight_loader
    assert bind(config, model) is None
    assert model.moe_layers[0].routed_experts.w13_weight.weight_loader is original


def test_slimquant_capability_stays_rejected(tmp_path):
    from vllm_hcu.model_executor.layers.quantization.slimquant_w4a8 import SlimQuantW4A8Int8AiterMoEMethod
    model = GenericMoE()
    method = SlimQuantW4A8Int8AiterMoEMethod(None, SimpleNamespace(moe_backend="auto"))
    del model.moe_layers[0].routed_experts.quant_method
    model.moe_layers[0].routed_experts.quant_method = method
    config, _ = config_and_map(tmp_path)
    with pytest.raises(ValueError, match="EPLB.*SlimQuant|SlimQuant.*EPLB"):
        bind(config, model)
    assert not method.supports_eplb
    assert model.moe_layers[0].routed_experts.quant_method is method


def test_fingerprints_disagree_before_binding(tmp_path, monkeypatch):
    model = GenericMoE()
    config, _ = config_and_map(tmp_path)
    import vllm.distributed as distributed
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(distributed, "get_ep_group", lambda: SimpleNamespace(world_size=2, cpu_group=None))
    def disagree(output, value, group):
        output[:] = [value, ("different",)]
    monkeypatch.setattr(torch.distributed, "all_gather_object", disagree)
    with pytest.raises(RuntimeError, match="fingerprint"):
        bind(config, model)
    assert not hasattr(model, "_vllm_hcu_static_eplb_plan")


@pytest.mark.parametrize("architecture", ["deepseek", "qwen3"])
def test_current_generic_causal_lm_auto_loader_loads_two_maps(tmp_path, architecture):
    from vllm.model_executor.models.deepseek_v2 import DeepseekV2ForCausalLM
    from vllm.model_executor.models.qwen3_moe import Qwen3MoeForCausalLM
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
    cls = DeepseekV2ForCausalLM if architecture == "deepseek" else Qwen3MoeForCausalLM
    model = cls.__new__(cls)
    torch.nn.Module.__init__(model)
    generic = GenericMoE(1)
    model.model = torch.nn.Module()
    model.model.layers = torch.nn.ModuleList()
    runners = []
    for item in generic.moe_layers:
        runner = MoERunner.__new__(MoERunner)
        torch.nn.Module.__init__(runner)
        runner.routed_experts = item.routed_experts
        layer = torch.nn.Module()
        layer.mlp = torch.nn.Module()
        layer.mlp.experts = runner
        model.model.layers.append(layer)
        runners.append(runner)
    model.moe_layers = runners
    for name in ("num_moe_layers", "num_logical_experts", "num_routed_experts",
                 "num_physical_experts", "num_local_physical_experts", "num_shared_experts",
                 "num_redundant_experts", "num_expert_groups", "expert_weights"):
        setattr(model, name, getattr(generic, name))
    config, _ = config_and_map(tmp_path, key=type(model).__name__)
    bind(config, model)
    checkpoint = [(f"model.layers.{layer}.mlp.experts.{logical}.{projection}.weight",
                   torch.full((2, 2), logical + 1.))
                  for layer in range(2) for logical in range(3)
                  for projection in ("gate_proj", "up_proj", "down_proj")]
    model.load_weights(iter(checkpoint))
    assert runners[0].routed_experts.w2_weight[:, 0, 0].tolist() == [3., 2.]
    assert runners[1].routed_experts.w2_weight[:, 0, 0].tolist() == [2., 3.]


def test_legacy_model_mapping_does_not_emit_initial_redundant_ids(tmp_path):
    from types import MethodType
    from vllm.model_executor.layers.fused_moe import fused_moe_make_expert_params_mapping
    model = GenericMoE()
    def get_expert_mapping(self):
        return fused_moe_make_expert_params_mapping(self, "gate_proj", "down_proj", "up_proj",
                                                   3, self.num_redundant_experts)
    model.get_expert_mapping = MethodType(get_expert_mapping, model)
    config, _ = config_and_map(tmp_path)
    bind(config, model)
    assert {row[2] for row in model.get_expert_mapping()} == {0, 1, 2}


def test_static_shared_experts_keep_current_aiter_physical_tail(tmp_path):
    model = GenericMoE()
    for layer in model.moe_layers:
        owner = layer.routed_experts
        owner.expert_map_manager.num_fused_shared_experts = 1
        owner.expert_map_manager._expert_map = torch.tensor([0, 1, -1, -1, 2])
        for name, param in list(owner.named_parameters()):
            replacement = torch.nn.Parameter(torch.zeros((3, *param.shape[1:])), requires_grad=False)
            replacement.weight_loader = owner.weight_loader
            if "scale" in name:
                replacement.quant_method = "channel"
            setattr(owner, name, replacement)
    config, _ = config_and_map(tmp_path)
    bind(config, model)
    for layer in model.moe_layers:
        owner = layer.routed_experts
        list(owner.load_weights(iter([("3.down_proj.weight", torch.full((2, 2), 9.))])))
        assert owner.w2_weight[:, 0, 0].tolist() == [0., 0., 9.]


def test_rebinding_detects_changed_consumer_row(tmp_path):
    model = GenericMoE()
    config, _ = config_and_map(tmp_path)
    bind(config, model)
    model.moe_layers[0].routed_experts._vllm_hcu_static_eplb_row = (1, 0, 2, 1)
    with pytest.raises(ValueError, match="row|binding"):
        bind(config, model)


def test_current_hcu_fused_channel_scale_owner_loads_two_maps(tmp_path, monkeypatch):
    from tests.models.static_eplb_test_utils import apply_current_moe_adapter
    apply_current_moe_adapter(monkeypatch)
    model = GenericMoE(1)
    config, _ = config_and_map(tmp_path)
    bind(config, model)
    fused = torch.tensor([1., 2., 3.])[:, None, None].expand(3, 4, 1).contiguous()
    for layer in model.moe_layers:
        list(layer.routed_experts.load_weights(iter([("gate_up_proj_scale", fused)])))
    assert model.moe_layers[0].routed_experts.w13_weight_scale[:, 0, 0].tolist() == [3., 2.]
    assert model.moe_layers[1].routed_experts.w13_weight_scale[:, 2, 0].tolist() == [2., 3.]


def test_direct_load_preserves_current_aiter_layout_installer(tmp_path):
    from vllm_hcu.model_executor.layers.quantization.compressed_tensors_moe_runtime import install_aiter_moe_weight_layout
    model = GenericMoE(1)
    config, _ = config_and_map(tmp_path)
    owner = model.moe_layers[0].routed_experts
    method, weights = owner.quant_method, (owner.w13_weight, owner.w2_weight)
    bind(config, model)
    for logical in range(3):
        owner.w2_weight.weight_loader(owner.w2_weight, torch.full((2, 2), logical + 1.),
                                      "w2_weight", "w2", logical)
    layout = SimpleNamespace(solution_type="triton", need_shuffle=False,
                             need_shuffle_scale=False, quant_type="per_token", config={})
    result = install_aiter_moe_weight_layout(owner, layout)
    assert result[0] is weights[0] and result[1] is weights[1]
    assert owner.quant_method is method
    assert owner.w2_weight[:, 0, 0].tolist() == [3., 2.]


def test_global_input_scale_stays_in_logical_index_space(tmp_path):
    model = GenericMoE(1)
    config, _ = config_and_map(tmp_path)
    owner = model.moe_layers[1].routed_experts
    owner.quant_method.use_global_sf = True
    owner.w2_input_scale = torch.nn.Parameter(torch.ones(3), requires_grad=False)
    owner.w2_input_scale.weight_loader = owner.weight_loader
    bind(config, model)
    owner.w2_input_scale.weight_loader(owner.w2_input_scale, torch.tensor(7.),
        "w2_input_scale", "w2", 2, return_success=True)
    assert owner.w2_input_scale.tolist() == [1., 1., 7.]


def test_direct_parameter_loader_rejects_unsplit_fused_tensor(tmp_path):
    model = GenericMoE()
    config, _ = config_and_map(tmp_path)
    bind(config, model)
    owner = model.moe_layers[0].routed_experts
    with pytest.raises(ValueError, match="split|fused"):
        owner.w2_weight.weight_loader(owner.w2_weight, torch.ones(3, 2, 2),
            "w2_weight", "w2", 0, return_success=True)
    assert torch.count_nonzero(owner.w2_weight) == 0
