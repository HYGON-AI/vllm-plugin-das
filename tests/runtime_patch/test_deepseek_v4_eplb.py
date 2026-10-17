# SPDX-License-Identifier: Apache-2.0
from types import ModuleType, SimpleNamespace

import pytest
import torch

from vllm_hcu.patch.worker.core_fix import patch_deepseek_v4_eplb as patch
from vllm_hcu.patch.worker.core_fix._common import PatchCompatibilityError


def target():
    module = ModuleType(patch.TARGET_MODULE)
    factory_calls = []

    def factory(**kwargs):
        factory_calls.append(kwargs)
        return SimpleNamespace(moe_config=SimpleNamespace(ep_size=8),
                               update_expert_map=lambda: None)

    class DeepseekV4MoE(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.n_routed_experts = config.n_routed_experts
            self.experts = module.FusedMoEFactory(num_experts=config.n_routed_experts)

    class DeepseekV4Model:
        def __init__(self, config):
            self.config = config

        def get_expert_mapping(self):
            return "original"

    class DeepseekV4ForCausalLM(torch.nn.Module):
        def __init__(self, config, moe_layers=2):
            super().__init__()
            self.config = config
            self.model = DeepseekV4Model(config)
            self.model.layers = [SimpleNamespace(ffn=None)] + [
                SimpleNamespace(ffn=DeepseekV4MoE(config)) for _ in range(moe_layers)]

    module.FusedMoEFactory = factory
    module.DeepseekV4MoE = DeepseekV4MoE
    module.DeepseekV4Model = DeepseekV4Model
    module.DeepseekV4ForCausalLM = DeepseekV4ForCausalLM
    module.fused_moe_make_expert_params_mapping = (
        lambda model, **kwargs: ("eplb", kwargs))
    return module, factory_calls


@pytest.fixture
def vllm_config(monkeypatch):
    import vllm.config as config_module

    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            enable_eplb=True, eplb_config=SimpleNamespace(num_redundant_experts=16)),
        model_config=SimpleNamespace(hf_config=SimpleNamespace(n_shared_experts=1)),
    )
    monkeypatch.setattr(config_module, "get_current_vllm_config", lambda: config)
    return config


HF = SimpleNamespace(n_routed_experts=256, n_group=None)


def test_eplb_builds_redundant_experts_and_moe_interface(vllm_config):
    from vllm.model_executor.models.interfaces import is_mixture_of_experts

    module, calls = target()
    assert patch.apply_to_module(module)
    assert not patch.apply_to_module(module)
    model = module.DeepseekV4ForCausalLM(HF)
    assert calls[0] == {"num_experts": 256, "enable_eplb": True,
                        "num_redundant_experts": 16}
    assert is_mixture_of_experts(model)
    assert (model.num_moe_layers, model.num_logical_experts, model.num_physical_experts,
            model.num_local_physical_experts, model.num_redundant_experts,
            model.num_shared_experts, model.num_expert_groups) == (2, 256, 272, 34, 16, 1, 1)
    assert model.moe_layers == [layer.ffn.experts for layer in model.model.layers[1:]]
    assert model.model.get_expert_mapping() == ("eplb", {
        "ckpt_gate_proj_name": "w1", "ckpt_down_proj_name": "w2",
        "ckpt_up_proj_name": "w3", "num_experts": 256, "num_redundant_experts": 16})


def test_eplb_disabled_keeps_upstream_paths(vllm_config):
    from vllm.model_executor.models.interfaces import is_mixture_of_experts

    vllm_config.parallel_config.enable_eplb = False
    module, calls = target()
    patch.apply_to_module(module)
    model = module.DeepseekV4ForCausalLM(HF)
    assert calls[0] == {"num_experts": 256}
    assert not is_mixture_of_experts(model)
    assert model.model.get_expert_mapping() == "original"


def test_rejects_indivisible_physical_experts_and_stale_marker(vllm_config):
    vllm_config.parallel_config.eplb_config.num_redundant_experts = 3
    module, _ = target()
    patch.apply_to_module(module)
    with pytest.raises(ValueError, match="divisible"):
        module.DeepseekV4ForCausalLM(HF)
    module.DeepseekV4Model.get_expert_mapping = lambda self: None
    with pytest.raises(PatchCompatibilityError, match="stale"):
        patch.apply_to_module(module)
