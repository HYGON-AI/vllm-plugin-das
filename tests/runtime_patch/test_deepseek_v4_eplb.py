# SPDX-License-Identifier: Apache-2.0
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from vllm_hcu.patch.worker.core_fix import patch_deepseek_v4_eplb as patch
from vllm_hcu.patch.worker.core_fix._common import PatchCompatibilityError


def target(monkeypatch):
    module = ModuleType(patch.TARGET_MODULE)
    mtp_module = ModuleType(patch.MTP_MODULE)
    factory_calls = []

    def factory(**kwargs):
        factory_calls.append(kwargs)
        return SimpleNamespace(moe_config=SimpleNamespace(ep_size=module._test_ep_size),
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

    class DeepSeekV4MTP(torch.nn.Module):
        def __init__(self, config, moe_layers=2):
            super().__init__()
            self.config = config
            self.model = SimpleNamespace(
                layers={
                    str(index): SimpleNamespace(
                        mtp_block=SimpleNamespace(ffn=DeepseekV4MoE(config))
                    )
                    for index in range(moe_layers)
                }
            )

    module.FusedMoEFactory = factory
    module.DeepseekV4MoE = DeepseekV4MoE
    module.DeepseekV4Model = DeepseekV4Model
    module.DeepseekV4ForCausalLM = DeepseekV4ForCausalLM
    module._test_ep_size = 8
    module.fused_moe_make_expert_params_mapping = (
        lambda model, **kwargs: ("eplb", kwargs))
    mtp_module.DeepSeekV4MTP = DeepSeekV4MTP
    mtp_module.fused_moe_make_expert_params_mapping = (
        lambda model, **kwargs: ("mtp-eplb", kwargs)
    )
    monkeypatch.setitem(sys.modules, patch.MTP_MODULE, mtp_module)
    module._test_mtp_module = mtp_module
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


def test_eplb_builds_redundant_experts_and_moe_interface(
    vllm_config, monkeypatch
):
    from vllm.model_executor.models.interfaces import is_mixture_of_experts

    module, calls = target(monkeypatch)
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


def test_eplb_disabled_keeps_upstream_paths(vllm_config, monkeypatch):
    from vllm.model_executor.models.interfaces import is_mixture_of_experts

    vllm_config.parallel_config.enable_eplb = False
    module, calls = target(monkeypatch)
    module._test_ep_size = 3
    patch.apply_to_module(module)
    model = module.DeepseekV4ForCausalLM(HF)
    assert calls[0] == {"num_experts": 256}
    assert not is_mixture_of_experts(model)
    assert model.model.get_expert_mapping() == "original"
    assert not hasattr(model.model.layers[1].ffn, "n_physical_experts")


def test_eplb_registers_native_mtp_and_maps_redundant_weights(
    vllm_config, monkeypatch
):
    from vllm.model_executor.models.interfaces import is_mixture_of_experts

    module, _ = target(monkeypatch)
    patch.apply_to_module(module)
    mtp_module = module._test_mtp_module

    class RegisteredCompatMTP(mtp_module.DeepSeekV4MTP):
        pass

    draft = RegisteredCompatMTP(HF)

    assert is_mixture_of_experts(draft)
    assert draft.num_moe_layers == 2
    assert draft.num_physical_experts == 272
    assert draft.num_local_physical_experts == 34
    assert draft.moe_layers == [
        layer.mtp_block.ffn.experts for layer in draft.model.layers.values()
    ]
    assert mtp_module.fused_moe_make_expert_params_mapping(
        draft,
        num_experts=256,
    ) == (
        "mtp-eplb",
        {"num_experts": 256, "num_redundant_experts": 16},
    )


def test_rejects_indivisible_physical_experts_and_stale_marker(
    vllm_config, monkeypatch
):
    vllm_config.parallel_config.eplb_config.num_redundant_experts = 3
    module, _ = target(monkeypatch)
    patch.apply_to_module(module)
    with pytest.raises(ValueError, match="divisible"):
        module.DeepseekV4ForCausalLM(HF)
    module.DeepseekV4Model.get_expert_mapping = lambda self: None
    with pytest.raises(PatchCompatibilityError, match="stale"):
        patch.apply_to_module(module)
