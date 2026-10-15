# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch import nn

from vllm.models.deepseek_v4.amd.mtp import DeepSeekV4MTP as NativeDeepSeekV4MTP

from vllm_hcu.models.deepseek_v4_mtp_compat import DeepSeekV4MTP


def test_scale_aliases_are_scoped_to_native_weight_loading(monkeypatch):
    model = DeepSeekV4MTP.__new__(DeepSeekV4MTP)
    nn.Module.__init__(model)
    model.proj = nn.Module()
    model.proj.register_parameter(
        "weight_scale", nn.Parameter(torch.ones(1), requires_grad=False)
    )

    observed = {}

    def fake_load_weights(self, weights):
        params = dict(self.named_parameters())
        observed.update(params)
        assert params["proj.weight_scale_inv"] is params["proj.weight_scale"]
        return {"proj.weight_scale_inv"}

    monkeypatch.setattr(NativeDeepSeekV4MTP, "load_weights", fake_load_weights)

    loaded = model.load_weights([])

    assert loaded == {"proj.weight_scale_inv"}
    assert "proj.weight_scale_inv" in observed
    assert "proj.weight_scale_inv" not in dict(model.named_parameters())


def test_underscore_scale_alias_is_available_during_loading(monkeypatch):
    model = DeepSeekV4MTP.__new__(DeepSeekV4MTP)
    nn.Module.__init__(model)
    model.register_parameter(
        "experts_weight_scale", nn.Parameter(torch.ones(1), requires_grad=False)
    )

    def fake_load_weights(self, weights):
        params = dict(self.named_parameters())
        assert params["experts_weight_scale_inv"] is params["experts_weight_scale"]
        return {"experts_weight_scale_inv"}

    monkeypatch.setattr(NativeDeepSeekV4MTP, "load_weights", fake_load_weights)

    assert model.load_weights([]) == {"experts_weight_scale_inv"}


def test_registry_resolves_adapter_and_native_loader_accepts_w8a8_scale(monkeypatch):
    from types import SimpleNamespace
    from vllm import ModelRegistry
    from vllm.models.deepseek_v4.amd import mtp as native_module
    from vllm_hcu.models import register_model

    register_model()
    resolved, architecture = ModelRegistry.resolve_model_cls(
        ["DeepSeekV4MTPModel"], SimpleNamespace(model_impl="vllm", convert_type="none")
    )
    assert architecture == "DeepSeekV4MTPModel"
    assert resolved is DeepSeekV4MTP
    model = resolved.__new__(resolved)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        num_attention_heads=1,
        n_routed_experts=0,
        num_hidden_layers=1,
        num_nextn_predict_layers=1,
    )
    model.model = nn.Module()
    model.model.mtp_start_layer_idx = 1
    model.model.num_mtp_layers = 1
    layer = nn.Module()
    layer.e_proj = nn.Module()
    parameter = nn.Parameter(torch.ones(1), requires_grad=False)
    layer.e_proj.register_parameter("weight_scale", parameter)
    model.model.layers = nn.ModuleDict({"1": layer})
    monkeypatch.setattr(
        native_module, "get_tensor_model_parallel_world_size", lambda: 1
    )
    monkeypatch.setattr(native_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        native_module, "fused_moe_make_expert_params_mapping", lambda *a, **k: []
    )
    # The native loader alone cannot find the W8A8 plain scale parameter.
    with pytest.raises(KeyError, match="weight_scale_inv"):
        NativeDeepSeekV4MTP.load_weights(
            model, [("mtp.0.e_proj.scale", torch.tensor([2.0]))]
        )
    # Run the real upstream loader, rather than substituting load_weights.
    model.load_weights([("mtp.0.e_proj.scale", torch.tensor([2.0]))])
    torch.testing.assert_close(parameter, torch.tensor([2.0]))
    assert "model.layers.1.e_proj.weight_scale_inv" not in dict(
        model.named_parameters()
    )
