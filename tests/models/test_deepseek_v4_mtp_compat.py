# SPDX-License-Identifier: Apache-2.0

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
        assert (
            params["experts_weight_scale_inv"]
            is params["experts_weight_scale"]
        )
        return {"experts_weight_scale_inv"}

    monkeypatch.setattr(NativeDeepSeekV4MTP, "load_weights", fake_load_weights)

    assert model.load_weights([]) == {"experts_weight_scale_inv"}
