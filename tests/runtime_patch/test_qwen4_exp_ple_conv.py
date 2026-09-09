# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from types import ModuleType, SimpleNamespace

import torch
import torch.nn.functional as torch_functional

from vllm_hcu.patch.worker.core_fix import patch_qwen4_exp_ple_conv as patch


def _target_module(calls):
    class Functional:
        silu = staticmethod(torch_functional.silu)

        @staticmethod
        def conv1d(*args, **kwargs):
            calls.append(("official", args, kwargs))
            return "official"

    class Qwen4ExpPLELayer:
        def _short_conv_fallback(self, inputs):
            return inputs

    class Qwen4ExpNGramEmbedding:
        def forward(self, input_ids, query_start_loc, ngram_context):
            return input_ids, query_start_loc, ngram_context

    module = ModuleType(patch.TARGET_MODULE)
    module.F = Functional
    module.Qwen4ExpPLELayer = Qwen4ExpPLELayer
    module.Qwen4ExpNGramEmbedding = Qwen4ExpNGramEmbedding
    return module, Qwen4ExpPLELayer, Qwen4ExpNGramEmbedding


def test_qwen4_exp_ple_routes_only_depthwise_conv_to_hcu(monkeypatch):
    calls = []
    module, _, _ = _target_module(calls)

    def hcu(inputs, weight, bias, *, padding, dilation):
        calls.append(("hcu", inputs, weight, bias, padding, dilation))
        return "hcu"

    monkeypatch.setattr(patch, "_depthwise_conv1d", hcu)
    assert patch.apply_to_module(module) is True
    assert patch.apply_to_module(module) is False

    inputs = torch.empty(1, 4, 8)
    weight = torch.empty(4, 1, 3)
    assert module.F.conv1d(inputs, weight, groups=4, dilation=2) == "hcu"
    assert module.F.conv1d(inputs, weight, groups=1) == "official"
    assert [call[0] for call in calls] == ["hcu", "official"]


def test_qwen4_exp_ple_profile_fallback_uses_hcu_conv(monkeypatch):
    calls = []
    module, ple_class, _ = _target_module(calls)

    def hcu(inputs, weight, bias, *, padding, dilation):
        calls.append(("hcu", tuple(inputs.shape), padding, dilation))
        return torch.zeros(1, 4, inputs.shape[-1] + padding)

    monkeypatch.setattr(patch, "_depthwise_conv1d", hcu)
    patch.apply_to_module(module)
    layer = ple_class()
    layer.conv1d = SimpleNamespace(weight=torch.empty(4, 1, 4), bias=None)
    layer.conv_state_len = 9
    layer.short_conv_dilation = 3

    output = layer._short_conv_fallback(torch.empty(5, 4))

    assert output.shape == (5, 4)
    assert calls == [("hcu", (1, 4, 5), 9, 3)]


def test_qwen4_exp_ngram_dynamic_preprocessing_is_behind_custom_op(monkeypatch):
    calls = []
    module, _, ngram_class = _target_module(calls)

    def run(input_ids, query_start_loc, ngram_context, output, layer_name):
        calls.append(
            (
                tuple(input_ids.shape),
                tuple(query_start_loc.shape),
                tuple(ngram_context.shape),
                layer_name,
            )
        )
        output.fill_(2)

    monkeypatch.setattr(patch, "_run_ple_ngram_custom_op", run)
    patch.apply_to_module(module)
    embedding = ngram_class()
    embedding.embedding_dim = 8
    embedding.ngram_embedding = SimpleNamespace(params_dtype=torch.bfloat16)
    embedding.layer_name = "language_model.model.layers.2.ple"

    output = embedding.forward(
        torch.arange(5),
        torch.tensor([0, 5]),
        torch.zeros(1, 2, dtype=torch.long),
    )

    assert output.shape == (5, 8)
    assert output.dtype == torch.bfloat16
    assert torch.all(output == 2)
    assert calls == [((5,), (2,), (1, 2), embedding.layer_name)]
