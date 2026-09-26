# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

import torch
from vllm.model_executor.layers.linear import (
    ReplicatedLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.models.utils import WeightsMapper

from vllm_hcu.model_executor.layers.quantization import slimquant_w4a8


def _mixed_precision_config() -> slimquant_w4a8.SlimQuantW4A8Int8Config:
    config = slimquant_w4a8.SlimQuantW4A8Int8Config.from_config(
        {
            "quant_method": "slimquant_w4a8",
            "ignore": [
                r"re:.*self_attn\.o_proj(?:\.|$)",
                r"re:.*mlp\.gate(?:\.|$)",
                r"re:.*hyper_connection.*",
            ],
            "mixed_precision": {
                "experts": {
                    "format": "pack-int4-nibble",
                    "modules": [
                        "model.language_model.layers.*.mlp.experts.gate_up_proj",
                        "model.language_model.layers.*.mlp.experts.down_proj",
                    ],
                },
                "w8a8_include": {
                    "modules": [
                        "model.language_model.layers.*.self_attn.q_proj",
                        "model.language_model.layers.*.self_attn.k_proj",
                        "model.language_model.layers.*.self_attn.v_proj",
                        "model.language_model.layers.*.mlp.shared_expert.gate_proj",
                        "model.language_model.layers.*.mlp.shared_expert.up_proj",
                        "model.language_model.layers.*.mlp.shared_expert.down_proj",
                        "model.language_model.layers.*.linear_attn.in_proj_qkv",
                        "model.language_model.layers.*.linear_attn.in_proj_z",
                    ],
                },
            },
        }
    )
    config.apply_vllm_mapper(
        WeightsMapper(
            orig_to_new_prefix={
                "model.language_model.": "model.",
            }
        )
    )
    config.packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
    }
    return config


def _linear() -> ReplicatedLinear:
    layer = ReplicatedLinear.__new__(ReplicatedLinear)
    torch.nn.Module.__init__(layer)
    return layer


def test_slimquant_experts_only_mixed_precision_preserves_legacy_linears() -> None:
    config = slimquant_w4a8.SlimQuantW4A8Int8Config.from_config(
        {
            "quant_method": "slimquant_w4a8",
            "mixed_precision": {"experts": {"weight_bits": 4}},
        }
    )

    assert config.w8a8_include is None
    method = config.get_quant_method(_linear(), "model.layers.0.self_attn.qkv_proj")
    assert isinstance(method, slimquant_w4a8.SlimQuantW4A8Int8LinearMethod)


def test_qwen38_mixed_precision_leaves_other_linears_unquantized() -> None:
    config = _mixed_precision_config()

    for prefix in (
        "model.layers.0.self_attn.o_proj",
        "model.layers.0.mlp.gate",
        "model.layers.0.attn_hyper_connection.input_mix_weight_up",
    ):
        layer = _linear()
        method = config.get_quant_method(layer, prefix)

        assert isinstance(method, UnquantizedLinearMethod), prefix


def test_qwen38_mixed_precision_maps_and_quantizes_fused_w8a8_layers() -> None:
    config = _mixed_precision_config()

    assert config.w8a8_include == [
        "model.layers.*.self_attn.q_proj",
        "model.layers.*.self_attn.k_proj",
        "model.layers.*.self_attn.v_proj",
        "model.layers.*.mlp.shared_expert.gate_proj",
        "model.layers.*.mlp.shared_expert.up_proj",
        "model.layers.*.mlp.shared_expert.down_proj",
        "model.layers.*.linear_attn.in_proj_qkv",
        "model.layers.*.linear_attn.in_proj_z",
    ]
    for prefix in (
        "model.layers.0.self_attn.qkv_proj",
        "model.layers.0.mlp.shared_expert.gate_up_proj",
        "model.layers.0.mlp.shared_expert.down_proj",
        "model.layers.0.linear_attn.in_proj_qkvz",
    ):
        layer = _linear()
        method = config.get_quant_method(layer, prefix)

        assert isinstance(method, slimquant_w4a8.SlimQuantW4A8Int8LinearMethod)
        assert isinstance(layer.scheme, slimquant_w4a8.CompressedTensorsW8A8Int8)
