# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""CPU behavior contracts for the GLM-5.1 sparse MTP runtime."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.nn as nn


REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def mtp_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Load deepseek_mtp with only its HCU-owned dependencies stubbed."""
    monkeypatch.delenv("VLLM_HCU_GLM_MTP_QUANT_PREFIX_FIX", raising=False)

    deepseek_v2 = ModuleType("vllm_hcu.models.deepseek_v2")

    class DeepseekV2DecoderLayer(nn.Module):
        pass

    class DeepseekV2MixtureOfExperts:
        pass

    deepseek_v2.DeepseekV2DecoderLayer = DeepseekV2DecoderLayer
    deepseek_v2.DeepseekV2MixtureOfExperts = DeepseekV2MixtureOfExperts
    deepseek_v2.DeepseekV2MoE = type("DeepseekV2MoE", (), {})
    deepseek_v2._try_load_quantized_indexer_wk = lambda *args, **kwargs: False
    deepseek_v2.get_spec_layer_idx_from_weight_name = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, deepseek_v2.__name__, deepseek_v2)

    lightly_cp_utils = ModuleType("vllm_hcu.v1.attention.lightly_cp_utils")
    lightly_cp_utils.lightly_cp_inputs_splitting = lambda *args: args[:4]
    monkeypatch.setitem(sys.modules, lightly_cp_utils.__name__, lightly_cp_utils)

    module_name = "vllm_hcu.models._test_glm51_deepseek_mtp"
    spec = importlib.util.spec_from_file_location(
        module_name,
        REPO / "vllm_hcu/models/deepseek_mtp.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def _uninitialized(module_type: type[nn.Module]) -> nn.Module:
    instance = object.__new__(module_type)
    nn.Module.__init__(instance)
    return instance


def test_main_v32_mtp_architecture_uses_hcu_mtp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm_hcu.models as models

    registrations: dict[str, str] = {}
    monkeypatch.setattr(
        models.ModelRegistry,
        "register_model",
        lambda architecture, implementation: registrations.__setitem__(
            architecture, implementation
        ),
    )

    models.register_model()

    assert registrations["DeepseekV32MTPModel"] == (
        "vllm_hcu.models.deepseek_mtp:DeepSeekMTP"
    )


def test_mtp_layer_returns_pre_norm_logits_and_post_norm_recycle_states(
    mtp_module: ModuleType,
) -> None:
    class FuseInputs(nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            midpoint = value.shape[-1] // 2
            return value[..., :midpoint] + value[..., midpoint:]

    class MTPBlock(nn.Module):
        def forward(self, **kwargs):
            hidden_states = kwargs["hidden_states"]
            return hidden_states * 2, hidden_states * 3

    class FinalNorm(nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return value + 7

    layer = _uninitialized(mtp_module.DeepSeekMultiTokenPredictorLayer)
    layer.enorm = nn.Identity()
    layer.hnorm = nn.Identity()
    layer.eh_proj = FuseInputs()
    layer.mtp_block = MTPBlock()
    layer.shared_head = FinalNorm()

    positions = torch.tensor([0, 1])
    previous = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    inputs = torch.tensor([[10.0, 20.0], [30.0, 40.0]])

    logits_hidden, recycle_hidden = layer(
        torch.tensor([1, 2]),
        positions,
        previous,
        inputs,
    )

    # Position zero masks inputs, then the block returns 2*x with 3*x residual.
    expected_logits_hidden = torch.tensor([[5.0, 10.0], [165.0, 220.0]])
    torch.testing.assert_close(logits_hidden, expected_logits_hidden)
    torch.testing.assert_close(recycle_hidden, expected_logits_hidden + 7)


def test_mtp_sparse_index_controls_share_and_compact_all_layers(
    mtp_module: ModuleType,
) -> None:
    class MLAAttention(nn.Module):
        def __init__(self, values: torch.Tensor) -> None:
            super().__init__()
            self.skip_topk = False
            self.topk_indices_buffer = values.clone()

    class SelfAttention(nn.Module):
        def __init__(self, mla_attn: nn.Module) -> None:
            super().__init__()
            self.mla_attn = mla_attn

    class MTPBlock(nn.Module):
        def __init__(self, mla_attn: nn.Module) -> None:
            super().__init__()
            self.self_attn = SelfAttention(mla_attn)

    class Layer(nn.Module):
        def __init__(self, mla_attn: nn.Module | None = None) -> None:
            super().__init__()
            if mla_attn is not None:
                self.mtp_block = MTPBlock(mla_attn)

    first = MLAAttention(torch.arange(10).view(5, 2))
    second = MLAAttention(torch.arange(20, 30).view(5, 2))
    predictor = _uninitialized(mtp_module.DeepSeekMultiTokenPredictor)
    predictor.layers = nn.ModuleDict(
        {"0": Layer(first), "1": Layer(), "2": Layer(second)}
    )

    originals = [first.topk_indices_buffer.clone(), second.topk_indices_buffer.clone()]
    predictor.set_skip_topk(True)
    predictor.compact_topk_indices(torch.tensor([3, 1]))

    assert first.skip_topk is True
    assert second.skip_topk is True
    torch.testing.assert_close(first.topk_indices_buffer[:2], originals[0][[3, 1]])
    torch.testing.assert_close(second.topk_indices_buffer[:2], originals[1][[3, 1]])


def test_mtp_lightly_cp_gathers_both_hidden_state_contracts(
    mtp_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Layer(nn.Module):
        def forward(
            self,
            input_ids,
            positions,
            previous_hidden_states,
            inputs_embeds,
            spec_step_idx,
        ):
            del input_ids, positions, inputs_embeds, spec_step_idx
            return previous_hidden_states + 10, previous_hidden_states + 20

    context = SimpleNamespace(
        enable_lightly_cp=True,
        gather_indexes_tensor=torch.tensor([2, 0]),
    )
    monkeypatch.setattr(mtp_module, "get_forward_context", lambda: context)
    monkeypatch.setattr(
        mtp_module,
        "lightly_cp_inputs_splitting",
        lambda hidden, positions, residual, inputs, tp_size, tp_rank: (
            hidden,
            positions,
            residual,
            inputs,
        ),
    )
    monkeypatch.setattr(
        mtp_module,
        "tensor_model_parallel_all_gather",
        lambda value, dim: torch.cat((value, value + 100), dim=dim),
    )

    predictor = _uninitialized(mtp_module.DeepSeekMultiTokenPredictor)
    predictor.mtp_start_layer_idx = 4
    predictor.num_mtp_layers = 1
    predictor.tp_size = 2
    predictor.tp_rank = 0
    predictor.layers = nn.ModuleDict({"4": Layer()})

    logits_hidden, recycle_hidden = predictor(
        torch.tensor([1, 2]),
        torch.tensor([0, 1]),
        torch.tensor([[1.0], [2.0]]),
        torch.zeros(2, 1),
    )

    torch.testing.assert_close(logits_hidden, torch.tensor([[111.0], [11.0]]))
    torch.testing.assert_close(recycle_hidden, torch.tensor([[121.0], [21.0]]))


def _glm_engine(quant_config, **kwargs):
    return SimpleNamespace(
        quant_config=quant_config,
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(model_type="glm_moe_dsa"),
        ),
        **kwargs,
    )


def _mixed_quant_config(target: str):
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
        CompressedTensorsConfig,
    )

    def scheme(bits, targets, quant_format):
        return {
            "targets": targets,
            "format": quant_format,
            "weights": {
                "num_bits": bits, "type": "int", "strategy": "channel",
                "symmetric": True, "dynamic": False, "group_size": -1,
            },
            "input_activations": {
                "num_bits": 8, "type": "int", "strategy": "token",
                "symmetric": True, "dynamic": True,
            },
        }

    return CompressedTensorsConfig.from_config({
        "quant_method": "compressed-tensors",
        "quantization_status": "compressed",
        "format": "pack-quantized",
        "config_groups": {
            "group_int4": scheme(4, [target], "pack-quantized"),
            "group_int8": scheme(8, ["Linear"], "int-quantized"),
        },
        "ignore": [r"re:^model\.decoder\.self_attn\.kv_b_proj$"],
    })


@pytest.mark.parametrize("layer_idx", [78, 79])
@pytest.mark.parametrize("target", [
    r"re:^model\.decoder\.mlp\.experts\.\d+\.(?:gate_proj|up_proj|down_proj)$",
    r"re:model\.decoder\.mlp\.experts\.\d+\.(?:gate_proj|up_proj|down_proj)$",
    r"re:^model[.]decoder[.]mlp\.experts\.\d+\.(?:gate_proj|up_proj|down_proj)$",
    r"re:^(?:model)[.](?:decoder)[.]mlp\.experts\.\d+\.(?:gate_proj|up_proj|down_proj)$",
    r"re:^model\x2edecoder\x2emlp\.experts\.\d+\.(?:gate_proj|up_proj|down_proj)$",
    "model.decoder.mlp.experts.0.down_proj",
])
def test_mtp_decoder_quant_targets_select_int4_without_mutating_target_config(
    mtp_module, layer_idx, target,
):
    qconfig = _mixed_quant_config(target)
    original_targets = dict(qconfig.target_scheme_map)
    original_ignore = list(qconfig.ignore)
    engine = _glm_engine(qconfig, compilation_config=object())
    prefix = f"model.layers.{layer_idx}"
    linear = nn.Linear(8, 8, bias=False)
    name = prefix + ".mlp.experts.0.down_proj"
    assert qconfig.get_scheme_dict(linear, name)["weights"].num_bits == 8

    adapted = mtp_module._mtp_quantization_config(engine, prefix)
    scheme = adapted.quant_config.get_scheme_dict(linear, name)
    assert scheme["weights"].num_bits == 4
    assert scheme["input_activations"].num_bits == 8
    assert scheme["input_activations"].dynamic
    assert scheme["format"] == "pack-quantized"
    assert adapted.quant_config.get_scheme_dict(
        linear, prefix + ".self_attn.kv_b_proj",
    ) is None
    assert adapted.quant_config.get_scheme_dict(
        linear, prefix + ".self_attn.q_b_proj",
    )["weights"].num_bits == 8
    assert adapted.quant_config.get_scheme_dict(
        linear, "model.layers.77.mlp.experts.0.down_proj",
    )["weights"].num_bits == 8
    assert adapted is not engine
    assert adapted.quant_config is not qconfig
    assert adapted.compilation_config is engine.compilation_config
    assert qconfig.target_scheme_map == original_targets
    assert qconfig.ignore == original_ignore
    assert qconfig.get_scheme_dict(linear, name)["weights"].num_bits == 8


def test_mtp_explicit_actual_prefix_ignore_and_target_take_priority(mtp_module):
    qconfig = _mixed_quant_config(r"re:^model\.decoder\.mlp\.experts\.0\.down_proj$")
    actual = r"re:^model\.layers\.78\.mlp\.experts\.0\.down_proj$"
    qconfig.target_scheme_map[actual] = qconfig.target_scheme_map["Linear"]
    qconfig.ignore.append(r"re:^model\.layers\.78\.self_attn\.")
    adapted = mtp_module._mtp_quantization_config(
        _glm_engine(qconfig), "model.layers.78",
    )
    linear = nn.Linear(8, 8, bias=False)
    assert adapted.quant_config.get_scheme_dict(
        linear, "model.layers.78.mlp.experts.0.down_proj",
    )["weights"].num_bits == 8
    assert adapted.quant_config.get_scheme_dict(
        linear, "model.layers.78.self_attn.q_b_proj",
    ) is None


@pytest.mark.parametrize("generic_first", [False, True])
@pytest.mark.parametrize("canonical", [
    r"re:^model\.decoder\.mlp\.experts\.\d+\.down_proj$",
    r"re:^model[.]decoder[.]mlp\.experts\.\d+\.down_proj$",
    r"re:^(model)[.](decoder)[.]mlp\.experts\.\d+\.down_proj$",
])
def test_mtp_alias_preserves_regex_first_match_priority(mtp_module, generic_first, canonical):
    generic = r"re:^model\..*\.down_proj$"
    qconfig = _mixed_quant_config(canonical)
    specific_scheme = qconfig.target_scheme_map[canonical]
    generic_scheme = qconfig.target_scheme_map["Linear"]
    rules = [(canonical, specific_scheme), (generic, generic_scheme)]
    if generic_first:
        rules.reverse()
    qconfig.target_scheme_map = dict(rules)
    original_rules = list(qconfig.target_scheme_map.items())
    linear = nn.Linear(8, 8, bias=False)
    original_bits = qconfig.get_scheme_dict(
        linear, "model.decoder.mlp.experts.0.down_proj",
    )["weights"].num_bits
    adapted = mtp_module._mtp_quantization_config(_glm_engine(qconfig), "model.layers.78")
    assert adapted.quant_config.get_scheme_dict(
        linear, "model.layers.78.mlp.experts.0.down_proj",
    )["weights"].num_bits == original_bits
    assert original_bits == (8 if generic_first else 4)
    assert list(qconfig.target_scheme_map.items()) == original_rules


def test_mtp_existing_alias_retains_its_position(mtp_module):
    canonical = r"re:^model\.decoder\.mlp\.experts\.0\.down_proj$"
    actual = r"re:^model\.layers\.78\.mlp\.experts\.0\.down_proj$"
    generic = r"re:^model\..*\.down_proj$"
    qconfig = _mixed_quant_config(canonical)
    four = qconfig.target_scheme_map[canonical]
    eight = qconfig.target_scheme_map["Linear"]
    qconfig.target_scheme_map = {canonical: four, generic: eight, actual: eight}
    adapted = mtp_module._mtp_quantization_config(_glm_engine(qconfig), "model.layers.78")
    assert list(adapted.quant_config.target_scheme_map) == [canonical, generic, actual]
    assert adapted.quant_config.target_scheme_map[actual] is eight


@pytest.mark.parametrize("generic_first", [False, True])
@pytest.mark.parametrize("backbone", [
    r"re:^model\.layers\.\d+\..*proj$",
    r"re:^model[.]layers[.]\d+[.].*proj$",
])
def test_mtp_without_decoder_rules_preserves_native_priority(mtp_module, generic_first, backbone):
    generic = r"re:.*\.experts\.\d+\.down_proj$"
    qconfig = _mixed_quant_config(generic)
    four = qconfig.target_scheme_map[generic]
    eight = qconfig.target_scheme_map["Linear"]
    rules = [(generic, four), (backbone, eight)]
    if not generic_first:
        rules.reverse()
    qconfig.target_scheme_map = dict(rules)
    before = list(qconfig.target_scheme_map.items())
    layer = nn.Linear(8, 8)
    name = "model.layers.78.mlp.experts.0.down_proj"
    expected = qconfig.get_scheme_dict(layer, name)
    adapted = mtp_module._mtp_quantization_config(_glm_engine(qconfig), "model.layers.78")
    assert adapted.quant_config.get_scheme_dict(layer, name) == expected
    assert expected["weights"].num_bits == (4 if generic_first else 8)
    assert list(qconfig.target_scheme_map.items()) == before


@pytest.mark.parametrize("prefix", ["model.layers.78", "model.layers.79"])
def test_mtp_backbone_regex_does_not_override_decoder_rule(mtp_module, prefix):
    canonical = r"re:^model[.]decoder[.]mlp[.]experts[.]\d+[.]down_proj$"
    backbone = r"re:^model[.]layers[.]\d+[.].*proj$"
    qconfig = _mixed_quant_config(canonical)
    four = qconfig.target_scheme_map[canonical]
    eight = qconfig.target_scheme_map["Linear"]
    qconfig.target_scheme_map = {canonical: four, backbone: eight}
    engine = _glm_engine(qconfig)
    engine.model_config.hf_config.num_hidden_layers = 78
    adapted = mtp_module._mtp_quantization_config(engine, prefix)
    assert adapted.quant_config.get_scheme_dict(
        nn.Linear(8, 8), prefix + ".mlp.experts.0.down_proj",
    )["weights"].num_bits == 4


@pytest.mark.parametrize("actual", [
    "model.layers.78.mlp.experts.0.down_proj",
    r"re:^model[.]layers[.]78[.]mlp[.]experts[.]0[.]down_proj$",
])
def test_mtp_actual_override_precedes_canonical_and_generic_rules(mtp_module, actual):
    canonical = r"re:^model[.]decoder[.]mlp[.]experts[.]\d+[.]down_proj$"
    qconfig = _mixed_quant_config(canonical)
    four = qconfig.target_scheme_map[canonical]
    eight = qconfig.target_scheme_map["Linear"]
    qconfig.target_scheme_map = {canonical: four, r"re:.*down_proj$": four, actual: eight}
    adapted = mtp_module._mtp_quantization_config(_glm_engine(qconfig), "model.layers.78")
    assert adapted.quant_config.get_scheme_dict(
        nn.Linear(8, 8), "model.layers.78.mlp.experts.0.down_proj",
    )["weights"].num_bits == 8
    assert qconfig.get_scheme_dict(
        nn.Linear(8, 8), "model.decoder.mlp.experts.0.down_proj",
    )["weights"].num_bits == 4


def test_mtp_canonical_match_preserves_fused_projection_overrides(mtp_module):
    canonical = r"re:^model[.]decoder[.]self_attn[.](q_proj|k_proj|v_proj)$"
    qconfig = _mixed_quant_config(canonical)
    qconfig.packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}
    four = qconfig.target_scheme_map[canonical]
    eight = qconfig.target_scheme_map["Linear"]
    qconfig.target_scheme_map = {canonical: four, r"re:.*proj$": four}
    for name in ("q_proj", "k_proj", "v_proj"):
        qconfig.target_scheme_map["model.layers.78.self_attn." + name] = eight
    adapted = mtp_module._mtp_quantization_config(_glm_engine(qconfig), "model.layers.78")
    assert adapted.quant_config.get_scheme_dict(
        nn.Linear(8, 8), "model.layers.78.self_attn.qkv_proj",
    )["weights"].num_bits == 8


@pytest.mark.parametrize("prefix", ["model.layers.78", "model.layers.79"])
@pytest.mark.parametrize("specific", ["gate_proj", "up_proj"])
def test_mtp_fusion_combines_decoder_and_generic_rules(mtp_module, prefix, specific):
    canonical = "model.decoder.mlp.shared_experts." + specific
    other = "up_proj" if specific == "gate_proj" else "gate_proj"
    qconfig = _mixed_quant_config(canonical)
    four, eight = qconfig.target_scheme_map[canonical], qconfig.target_scheme_map["Linear"]
    qconfig.target_scheme_map = {canonical: eight, "re:.*[.]" + other + "$": eight,
                                "Linear": four}
    qconfig.packed_modules_mapping = {"gate_up_proj": ["gate_proj", "up_proj"]}
    engine = _glm_engine(qconfig)
    engine.model_config.hf_config.num_hidden_layers = 78
    layer = nn.Linear(8, 8)
    expected = qconfig.get_scheme_dict(layer, "model.decoder.mlp.shared_experts.gate_up_proj")
    assert expected["weights"].num_bits == 8
    assert qconfig.get_scheme_dict(
        layer, prefix + ".mlp.shared_experts.gate_up_proj",
    )["weights"].num_bits == 4
    adapted = mtp_module._mtp_quantization_config(engine, prefix)
    assert adapted.quant_config.get_scheme_dict(
        layer, prefix + ".mlp.shared_experts.gate_up_proj",
    ) == expected


@pytest.mark.parametrize("prefix", ["model.layers.78", "model.layers.79"])
@pytest.mark.parametrize("specific", ["gate_proj", "up_proj"])
def test_mtp_fusion_combines_actual_override_and_generic_rules(mtp_module, prefix, specific):
    canonical = r"re:^model[.]decoder[.]mlp[.]shared_experts[.](gate_proj|up_proj)$"
    actual = prefix + ".mlp.shared_experts." + specific
    other = "up_proj" if specific == "gate_proj" else "gate_proj"
    qconfig = _mixed_quant_config(canonical)
    four, eight = qconfig.target_scheme_map[canonical], qconfig.target_scheme_map["Linear"]
    qconfig.target_scheme_map = {canonical: four, "re:.*[.]" + other + "$": eight,
                                actual: eight, "Linear": four}
    qconfig.packed_modules_mapping = {"gate_up_proj": ["gate_proj", "up_proj"]}
    engine = _glm_engine(qconfig)
    engine.model_config.hf_config.num_hidden_layers = 78
    layer = nn.Linear(8, 8)
    expected = qconfig.get_scheme_dict(layer, prefix + ".mlp.shared_experts.gate_up_proj")
    assert expected["weights"].num_bits == 8
    adapted = mtp_module._mtp_quantization_config(engine, prefix)
    assert adapted.quant_config.get_scheme_dict(
        layer, prefix + ".mlp.shared_experts.gate_up_proj",
    ) == expected


@pytest.mark.parametrize("prefix", ["model.layers.78", "model.layers.79"])
def test_mtp_shadowed_decoder_rules_do_not_change_native_fusion(mtp_module, prefix):
    canonical = r"re:^model[.]decoder[.]mlp[.]shared_experts[.](gate_proj|up_proj)$"
    qconfig = _mixed_quant_config(canonical)
    four, eight = qconfig.target_scheme_map[canonical], qconfig.target_scheme_map["Linear"]
    qconfig.target_scheme_map = {
        r"re:.*[.](gate_proj|up_proj)$": four, canonical: eight,
        r"re:^model[.]layers[.]\d+[.].*[.]gate_up_proj$": eight, "Linear": four,
    }
    qconfig.packed_modules_mapping = {"gate_up_proj": ["gate_proj", "up_proj"]}
    engine = _glm_engine(qconfig)
    engine.model_config.hf_config.num_hidden_layers = 78
    layer = nn.Linear(8, 8)
    expected = qconfig.get_scheme_dict(layer, prefix + ".mlp.shared_experts.gate_up_proj")
    assert expected["weights"].num_bits == 8
    assert qconfig.get_scheme_dict(
        layer, "model.decoder.mlp.shared_experts.gate_up_proj",
    )["weights"].num_bits == 4
    adapted = mtp_module._mtp_quantization_config(engine, prefix)
    assert adapted.quant_config.get_scheme_dict(
        layer, prefix + ".mlp.shared_experts.gate_up_proj",
    ) == expected


@pytest.mark.parametrize("ignore", [
    r"re:^model[.]decoder[.]self_attn[.]kv_b_proj$",
    r"re:^(model)[.](decoder)[.]self_attn[.]kv_b_proj$",
])
def test_mtp_canonical_ignore_accepts_equivalent_regex(mtp_module, ignore):
    qconfig = _mixed_quant_config("Linear")
    qconfig.ignore = [ignore]
    adapted = mtp_module._mtp_quantization_config(_glm_engine(qconfig), "model.layers.78")
    assert adapted.quant_config.get_scheme_dict(
        nn.Linear(8, 8), "model.layers.78.self_attn.kv_b_proj",
    ) is None
    assert qconfig.ignore == [ignore]


@pytest.mark.parametrize("prefix", ["model.layers.78", "model.layers.79"])
@pytest.mark.parametrize("ignored", [
    ("canonical.gate_up_proj", "actual.gate_proj"),
    ("canonical.gate_proj", "canonical.up_proj", "actual.gate_proj"),
    ("canonical.gate_proj", "actual.up_proj"),
    ("actual.gate_up_proj", "canonical.gate_proj"),
])
def test_mtp_fused_ignores_combine_both_prefixes(mtp_module, prefix, ignored):
    names = {
        "canonical": "model.decoder.mlp.shared_experts",
        "actual": prefix + ".mlp.shared_experts",
    }
    qconfig = _mixed_quant_config("Linear")
    qconfig.ignore = [names[side] + "." + projection
                      for side, projection in (value.split(".") for value in ignored)]
    original_ignore = list(qconfig.ignore)
    qconfig.packed_modules_mapping = {"gate_up_proj": ["gate_proj", "up_proj"]}
    adapted = mtp_module._mtp_quantization_config(_glm_engine(qconfig), prefix)
    assert adapted.quant_config.get_scheme_dict(
        nn.Linear(8, 8), names["actual"] + ".gate_up_proj",
    ) is None
    assert qconfig.ignore == original_ignore
    assert adapted.quant_config.ignore == original_ignore


@pytest.mark.parametrize("side", ["model.decoder", "model.layers.78"])
def test_mtp_fused_partial_ignore_still_rejects_mixed_schemes(mtp_module, side):
    qconfig = _mixed_quant_config("Linear")
    qconfig.ignore = [side + ".mlp.shared_experts.gate_proj"]
    qconfig.packed_modules_mapping = {"gate_up_proj": ["gate_proj", "up_proj"]}
    adapted = mtp_module._mtp_quantization_config(_glm_engine(qconfig), "model.layers.78")
    with pytest.raises(ValueError, match="different quantization schemes"):
        adapted.quant_config.get_scheme_dict(
            nn.Linear(8, 8), "model.layers.78.mlp.shared_experts.gate_up_proj",
        )


@pytest.mark.parametrize("quant_config", [None, SimpleNamespace()])
def test_non_compressed_tensors_mtp_config_is_unchanged(mtp_module, quant_config):
    engine = _glm_engine(quant_config)
    assert mtp_module._mtp_quantization_config(engine, "model.layers.78") is engine


def test_legacy_mtp_quantization_ignore_remains_unchanged(mtp_module):
    qconfig = _mixed_quant_config("Linear")
    qconfig.ignore = [r"re:^model\.layers\.78(?:\..*)?$"]
    engine = _glm_engine(qconfig)
    adapted = mtp_module._mtp_quantization_config(engine, "model.layers.78")
    assert adapted.quant_config.ignore == qconfig.ignore
    assert adapted.quant_config.get_scheme_dict(
        nn.Linear(8, 8, bias=False), "model.layers.78.mlp.experts.0.down_proj",
    ) is None


@pytest.mark.parametrize("converted_draft", [False, True])
def test_mtp_constructor_passes_private_quant_config_to_decoder(
    mtp_module, monkeypatch, converted_draft,
):
    from copy import deepcopy
    from transformers import PretrainedConfig
    from vllm.config.speculative import SpeculativeConfig

    qconfig = _mixed_quant_config(r"re:^model\.decoder\.mlp\.experts\.\d+\.down_proj$")
    hf_config = PretrainedConfig(
        hidden_size=8, rms_norm_eps=1e-6, num_hidden_layers=78,
        num_nextn_predict_layers=1, architectures=["GlmMoeDsaForCausalLM"],
    )
    hf_config.model_type = "glm_moe_dsa"
    target_config = SimpleNamespace(hf_config=hf_config)
    draft_hf_config = deepcopy(hf_config)
    if converted_draft:
        draft_hf_config = SpeculativeConfig.hf_config_override(draft_hf_config)
        assert draft_hf_config.model_type == "deepseek_mtp"
    draft_config = SimpleNamespace(hf_config=draft_hf_config)
    engine = _glm_engine(
        qconfig,
        speculative_config=SimpleNamespace(
            target_model_config=target_config,
            draft_model_config=draft_config,
        ),
    )
    engine.model_config = draft_config if converted_draft else target_config
    observed = {}

    class Decoder(nn.Module):
        def __init__(self, vllm_config, prefix, **kwargs):
            super().__init__()
            observed["config"] = vllm_config
            observed["scheme"] = vllm_config.quant_config.get_scheme_dict(
                nn.Linear(8, 8, bias=False), prefix + ".mlp.experts.0.down_proj",
            )

    monkeypatch.setattr(mtp_module, "DeepseekV2DecoderLayer", Decoder)
    monkeypatch.setattr(mtp_module, "RMSNorm", lambda *args, **kwargs: nn.Identity())
    monkeypatch.setattr(mtp_module, "SharedHead", lambda **kwargs: nn.Identity())
    layer = mtp_module.DeepSeekMultiTokenPredictorLayer(engine, "model.layers.78")
    assert isinstance(layer.mtp_block, Decoder)
    assert observed["config"] is not engine
    assert observed["scheme"]["weights"].num_bits == 4
    assert engine.quant_config is qconfig


@pytest.mark.parametrize("model_type", ["deepseek_v3", "deepseek_v32"])
def test_converted_deepseek_draft_does_not_enable_glm_fix(mtp_module, model_type):
    from copy import deepcopy
    from transformers import PretrainedConfig
    from vllm.config.speculative import SpeculativeConfig

    target_hf = PretrainedConfig(
        num_hidden_layers=78, num_nextn_predict_layers=1,
        architectures=["DeepseekV3ForCausalLM"],
    )
    target_hf.model_type = model_type
    draft_hf = SpeculativeConfig.hf_config_override(deepcopy(target_hf))
    assert draft_hf.model_type == "deepseek_mtp"
    qconfig = _mixed_quant_config(r"re:^model\.decoder\.mlp\.experts\.\d+\.down_proj$")
    engine = _glm_engine(qconfig, speculative_config=SimpleNamespace(
        target_model_config=SimpleNamespace(hf_config=target_hf),
    ))
    engine.model_config = SimpleNamespace(hf_config=draft_hf)
    assert mtp_module._mtp_quantization_config(engine, "model.layers.78") is engine


def test_eagle_loader_uses_original_target_for_glm_quant_aliases(mtp_module, monkeypatch):
    from copy import deepcopy
    from transformers import PretrainedConfig
    from vllm.config.speculative import SpeculativeConfig
    from vllm_hcu.patch.worker.framework_opt import patch_eagle_utils as eagle

    target_hf = PretrainedConfig(
        num_hidden_layers=78, num_nextn_predict_layers=1,
        architectures=["GlmMoeDsaForCausalLM"],
    )
    target_hf.model_type = "glm_moe_dsa"
    target_config = SimpleNamespace(hf_config=target_hf)
    draft_hf = SpeculativeConfig.hf_config_override(deepcopy(target_hf))
    draft_config = SimpleNamespace(hf_config=draft_hf)
    qconfig = _mixed_quant_config(r"re:^model\.decoder\.mlp\.experts\.\d+\.down_proj$")
    engine = _glm_engine(qconfig, speculative_config=SimpleNamespace(
        target_model_config=target_config, draft_model_config=draft_config,
    ), compilation_config=SimpleNamespace(static_forward_context={}, static_all_moe_layers=[]))
    engine.model_config = target_config
    result = object()

    def load_eagle_model(target_model, vllm_config):
        assert vllm_config.model_config is draft_config
        assert vllm_config.model_config.hf_config.model_type == "deepseek_mtp"
        adapted = mtp_module._mtp_quantization_config(vllm_config, "model.layers.78")
        assert adapted is not vllm_config
        assert adapted.quant_config.get_scheme_dict(
            nn.Linear(8, 8, bias=False), "model.layers.78.mlp.experts.0.down_proj",
        )["weights"].num_bits == 4
        return result

    def should_share(eagle, flag, draft, target):
        return False

    module = ModuleType(eagle.TARGET_MODULE)
    module.load_eagle_model = load_eagle_model
    module._should_share = should_share
    monkeypatch.setattr(eagle, "get_hcu_config", lambda _: SimpleNamespace(enable_multi_layers_mtp=False))
    eagle.apply_to_module(module)
    assert module.load_eagle_model(object(), engine) is result
    assert engine.model_config is target_config
    assert engine.quant_config is qconfig


@pytest.mark.parametrize("value", [None, "1", "true", "TRUE"])
def test_mtp_prefix_fix_is_enabled_by_default(mtp_module, monkeypatch, value):
    if value is None:
        monkeypatch.delenv("VLLM_HCU_GLM_MTP_QUANT_PREFIX_FIX", raising=False)
    else:
        monkeypatch.setenv("VLLM_HCU_GLM_MTP_QUANT_PREFIX_FIX", value)
    assert mtp_module.henvs.VLLM_HCU_GLM_MTP_QUANT_PREFIX_FIX
    qconfig = _mixed_quant_config(r"re:^model\.decoder\.mlp\.experts\.\d+\.down_proj$")
    engine = _glm_engine(qconfig)
    adapted = mtp_module._mtp_quantization_config(engine, "model.layers.78")
    assert adapted is not engine
    linear = nn.Linear(8, 8, bias=False)
    name = "model.layers.78.mlp.experts.0.down_proj"
    assert adapted.quant_config.get_scheme_dict(linear, name)["weights"].num_bits == 4
    assert qconfig.get_scheme_dict(linear, name)["weights"].num_bits == 8


@pytest.mark.parametrize("value", ["0", "false", "unexpected"])
def test_mtp_prefix_fix_can_be_disabled(mtp_module, monkeypatch, value):
    monkeypatch.setenv("VLLM_HCU_GLM_MTP_QUANT_PREFIX_FIX", value)
    qconfig = _mixed_quant_config(r"re:^model\.decoder\.mlp\.experts\.\d+\.down_proj$")
    engine = _glm_engine(qconfig)
    assert mtp_module._mtp_quantization_config(engine, "model.layers.78") is engine
    assert qconfig.get_scheme_dict(
        nn.Linear(8, 8, bias=False), "model.layers.78.mlp.experts.0.down_proj",
    )["weights"].num_bits == 8


@pytest.mark.parametrize("model_type", [
    "deepseek_v2", "deepseek_v3", "deepseek_v32", "mimo_v2_flash",
    "qwen3_moe", "glm4_moe", None,
])
def test_non_glm_dsa_model_keeps_original_config_with_default_enabled(
    mtp_module, model_type,
):
    qconfig = _mixed_quant_config(r"re:^model\.decoder\.mlp\.experts\.\d+\.down_proj$")
    engine = _glm_engine(qconfig)
    engine.model_config.hf_config.model_type = model_type
    assert mtp_module.henvs.VLLM_HCU_GLM_MTP_QUANT_PREFIX_FIX
    original_targets = dict(qconfig.target_scheme_map)
    original_ignore = list(qconfig.ignore)
    assert mtp_module._mtp_quantization_config(engine, "model.layers.78") is engine
    assert qconfig.target_scheme_map == original_targets
    assert qconfig.ignore == original_ignore
