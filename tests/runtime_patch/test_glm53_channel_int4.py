# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Packing conversion, AITER gemm2 layout, and quant detection for GLM-5.3 channel INT4."""

import ast
import sys
import time
from collections.abc import Iterable
from functools import partial
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import cast

import pytest
import torch

from vllm_hcu.model_executor.layers.quantization.glm_channel_int4 import (
    convert_compressed_tensors_int4_to_slimquant,
    is_glm_channel_int4_w4a8,
    normalize_routed_expert_checkpoint_name,
    repack_and_shuffle_w4a8_gemm2,
    repack_slimquant_w4a8_for_aiter,
)
from vllm_hcu.model_executor.layers.quantization.slimquant_facade import (
    GlmMoeDsaW4A8Facade,
)


def _pack_compressed_tensors_int4(values: torch.Tensor) -> torch.Tensor:
    """Pack into int32 with compressed-tensors uint4b8 (value plus 8), low nibble first."""
    if values.shape[-1] % 8:
        raise AssertionError("K must be a multiple of 8")
    nibbles = ((values.to(torch.int32) + 8) & 0xF).reshape(*values.shape[:-1], -1, 8)
    packed = torch.zeros(
        *values.shape[:-1],
        values.shape[-1] // 8,
        dtype=torch.int32,
    )
    for index in range(8):
        packed |= nibbles[..., index] << (4 * index)
    return packed


def _unpack_slimquant_int4(packed: torch.Tensor) -> torch.Tensor:
    raw = packed.view(torch.uint8)
    high = ((raw >> 4) & 0xF).to(torch.int16)
    low = (raw & 0xF).to(torch.int16)
    high = torch.where(high >= 8, high - 16, high)
    low = torch.where(low >= 8, low - 16, low)
    return torch.stack((high, low), dim=-1).flatten(-2).to(torch.int8)


def _quant_config() -> dict:
    return {
        "quant_method": "compressed-tensors",
        "format": "pack-quantized",
        "config_groups": {
            "group_int4": {
                "targets": ["re:.*\\.mlp\\.experts.*"],
                "format": "pack-quantized",
                "weights": {
                    "num_bits": 4,
                    "strategy": "channel",
                    "type": "int",
                    "symmetric": True,
                },
                "input_activations": {
                    "num_bits": 8,
                    "strategy": "token",
                    "dynamic": True,
                    "type": "int",
                },
            },
            "group_int8": {
                "targets": ["Linear"],
                "format": "int-quantized",
                "weights": {
                    "num_bits": 8,
                    "strategy": "channel",
                    "type": "int",
                },
                "input_activations": {
                    "num_bits": 8,
                    "strategy": "token",
                    "dynamic": True,
                    "type": "int",
                },
            },
        },
        "ignore": ["lm_head"],
    }


def _glm_hf_config() -> SimpleNamespace:
    return SimpleNamespace(
        model_type="glm_moe_dsa",
        architectures=["GlmMoeDsaForCausalLM"],
        num_hidden_layers=61,
    )


def test_compressed_tensors_int4_matches_slimquant_nibbles() -> None:
    values = torch.tensor(
        [[1, -1, 7, -8, 0, 2, 3, 4], [-7, 5, -3, 6, 1, 1, -8, 7]],
        dtype=torch.int8,
    )
    packed = _pack_compressed_tensors_int4(values)
    assert tuple(packed.shape) == (2, 1)
    converted = convert_compressed_tensors_int4_to_slimquant(packed)
    assert converted.dtype == torch.int8
    assert tuple(converted.shape) == (2, 4)
    assert torch.equal(_unpack_slimquant_int4(converted), values)


def test_int4_converter_rejects_unpacked_storage() -> None:
    with pytest.raises(ValueError, match="int32"):
        convert_compressed_tensors_int4_to_slimquant(torch.zeros(2, 4, dtype=torch.int8))


def test_aiter_w4a8_repack_regroups_eight_nibbles() -> None:
    # SlimQuant bytes encode [0,1], [2,3], [4,5], [6,7].
    packed = torch.tensor(
        [[0x01, 0x23, 0x45, 0x67]], dtype=torch.uint8
    ).view(torch.int8)
    repacked = repack_slimquant_w4a8_for_aiter(packed)
    assert repacked.tolist() == [[0x04, 0x15, 0x26, 0x37]]


def test_gemm2_repack_shuffles_each_regrouped_expert() -> None:
    weight = torch.arange(2 * 3 * 4, dtype=torch.uint8).reshape(2, 3, 4)
    weight = weight.view(torch.int8)
    seen = []

    def fake_shuffle(expert: torch.Tensor) -> torch.Tensor:
        seen.append((expert.dtype, tuple(expert.shape)))
        return expert.flip(0)

    result = repack_and_shuffle_w4a8_gemm2(weight, shuffle_fn=fake_shuffle)

    regrouped = repack_slimquant_w4a8_for_aiter(weight).view(torch.uint8)
    expected = torch.stack([regrouped[index].flip(0) for index in range(2)])
    assert seen == [(torch.uint8, (3, 4)), (torch.uint8, (3, 4))]
    assert result.dtype == torch.int8
    assert torch.equal(result.view(torch.uint8), expected)


def test_gemm2_repack_rejects_non_expert_rank() -> None:
    with pytest.raises(ValueError, match=r"\[E, N, K/2\]"):
        repack_and_shuffle_w4a8_gemm2(
            torch.zeros(3, 4, dtype=torch.int8), shuffle_fn=lambda tensor: tensor
        )


def test_expert_int4_after_ignored_dense_layers_selects_w4a8() -> None:
    config = _glm_hf_config()
    quant = _quant_config()
    quant["config_groups"]["group_int4"]["targets"] = [
        "re:^model\\.layers\\.([3-9]|[1-9]\\d+)\\.mlp\\.experts.*"
    ]
    quant["ignore"] = ["re:^model\\.layers\\.[0-2]\\..*", "lm_head"]
    assert is_glm_channel_int4_w4a8(quant, config) is True
    assert (
        GlmMoeDsaW4A8Facade.override_quantization_method(quant, None, config)
        == "glm_moe_dsa_w4a8"
    )


def test_anchored_expert_targets_past_the_dense_prefix_select_w4a8() -> None:
    config = _glm_hf_config()
    for index in (8, 60):
        quant = _quant_config()
        quant["config_groups"]["group_int4"]["targets"] = [
            f"re:^model\\.layers\\.{index}\\.mlp\\.experts.*"
        ]
        assert is_glm_channel_int4_w4a8(quant, config) is True
        assert (
            GlmMoeDsaW4A8Facade.override_quantization_method(quant, None, config)
            == "glm_moe_dsa_w4a8"
        )


def test_omitted_group_format_still_selects_channel_int4() -> None:
    config = _glm_hf_config()
    quant = _quant_config()
    del quant["config_groups"]["group_int4"]["format"]
    assert is_glm_channel_int4_w4a8(quant, config) is True
    assert (
        GlmMoeDsaW4A8Facade.override_quantization_method(quant, None, config)
        == "glm_moe_dsa_w4a8"
    )
    config = _glm_hf_config()
    quant = _quant_config()
    quant["config_groups"]["group_int4"]["targets"] = ["re:.*self_attn.*"]
    quant["ignore"] = ["re:.*mlp\\.experts.*"]
    assert is_glm_channel_int4_w4a8(quant) is False
    assert (
        GlmMoeDsaW4A8Facade.override_quantization_method(quant, None, config) is None
    )


def _int4_scheme() -> dict:
    return {
        "weights": {
            "num_bits": 4,
            "strategy": "channel",
            "type": "int",
            "symmetric": True,
        },
        "input_activations": {
            "num_bits": 8,
            "strategy": "token",
            "dynamic": True,
            "type": "int",
        },
        "format": "pack-quantized",
    }


def _int8_scheme() -> dict:
    return {
        "weights": {"num_bits": 8, "strategy": "channel", "type": "int"},
        "input_activations": {
            "num_bits": 8,
            "strategy": "token",
            "dynamic": True,
            "type": "int",
        },
        "format": "int-quantized",
    }


def _selection_config(**overrides: object) -> SimpleNamespace:
    config = SimpleNamespace(
        ignore=[],
        packed_modules_mapping={},
        target_scheme_map={
            "re:.*\\.mlp\\.experts.*": _int4_scheme(),
            "Linear": _int8_scheme(),
        },
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


@pytest.mark.parametrize("symmetric", ["omitted", True, False, None])
@pytest.mark.parametrize("args_as_objects", [False, True])
def test_int4_weight_symmetry_defaults_only_when_omitted(
    symmetric, args_as_objects
) -> None:
    from vllm_hcu.model_executor.layers.quantization.glm_channel_int4 import (
        routed_expert_uses_glm_channel_int4,
    )

    quant = _quant_config()
    group = quant["config_groups"]["group_int4"]
    if symmetric == "omitted":
        del group["weights"]["symmetric"]
    else:
        group["weights"]["symmetric"] = symmetric
    if args_as_objects:
        group["weights"] = SimpleNamespace(**group["weights"])
        group["input_activations"] = SimpleNamespace(**group["input_activations"])
    expected = symmetric == "omitted" or symmetric is True
    assert is_glm_channel_int4_w4a8(quant, _glm_hf_config()) is expected
    assert GlmMoeDsaW4A8Facade.override_quantization_method(
        quant, None, _glm_hf_config()
    ) == ("glm_moe_dsa_w4a8" if expected else None)
    config = _selection_config(target_scheme_map={group["targets"][0]: group})
    assert routed_expert_uses_glm_channel_int4(
        config, SimpleNamespace(), "model.layers.0.mlp.experts"
    ) is expected


@pytest.mark.parametrize("group_name,field_path", [
    ("group_int8", ("weights", "num_bits")),
    ("group_int8", ("input_activations", "num_bits")),
    ("group_int4", ("input_activations", "num_bits")),
])
def test_quant_num_bits_defaults_are_applied_to_raw_detection(
    group_name, field_path
) -> None:
    quant = _quant_config()
    group = quant["config_groups"][group_name]
    del group[field_path[0]][field_path[1]]
    assert is_glm_channel_int4_w4a8(quant, _glm_hf_config()) is True
    assert GlmMoeDsaW4A8Facade.override_quantization_method(
        quant, None, _glm_hf_config()
    ) == "glm_moe_dsa_w4a8"


@pytest.mark.parametrize("group_name", ["group_int4", "group_int8"])
@pytest.mark.parametrize("field", ["weights", "input_activations"])
@pytest.mark.parametrize("quant_type", ["omitted", "int", "float", None])
def test_quant_type_defaults_and_explicit_overrides(
    group_name, field, quant_type
) -> None:
    from vllm_hcu.model_executor.layers.quantization.glm_channel_int4 import (
        routed_expert_uses_glm_channel_int4,
    )
    from vllm_hcu.model_executor.layers.quantization.glm_moe_dsa_w4a8 import (
        GlmMoeDsaChannelW4A8Config,
    )

    quant = _quant_config()
    group = quant["config_groups"][group_name]
    if quant_type == "omitted":
        del group[field]["type"]
    else:
        group[field]["type"] = quant_type
    expected = quant_type in ("omitted", "int")
    assert is_glm_channel_int4_w4a8(quant, _glm_hf_config()) is expected
    for config_cls in (GlmMoeDsaW4A8Facade, GlmMoeDsaChannelW4A8Config):
        assert config_cls.override_quantization_method(
            quant, None, _glm_hf_config()
        ) == ("glm_moe_dsa_w4a8" if expected else None)
    if group_name == "group_int4":
        for args_as_objects in (False, True):
            scheme = dict(group)
            if args_as_objects:
                scheme["weights"] = SimpleNamespace(**group["weights"])
                scheme["input_activations"] = SimpleNamespace(
                    **group["input_activations"]
                )
            config = _selection_config(
                target_scheme_map={group["targets"][0]: scheme}
            )
            assert routed_expert_uses_glm_channel_int4(
                config, SimpleNamespace(), "model.layers.3.mlp.experts"
            ) is expected


_OMIT = object()


@pytest.mark.parametrize("group_name", ["group_int4", "group_int8"])
@pytest.mark.parametrize(
    "strategy,group_size,expected",
    [
        # QuantizationArgs infers CHANNEL from group_size=-1 when strategy is
        # omitted or null, and parses enum strings case-insensitively.
        (_OMIT, -1, True),
        (None, -1, True),
        ("channel", -1, True),
        ("CHANNEL", _OMIT, True),
        ("Channel", -1, True),
        # Every other combination is either another strategy or rejected by
        # the schema, which native parsing would also refuse to load.
        (_OMIT, _OMIT, False),
        (None, _OMIT, False),
        (_OMIT, 128, False),
        ("channel", 128, False),
        (_OMIT, -2, False),
        ("tensor", _OMIT, False),
        ("token", _OMIT, False),
        ("not-a-strategy", _OMIT, False),
    ],
)
def test_channel_strategy_follows_the_quantization_schema(
    glm_quant_method_labels, group_name, strategy, group_size, expected
) -> None:
    from vllm.model_executor.layers.fused_moe import RoutedExperts

    glm_moe_dsa_w4a8 = glm_quant_method_labels
    quant = _quant_config()
    weights = quant["config_groups"][group_name]["weights"]
    for key, value in (("strategy", strategy), ("group_size", group_size)):
        if value is _OMIT:
            weights.pop(key, None)
        else:
            weights[key] = value
    hf_config = _glm_hf_config()
    assert is_glm_channel_int4_w4a8(quant, hf_config) is expected
    for config_cls in (
        GlmMoeDsaW4A8Facade,
        glm_moe_dsa_w4a8.GlmMoeDsaChannelW4A8Config,
    ):
        assert config_cls.override_quantization_method(
            quant, None, hf_config
        ) == ("glm_moe_dsa_w4a8" if expected else None)
    if expected and group_name == "group_int4":
        # The parsed config must reach the same decision during layer selection.
        config = glm_moe_dsa_w4a8.GlmMoeDsaChannelW4A8Config.from_config(quant)
        config.packed_modules_mapping = {}
        layer = RoutedExperts.__new__(RoutedExperts)
        layer.moe_config = SimpleNamespace(num_logical_experts=2)
        assert config.get_quant_method(layer, "model.layers.4.mlp.experts") == "int4"


@pytest.mark.parametrize("field", ["weights", "input_activations"])
@pytest.mark.parametrize("group_name", ["group_int4", "group_int8"])
@pytest.mark.parametrize("quant_type", ["INT", "Int", "int"])
def test_quant_type_is_parsed_case_insensitively(group_name, field, quant_type) -> None:
    quant = _quant_config()
    quant["config_groups"][group_name][field]["type"] = quant_type
    assert is_glm_channel_int4_w4a8(quant, _glm_hf_config()) is True


def test_schema_rejected_group_selects_nothing() -> None:
    """A group native parsing cannot load must not select the GLM method."""
    quant = _quant_config()
    quant["config_groups"]["group_int4"]["weights"]["num_bits"] = "four"
    assert is_glm_channel_int4_w4a8(quant, _glm_hf_config()) is False
    quant = _quant_config()
    quant["config_groups"]["group_int8"]["input_activations"] = ["token"]
    assert is_glm_channel_int4_w4a8(quant, _glm_hf_config()) is False


@pytest.mark.parametrize("symmetric", [None, True, False])
@pytest.mark.parametrize("args_as_objects", [False, True])
def test_int4_activation_symmetry_is_respected(symmetric, args_as_objects) -> None:
    from vllm_hcu.model_executor.layers.quantization.glm_channel_int4 import (
        routed_expert_uses_glm_channel_int4,
    )

    quant = _quant_config()
    group = quant["config_groups"]["group_int4"]
    if symmetric is not None:
        group["input_activations"]["symmetric"] = symmetric
    if args_as_objects:
        group["weights"] = SimpleNamespace(**group["weights"])
        group["input_activations"] = SimpleNamespace(**group["input_activations"])
    expected = symmetric is not False
    assert is_glm_channel_int4_w4a8(quant, _glm_hf_config()) is expected
    assert GlmMoeDsaW4A8Facade.override_quantization_method(
        quant, None, _glm_hf_config()
    ) == ("glm_moe_dsa_w4a8" if expected else None)
    config = _selection_config(target_scheme_map={group["targets"][0]: group})
    assert routed_expert_uses_glm_channel_int4(
        config, SimpleNamespace(), "model.layers.0.mlp.experts"
    ) is expected


def test_int8_linear_symmetry_stays_on_compressed_tensors() -> None:
    quant = _quant_config()
    quant["config_groups"]["group_int8"]["input_activations"]["symmetric"] = False
    assert is_glm_channel_int4_w4a8(quant, _glm_hf_config())


@pytest.mark.parametrize("symmetric", [_OMIT, True, False, None])
def test_int8_linear_weights_must_be_symmetric_like_native(symmetric) -> None:
    """Native dynamic-token W8A8 rejects asymmetric weights, as does detection."""
    quant = _quant_config()
    weights = quant["config_groups"]["group_int8"]["weights"]
    if symmetric is _OMIT:
        weights.pop("symmetric", None)
    else:
        weights["symmetric"] = symmetric
    expected = symmetric is _OMIT or symmetric is True
    assert is_glm_channel_int4_w4a8(quant, _glm_hf_config()) is expected
    assert GlmMoeDsaW4A8Facade.override_quantization_method(
        quant, None, _glm_hf_config()
    ) == ("glm_moe_dsa_w4a8" if expected else None)


@pytest.mark.parametrize(
    "group_format",
    ["int-quantized", "naive-quantized", "float-quantized", "pack-quantized", _OMIT],
)
@pytest.mark.parametrize("top_format", ["int-quantized", "pack-quantized", _OMIT])
def test_int8_linear_group_needs_an_activation_quantization_format(
    group_format, top_format
) -> None:
    """Native builds dynamic-token W8A8 only for activation-quantization formats."""
    from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
        is_activation_quantization_format,
    )

    quant = _quant_config()
    group = quant["config_groups"]["group_int8"]
    for owner, value, key in ((group, group_format, "format"), (quant, top_format, "format")):
        if value is _OMIT:
            owner.pop(key, None)
        else:
            owner[key] = value
    effective = top_format if group_format is _OMIT else group_format
    expected = is_activation_quantization_format(None if effective is _OMIT else effective)
    assert is_glm_channel_int4_w4a8(quant, _glm_hf_config()) is expected
    assert GlmMoeDsaW4A8Facade.override_quantization_method(
        quant, None, _glm_hf_config()
    ) == ("glm_moe_dsa_w4a8" if expected else None)


@pytest.mark.parametrize("value", [["token"], "channel", 4, 1.5, True, b"channel"])
def test_non_dict_json_values_are_rejected_not_passed_through(value) -> None:
    from vllm_hcu.model_executor.layers.quantization.glm_channel_int4 import (
        _parse_quant_args,
    )

    assert _parse_quant_args(value) is None


def test_none_and_already_parsed_arguments_pass_through_the_parser() -> None:
    from vllm_hcu.model_executor.layers.quantization.glm_channel_int4 import (
        _parse_quant_args,
    )

    parsed = SimpleNamespace(num_bits=4)
    assert _parse_quant_args(parsed) is parsed
    assert _parse_quant_args(None) is None


@pytest.mark.parametrize(
    "error", [TypeError("odd value"), NotImplementedError("float width")]
)
def test_schema_errors_beyond_value_error_select_nothing(monkeypatch, error) -> None:
    """compressed-tensors validators can raise these as well as ValueError."""
    from compressed_tensors.quantization import QuantizationArgs

    def fail(value):
        raise error

    monkeypatch.setattr(QuantizationArgs, "model_validate", staticmethod(fail))
    assert is_glm_channel_int4_w4a8(_quant_config(), _glm_hf_config()) is False


@pytest.mark.parametrize(
    "architecture",
    ["Glm5NextForCausalLM", "Glm5NextForConditionalGeneration", "Glm5NextMTPModel"],
)
def test_glm5next_checkpoints_are_not_auto_selected(architecture) -> None:
    """Glm5Next loads through vllm.models.glm5next, not the patched DeepSeek loaders."""
    from vllm_hcu.model_executor.layers.quantization.glm_moe_dsa_w4a8 import (
        GlmMoeDsaChannelW4A8Config,
    )

    hf_config = SimpleNamespace(
        model_type="glm5next", architectures=[architecture], num_hidden_layers=61
    )
    quant = _quant_config()
    assert is_glm_channel_int4_w4a8(quant, hf_config) is True
    for config_cls in (GlmMoeDsaW4A8Facade, GlmMoeDsaChannelW4A8Config):
        assert config_cls.override_quantization_method(quant, None, hf_config) is None


@pytest.mark.parametrize("target", ["re:.*\\.mlp\\.experts.*", "RoutedExperts"])
def test_auto_detection_honors_projection_ignores(target) -> None:
    quant = _quant_config()
    quant["config_groups"]["group_int4"]["targets"] = [target]
    quant["ignore"] = ["re:.*experts\\.\\d+\\.(gate_proj|up_proj|down_proj)$"]
    assert is_glm_channel_int4_w4a8(quant, _glm_hf_config()) is False
    # One excluded layer must not hide other valid routed experts.
    quant["ignore"] = ["re:^model\\.layers\\.0\\.mlp\\.experts\\.0\\.gate_proj$"]
    assert is_glm_channel_int4_w4a8(quant, _glm_hf_config()) is True


@pytest.mark.parametrize("target_kind", ["projection", "owner", "class"])
@pytest.mark.parametrize("projection", ["gate_proj", "up_proj", "down_proj"])
def test_mtp_checkpoint_projection_ignores_apply_before_dispatch(
    target_kind, projection
) -> None:
    from vllm_hcu.model_executor.layers.quantization.glm_channel_int4 import (
        routed_expert_uses_glm_channel_int4,
    )

    class RoutedExperts:
        pass

    layer = RoutedExperts()
    prefix = "model.layers.61.mtp_block.mlp.experts"
    checkpoint_prefix = "model.layers.61.mlp.experts"
    targets = {
        "projection": "re:^model\\.layers\\.61\\.mlp\\.experts\\.0\\.(gate_proj|up_proj|down_proj)$",
        "owner": "re:.*\\.mlp\\.experts.*",
        "class": "RoutedExperts",
    }
    config = _selection_config(
        ignore=[f"{checkpoint_prefix}.0.{projection}"],
        target_scheme_map={targets[target_kind]: _int4_scheme()},
    )
    for candidate in (prefix, checkpoint_prefix):
        assert routed_expert_uses_glm_channel_int4(
            config, layer, candidate, num_hidden_layers=61
        ) is False
    config.ignore = []
    assert routed_expert_uses_glm_channel_int4(
        config, layer, prefix, num_hidden_layers=61
    ) is True


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("projection", ["gate_proj", "up_proj", "down_proj"])
def test_mtp_backbone_alias_honors_projection_ignores(wrapped, projection) -> None:
    from vllm_hcu.model_executor.layers.quantization.glm_channel_int4 import (
        routed_expert_uses_glm_channel_int4,
    )

    target = "re:^model\\.layers\\.0\\.mlp\\.experts\\.0\\.(gate_proj|up_proj|down_proj)$"
    config = _selection_config(
        ignore=[f"model.layers.0.mlp.experts.0.{projection}"],
        target_scheme_map={target: _int4_scheme()},
    )
    prefix = "model.layers.61.mlp.experts"
    if wrapped:
        prefix = "model.layers.61.mtp_block.mlp.experts"
    assert routed_expert_uses_glm_channel_int4(
        config, SimpleNamespace(), prefix, num_hidden_layers=61
    ) is False
    # Backbone targets do not change MTP's quantization scheme.
    config.target_scheme_map[target.replace("layers\\.0", "layers\\.1")] = _int4_scheme()
    assert routed_expert_uses_glm_channel_int4(
        config, SimpleNamespace(), prefix, num_hidden_layers=61
    ) is False
    config.target_scheme_map = {
        "re:^model\\.layers\\.\\d+\\.mlp\\.experts\\.0\\.(gate_proj|up_proj|down_proj)$": (
            _int4_scheme()
        )
    }
    assert routed_expert_uses_glm_channel_int4(
        config, SimpleNamespace(), prefix, num_hidden_layers=61
    ) is True


@pytest.mark.parametrize("projection", ["gate_proj", "up_proj", "down_proj"])
def test_wrapped_backbone_alias_honors_checkpoint_projection_ignores(projection):
    from vllm_hcu.model_executor.layers.quantization.glm_channel_int4 import (
        routed_expert_uses_glm_channel_int4,
    )

    config = _selection_config(
        ignore=[f"model.layers.0.mlp.experts.0.{projection}"],
        target_scheme_map={
            "re:^model\\.layers\\.0\\.mtp_block\\.mlp\\.experts.*": _int4_scheme(),
        },
    )
    prefix = "model.layers.61.mtp_block.mlp.experts"
    assert routed_expert_uses_glm_channel_int4(
        config, SimpleNamespace(), prefix, num_hidden_layers=61
    ) is False
    config.ignore = []
    assert routed_expert_uses_glm_channel_int4(
        config, SimpleNamespace(), prefix, num_hidden_layers=61
    ) is False


def test_routed_expert_selection_respects_targets_ignore_and_mtp() -> None:
    from vllm_hcu.model_executor.layers.quantization.glm_channel_int4 import (
        routed_expert_uses_glm_channel_int4,
    )

    layer = SimpleNamespace()
    expert = "model.layers.0.mlp.experts"
    mtp = "model.layers.61.mtp_block.mlp.experts"
    assert routed_expert_uses_glm_channel_int4(
        _selection_config(), layer, expert
    )
    assert routed_expert_uses_glm_channel_int4(_selection_config(), layer, mtp)
    assert (
        routed_expert_uses_glm_channel_int4(
            _selection_config(ignore=["re:.*mlp\\.experts.*"]),
            layer,
            expert,
        )
        is False
    )
    assert (
        routed_expert_uses_glm_channel_int4(
            _selection_config(ignore=["re:.*mtp_block.*"]),
            layer,
            mtp,
        )
        is False
    )
    non_expert = _selection_config(
        target_scheme_map={
            "re:.*self_attn.*": _int4_scheme(),
            "Linear": _int8_scheme(),
        }
    )
    assert routed_expert_uses_glm_channel_int4(non_expert, layer, expert) is False
    strict = _selection_config(
        target_scheme_map={
            "re:^model\\.layers\\.\\d+\\.mlp\\.experts.*": _int4_scheme(),
        }
    )
    assert routed_expert_uses_glm_channel_int4(strict, layer, expert)
    assert routed_expert_uses_glm_channel_int4(strict, layer, mtp)
    proj_targets = _selection_config(
        target_scheme_map={
            "re:.*\\.mlp\\.experts\\.\\d+\\.(gate_proj|up_proj|down_proj)$": (
                _int4_scheme()
            ),
        }
    )
    assert routed_expert_uses_glm_channel_int4(proj_targets, layer, expert)
    split_targets = _selection_config(
        target_scheme_map={
            "re:.*\\.mlp\\.experts\\.\\d+\\.gate_proj$": _int4_scheme(),
            "re:.*\\.mlp\\.experts\\.\\d+\\.up_proj$": _int4_scheme(),
            "re:.*\\.mlp\\.experts\\.\\d+\\.down_proj$": _int4_scheme(),
        }
    )
    assert routed_expert_uses_glm_channel_int4(split_targets, layer, expert)
    mixed_targets = _selection_config(
        target_scheme_map={
            "re:.*\\.mlp\\.experts\\.\\d+\\.gate_proj$": _int4_scheme(),
            "re:.*\\.mlp\\.experts\\.\\d+\\.up_proj$": _int8_scheme(),
            "re:.*\\.mlp\\.experts\\.\\d+\\.down_proj$": _int8_scheme(),
        }
    )
    assert (
        routed_expert_uses_glm_channel_int4(mixed_targets, layer, expert) is False
    )
    layer0_only = _selection_config(
        target_scheme_map={
            "re:^model\\.layers\\.0\\.mlp\\.experts.*": _int4_scheme(),
        }
    )
    mtp_expert = "model.layers.61.mlp.experts"
    backbone_expert = "model.layers.3.mlp.experts"
    assert routed_expert_uses_glm_channel_int4(
        layer0_only, layer, mtp_expert, num_hidden_layers=61
    ) is False
    assert routed_expert_uses_glm_channel_int4(
        layer0_only, layer, mtp, num_hidden_layers=61
    ) is False
    assert (
        routed_expert_uses_glm_channel_int4(
            layer0_only, layer, backbone_expert, num_hidden_layers=61
        )
        is False
    )
    assert (
        routed_expert_uses_glm_channel_int4(
            layer0_only, layer, mtp_expert, num_hidden_layers=80
        )
        is False
    )
    assert (
        routed_expert_uses_glm_channel_int4(
            strict, layer, "model.layers.0.mlp.shared_experts"
        )
        is False
    )
    class RoutedExperts:
        pass

    class_layer = RoutedExperts()
    class_targets = _selection_config(
        target_scheme_map={"RoutedExperts": _int4_scheme()}
    )
    assert routed_expert_uses_glm_channel_int4(
        class_targets, class_layer, "model.layers.8.mlp.experts"
    )
    linear_only = _selection_config(
        target_scheme_map={"Linear": _int4_scheme()}
    )
    assert (
        routed_expert_uses_glm_channel_int4(
            linear_only, class_layer, "model.layers.8.mlp.experts"
        )
        is False
    )
    omitted_format = _int4_scheme()
    omitted_format.pop("format")
    format_config = _selection_config(
        quant_format="pack-quantized",
        target_scheme_map={"re:.*\\.mlp\\.experts.*": omitted_format},
    )
    assert routed_expert_uses_glm_channel_int4(
        format_config, layer, "model.layers.0.mlp.experts"
    )
    projection_ignore = _selection_config(
        ignore=["re:.*experts\\.\\d+\\.(gate_proj|up_proj|down_proj)$"]
    )
    assert (
        routed_expert_uses_glm_channel_int4(
            projection_ignore, layer, "model.layers.0.mlp.experts"
        )
        is False
    )


def test_mtp_does_not_inherit_single_exact_backbone_expert_scheme() -> None:
    from vllm_hcu.model_executor.layers.quantization.glm_channel_int4 import (
        routed_expert_uses_glm_channel_int4,
    )

    config = _selection_config(
        target_scheme_map={
            "re:^model\\.layers\\.3\\.mlp\\.experts.*": _int4_scheme(),
        }
    )
    assert routed_expert_uses_glm_channel_int4(
        config, SimpleNamespace(), "model.layers.3.mlp.experts", num_hidden_layers=4
    ) is True
    assert routed_expert_uses_glm_channel_int4(
        config, SimpleNamespace(), "model.layers.4.mlp.experts", num_hidden_layers=4
    ) is False


@pytest.mark.parametrize("indexes", ["(3|4)", "[34]", "[0-4]"])
@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("mtp_scheme", [None, "int4", "int8"])
def test_mtp_requires_its_own_quantization_target(
    monkeypatch, indexes, wrapped, mtp_scheme
) -> None:
    from vllm.model_executor.layers.fused_moe import RoutedExperts
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
        CompressedTensorsConfig,
    )
    from vllm_hcu.model_executor.layers.quantization import glm_moe_dsa_w4a8
    from vllm_hcu.model_executor.layers.quantization.glm_channel_int4 import (
        routed_expert_uses_glm_channel_int4,
    )

    hf_config = _glm_hf_config()
    hf_config.num_hidden_layers = 5
    hf_config.first_k_dense_replace = 3
    hf_config.num_nextn_predict_layers = 1
    monkeypatch.setattr(
        "vllm.config.get_current_vllm_config",
        lambda: SimpleNamespace(model_config=SimpleNamespace(hf_config=hf_config)),
    )
    monkeypatch.setattr(
        glm_moe_dsa_w4a8, "GlmMoeDsaW4A8MoEMethod", lambda *args: "int4"
    )
    monkeypatch.setattr(
        CompressedTensorsConfig, "get_quant_method", lambda *args: "parent"
    )
    quant = _quant_config()
    target = rf"re:^model\.layers\.{indexes}\.mlp\.experts.*"
    quant["config_groups"]["group_int4"]["targets"] = [target]
    assert GlmMoeDsaW4A8Facade.override_quantization_method(
        quant, None, hf_config
    ) == "glm_moe_dsa_w4a8"
    config = glm_moe_dsa_w4a8.GlmMoeDsaChannelW4A8Config.__new__(
        glm_moe_dsa_w4a8.GlmMoeDsaChannelW4A8Config
    )
    config.ignore = []
    config.packed_modules_mapping = {}
    config.target_scheme_map = {target: _int4_scheme(), "Linear": _int8_scheme()}
    layer = RoutedExperts.__new__(RoutedExperts)
    layer.moe_config = SimpleNamespace()
    for index in (3, 4):
        assert (
            config.get_quant_method(layer, f"model.layers.{index}.mlp.experts")
            == "int4"
        )
    checkpoint_prefix = "model.layers.5.mlp.experts"
    prefix = checkpoint_prefix
    if wrapped:
        prefix = "model.layers.5.mtp_block.mlp.experts"
    if mtp_scheme is not None:
        config.target_scheme_map[checkpoint_prefix] = (
            _int4_scheme() if mtp_scheme == "int4" else _int8_scheme()
        )
    expected = mtp_scheme == "int4"
    assert routed_expert_uses_glm_channel_int4(
        config, layer, prefix, num_hidden_layers=5
    ) is expected
    assert config.get_quant_method(layer, prefix) == ("int4" if expected else "parent")
    config.ignore = [f"{checkpoint_prefix}.0.down_proj"]
    assert config.get_quant_method(layer, prefix) == "parent"


def test_get_quant_method_follows_prefix_scheme(monkeypatch) -> None:
    from vllm.model_executor.layers.fused_moe import RoutedExperts
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
        CompressedTensorsConfig,
    )
    from vllm_hcu.model_executor.layers.quantization import glm_moe_dsa_w4a8

    monkeypatch.setattr(
        glm_moe_dsa_w4a8,
        "GlmMoeDsaW4A8MoEMethod",
        lambda config, moe: "int4",
    )
    monkeypatch.setattr(
        glm_moe_dsa_w4a8,
        "UnquantizedFusedMoEMethod",
        lambda moe: "unquantized",
    )
    monkeypatch.setattr(
        CompressedTensorsConfig,
        "get_quant_method",
        lambda self, layer, prefix: "parent",
    )
    config = glm_moe_dsa_w4a8.GlmMoeDsaChannelW4A8Config.__new__(
        glm_moe_dsa_w4a8.GlmMoeDsaChannelW4A8Config
    )
    config.ignore = []
    config.packed_modules_mapping = {}
    config.target_scheme_map = {
        "re:.*\\.mlp\\.experts.*": _int4_scheme(),
        "Linear": _int8_scheme(),
    }
    layer = RoutedExperts.__new__(RoutedExperts)
    layer.moe_config = SimpleNamespace()
    assert config.get_quant_method(layer, "model.layers.0.mlp.experts") == "int4"
    assert (
        config.get_quant_method(layer, "model.layers.61.mtp_block.mlp.experts")
        == "int4"
    )
    config.ignore = ["re:.*mlp\\.experts.*"]
    assert (
        config.get_quant_method(layer, "model.layers.0.mlp.experts")
        == "unquantized"
    )
    config.ignore = []
    config.target_scheme_map = {
        "re:.*self_attn.*": _int4_scheme(),
        "Linear": _int8_scheme(),
    }
    assert config.get_quant_method(layer, "model.layers.0.mlp.experts") == "parent"
    config.target_scheme_map = {
        "re:^model\\.layers\\.0\\.mlp\\.experts.*": _int4_scheme(),
    }
    assert config.get_quant_method(layer, "model.layers.61.mlp.experts") == "parent"
    assert config.get_quant_method(layer, "model.layers.3.mlp.experts") == "parent"
    config.ignore = ["re:.*experts\\.\\d+\\.(gate_proj|up_proj|down_proj)$"]
    config.target_scheme_map = {"re:.*\\.mlp\\.experts.*": _int4_scheme()}
    assert config.get_quant_method(layer, "model.layers.0.mlp.experts") == "parent"
    config.ignore = []
    omitted_format = _int4_scheme()
    omitted_format.pop("format")
    config.quant_format = "pack-quantized"
    config.target_scheme_map = {"RoutedExperts": omitted_format}
    assert config.get_quant_method(layer, "model.layers.60.mlp.experts") == "int4"
    assert config.get_quant_method(layer, "model.layers.61.mlp.experts") == "int4"
    assert (
        config.get_quant_method(layer, "model.layers.61.mtp_block.mlp.experts")
        == "int4"
    )


@pytest.fixture
def glm_quant_method_labels(monkeypatch):
    """Keep native parsing/dispatch and replace only backend constructors."""
    from vllm_hcu.model_executor.layers.quantization import glm_moe_dsa_w4a8

    monkeypatch.setattr(
        glm_moe_dsa_w4a8,
        "GlmMoeDsaW4A8MoEMethod",
        lambda config, moe: "int4",
    )
    monkeypatch.setattr(
        glm_moe_dsa_w4a8,
        "UnquantizedFusedMoEMethod",
        lambda moe: "unquantized",
    )
    # Keep parser, parent get_quant_method, MoE dispatch and matching real.
    # Only replace final backend constructors so this is a CPU contract test.
    native_package = (
        "vllm.model_executor.layers.quantization.compressed_tensors."
        "compressed_tensors_moe"
    )
    for module_name, method_name, result in (
        (
            "compressed_tensors_moe_w8a8_int8",
            "CompressedTensorsW8A8Int8MoEMethod",
            "int8",
        ),
        (
            "compressed_tensors_moe_w4a8_int8",
            "CompressedTensorsW4A8Int8MoEMethod",
            "native-int4",
        ),
    ):
        name = f"{native_package}.{module_name}"
        module = ModuleType(name)
        setattr(module, method_name, partial(lambda label, *args: label, result))
        monkeypatch.setitem(sys.modules, name, module)
    return glm_moe_dsa_w4a8


@pytest.mark.parametrize(
    "projection_targets",
    [
        [
            "model.layers.4.mlp.experts.0.gate_proj",
            "model.layers.4.mlp.experts.0.up_proj",
            "model.layers.4.mlp.experts.0.down_proj",
        ],
        [
            r"re:^model\.layers\.4\.mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)$",
        ],
    ],
)
@pytest.mark.parametrize("layer_order", [(3, 4), (4, 3)])
def test_projection_targets_precede_generated_routed_experts_target(
    glm_quant_method_labels, projection_targets, layer_order
) -> None:
    """Resolve both layers with one parsed config and the actual parent dispatch."""
    from vllm.model_executor.layers.fused_moe import RoutedExperts

    glm_moe_dsa_w4a8 = glm_quant_method_labels

    hf_config = _glm_hf_config()
    hf_config.num_hidden_layers = 5
    hf_config.first_k_dense_replace = 3
    quant = _quant_config()
    quant["config_groups"]["group_int4"]["targets"] = projection_targets
    assert GlmMoeDsaW4A8Facade.override_quantization_method(
        quant, None, hf_config
    ) == "glm_moe_dsa_w4a8"
    config = glm_moe_dsa_w4a8.GlmMoeDsaChannelW4A8Config.from_config(quant)
    config.packed_modules_mapping = {}
    assert "RoutedExperts" not in config.target_scheme_map
    for index in layer_order:
        layer = RoutedExperts.__new__(RoutedExperts)
        layer.moe_config = SimpleNamespace()
        assert config.get_quant_method(layer, f"model.layers.{index}.mlp.experts") == (
            "int8" if index == 3 else "int4"
        )
        if index == 3:
            assert config.target_scheme_map["RoutedExperts"] is (
                config.target_scheme_map["Linear"]
            )


@pytest.mark.parametrize("local_bits,owner_bits", [(8, 4), (4, 8)])
@pytest.mark.parametrize("projection_regex", [False, True])
@pytest.mark.parametrize("local_first", [True, False])
@pytest.mark.parametrize("layer_order", [(3, 4), (4, 3)])
def test_projection_target_order_matches_native_with_broad_owner(
    glm_quant_method_labels,
    local_bits,
    owner_bits,
    projection_regex,
    local_first,
    layer_order,
) -> None:
    from vllm.model_executor.layers.fused_moe import RoutedExperts

    schemes = {4: _int4_scheme(), 8: _int8_scheme()}
    prefix = "model.layers.4.mlp.experts"
    projections = ("gate_proj", "up_proj", "down_proj")
    targets = (
        [r"re:^model\.layers\.4\.mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)$"]
        if projection_regex else [
            f"{prefix}.{index}.{proj}"
            for index in range(2)
            for proj in projections
        ]
    )
    groups = [
        ("local", dict(schemes[local_bits], targets=targets)),
        ("owner", dict(schemes[owner_bits], targets=[r"re:.*\.mlp\.experts.*"])),
    ]
    if not local_first:
        groups.reverse()
    quant = _quant_config()
    quant["config_groups"] = {
        **dict(groups),
        "linears": dict(_int8_scheme(), targets=["Linear"]),
    }
    hf_config = _glm_hf_config()
    assert GlmMoeDsaW4A8Facade.override_quantization_method(
        quant, None, hf_config
    ) == "glm_moe_dsa_w4a8"
    config = glm_quant_method_labels.GlmMoeDsaChannelW4A8Config.from_config(quant)
    config.packed_modules_mapping = {}
    for index in layer_order:
        layer = RoutedExperts.__new__(RoutedExperts)
        layer.moe_config = SimpleNamespace(num_logical_experts=2)
        current_prefix = f"model.layers.{index}.mlp.experts"
        expected_bits = local_bits if local_first and index == 4 else owner_bits
        native_bits = [
            config.get_scheme_dict(layer, f"{current_prefix}.0.{proj}")[
                "weights"
            ].num_bits
            for proj in projections
        ]
        assert native_bits == [expected_bits] * 3
        expected_method = "int4" if expected_bits == 4 else "int8"
        assert config.get_quant_method(layer, current_prefix) == expected_method
    # Re-select after any parent-generated class alias has been installed.
    layer = RoutedExperts.__new__(RoutedExperts)
    layer.moe_config = SimpleNamespace(num_logical_experts=2)
    assert config.get_quant_method(layer, prefix) == (
        "int4" if (local_bits if local_first else owner_bits) == 4 else "int8"
    )


@pytest.mark.parametrize("local_bits,owner_bits", [(8, 4), (4, 8)])
@pytest.mark.parametrize("projection_regex", [False, True])
def test_mixed_projections_are_not_hidden_by_broad_owner(
    glm_quant_method_labels, local_bits, owner_bits, projection_regex
) -> None:
    from vllm.model_executor.layers.fused_moe import RoutedExperts

    schemes = {4: _int4_scheme(), 8: _int8_scheme()}
    prefix = "model.layers.4.mlp.experts"
    target = (
        r"re:^model\.layers\.4\.mlp\.experts\.\d+\.gate_proj$"
        if projection_regex else f"{prefix}.0.gate_proj"
    )
    quant = _quant_config()
    quant["config_groups"] = {
        "local": dict(schemes[local_bits], targets=[target]),
        "owner": dict(schemes[owner_bits], targets=[r"re:.*\.mlp\.experts.*"]),
        "linears": dict(_int8_scheme(), targets=["Linear"]),
    }
    config = glm_quant_method_labels.GlmMoeDsaChannelW4A8Config.from_config(quant)
    config.packed_modules_mapping = {}
    layer = RoutedExperts.__new__(RoutedExperts)
    layer.moe_config = SimpleNamespace()
    with pytest.raises(ValueError, match="All MoE projections need to have same"):
        config.get_quant_method(layer, prefix)


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("local_bits,owner_bits", [(8, 4), (4, 8)])
@pytest.mark.parametrize("local_first", [True, False])
def test_projection_target_order_across_checkpoint_aliases(
    wrapped, local_bits, owner_bits, local_first
) -> None:
    from vllm.model_executor.layers.fused_moe import RoutedExperts
    from vllm_hcu.model_executor.layers.quantization.glm_channel_int4 import (
        routed_expert_uses_glm_channel_int4,
    )

    schemes = {4: _int4_scheme(), 8: _int8_scheme()}
    local = (
        r"re:^model\.layers\.61\.mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)$",
        schemes[local_bits],
    )
    owner = (r"re:.*\.mlp\.experts.*", schemes[owner_bits])
    targets = dict((local, owner) if local_first else (owner, local))
    prefix = "model.layers.61.mlp.experts"
    if wrapped:
        prefix = "model.layers.61.mtp_block.mlp.experts"
    config = _selection_config(target_scheme_map=targets)
    layer = RoutedExperts.__new__(RoutedExperts)
    assert routed_expert_uses_glm_channel_int4(config, layer, prefix) is (
        (local_bits if local_first else owner_bits) == 4
    )


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("owner_regex", [False, True])
@pytest.mark.parametrize(
    "owner_bits,gate_bits,class_bits,expected",
    [
        (4, None, 8, True),
        (8, None, 4, False),
        (4, 8, 4, False),
        (8, 4, 4, True),
        (4, 4, 8, False),
        (4, 4, None, False),
    ],
)
def test_fused_owner_fallback_requires_no_projection_name_match(
    wrapped, owner_regex, owner_bits, gate_bits, class_bits, expected
) -> None:
    from vllm.model_executor.layers.fused_moe import RoutedExperts
    from vllm_hcu.model_executor.layers.quantization.glm_channel_int4 import (
        routed_expert_uses_glm_channel_int4,
    )

    schemes = {4: _int4_scheme(), 8: _int8_scheme()}
    checkpoint_prefix = "model.layers.61.mlp.experts"
    owner_target = (
        r"re:^model\.layers\.61\.mlp\.experts$" if owner_regex else checkpoint_prefix
    )
    targets = {owner_target: schemes[owner_bits]}
    if gate_bits is not None:
        targets[f"{checkpoint_prefix}.0.gate_proj"] = schemes[gate_bits]
    if class_bits is not None:
        targets["RoutedExperts"] = schemes[class_bits]
    config = _selection_config(target_scheme_map=targets)
    prefix = (
        "model.layers.61.mtp_block.mlp.experts" if wrapped else checkpoint_prefix
    )
    layer = RoutedExperts.__new__(RoutedExperts)
    assert routed_expert_uses_glm_channel_int4(config, layer, prefix) is expected


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize(
    "class_bits,projection_bits,expected",
    [
        (8, (4, 4, 4), True),
        (4, (8, 8, 8), False),
        (4, (4, None, None), True),
        (8, (4, None, None), False),
        (4, (None, None, 8), False),
        (4, (None, None, None), True),
        (8, (None, None, None), False),
        (4, (4, 8, 4), False),
    ],
)
def test_expert_projection_targets_and_class_fallback_agree(
    wrapped, class_bits, projection_bits, expected
) -> None:
    from vllm.model_executor.layers.fused_moe import RoutedExperts
    from vllm_hcu.model_executor.layers.quantization.glm_channel_int4 import (
        routed_expert_uses_glm_channel_int4,
    )

    schemes = {4: _int4_scheme(), 8: _int8_scheme()}
    checkpoint_prefix = "model.layers.61.mlp.experts"
    runtime_prefix = (
        "model.layers.61.mtp_block.mlp.experts" if wrapped else checkpoint_prefix
    )
    scheme_map = {"RoutedExperts": schemes[class_bits]}
    for projection, bits in zip(
        ("gate_proj", "up_proj", "down_proj"), projection_bits
    ):
        if bits is not None:
            scheme_map[f"{checkpoint_prefix}.0.{projection}"] = schemes[bits]
    config = _selection_config(target_scheme_map=scheme_map)
    layer = RoutedExperts.__new__(RoutedExperts)
    assert routed_expert_uses_glm_channel_int4(config, layer, runtime_prefix) is expected
    config.ignore = [f"{checkpoint_prefix}.0.down_proj"]
    assert routed_expert_uses_glm_channel_int4(config, layer, runtime_prefix) is False


# Native matches the module class name with ``re:`` targets through re.match and
# with plain targets by case-insensitive containment.
_CLASS_TARGETS_SELECTING_ROUTED_EXPERTS = [
    "RoutedExperts",
    "re:^RoutedExperts$",
    "re:Routed.*",
    "re:.*Experts$",
    "Experts",
    "routedexperts",
]


@pytest.mark.parametrize("target", _CLASS_TARGETS_SELECTING_ROUTED_EXPERTS)
def test_class_name_targets_select_routed_experts_like_native(
    glm_quant_method_labels, target
) -> None:
    from vllm.model_executor.layers.fused_moe import RoutedExperts

    glm_moe_dsa_w4a8 = glm_quant_method_labels
    quant = _quant_config()
    quant["config_groups"]["group_int4"]["targets"] = [target]
    hf_config = _glm_hf_config()
    assert is_glm_channel_int4_w4a8(quant, hf_config) is True
    assert GlmMoeDsaW4A8Facade.override_quantization_method(
        quant, None, hf_config
    ) == "glm_moe_dsa_w4a8"
    config = glm_moe_dsa_w4a8.GlmMoeDsaChannelW4A8Config.from_config(quant)
    config.packed_modules_mapping = {}
    # Native resolves experts through the class name here, and the parent adds
    # a RoutedExperts -> Linear alias behind the INT4 target. The first target
    # in config order must keep winning on every layer.
    for index in (3, 4, 3):
        layer = RoutedExperts.__new__(RoutedExperts)
        layer.moe_config = SimpleNamespace(num_logical_experts=2)
        assert config.get_quant_method(
            layer, f"model.layers.{index}.mlp.experts"
        ) == "int4"


@pytest.mark.parametrize(
    "target",
    ["re:^Other$", "re:^routedexperts$", "SharedExperts", "Linear"],
)
def test_class_name_targets_that_miss_routed_experts_select_nothing(target) -> None:
    quant = _quant_config()
    quant["config_groups"]["group_int4"]["targets"] = [target]
    assert is_glm_channel_int4_w4a8(quant, _glm_hf_config()) is False


@pytest.mark.parametrize("layer_order", [(3, 4), (4, 3)])
@pytest.mark.parametrize("class_target", ["RoutedExperts", "re:^RoutedExperts$"])
def test_projection_path_targets_precede_class_name_targets(
    glm_quant_method_labels, class_target, layer_order
) -> None:
    """Native tries the layer path before the class name, for either spelling."""
    from vllm.model_executor.layers.fused_moe import RoutedExperts

    prefix = "model.layers.3.mlp.experts"
    projections = ("gate_proj", "up_proj", "down_proj")
    quant = _quant_config()
    quant["config_groups"] = {
        "classes": dict(_int4_scheme(), targets=[class_target]),
        "layer3": dict(
            _int8_scheme(),
            targets=[f"{prefix}.{e}.{p}" for e in range(2) for p in projections],
        ),
        "linears": dict(_int8_scheme(), targets=["Linear"]),
    }
    config = glm_quant_method_labels.GlmMoeDsaChannelW4A8Config.from_config(quant)
    config.packed_modules_mapping = {}
    for index in layer_order:
        layer = RoutedExperts.__new__(RoutedExperts)
        layer.moe_config = SimpleNamespace(num_logical_experts=2)
        assert config.get_quant_method(
            layer, f"model.layers.{index}.mlp.experts"
        ) == ("int8" if index == 3 else "int4")


@pytest.mark.parametrize(
    "targets",
    [
        [],
        ["Linear"],
        ["RoutedExperts"],
        ["re:^RoutedExperts$"],
        ["Linear", "RoutedExperts"],
        ["RoutedExperts", "re:^RoutedExperts$"],
        ["re:^RoutedExperts$", "Linear", "RoutedExperts"],
        ["Linear", "re:^Routed.*"],
        ["re:^Other$", "Experts"],
        ["experts", "RoutedExperts"],
        ["Linear", "re:^routedexperts$"],
        ["re:.*Experts$", "re:^Routed"],
    ],
)
def test_class_target_matching_agrees_with_native_find_matched_target(targets) -> None:
    """Differential check on the concrete class name, including target order."""
    try:
        from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
            find_matched_target,
        )
    except ImportError:
        pytest.skip("native find_matched_target is unavailable")
    from vllm.model_executor.layers.fused_moe import RoutedExperts
    from vllm_hcu.model_executor.layers.quantization.glm_channel_int4 import (
        _match_target_by_class,
    )

    layer = RoutedExperts.__new__(RoutedExperts)
    try:
        expected = find_matched_target(
            "model.layers.4.mlp.experts.0.gate_proj", layer, targets
        )
    except ValueError:
        expected = None
    assert _match_target_by_class(layer, dict.fromkeys(targets)) == expected


def test_class_targets_also_match_ancestor_class_names() -> None:
    """A renamed RoutedExperts subclass still resolves its base-class targets."""
    from vllm.model_executor.layers.fused_moe import RoutedExperts
    from vllm_hcu.model_executor.layers.quantization.glm_channel_int4 import (
        _match_target_by_class,
    )

    layer_class = type("RenamedExperts", (RoutedExperts,), {})
    layer = layer_class.__new__(layer_class)
    for target in ("RoutedExperts", "re:^RoutedExperts$"):
        assert _match_target_by_class(layer, {"Linear": None, target: None}) == target
    assert _match_target_by_class(layer, {"Linear": None, "re:^Other$": None}) is None


def test_class_target_is_only_resolved_for_projections_without_a_path_target(
    monkeypatch,
) -> None:
    from vllm.model_executor.layers.fused_moe import RoutedExperts
    from vllm_hcu.model_executor.layers.quantization import glm_channel_int4

    calls = []
    real = glm_channel_int4._match_target_by_class

    def record(*args):
        calls.append(args)
        return real(*args)

    monkeypatch.setattr(glm_channel_int4, "_match_target_by_class", record)
    layer = RoutedExperts.__new__(RoutedExperts)
    prefix = "model.layers.3.mlp.experts"
    # Every projection matches a path target, so the class is never consulted.
    assert glm_channel_int4.routed_expert_uses_glm_channel_int4(
        _selection_config(), layer, prefix
    ) is True
    assert calls == []
    # No path target matches, so the class target decides, once.
    config = _selection_config(
        target_scheme_map={"RoutedExperts": _int4_scheme(), "Linear": _int8_scheme()}
    )
    assert glm_channel_int4.routed_expert_uses_glm_channel_int4(
        config, layer, prefix
    ) is True
    assert len(calls) == 1


@pytest.mark.parametrize("kind", ["target", "ignore"])
@pytest.mark.parametrize("expert_id", [0, 1])
@pytest.mark.parametrize("projection_regex", [False, True])
@pytest.mark.parametrize("wrapped", [False, True])
def test_heterogeneous_experts_fail_before_fused_method_creation(
    glm_quant_method_labels, kind, expert_id, projection_regex, wrapped
) -> None:
    from vllm.model_executor.layers.fused_moe import RoutedExperts
    from vllm_hcu.model_executor.layers.quantization.glm_channel_int4 import (
        routed_expert_uses_glm_channel_int4,
    )

    checkpoint_prefix = "model.layers.4.mlp.experts"
    prefix = (
        "model.layers.4.mtp_block.mlp.experts" if wrapped else checkpoint_prefix
    )
    projections = ("gate_proj", "up_proj", "down_proj")
    targets = (
        [rf"re:^model\.layers\.4\.mlp\.experts\.{expert_id}\."
         r"(gate_proj|up_proj|down_proj)$"]
        if projection_regex else [
            f"{checkpoint_prefix}.{expert_id}.{proj}" for proj in projections
        ]
    )
    quant = _quant_config()
    if kind == "target":
        quant["config_groups"]["group_int4"]["targets"] = targets
    else:
        quant["ignore"] = targets
    hf_config = _glm_hf_config()
    hf_config.n_routed_experts = 2
    assert GlmMoeDsaW4A8Facade.override_quantization_method(
        quant, None, hf_config
    ) == "glm_moe_dsa_w4a8"
    config = glm_quant_method_labels.GlmMoeDsaChannelW4A8Config.from_config(quant)
    config.packed_modules_mapping = {}
    layer = RoutedExperts.__new__(RoutedExperts)
    # Validate global logical IDs even when EP only owns one expert and EPLB
    # has allocated extra physical slots.
    layer.moe_config = SimpleNamespace(
        num_logical_experts=2, num_experts=4, num_local_experts=1
    )
    layer.global_num_experts = 4
    layer.local_num_experts = 1
    config._add_fused_moe_to_target_scheme_map()
    for index in range(2):
        schemes = [
            config.get_scheme_dict(layer, f"{checkpoint_prefix}.{index}.{proj}")
            for proj in projections
        ]
        bits = [scheme["weights"].num_bits if scheme else None for scheme in schemes]
        expected = (4 if index == expert_id else 8) if kind == "target" else (
            None if index == expert_id else 4
        )
        assert bits == [expected] * 3
    for select in (
        lambda: routed_expert_uses_glm_channel_int4(config, layer, prefix),
        lambda: config.get_quant_method(layer, prefix),
    ):
        with pytest.raises(NotImplementedError, match="all logical experts") as error:
            select()
        assert prefix in str(error.value)
        assert ".0." in str(error.value) and ".1." in str(error.value)


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("projection", ["gate_proj", "up_proj", "down_proj"])
def test_single_ignored_projection_is_not_hidden_by_other_experts(
    glm_quant_method_labels, wrapped, projection
) -> None:
    from vllm.model_executor.layers.fused_moe import RoutedExperts

    quant = _quant_config()
    quant["ignore"] = [f"model.layers.4.mlp.experts.1.{projection}"]
    config = glm_quant_method_labels.GlmMoeDsaChannelW4A8Config.from_config(quant)
    config.packed_modules_mapping = {}
    layer = RoutedExperts.__new__(RoutedExperts)
    layer.moe_config = SimpleNamespace(num_logical_experts=2)
    prefix = "model.layers.4.mlp.experts"
    if wrapped:
        prefix = "model.layers.4.mtp_block.mlp.experts"
    with pytest.raises(NotImplementedError, match="all logical experts"):
        config.get_quant_method(layer, prefix)


@pytest.mark.parametrize("target_kind", ["projections", "owner", "class"])
@pytest.mark.parametrize("wrapped", [False, True])
def test_all_logical_experts_keep_uniform_int4_selection(
    glm_quant_method_labels, target_kind, wrapped
) -> None:
    from vllm.model_executor.layers.fused_moe import RoutedExperts

    prefix = "model.layers.4.mlp.experts"
    quant = _quant_config()
    targets = {
        "projections": [
            f"{prefix}.{index}.{proj}"
            for index in range(2)
            for proj in ("gate_proj", "up_proj", "down_proj")
        ],
        "owner": [prefix],
        "class": ["RoutedExperts"],
    }
    quant["config_groups"]["group_int4"]["targets"] = targets[target_kind]
    # EPLB physical slots are not checkpoint expert IDs and must not be probed.
    quant["ignore"] = [f"{prefix}.2.gate_proj"]
    config = glm_quant_method_labels.GlmMoeDsaChannelW4A8Config.from_config(quant)
    config.packed_modules_mapping = {}
    layer = RoutedExperts.__new__(RoutedExperts)
    layer.moe_config = SimpleNamespace(
        num_logical_experts=2, num_experts=4, num_local_experts=1
    )
    layer.global_num_experts = 4
    layer.local_num_experts = 1
    if wrapped:
        prefix = "model.layers.4.mtp_block.mlp.experts"
    assert config.get_quant_method(layer, prefix) == "int4"


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("ignore_kind", ["owner", "projections"])
def test_all_ignored_experts_do_not_select_glm_int4(wrapped, ignore_kind) -> None:
    from vllm.model_executor.layers.fused_moe import RoutedExperts
    from vllm_hcu.model_executor.layers.quantization.glm_channel_int4 import (
        routed_expert_uses_glm_channel_int4,
    )

    checkpoint_prefix = "model.layers.4.mlp.experts"
    ignores = {
        "owner": [checkpoint_prefix],
        "projections": [
            f"{checkpoint_prefix}.{index}.{proj}"
            for index in range(2)
            for proj in ("gate_proj", "up_proj", "down_proj")
        ],
    }
    config = _selection_config(ignore=ignores[ignore_kind])
    layer = RoutedExperts.__new__(RoutedExperts)
    layer.moe_config = SimpleNamespace(num_logical_experts=2)
    prefix = (
        "model.layers.4.mtp_block.mlp.experts" if wrapped else checkpoint_prefix
    )
    assert routed_expert_uses_glm_channel_int4(config, layer, prefix) is False
    quant = _quant_config()
    quant["ignore"] = ignores[ignore_kind]
    hf_config = _glm_hf_config()
    hf_config.n_routed_experts = 2
    quant["config_groups"]["group_int4"]["targets"] = [
        r"re:^model\.layers\.4\.mlp\.experts.*"
    ]
    assert GlmMoeDsaW4A8Facade.override_quantization_method(
        quant, None, hf_config
    ) is None


@pytest.mark.parametrize(
    "owner_ignore",
    [
        "model.layers.4.mlp.experts",
        r"re:^model\.layers\.4\.mlp\.experts$",
    ],
)
@pytest.mark.parametrize("wrapped", [False, True])
def test_owner_ignore_bypasses_parent_moe_dispatch(
    glm_quant_method_labels, monkeypatch, owner_ignore, wrapped
) -> None:
    from vllm.model_executor.layers.fused_moe import RoutedExperts
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
        CompressedTensorsConfig,
    )

    quant = _quant_config()
    quant["ignore"] = [owner_ignore]
    quant["config_groups"]["group_int4"]["targets"] = [
        r"re:.*\.mlp\.experts.*"
    ]
    config = glm_quant_method_labels.GlmMoeDsaChannelW4A8Config.from_config(quant)
    config.packed_modules_mapping = {}
    layer = RoutedExperts.__new__(RoutedExperts)
    layer.moe_config = SimpleNamespace(num_logical_experts=2)
    prefix = (
        "model.layers.4.mtp_block.mlp.experts" if wrapped
        else "model.layers.4.mlp.experts"
    )
    # Another layer still uses GLM INT4, so the facade remains active.
    assert config.get_quant_method(
        layer, "model.layers.3.mlp.experts"
    ) == "int4"
    monkeypatch.setattr(
        CompressedTensorsConfig,
        "get_quant_method",
        lambda *args: (_ for _ in ()).throw(
            AssertionError("owner ignore must not call parent MoE dispatch")
        ),
    )
    assert config.get_quant_method(layer, prefix) == "unquantized"


@pytest.mark.parametrize("projection", ["gate_proj", "up_proj", "down_proj"])
def test_last_logical_expert_is_checked_without_ep_locality(projection) -> None:
    from vllm_hcu.model_executor.layers.quantization.glm_channel_int4 import (
        routed_expert_uses_glm_channel_int4,
    )

    prefix = "model.layers.4.mlp.experts"
    layer = SimpleNamespace(
        moe_config=SimpleNamespace(num_logical_experts=16, num_local_experts=1),
        local_num_experts=1,
    )
    config = _selection_config(ignore=[f"{prefix}.15.{projection}"])
    with pytest.raises(NotImplementedError, match=r"experts\.15\."):
        routed_expert_uses_glm_channel_int4(config, layer, prefix)


def _routed_experts_layer(fused_shared_experts):
    """RoutedExperts as native leaves it before quant selection.

    ``num_logical_experts`` counts routed experts only. Fused shared experts are
    extra slots tracked by the expert map manager, or absent when None.
    """
    from vllm.model_executor.layers.fused_moe import RoutedExperts

    layer = RoutedExperts.__new__(RoutedExperts)
    layer.moe_config = SimpleNamespace(num_logical_experts=2)
    if fused_shared_experts is not None:
        layer.expert_map_manager = SimpleNamespace(
            num_fused_shared_experts=fused_shared_experts
        )
    return layer


def _fused_shared_scenario(glm_module, scenario):
    """(config, prefix, label) for each way get_quant_method can resolve a layer."""
    quant = _quant_config()
    prefix, label = "model.layers.4.mlp.experts", "int4"
    if scenario == "ignored-owner":
        quant["ignore"] = [prefix]
        label = "unquantized"
    elif scenario == "int8-experts":
        projections = ("gate_proj", "up_proj", "down_proj")
        prefix, label = "model.layers.3.mlp.experts", "int8"
        quant["config_groups"] = {
            "int4": dict(
                _int4_scheme(), targets=[r"re:^model\.layers\.4\.mlp\.experts.*"]
            ),
            "int8_layer": dict(
                _int8_scheme(),
                targets=[f"{prefix}.{e}.{p}" for e in range(2) for p in projections],
            ),
            "linears": dict(_int8_scheme(), targets=["Linear"]),
        }
    config = glm_module.GlmMoeDsaChannelW4A8Config.from_config(quant)
    config.packed_modules_mapping = {}
    return config, prefix, label


@pytest.mark.parametrize("scenario", ["int4", "ignored-owner", "int8-experts"])
def test_layers_that_fused_shared_experts_are_rejected_on_every_branch(
    glm_quant_method_labels, scenario
) -> None:
    config, prefix, label = _fused_shared_scenario(glm_quant_method_labels, scenario)
    # Not fused (no manager at all, or none fused): selection is unchanged.
    for fused in (None, 0):
        assert config.get_quant_method(_routed_experts_layer(fused), prefix) == label
    for fused in (1, 2):
        with pytest.raises(NotImplementedError, match="fused shared experts") as error:
            config.get_quant_method(_routed_experts_layer(fused), prefix)
        message = str(error.value)
        assert repr(prefix) in message
        assert f"fuses {fused} of them" in message
        assert "supports_fused_shared_experts" in message


def test_fused_shared_experts_are_rejected_for_mtp_prefixes(
    glm_quant_method_labels,
) -> None:
    config, _, _ = _fused_shared_scenario(glm_quant_method_labels, "int4")
    prefix = "model.layers.4.mtp_block.mlp.experts"
    assert config.get_quant_method(_routed_experts_layer(0), prefix) == "int4"
    with pytest.raises(NotImplementedError, match="fused shared experts"):
        config.get_quant_method(_routed_experts_layer(1), prefix)


_MODELS = Path(__file__).parents[2] / "vllm_hcu/models"


def _load_fused_shared_experts_gate(flag):
    """The production gate, extracted so the full model module is not imported."""
    path = _MODELS / "deepseek_v2.py"
    func = next(
        node
        for node in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(node, ast.FunctionDef)
        and node.name == "is_fused_shared_experts_enabled"
    )
    warnings = []
    namespace = {
        "QuantizationConfig": object,
        "rocm_aiter_ops": SimpleNamespace(
            is_fusion_moe_shared_experts_enabled=lambda: flag
        ),
        "logger": SimpleNamespace(
            warning_once=lambda message, *args: warnings.append(message % args)
        ),
    }
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            func,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace["is_fused_shared_experts_enabled"], warnings


def test_glm_config_does_not_support_fused_shared_experts() -> None:
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
        CompressedTensorsConfig,
    )
    from vllm_hcu.model_executor.layers.quantization.glm_moe_dsa_w4a8 import (
        GlmMoeDsaChannelW4A8Config,
    )

    assert GlmMoeDsaChannelW4A8Config.supports_fused_shared_experts is False
    # Every other config keeps today's behaviour: the plain config still fuses.
    gate, _ = _load_fused_shared_experts_gate(True)
    assert gate(CompressedTensorsConfig.__new__(CompressedTensorsConfig)) is True


@pytest.mark.parametrize("flag", [False, True])
def test_fused_shared_experts_gate_follows_the_quant_config(flag) -> None:
    from vllm_hcu.model_executor.layers.quantization.glm_moe_dsa_w4a8 import (
        GlmMoeDsaChannelW4A8Config,
    )

    gate, warnings = _load_fused_shared_experts_gate(flag)
    glm_config = GlmMoeDsaChannelW4A8Config.__new__(GlmMoeDsaChannelW4A8Config)
    assert gate(None) is flag
    assert gate(SimpleNamespace()) is flag
    assert warnings == []
    # Only this config keeps its shared experts separate, and says so.
    assert gate(glm_config) is False
    if flag:
        assert len(warnings) == 1 and "GlmMoeDsaChannelW4A8Config" in warnings[0]
    else:
        assert warnings == []


class _FusionFlagUses(ast.NodeVisitor):
    """Where model code reads the AITER fusion flag or calls the gate."""

    def __init__(self):
        self.scope, self.raw, self.gated = [], [], []

    def _enter(self, node):
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    visit_ClassDef = visit_FunctionDef = _enter

    def visit_Call(self, node):
        func = node.func
        where = ".".join(self.scope)
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "is_fusion_moe_shared_experts_enabled"
        ):
            self.raw.append(where)
        elif isinstance(func, ast.Name) and func.id == "is_fused_shared_experts_enabled":
            self.gated.append(where)
        self.generic_visit(node)


def _fusion_flag_uses(filename):
    visitor = _FusionFlagUses()
    visitor.visit(ast.parse((_MODELS / filename).read_text(encoding="utf-8")))
    return visitor


def test_model_code_decides_shared_expert_fusion_only_through_the_gate() -> None:
    """The model, its loader and the MTP loader must agree on one decision."""
    v2, mtp = _fusion_flag_uses("deepseek_v2.py"), _fusion_flag_uses("deepseek_mtp.py")
    # The invariant: the AITER flag is read in the gate and nowhere else here.
    assert v2.raw == ["is_fused_shared_experts_enabled"]
    assert mtp.raw == []
    # Each decision point goes through the gate (a new consumer may be added).
    assert {"DeepseekV2MoE.__init__", "DeepseekV2Model.load_weights"} <= set(v2.gated)
    assert "DeepSeekMTP.load_weights" in mtp.gated


@pytest.mark.parametrize(
    "filename,class_name",
    [("deepseek_v2.py", "DeepseekV2Model"), ("deepseek_mtp.py", "DeepSeekMTP")],
)
def test_loaders_that_use_the_gate_keep_the_quant_config(filename, class_name) -> None:
    """load_weights reads self.quant_config, so __init__ must have stored it."""
    tree = ast.parse((_MODELS / filename).read_text(encoding="utf-8"))
    model = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    init = next(
        node
        for node in model.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    assert any(
        isinstance(target, ast.Attribute)
        and target.attr == "quant_config"
        and isinstance(target.value, ast.Name)
        and target.value.id == "self"
        for node in ast.walk(init)
        if isinstance(node, ast.Assign)
        for target in node.targets
    )


def test_expert_loaders_convert_weights_and_scales() -> None:
    import inspect

    from vllm_hcu.model_executor.layers.quantization.glm_moe_dsa_w4a8 import (
        _wrap_checkpoint_loader,
    )

    loaded = []
    destinations = []

    def record(param, weight, *args, **kwargs):
        destinations.append(param)
        loaded.append(weight)

    scale_param = torch.nn.Parameter(torch.ones(2, 1), requires_grad=False)
    scale_param.weight_loader = record
    _wrap_checkpoint_loader(scale_param, scale=True)
    scale = torch.tensor([[0.5], [0.25]], dtype=torch.bfloat16)
    scale_param.weight_loader(scale_param, scale, "w13_weight_scale", shard_id="w1")
    assert loaded[0].dtype == torch.float32
    assert torch.equal(loaded[0], scale.float() / 16.0)

    weight_param = torch.nn.Parameter(
        torch.zeros(2, 4, dtype=torch.int8), requires_grad=False
    )
    weight_param.weight_loader = record
    _wrap_checkpoint_loader(weight_param, scale=False)
    values = torch.tensor(
        [[1, -1, 7, -8, 0, 2, 3, 4], [-7, 5, -3, 6, 1, 1, -8, 7]],
        dtype=torch.int8,
    )
    weight_param.weight_loader(weight_param, _pack_compressed_tensors_int4(values))
    assert loaded[1].dtype == torch.int8
    assert torch.equal(_unpack_slimquant_int4(loaded[1]), values)
    # Layerwise reload rebinds arguments["param"]; the name has to be visible.
    assert next(iter(inspect.signature(weight_param.weight_loader).parameters)) == (
        "param"
    )
    replacement = torch.nn.Parameter(
        torch.zeros_like(weight_param), requires_grad=False
    )
    bound = inspect.signature(weight_param.weight_loader).bind(
        weight_param, _pack_compressed_tensors_int4(values)
    )
    bound.apply_defaults()
    bound.arguments["param"] = replacement
    weight_param.weight_loader(*bound.args, **bound.kwargs)
    assert destinations[-1] is replacement


def test_layerwise_reload_writes_converted_weights_and_scales() -> None:
    from vllm.model_executor.model_loader.reload.layerwise import (
        finalize_layerwise_reload,
        initialize_layerwise_reload,
        record_metadata_for_reloading,
    )
    from vllm_hcu.model_executor.layers.quantization.glm_moe_dsa_w4a8 import (
        _wrap_checkpoint_loader,
    )

    devices: list[str] = []

    def base_loader(param, loaded_weight, *args, **kwargs):
        devices.append(param.device.type)
        if param.is_meta:
            # Meta parameters reject copy_. Count the converted payload so
            # layerwise reload still advances to the materialized parameter.
            counted = loaded_weight.detach().cpu().clone()
            counted.copy_(loaded_weight.detach().cpu())
            return
        param.data.copy_(loaded_weight)

    layer = torch.nn.Module()
    weight = torch.nn.Parameter(torch.zeros(2, 4, dtype=torch.int8), requires_grad=False)
    scale = torch.nn.Parameter(torch.ones(2, 1, dtype=torch.float32), requires_grad=False)
    weight.weight_loader = base_loader
    scale.weight_loader = base_loader
    layer.register_parameter("w13_weight", weight)
    layer.register_parameter("w13_weight_scale", scale)
    _wrap_checkpoint_loader(layer.w13_weight, scale=False)
    _wrap_checkpoint_loader(layer.w13_weight_scale, scale=True)

    first_values = torch.tensor(
        [[1, -1, 7, -8, 0, 2, 3, 4], [-7, 5, -3, 6, 1, 1, -8, 7]],
        dtype=torch.int8,
    )
    first_scale = torch.tensor([[0.5], [0.25]], dtype=torch.bfloat16)
    layer.w13_weight.weight_loader(
        layer.w13_weight, _pack_compressed_tensors_int4(first_values)
    )
    layer.w13_weight_scale.weight_loader(layer.w13_weight_scale, first_scale)
    assert devices == ["cpu", "cpu"]
    assert torch.equal(_unpack_slimquant_int4(layer.w13_weight.data), first_values)
    assert torch.equal(layer.w13_weight_scale.data, first_scale.float() / 16.0)

    reloaded_values = torch.tensor(
        [[2, -2, 1, -1, 7, -8, 0, 3], [4, -7, 5, -3, 6, 1, -8, 7]],
        dtype=torch.int8,
    )
    reloaded_scale = torch.tensor([[1.0], [0.5]], dtype=torch.bfloat16)
    devices.clear()
    record_metadata_for_reloading(layer)
    initialize_layerwise_reload(layer)
    layer.w13_weight.weight_loader(
        layer.w13_weight, _pack_compressed_tensors_int4(reloaded_values)
    )
    layer.w13_weight_scale.weight_loader(layer.w13_weight_scale, reloaded_scale)
    finalize_layerwise_reload(layer, None)

    assert devices == ["meta", "meta", "cpu", "cpu"]
    assert torch.equal(_unpack_slimquant_int4(layer.w13_weight.data), reloaded_values)
    assert torch.equal(layer.w13_weight_scale.data, reloaded_scale.float() / 16.0)


def test_glm53_quant_config_selects_mixed_w4a8() -> None:
    config = _glm_hf_config()
    assert GlmMoeDsaW4A8Facade.override_quantization_method(
        _quant_config(), None, config
    ) == "glm_moe_dsa_w4a8"
    assert GlmMoeDsaW4A8Facade.override_quantization_method(
        _quant_config(), "glm_moe_dsa_w4a8", config
    ) == "glm_moe_dsa_w4a8"
    assert is_glm_channel_int4_w4a8(_quant_config())
    # An explicit different -q keeps its own resolution path.
    assert GlmMoeDsaW4A8Facade.override_quantization_method(
        _quant_config(), "slimquant_marlin", config
    ) is None
    int8_only = {
        "quant_method": "compressed-tensors",
        "format": "int-quantized",
        "config_groups": {"group_int8": _quant_config()["config_groups"]["group_int8"]},
    }
    assert GlmMoeDsaW4A8Facade.override_quantization_method(
        int8_only, None, config
    ) is None
    other = SimpleNamespace(model_type="deepseek_v3", architectures=["DeepseekV3ForCausalLM"])
    assert GlmMoeDsaW4A8Facade.override_quantization_method(
        _quant_config(), None, other
    ) is None


def test_expert_checkpoint_names_drop_shape_and_rebind_packed_weights() -> None:
    packed = "model.layers.78.mlp.experts.0.gate_proj.weight_packed"
    shape = "model.layers.3.mlp.experts.0.down_proj.weight_shape"
    assert normalize_routed_expert_checkpoint_name(packed, True) == (
        "model.layers.78.mlp.experts.0.gate_proj.weight"
    )
    assert normalize_routed_expert_checkpoint_name(shape, True) is None
    # Packed-named expert parameters (compressed-tensors W4A16) keep both names.
    assert normalize_routed_expert_checkpoint_name(packed, False) == packed
    assert normalize_routed_expert_checkpoint_name(shape, False) == shape
    assert normalize_routed_expert_checkpoint_name(
        "model.layers.0.mlp.down_proj.weight", True
    ) == "model.layers.0.mlp.down_proj.weight"
    params = {
        "model.layers.3.mlp.experts.w13_weight_packed": object(),
        "model.layers.4.mlp.experts.w13_weight": object(),
        "model.layers.61.mtp_block.mlp.experts.w13_weight": object(),
    }
    w4a16 = "model.layers.3.mlp.experts.0.gate_proj.weight_packed"
    w4a8 = "model.layers.4.mlp.experts.0.gate_proj.weight_packed"
    mtp = "model.layers.61.mlp.experts.0.down_proj.weight_packed"
    assert normalize_routed_expert_checkpoint_name(w4a16, params_dict=params) == w4a16
    assert normalize_routed_expert_checkpoint_name(
        "model.layers.3.mlp.experts.0.gate_proj.weight_shape", params_dict=params
    ) == "model.layers.3.mlp.experts.0.gate_proj.weight_shape"
    assert normalize_routed_expert_checkpoint_name(w4a8, params_dict=params) == (
        "model.layers.4.mlp.experts.0.gate_proj.weight"
    )
    assert (
        normalize_routed_expert_checkpoint_name(
            "model.layers.4.mlp.experts.0.up_proj.weight_shape", params_dict=params
        )
        is None
    )
    assert normalize_routed_expert_checkpoint_name(mtp, params_dict=params) == (
        "model.layers.61.mlp.experts.0.down_proj.weight"
    )
    nested = {
        "layers.10.mlp.experts.routed_experts.w13_weight": object(),
        "layers.10.mlp.experts.routed_experts.w2_weight": object(),
        "layers.11.mlp.experts.routed_experts.w2_weight_packed": object(),
    }
    nested_w4a8 = "layers.10.mlp.experts.0.down_proj.weight_packed"
    nested_w4a16 = "layers.11.mlp.experts.0.down_proj.weight_packed"
    assert normalize_routed_expert_checkpoint_name(
        nested_w4a8, params_dict=nested
    ) == "layers.10.mlp.experts.0.down_proj.weight"
    assert (
        normalize_routed_expert_checkpoint_name(
            "layers.10.mlp.experts.0.down_proj.weight_shape", params_dict=nested
        )
        is None
    )
    assert (
        normalize_routed_expert_checkpoint_name(nested_w4a16, params_dict=nested)
        == nested_w4a16
    )


def _glm_aiter_method(moe_backend: str):
    from vllm_hcu.model_executor.layers.quantization.glm_moe_dsa_w4a8 import (
        GlmMoeDsaW4A8MoEMethod,
    )

    method = GlmMoeDsaW4A8MoEMethod.__new__(GlmMoeDsaW4A8MoEMethod)
    method.moe = SimpleNamespace(moe_backend=moe_backend)
    layer = torch.nn.Module()
    layer.w13_weight = torch.nn.Parameter(
        torch.arange(16, dtype=torch.int8).reshape(1, 4, 4), requires_grad=False
    )
    layer.w2_weight = torch.nn.Parameter(
        torch.arange(8, dtype=torch.int8).reshape(1, 4, 2), requires_grad=False
    )
    return method, layer


def _patch_glm_post_load(monkeypatch, config):
    from vllm_hcu.model_executor.layers.fused_moe import deepep_runtime
    from vllm_hcu.model_executor.layers.quantization import (
        compressed_tensors_moe_runtime,
        glm_moe_dsa_w4a8,
        slimquant_w4a8,
    )

    calls = {"repack": 0, "parent": []}

    def fake_repack(weight):
        calls["repack"] += 1
        return weight + 1

    def fake_parent(self, layer):
        calls["parent"].append(
            (
                bool(getattr(config, "need_shuffle", False)),
                bool(getattr(config, "need_shuffle_scale", False)),
            )
        )

    monkeypatch.setattr(
        deepep_runtime, "slimquant_w4a8_uses_deepep_auto", lambda moe: False
    )
    monkeypatch.setattr(
        compressed_tensors_moe_runtime,
        "prewarm_aiter_w4a8_moe",
        lambda method, layer: config,
    )
    monkeypatch.setattr(glm_moe_dsa_w4a8, "repack_and_shuffle_w4a8_gemm2", fake_repack)
    monkeypatch.setattr(
        slimquant_w4a8.SlimQuantW4A8Int8AiterMoEMethod,
        "process_weights_after_loading",
        fake_parent,
    )
    return calls


def test_glm_aiter_post_load_installs_gemm2_layout_once(monkeypatch) -> None:
    config = SimpleNamespace(need_shuffle=True, need_shuffle_scale=True)
    calls = _patch_glm_post_load(monkeypatch, config)
    method, layer = _glm_aiter_method("auto")
    original_w13 = layer.w13_weight.detach().clone()

    method.process_weights_after_loading(layer)
    method.process_weights_after_loading(layer)

    assert calls["repack"] == 2  # w13 and w2, only on the first call
    assert layer._glm_w4a8_gemm2_shuffled is True
    assert torch.equal(layer.w13_weight, original_w13 + 1)
    # The parent registers the gemm2 layout without generic AITER shuffles.
    assert calls["parent"] == [(False, False), (False, False)]


def test_glm_aiter_post_load_repacks_replaced_weights(monkeypatch) -> None:
    config = SimpleNamespace(need_shuffle=True, need_shuffle_scale=True)
    calls = _patch_glm_post_load(monkeypatch, config)
    method, layer = _glm_aiter_method("auto")
    method.process_weights_after_loading(layer)
    assert calls["repack"] == 2

    layer.w13_weight = torch.nn.Parameter(
        torch.arange(16, dtype=torch.int8).reshape(1, 4, 4), requires_grad=False
    )
    layer.w2_weight = torch.nn.Parameter(
        torch.arange(8, dtype=torch.int8).reshape(1, 4, 2), requires_grad=False
    )
    fresh = layer.w13_weight.detach().clone()
    method.process_weights_after_loading(layer)

    assert calls["repack"] == 4
    assert torch.equal(layer.w13_weight, fresh + 1)
    assert layer._glm_w4a8_gemm2_shuffled is True


@pytest.mark.parametrize("glm_method", [False, True])
@pytest.mark.parametrize("installed_layout", [None, "gemm2", "native"])
def test_reload_scale_refresh_is_scoped_to_glm_installed_layout(
    glm_method, installed_layout
):
    from vllm_hcu.model_executor.layers.quantization.glm_moe_dsa_w4a8 import (
        commit_installed_glm_w4a8_gemm2_generations,
    )
    from vllm_hcu.model_executor.layers.quantization.slimquant_w4a8 import (
        SlimQuantW4A8Int8AiterMoEMethod,
    )

    method, layer = _glm_aiter_method("auto")
    if not glm_method:
        method = SlimQuantW4A8Int8AiterMoEMethod.__new__(
            SlimQuantW4A8Int8AiterMoEMethod
        )
    layer.quant_method = method
    layer.w13_weight_scale = torch.nn.Parameter(torch.ones(1), requires_grad=False)
    layer.w2_weight_scale = torch.nn.Parameter(torch.ones(1), requires_grad=False)
    layer.w13_input_scale = None
    layer.w2_input_scale = None
    quant_config = method.get_fused_moe_quant_config(layer)
    old_w1_scale = quant_config.w1_scale
    old_w2_scale = quant_config.w2_scale
    layer.w13_weight_scale = torch.nn.Parameter(torch.full((1,), 7.0), requires_grad=False)
    layer.w2_weight_scale = torch.nn.Parameter(torch.full((1,), 11.0), requires_grad=False)
    if installed_layout == "gemm2":
        layer._glm_w4a8_gemm2_generation = ("previous owners",)
    elif installed_layout == "native":
        layer.w13_weight._hcu_vllm_w4a8_unpacked = True
        layer.w2_weight._hcu_vllm_w4a8_unpacked = True

    commit_installed_glm_w4a8_gemm2_generations(layer)

    assert method.moe_quant_config is quant_config
    if glm_method and installed_layout is not None:
        assert quant_config.w1_scale is layer.w13_weight_scale
        assert quant_config.w2_scale is layer.w2_weight_scale
    else:
        assert quant_config.w1_scale is old_w1_scale
        assert quant_config.w2_scale is old_w2_scale


def test_reload_scale_refresh_rebinds_deepep_auto_kernel_configs() -> None:
    from vllm_hcu.model_executor.layers.quantization.glm_moe_dsa_w4a8 import (
        GlmMoeDsaW4A8MoEMethod,
        commit_installed_glm_w4a8_gemm2_generations,
    )

    def config(w1, w2):
        return SimpleNamespace(
            _w1=SimpleNamespace(scale=w1),
            _w2=SimpleNamespace(scale=w2),
        )

    old_w1 = torch.full((2, 8, 1), 2.0)
    old_w2 = torch.full((2, 4, 1), 3.0)
    method_config = config(old_w1, old_w2)
    kernel_config = config(old_w1, old_w2)
    ht_config = config(old_w1, old_w2)
    ll_config = config(old_w1, old_w2)
    method = GlmMoeDsaW4A8MoEMethod.__new__(GlmMoeDsaW4A8MoEMethod)
    method.moe_quant_config = method_config
    method.moe_kernel = SimpleNamespace(
        fused_experts=SimpleNamespace(
            quant_config=kernel_config,
            ht_experts=SimpleNamespace(quant_config=ht_config),
            ll_experts=SimpleNamespace(quant_config=ll_config),
        )
    )
    layer = torch.nn.Module()
    layer.quant_method = method
    layer.w13_weight_scale = torch.nn.Parameter(
        torch.full((2, 8, 1), 7.0), requires_grad=False
    )
    layer.w2_weight_scale = torch.nn.Parameter(
        torch.full((2, 4, 1), 11.0), requires_grad=False
    )

    commit_installed_glm_w4a8_gemm2_generations(layer)

    for quant_config in (method_config, kernel_config, ht_config, ll_config):
        assert quant_config._w1.scale is layer.w13_weight_scale
        assert quant_config._w2.scale is layer.w2_weight_scale


def _v2_reload_runner(model):
    from vllm_hcu.v1.hcu_model_runner_v2 import HcuGPUModelRunnerV2

    # Exercise the inherited reload implementation without allocating a model
    # runner's device buffers or initializing unrelated LoRA/MM caches.
    runner = object.__new__(HcuGPUModelRunnerV2)
    runner.model = model
    runner.model_config = SimpleNamespace(quantization="glm_moe_dsa_w4a8")
    runner.load_config = SimpleNamespace()
    runner.lora_config = None
    runner.reset_lora_state = lambda: None
    runner.reset_encoder_cache = lambda: None
    runner.reset_mm_cache = lambda: None
    return runner


def _legacy_reload_method():
    """Extract the production legacy reload method without importing its runner.

    The full HCU legacy runner imports optional communication APIs that are not
    present in every CPU contract lane. This keeps the test on the production
    method body while isolating those unrelated imports.
    """
    source_path = Path(__file__).parents[2] / "vllm_hcu/v1/hcu_model_runner.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    runner_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "GPUModelRunner"
    )
    method = next(
        node
        for node in runner_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "reload_weights"
    )
    namespace = {
        "torch": torch,
        "Iterable": Iterable,
        "cast": cast,
        "initialize_layerwise_reload": sys.modules[
            "vllm.model_executor.model_loader.reload.layerwise"
        ].initialize_layerwise_reload,
        "finalize_layerwise_reload": sys.modules[
            "vllm.model_executor.model_loader.reload.layerwise"
        ].finalize_layerwise_reload,
        "logger": SimpleNamespace(
            warning=lambda *args, **kwargs: None,
            warning_once=lambda *args, **kwargs: None,
            info_once=lambda *args, **kwargs: None,
        ),
        "time": time,
    }
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
            str(source_path),
            "exec",
        ),
        namespace,
    )
    return namespace["reload_weights"]


@pytest.mark.parametrize("runner_kind", ["legacy", "v2"])
@pytest.mark.parametrize("installed_layout", ["gemm2", "native"])
def test_reload_weights_keeps_scale_owners_for_later_forward(
    monkeypatch, runner_kind, installed_layout
) -> None:
    from vllm.model_executor.model_loader.reload.layerwise import (
        record_metadata_for_reloading,
    )
    from vllm_hcu.model_executor.layers.quantization import (
        compressed_tensors_moe_runtime,
    )
    from vllm_hcu.model_executor.layers.quantization.glm_moe_dsa_w4a8 import (
        _wrap_checkpoint_loader,
        glm_expert_weights_use_gemm2_layout,
    )
    from vllm_hcu.model_executor.layers.quantization.slimquant_w4a8 import (
        SlimQuantW4A8Int8AiterMoEMethod,
    )
    gemm2 = installed_layout == "gemm2"
    config = (
        SimpleNamespace(
            need_shuffle=True, need_shuffle_scale=True,
            quant_type="w4a8", solution_type="moe_c",
        )
        if gemm2 else None
    )
    parent_post_load = SlimQuantW4A8Int8AiterMoEMethod.process_weights_after_loading
    calls = _patch_glm_post_load(monkeypatch, config)
    # Keep the real quant-config construction and scale binding during reload.
    monkeypatch.setattr(
        SlimQuantW4A8Int8AiterMoEMethod,
        "process_weights_after_loading",
        parent_post_load,
    )
    method, _layer = _glm_aiter_method("auto")
    method.moe.activation = "silu"
    method.moe.num_experts = 2

    class ExpertModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.quant_method = method
            self.w13_weight = torch.nn.Parameter(
                torch.zeros((2, 16, 4), dtype=torch.int8),
                requires_grad=False,
            )
            self.w2_weight = torch.nn.Parameter(
                torch.zeros((2, 8, 4), dtype=torch.int8),
                requires_grad=False,
            )
            self.w13_weight_scale = torch.nn.Parameter(
                torch.ones((2, 16, 1)), requires_grad=False
            )
            self.w2_weight_scale = torch.nn.Parameter(
                torch.ones((2, 8, 1)), requires_grad=False
            )
            self.register_parameter("w13_input_scale", None)
            self.register_parameter("w2_input_scale", None)

        def load_weights(self, weights):
            params = dict(self.named_parameters())
            loaded = set()
            for name, tensor in weights:
                param = params[name]
                param.weight_loader(param, tensor)
                loaded.add(name)
            return loaded

    def base_loader(param, loaded_weight, *args, **kwargs):
        if param.is_meta:
            counted = loaded_weight.detach().cpu().clone()
            counted.copy_(loaded_weight.detach().cpu())
            return
        param.data.copy_(loaded_weight)

    model = ExpertModel()
    for name, param in model.named_parameters():
        param.weight_loader = base_loader
        _wrap_checkpoint_loader(param, scale=name.endswith("_scale"))
    model.global_num_experts = 2
    model.swiglu_limit = None
    model._expert_map = None
    model.expert_mask = None
    # Native reload metadata describes checkpoint storage before post-load
    # installs kernel layout attributes on the runtime Parameters.
    record_metadata_for_reloading(model)
    if gemm2:
        compressed_tensors_moe_runtime.install_aiter_moe_weight_layout(
            model,
            SimpleNamespace(
                quant_type="w4a8",
                solution_type="moe_c",
                need_shuffle=False,
                need_shuffle_scale=False,
            ),
            logical_shape=(2, 16, 8, 8),
        )
    else:
        method.process_weights_after_loading(model)
    original_w13 = model.w13_weight
    original_w2 = model.w2_weight
    original_w13_scale = model.w13_weight_scale
    original_w2_scale = model.w2_weight_scale
    if runner_kind == "v2":
        runner = _v2_reload_runner(model)
        reload_weights = runner.reload_weights
    else:
        runner = SimpleNamespace(
            get_model=lambda: model,
            model_config=SimpleNamespace(quantization="glm_moe_dsa_w4a8"),
            load_config=SimpleNamespace(),
        )
        reload_weights = partial(_legacy_reload_method(), runner)
    checkpoint = [
        ("w13_weight", _pack_compressed_tensors_int4(
            torch.ones((2, 16, 8), dtype=torch.int8)
        )),
        ("w2_weight", _pack_compressed_tensors_int4(
            torch.ones((2, 8, 8), dtype=torch.int8)
        )),
        ("w13_weight_scale", torch.full_like(original_w13_scale, 32.0)),
        ("w2_weight_scale", torch.full_like(original_w2_scale, 48.0)),
    ]
    reload_weights(iter(checkpoint), is_checkpoint_format=True)

    assert model.w13_weight is original_w13
    assert model.w2_weight is original_w2
    assert calls["repack"] == (2 if gemm2 else 0)
    assert glm_expert_weights_use_gemm2_layout(model) is gemm2
    assert torch.equal(
        model.w13_weight, torch.full_like(original_w13, 0x12 if gemm2 else 1)
    )
    assert model.w13_weight_scale is original_w13_scale
    assert model.w2_weight_scale is original_w2_scale
    assert method.moe_quant_config.w1_scale is original_w13_scale
    assert method.moe_quant_config.w2_scale is original_w2_scale
    compensation = 1.0 if gemm2 else 16.0
    calls_forward = _forward_reloaded_weights(monkeypatch, model)
    assert not calls_forward["execute"].get("use_weight_shuffle", False)
    assert torch.all(calls_forward["execute"]["w1_scale"] == 2.0 * compensation)
    assert torch.all(calls_forward["execute"]["w2_scale"] == 3.0 * compensation)
    installed_quant_config = method.moe_quant_config

    kernel = [
        ("w13_weight", model.w13_weight.detach().clone()),
        ("w2_weight", model.w2_weight.detach().clone()),
        ("w13_weight_scale", torch.full_like(original_w13_scale, 7.0)),
        ("w2_weight_scale", torch.full_like(original_w2_scale, 11.0)),
    ]
    reload_weights(iter(kernel), is_checkpoint_format=False)
    assert calls["repack"] == (2 if gemm2 else 0)
    assert glm_expert_weights_use_gemm2_layout(model) is gemm2
    assert method.moe_quant_config is installed_quant_config
    for num_tokens in (1, 3):
        calls_forward = _forward_reloaded_weights(monkeypatch, model, num_tokens)
        assert not calls_forward["execute"].get("use_weight_shuffle", False)
        if gemm2:
            assert calls_forward["execute"]["w1_scale"] is original_w13_scale
            assert calls_forward["execute"]["w2_scale"] is original_w2_scale
        assert torch.all(calls_forward["execute"]["w1_scale"] == 7.0 * compensation)
        assert torch.all(calls_forward["execute"]["w2_scale"] == 11.0 * compensation)

    # A later checkpoint reload must convert its fresh bytes exactly once and
    # restore the same runtime owners after a kernel-format update.
    reload_weights(iter(checkpoint), is_checkpoint_format=True)
    assert calls["repack"] == (4 if gemm2 else 0)
    assert glm_expert_weights_use_gemm2_layout(model) is gemm2
    assert model.w13_weight is original_w13
    assert model.w2_weight is original_w2
    assert method.moe_quant_config.w1_scale is original_w13_scale
    assert method.moe_quant_config.w2_scale is original_w2_scale
    calls_forward = _forward_reloaded_weights(monkeypatch, model)
    assert not calls_forward["execute"].get("use_weight_shuffle", False)
    assert torch.all(calls_forward["execute"]["w1_scale"] == 2.0 * compensation)
    assert torch.all(calls_forward["execute"]["w2_scale"] == 3.0 * compensation)


def test_v2_reload_failure_does_not_commit_gemm2_generation(monkeypatch) -> None:
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    method, model = _glm_aiter_method("auto")
    model.quant_method = method
    generation = ("unprocessed weights",)
    model._glm_w4a8_gemm2_generation = generation
    runner = _v2_reload_runner(model)
    weights = iter(())
    failure = RuntimeError("upstream reload failed")

    def failed_reload(self, *args, **kwargs):
        assert self is runner
        assert args == (weights,)
        assert kwargs == {"is_checkpoint_format": False}
        raise failure

    monkeypatch.setattr(GPUModelRunner, "reload_weights", failed_reload)
    with pytest.raises(RuntimeError, match="upstream reload failed") as exc_info:
        runner.reload_weights(weights, is_checkpoint_format=False)
    assert exc_info.value is failure
    assert model._glm_w4a8_gemm2_generation is generation
    assert not getattr(model, "_glm_w4a8_gemm2_shuffled", False)


def test_v2_reload_preserves_non_glm_parameter_owners() -> None:
    model = torch.nn.Linear(4, 3, bias=False)
    model.requires_grad_(False)
    original_weight = model.weight
    runner = _v2_reload_runner(model)
    runner.model_config.quantization = None
    runner.reload_weights(
        iter([("weight", torch.full_like(original_weight, 5.0))]),
        is_checkpoint_format=False,
    )
    assert model.weight is original_weight
    assert torch.all(model.weight == 5.0)
    assert not hasattr(model, "_glm_w4a8_gemm2_generation")


def _forward_reloaded_weights(monkeypatch, layer, num_tokens=3):
    from vllm_hcu.model_executor.layers.quantization import (
        compressed_tensors_moe_runtime,
    )

    calls: dict[str, object] = {}

    class MoeQuantType:
        W4A8 = "w4a8"

    def get_config(**kwargs: object):
        return True, SimpleNamespace(
            quant_type="w4a8",
            solution_type="moe_c",
            need_shuffle=True,
            need_shuffle_scale=False,
        )

    def aiter_moe(**kwargs: object):
        calls["execute"] = kwargs
        return torch.zeros_like(kwargs["hidden_states"], dtype=kwargs["output_dtype"])

    def native_moe(x, *args, **kwargs):
        calls["execute"] = kwargs
        return torch.zeros_like(x)

    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.layers.fused_moe.fused_moe",
        _module(
            "vllm.model_executor.layers.fused_moe.fused_moe",
            fused_experts_impl=native_moe,
        ),
    )

    monkeypatch.setitem(
        sys.modules,
        "aiter.moe",
        _module(
            "aiter.moe",
            MoeQuantType=MoeQuantType,
            get_aiter_moe_config=get_config,
            aiter_moe=aiter_moe,
        ),
    )
    compressed_tensors_moe_runtime.apply_aiter_w4a8_moe(
        layer.quant_method,
        layer,
        torch.ones((num_tokens, layer.w2_weight.shape[1]), dtype=torch.bfloat16),
        torch.ones((num_tokens, 2), dtype=torch.float32),
        torch.zeros((num_tokens, 2), dtype=torch.int32),
    )
    return calls


def test_glm_triton_post_load_keeps_slimquant_packing(monkeypatch) -> None:
    config = SimpleNamespace(need_shuffle=True, need_shuffle_scale=True)
    calls = _patch_glm_post_load(monkeypatch, config)
    method, layer = _glm_aiter_method("triton")
    original_w13 = layer.w13_weight.detach().clone()

    method.process_weights_after_loading(layer)

    assert calls["repack"] == 0
    assert not hasattr(layer, "_glm_w4a8_gemm2_shuffled")
    assert torch.equal(layer.w13_weight, original_w13)
    assert calls["parent"] == [(True, True)]


def _module(name: str, **attributes: object) -> ModuleType:
    module = ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


def _run_apply_with_shuffling_config(
    monkeypatch, *, gemm2_installed: bool, stale_boolean: bool = False
):
    from vllm_hcu.model_executor.layers.quantization import (
        compressed_tensors_moe_runtime,
    )

    calls: dict[str, object] = {}

    class MoeQuantType:
        W4A8 = "w4a8"

    def get_config(**kwargs: object):
        # Larger M selects a config that asks for the generic AITER shuffle.
        return True, SimpleNamespace(
            quant_type="w4a8",
            solution_type="moe_c",
            need_shuffle=True,
            need_shuffle_scale=False,
        )

    def aiter_moe(**kwargs: object):
        calls["execute"] = kwargs
        return torch.zeros((3, 4), dtype=kwargs["output_dtype"])

    monkeypatch.setitem(
        sys.modules,
        "aiter.moe",
        _module(
            "aiter.moe",
            MoeQuantType=MoeQuantType,
            get_aiter_moe_config=get_config,
            aiter_moe=aiter_moe,
        ),
    )
    w1 = torch.zeros((2, 8, 2), dtype=torch.int8)
    w2 = torch.zeros((2, 4, 2), dtype=torch.int8)
    layer = SimpleNamespace(
        w13_weight=w1,
        w2_weight=w2,
        swiglu_limit=None,
        global_num_experts=2,
        _expert_map=None,
        expert_mask=None,
    )
    # Load time installed the gemm2 layout from an M=1 config without shuffle.
    compressed_tensors_moe_runtime.install_aiter_moe_weight_layout(
        layer,
        SimpleNamespace(
            quant_type="w4a8",
            solution_type="moe_c",
            need_shuffle=False,
            need_shuffle_scale=False,
        ),
        logical_shape=(2, 8, 4, 4),
    )
    if gemm2_installed:
        from vllm_hcu.model_executor.layers.quantization.glm_moe_dsa_w4a8 import (
            _expert_weight_generation,
        )

        layer._glm_w4a8_gemm2_generation = _expert_weight_generation(layer)
        layer._glm_w4a8_gemm2_shuffled = True
    elif stale_boolean:
        layer._glm_w4a8_gemm2_shuffled = True
    method = SimpleNamespace(
        moe=SimpleNamespace(num_experts=2),
        moe_quant_config=SimpleNamespace(
            w1_scale=torch.ones((2, 8, 1)),
            w2_scale=torch.ones((2, 4, 1)),
            a1_scale=None,
            a2_scale=None,
        ),
    )
    compressed_tensors_moe_runtime.apply_aiter_w4a8_moe(
        method,
        layer,
        torch.ones((3, 4), dtype=torch.bfloat16),
        torch.ones((3, 2), dtype=torch.float32),
        torch.zeros((3, 2), dtype=torch.int32),
    )
    return calls


def test_glm_gemm2_layout_survives_m_dependent_aiter_configs(monkeypatch) -> None:
    calls = _run_apply_with_shuffling_config(monkeypatch, gemm2_installed=True)
    assert calls["execute"]["use_weight_shuffle"] is False


def test_stale_gemm2_boolean_does_not_cover_replaced_weights(monkeypatch) -> None:
    from vllm_hcu.model_executor.layers.quantization.compressed_tensors_moe_runtime import (
        HcuCompressedTensorsMoeError,
    )

    with pytest.raises(
        HcuCompressedTensorsMoeError,
        match="AITER config layout does not match installed weights",
    ):
        _run_apply_with_shuffling_config(
            monkeypatch, gemm2_installed=False, stale_boolean=True
        )


def test_unmarked_layout_still_rejects_shuffling_config(monkeypatch) -> None:
    from vllm_hcu.model_executor.layers.quantization.compressed_tensors_moe_runtime import (
        HcuCompressedTensorsMoeError,
    )

    with pytest.raises(
        HcuCompressedTensorsMoeError,
        match="AITER config layout does not match installed weights",
    ):
        _run_apply_with_shuffling_config(monkeypatch, gemm2_installed=False)
