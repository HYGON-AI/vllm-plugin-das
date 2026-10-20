# SPDX-License-Identifier: Apache-2.0
"""Explicit HYV4 formats must not change the ordinary quantization owner."""
import importlib
import json
import os
import subprocess
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace

import pytest
import torch

from tests.models.hy_v4.test_weight_loading import checkpoint_model

FORMAT = "hy4-w4a8-custom-v1"
PACKING = "signed nibble, even input in low bits"


def compressed_w4a8_config(target="re:.*\\.mlp\\.experts\\.\\d+\\.(?:gate_proj|up_proj|down_proj)$"):
    return {
        "format": "pack-quantized",
        "config_groups": {
            "experts": {
                "format": "pack-quantized",
                "targets": [target],
                "weights": {
                    "num_bits": 4,
                    "type": "int",
                    "strategy": "channel",
                    "symmetric": True,
                    "dynamic": False,
                    "group_size": -1,
                },
                "input_activations": {
                    "num_bits": 8,
                    "type": "int",
                    "strategy": "token",
                    "symmetric": True,
                    "dynamic": True,
                },
            }
        },
    }


def module(leaf):
    name = "vllm_hcu.model_executor.layers.quantization." + leaf
    assert importlib.util.find_spec(name) is not None, f"Missing HYV4 adapter: {leaf}"
    return importlib.import_module(name)


def manifest(tmp_path, names=None):
    path = tmp_path / "conversion.json"
    path.write_text(json.dumps({"provenance": {"format": FORMAT},
        "quantized_tensors": {name: {"packing": PACKING} for name in (names or [
            "model.layers.1.mlp.experts.gate_up_proj",
            "model.layers.1.mlp.shared_experts.gate_proj.weight",
            "model.layers.1.mlp.shared_experts.up_proj.weight",
            "model.mtp_layers.0.mlp.shared_experts.down_proj.weight",
        ])}}))
    return path


def registry():
    from vllm_hcu.patch.platform.core_fix import patch_slimquant_registry as patch
    target = ModuleType(patch.TARGET_MODULE)
    target.QUANTIZATION_METHODS = []
    target._CUSTOMIZED_METHOD_TO_QUANT_CONFIG = {}
    def register(name):
        def decorate(cls):
            target.QUANTIZATION_METHODS.append(name)
            target._CUSTOMIZED_METHOD_TO_QUANT_CONFIG[name] = cls
            return cls
        return decorate
    target.register_quantization_config = register
    patch.apply_to_module(target)
    return target._CUSTOMIZED_METHOD_TO_QUANT_CONFIG["slimquant_w4a8"]


def test_registry_materializes_only_explicit_custom_format(tmp_path):
    config = registry().from_config({"quant_method": "slimquant_w4a8",
        "checkpoint_format": FORMAT, "conversion_manifest": str(manifest(tmp_path))})
    assert type(config).__name__ == "HYV4W4A8Config"
    from vllm_hcu.model_executor.layers.quantization.slimquant_w4a8 import SlimQuantW4A8Int8Config
    assert isinstance(config, SlimQuantW4A8Int8Config)


@pytest.mark.parametrize("metadata", [{}, {"checkpoint_format": "unknown"},
                                      {"checkpoint_format": "HY4-W4A8-CUSTOM-V1"}])
def test_ordinary_slimquant_keeps_current_selection(metadata):
    from vllm_hcu.model_executor.layers.quantization.slimquant_w4a8 import SlimQuantW4A8Int8Config
    assert type(registry().from_config({"quant_method": "slimquant_w4a8", **metadata})) is SlimQuantW4A8Int8Config


def test_channel_fp8_does_not_select_hyv4_w4a8():
    # Selection fields from the supplied Channel-FP8 compression_config.
    # Keep the contract portable to CI without that large checkpoint.
    config = {"quant_method": "compressed-tensors", "format": "float-quantized",
              "config_groups": {"group_0": {"targets": ["Linear"],
                  "weights": {"num_bits": 8, "type": "float", "strategy": "channel",
                              "dynamic": False, "symmetric": True},
                  "input_activations": {"num_bits": 8, "type": "float", "strategy": "token",
                                        "dynamic": True, "symmetric": True}}}}
    facade = registry()
    for user_quant in (None, "slimquant_w4a8"):
        assert facade.override_quantization_method(config, user_quant) is None
    from vllm.model_executor.layers.quantization import get_quantization_config
    selected = get_quantization_config(config["quant_method"]).from_config(config)
    assert type(selected).__name__ == "CompressedTensorsConfig"


def test_hyv4_compressed_tensors_w4a8_experts_select_hcu_method():
    from compressed_tensors.quantization import QuantizationArgs
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe import (
        CompressedTensorsMoEMethod,
    )
    from vllm_hcu.patch.worker.op_opt import (
        patch_compressed_tensors_moe_w4a8_int8,
    )

    patch_compressed_tensors_moe_w4a8_int8.apply()

    weight_quant = QuantizationArgs(
        num_bits=4,
        type="int",
        strategy="channel",
        symmetric=True,
        dynamic=False,
        group_size=-1,
    )
    input_quant = QuantizationArgs(
        num_bits=8,
        type="int",
        strategy="token",
        symmetric=True,
        dynamic=True,
    )

    class FakeCompressedTensorsConfig:
        def _add_fused_moe_to_target_scheme_map(self):
            pass

        def get_scheme_dict(self, layer, name):
            return {
                "weights": weight_quant,
                "input_activations": input_quant,
                "format": "pack-quantized",
            }

        def __getattr__(self, name):
            if name == "_is_dynamic_token_w4a8_int":
                return lambda weights, inputs: True
            if name.startswith("_is_"):
                return lambda *args: False
            raise AttributeError(name)

    layer = SimpleNamespace(
        moe_config=SimpleNamespace(
            moe_backend="aiter",
            has_bias=False,
            moe_parallel_config=SimpleNamespace(
                all2all_backend=None,
                use_ep=False,
            ),
        )
    )
    method = CompressedTensorsMoEMethod.get_moe_method(
        FakeCompressedTensorsConfig(), layer, "model.layers.1.mlp.experts"
    )

    assert type(method).__name__ == "HcuCompressedTensorsW4A8Int8MoEMethod"


def test_hyv4_compressed_tensors_packed_experts_adapt_to_hcu_layout():
    from compressed_tensors.compressors.pack_quantized.helpers import pack_to_int32
    from vllm_hcu.model_executor.layers.quantization.compressed_tensors_w4a8_moe import (
        repack_compressed_tensors_int4,
    )

    class CompressedTensorsConfig:
        config = compressed_w4a8_config()

        @staticmethod
        def get_name():
            return "compressed-tensors"

    values = torch.tensor([[-8, -7, -1, 0, 1, 2, 6, 7]], dtype=torch.int8)
    packed = pack_to_int32(values, 4)
    scale = torch.ones(1, 1, dtype=torch.bfloat16)
    prefix = "model.layers.1.mlp.experts.0.gate_proj"
    adapted = dict(module("hyv4_w4a8_weights").adapt_checkpoint_weights(
        CompressedTensorsConfig(),
        [
            (prefix + ".weight_packed", packed),
            (prefix + ".weight_scale", scale),
            (prefix + ".weight_shape", torch.tensor([1, 8])),
        ],
    ))

    assert set(adapted) == {prefix + ".weight", prefix + ".weight_scale"}
    assert adapted[prefix + ".weight"].data_ptr() == packed.data_ptr()
    assert adapted[prefix + ".weight_scale"].data_ptr() == scale.data_ptr()
    converted = repack_compressed_tensors_int4(adapted[prefix + ".weight"])
    torch.testing.assert_close(
        converted.view(torch.uint8),
        torch.tensor([[0x89, 0xF0, 0x12, 0x67]], dtype=torch.uint8),
    )
    random_values = torch.randint(-8, 8, (7, 24), dtype=torch.int8)
    random_packed = pack_to_int32(random_values, 4)
    expected = (random_values.to(torch.int16) & 0xF).to(torch.uint8)
    expected = ((expected[:, 0::2] << 4) | expected[:, 1::2]).view(torch.int8)
    torch.testing.assert_close(
        repack_compressed_tensors_int4(random_packed), expected
    )


def test_standard_packed_experts_load_without_hyv4_weight_adapter():
    """The global W4A8 route must accept native packed checkpoint names."""
    from vllm.model_executor.layers.fused_moe.routed_experts import (
        RoutedExperts,
    )
    from vllm_hcu.model_executor.layers.quantization.compressed_tensors_w4a8_moe import (
        HcuCompressedTensorsW4A8Int8MoEMethod,
    )
    from vllm_hcu.patch.worker.op_opt.moe import patch_layer

    patch_layer.apply()
    loaded: list[tuple[torch.Tensor, str, int]] = []
    parameter = torch.nn.Parameter(
        torch.empty((1, 1), dtype=torch.int8), requires_grad=False
    )

    def weight_loader(
        *, param, loaded_weight, shard_id, expert_id, **kwargs
    ) -> bool:
        loaded.append((loaded_weight, shard_id, expert_id))
        return True

    parameter.weight_loader = weight_loader
    owner = SimpleNamespace(
        layer_name="experts",
        moe_config=SimpleNamespace(hidden_dim_unpadded=8),
        quant_method=object.__new__(HcuCompressedTensorsW4A8Int8MoEMethod),
        w13_weight=parameter,
        get_expert_mapping=lambda include_fused: [
            ("w13_", "experts.0.gate_proj.", 0, "w1")
        ],
    )
    packed = torch.ones((2, 1), dtype=torch.int32)
    metadata = torch.tensor([2, 8], dtype=torch.int64)

    loaded_names = list(
        RoutedExperts.load_weights(
            owner,
            [
                ("0.gate_proj.weight_shape", metadata),
                ("0.gate_proj.weight_packed", packed),
            ],
        )
    )

    assert loaded_names == ["w13_weight"]
    assert len(loaded) == 1
    loaded_weight, shard_id, expert_id = loaded[0]
    torch.testing.assert_close(loaded_weight, packed)
    assert (shard_id, expert_id) == ("w1", 0)


def test_hyv4_compressed_tensors_preserves_unrelated_expert_scheme():
    class CompressedTensorsConfig:
        config = compressed_w4a8_config()
        config["config_groups"]["experts"]["weights"].update(
            strategy="group", group_size=128
        )
        config["config_groups"]["experts"]["input_activations"] = None

        @staticmethod
        def get_name():
            return "compressed-tensors"

    prefix = "model.layers.1.mlp.experts.0.gate_proj"
    original = [
        (prefix + ".weight_packed", torch.ones(1, 1, dtype=torch.int32)),
        (prefix + ".weight_scale", torch.ones(1, 1)),
        (prefix + ".weight_shape", torch.tensor([1, 8])),
    ]
    adapted = list(module("hyv4_w4a8_weights").adapt_checkpoint_weights(
        CompressedTensorsConfig(), iter(original)
    ))

    assert [name for name, _ in adapted] == [name for name, _ in original]
    assert all(actual is expected for (_, actual), (_, expected) in zip(adapted, original))


def test_hcu_compressed_tensors_w4a8_selector_is_strict():
    from compressed_tensors.quantization import QuantizationArgs
    from vllm_hcu.patch.worker.op_opt.patch_compressed_tensors_moe_w4a8_int8 import (
        _is_hcu_w4a8_scheme,
    )

    def schemes(*, weight_type="int", activation_symmetric=True, has_bias=False):
        weight = QuantizationArgs(
            num_bits=4, type=weight_type, strategy="channel", symmetric=True,
            dynamic=False, group_size=-1,
        )
        activation = QuantizationArgs(
            num_bits=8, type="int", strategy="token",
            symmetric=activation_symmetric, dynamic=True,
        )
        return weight, activation, SimpleNamespace(has_bias=has_bias)

    assert _is_hcu_w4a8_scheme(*schemes())
    assert not _is_hcu_w4a8_scheme(*schemes(weight_type="float"))
    assert not _is_hcu_w4a8_scheme(*schemes(activation_symmetric=False))
    assert not _is_hcu_w4a8_scheme(*schemes(has_bias=True))


def test_compressed_tensors_w4a8_conversion_runs_in_parameter_loader():
    from compressed_tensors.compressors.pack_quantized.helpers import pack_to_int32
    from compressed_tensors.quantization import QuantizationArgs
    from vllm_hcu.model_executor.layers.quantization.compressed_tensors_w4a8_moe import (
        HcuCompressedTensorsW4A8Int8MoEMethod,
    )

    weight_quant = QuantizationArgs(
        num_bits=4, type="int", strategy="channel", symmetric=True,
        dynamic=False, group_size=-1,
    )
    input_quant = QuantizationArgs(
        num_bits=8, type="int", strategy="token", symmetric=True,
        dynamic=True,
    )
    moe = SimpleNamespace(
        moe_backend="aiter",
        has_bias=False,
        moe_parallel_config=SimpleNamespace(
            all2all_backend=None, use_ep=False, tp_size=1,
        ),
    )
    method = HcuCompressedTensorsW4A8Int8MoEMethod(
        weight_quant, input_quant, moe,
    )
    layer = torch.nn.Module()
    layer.moe_config = moe
    loaded = []

    def loader(param, value, *args, **kwargs):
        loaded.append(value)

    method.create_weights(
        layer,
        num_experts=1,
        hidden_size=8,
        intermediate_size_per_partition=4,
        params_dtype=torch.bfloat16,
        weight_loader=loader,
    )
    values = torch.tensor(
        [[-8, -7, -1, 0, 1, 2, 6, 7]] * 4, dtype=torch.int8
    )
    layer.w13_weight.weight_loader(
        layer.w13_weight, pack_to_int32(values, 4), "weight", "w1", 0,
    )
    layer.w13_weight_scale.weight_loader(
        layer.w13_weight_scale, torch.ones(4, 1, dtype=torch.bfloat16),
        "scale", "w1", 0,
    )
    with pytest.raises(ValueError, match="requires INT32 shape"):
        layer.w13_weight.weight_loader(
            layer.w13_weight, pack_to_int32(values[:3], 4),
            "weight", "w1", 0,
        )
    with pytest.raises(ValueError, match="scale requires shape"):
        layer.w13_weight_scale.weight_loader(
            layer.w13_weight_scale,
            torch.ones(3, 1, dtype=torch.bfloat16),
            "scale", "w1", 0,
        )

    torch.testing.assert_close(
        loaded[0].view(torch.uint8),
        torch.tensor([[0x89, 0xF0, 0x12, 0x67]] * 4, dtype=torch.uint8),
    )
    torch.testing.assert_close(
        loaded[1], torch.full((4, 1), 1.0 / 16.0, dtype=torch.bfloat16),
    )


def test_custom_manifest_selects_fused_modules_and_mtp_names(tmp_path):
    config = module("hyv4_w4a8").HYV4W4A8Config(str(manifest(tmp_path)))
    assert config.quantized_modules == frozenset({"model.layers.1.mlp.experts",
        "model.layers.1.mlp.shared_experts.gate_up_proj",
        "model.mtp_layers.0.mlp.shared_experts.down_proj"})
    from vllm_hcu.models.hy_v4.mtp import _remap_mtp_quant_exclusions
    draft = _remap_mtp_quant_exclusions(config, 78, 1)
    assert "model.layers.78.mlp.shared_experts.down_proj" in draft.quantized_modules
    assert "model.layers.78.mlp.shared_experts.down_proj" not in config.quantized_modules


@pytest.mark.parametrize("change", ["format", "packing", "bits", "group_size", "symmetric", "empty"])
def test_custom_manifest_fails_closed(tmp_path, change):
    path = manifest(tmp_path)
    data = json.loads(path.read_text())
    if change == "format":
        data["provenance"]["format"] = "slimquant_w4a8"
    elif change == "empty":
        data["quantized_tensors"] = {}
    else:
        metadata = next(iter(data["quantized_tensors"].values()))
        metadata[change] = {"packing": "unsigned", "bits": 8,
                            "group_size": 128, "symmetric": False}[change]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="HYV4|HY V4|hy4"):
        module("hyv4_w4a8").HYV4W4A8Config(str(path))


def test_custom_stream_preserves_fused_scale_shape_and_identity_smoothing():
    adapter = module("hyv4_w4a8_weights")
    packed = torch.tensor([[[0x87, 0x10]]], dtype=torch.uint8)
    prefix = "model.layers.1.mlp.experts.gate_up_proj"
    result = dict(adapter.adapt_weights([(prefix + ".int4_packed", packed),
        (prefix + ".scale", torch.tensor([[0.5]])),
        (prefix + ".input_scale", torch.ones(1, 4))]))
    assert set(result) == {prefix, prefix + "_scale"}
    assert result[prefix].data_ptr() == packed.data_ptr()
    torch.testing.assert_close(result[prefix + "_scale"], torch.tensor([[[0.5]]]))


@pytest.mark.parametrize("value", [torch.tensor([2.]), torch.tensor([float("nan")]), torch.empty(0)])
def test_custom_rejects_nonidentity_or_empty_smoothing(value):
    with pytest.raises(ValueError, match="identity input_scale"):
        list(module("hyv4_w4a8_weights").adapt_weights([("test.input_scale", value)]))


@pytest.mark.parametrize("value", [torch.tensor([-1.]), torch.tensor([float("inf")]),
                                  torch.ones(2, 2, 2), torch.ones(1, dtype=torch.int32)])
def test_custom_rejects_invalid_channel_scales(value):
    with pytest.raises(ValueError, match="scale"):
        list(module("hyv4_w4a8_weights").adapt_weights([("test.weight.scale", value)]))


def test_custom_target_loader_keeps_strict_alias_ledger(tmp_path, checkpoint_model):
    model = checkpoint_model
    model.quant_config = module("hyv4_w4a8").HYV4W4A8Config(str(manifest(tmp_path)))
    values = [(name, torch.ones_like(value)) for name, value in model.named_parameters()]
    assert model.load_weights(iter(values + [("model.norm.input_scale", torch.ones(2))])) == set(dict(model.named_parameters()))
    with pytest.raises(ValueError, match="identity input_scale"):
        model.load_weights(iter(values + [("model.norm.input_scale", torch.tensor([2.]))]))
    with pytest.raises(RuntimeError, match="Duplicate"):
        model.load_weights(iter(values + [values[0]]))
    with pytest.raises(RuntimeError, match="Missing"):
        model.load_weights(iter(values[1:]))


@pytest.mark.parametrize("field,value", [("bits", 8), ("group_size", 128), ("symmetric", False)])
def test_explicit_config_rejects_incompatible_root_metadata(tmp_path, field, value):
    with pytest.raises(ValueError, match="HYV4"):
        registry().from_config({"quant_method": "slimquant_w4a8", "checkpoint_format": FORMAT,
            "conversion_manifest": str(manifest(tmp_path)), field: value})


def test_channel_fp8_format_label_cannot_materialize_hyv4(tmp_path):
    from vllm_hcu.model_executor.layers.quantization.slimquant_w4a8 import SlimQuantW4A8Int8Config
    values = {"quant_method": "compressed-tensors", "checkpoint_format": FORMAT,
              "conversion_manifest": str(manifest(tmp_path))}
    assert type(registry().from_config(values)) is SlimQuantW4A8Int8Config
    with pytest.raises(ValueError, match="explicit"):
        module("hyv4_w4a8").HYV4W4A8Config.from_config(values)


@pytest.mark.parametrize("kind", ["custom", "native"])
def test_smoothing_duplicate_cannot_bypass_checkpoint_accounting(kind):
    adapter = (module("hyv4_w4a8_weights").adapt_weights if kind == "custom"
               else module("hyv4_native").adapt_native_weights)
    with pytest.raises(RuntimeError, match="Duplicate"):
        list(adapter([("test.input_scale", torch.ones(4)),
                      ("test.input_scale", torch.ones(4))]))


@pytest.mark.parametrize("format_name,script,manifest_key", [
    (FORMAT, "serve_custom_w4a8.sh", "conversion_manifest"),
    ("hy4_w4a8_v1", "serve_native_w4a8.sh", "checkpoint_index"),
])
def test_serve_script_exposes_overrides_without_setting_plugins(tmp_path, format_name, script, manifest_key):
    executable = tmp_path / "vllm"
    executable.write_text("#!/usr/bin/env python3\nimport json, os, sys\nprint(json.dumps({'args':sys.argv[1:], 'plugins':os.getenv('VLLM_PLUGINS')}))\n")
    executable.chmod(0o755)
    env = {**os.environ, "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"]}
    env.pop("VLLM_PLUGINS", None)
    path = Path("tools/hy_v4") / script
    assert path.exists(), "Missing W4A8 serve script"
    run = subprocess.run(["bash", str(path), "/checkpoint", "--port", "8123"],
                         env=env, text=True, capture_output=True, check=True)
    result = json.loads(run.stdout)
    assert result["plugins"] is None
    args = result["args"]
    assert args[:2] == ["serve", "/checkpoint"]
    override = json.loads(args[args.index("--hf-overrides") + 1])["quantization_config"]
    assert override["checkpoint_format"] == format_name
    assert override["quant_method"] == "slimquant_w4a8"
    assert override[manifest_key].startswith("/checkpoint/")
    assert args[-2:] == ["--port", "8123"]
