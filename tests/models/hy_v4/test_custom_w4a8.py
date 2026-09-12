# SPDX-License-Identifier: Apache-2.0
"""Explicit HYV4 formats must not change the ordinary quantization owner."""
import importlib
import json
import os
import subprocess
from pathlib import Path
from types import ModuleType

import pytest
import torch

from tests.models.hy_v4.test_weight_loading import checkpoint_model

FORMAT = "hy4-w4a8-custom-v1"
PACKING = "signed nibble, even input in low bits"


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
