# SPDX-License-Identifier: Apache-2.0
import json

import pytest
import torch

from tests.models.hy_v4.test_custom_w4a8 import module, registry
from tests.models.hy_v4.test_weight_loading import checkpoint_model
from tests.models.hy_v4.test_mtp import _minimal_draft, _mtp, _weights


def native_index(tmp_path):
    path = tmp_path / "hy4-checkpoint.index.json"
    path.write_text(json.dumps({"format": "hy4_w4a8_v1", "complete": True,
        "parameters": {"model.layers.1.mlp.experts.0.gate_proj.weight": {"kind": "quantized"},
            "model.mtp_layers.0.mlp.shared_experts.up_proj.weight": {"kind": "quantized"},
            "lm_head.weight": {"kind": "retained"}}}))
    (tmp_path / "config.json").write_text(json.dumps({"num_hidden_layers": 2}))
    return path


def test_native_registry_and_manifest_names(tmp_path):
    config = registry().from_config({"quant_method": "slimquant_w4a8",
        "checkpoint_format": "hy4_w4a8_v1", "checkpoint_index": str(native_index(tmp_path))})
    assert type(config).__name__ == "HYV4NativeW4A8Config"
    assert config.quantized_modules == frozenset({"model.layers.1.mlp.experts",
        "model.layers.2.mlp.shared_experts.gate_up_proj"})


@pytest.mark.parametrize("field,value", [("format", "gptq"), ("complete", False),
                                        ("complete", 1), ("parameters", {}),
                                        ("parameters", {"x": {"kind": "unknown"}})])
def test_native_manifest_rejects_unsupported_metadata(tmp_path, field, value):
    path = native_index(tmp_path)
    data = json.loads(path.read_text())
    data[field] = value
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="HYV4|hy4"):
        module("hyv4_native").HYV4NativeW4A8Config(str(path))


def test_native_uint8_and_scale_storage_mapping():
    packed = torch.tensor([[0x21, 0xf0]], dtype=torch.uint8)
    values = dict(module("hyv4_native").adapt_native_weights([
        ("test.weight.packed", packed), ("test.weight.scale", torch.tensor([[.5]])),
        ("lm_head.weight.weight", torch.ones(2)),
        ("model.hc_head.hc_head_base.weight", torch.ones(2))]))
    assert values["test.weight"].data_ptr() == packed.data_ptr()
    torch.testing.assert_close(values["test.weight_scale"], torch.tensor([[.5]]))
    assert "lm_head.weight" in values and "model.hc_head.hc_head_base" in values


@pytest.mark.parametrize("suffix,value", [("packed", torch.ones(1, 2, dtype=torch.int8)),
    ("packed", torch.ones(1, 2, 3, dtype=torch.uint8)),
    ("scale", torch.ones(2)), ("scale", torch.ones(2, 2)),
    ("scale", torch.tensor([[float("nan")]])), ("scale", torch.zeros(1, 1))])
def test_native_tensor_validation(suffix, value):
    with pytest.raises(ValueError, match="HYV4|native"):
        list(module("hyv4_native").adapt_native_weights([("test.weight." + suffix, value)]))


def test_native_target_retained_names_and_strict_ledger(tmp_path, checkpoint_model):
    checkpoint_model.quant_config = module("hyv4_native").HYV4NativeW4A8Config(str(native_index(tmp_path)))
    weights = [(name + ".weight", torch.ones_like(value)) for name, value in checkpoint_model.named_parameters()]
    assert checkpoint_model.load_weights(iter(weights)) == set(dict(checkpoint_model.named_parameters()))
    with pytest.raises(RuntimeError, match="Duplicate"):
        checkpoint_model.load_weights(iter(weights + [weights[0]]))


def test_native_mtp_retained_names_keep_target_filter_and_ledger(tmp_path, monkeypatch):
    model = _minimal_draft(_mtp(), monkeypatch)
    model.quant_config = module("hyv4_native").HYV4NativeW4A8Config(str(native_index(tmp_path)))
    weights = [(name + ".weight", value) for name, value in _weights()]
    assert model.load_weights(iter(weights)) == set(dict(model.named_parameters()))
    with pytest.raises(RuntimeError, match="Duplicate"):
        model.load_weights(iter(weights + [weights[0]]))


def test_native_manifest_rejects_second_mtp_block(tmp_path):
    path = native_index(tmp_path)
    data = json.loads(path.read_text())
    data["parameters"]["model.mtp_layers.1.mlp.experts.0.gate_proj.weight"] = {"kind": "quantized"}
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="MTP"):
        module("hyv4_native").HYV4NativeW4A8Config(str(path))


@pytest.mark.parametrize("value", [torch.ones(4), torch.full((4,), 2.)])
def test_native_writer_smoothing_storage_suffix_is_validated(value):
    adapter = module("hyv4_native").adapt_native_weights
    values = [("test.input_scale.weight", value)]
    if torch.all(value == 1):
        assert list(adapter(values)) == []
    else:
        with pytest.raises(ValueError, match="identity input_scale"):
            list(adapter(values))
