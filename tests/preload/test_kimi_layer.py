# SPDX-License-Identifier: Apache-2.0
import json

import pytest
import torch
from safetensors.torch import save_file
from vllm_hcu.preload.kimi_layer import selected_weights


def test_layer_zero_selection_and_limit(tmp_path, monkeypatch):
    zero = "language_model.model.layers.0.input_layernorm.weight"
    other = "language_model.model.layers.1.input_layernorm.weight"
    save_file({zero: torch.ones(4), other: torch.ones(5)}, str(tmp_path / "part.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
        zero: "part.safetensors", other: "part.safetensors"}}))
    paths, total, digest = selected_weights(str(tmp_path))
    assert list(paths) == [zero]
    assert total == 16 and len(digest) == 64
    monkeypatch.setattr("vllm_hcu.preload.kimi_layer.MAX_RAW_BYTES", 15)
    with pytest.raises(ValueError, match="exceed"):
        selected_weights(str(tmp_path))


def test_partial_prefix_selects_only_requested_layers(tmp_path):
    names = {
        f"language_model.model.layers.{index}.input_layernorm.weight":
        torch.ones(index + 4)
        for index in range(4)
    }
    from safetensors.torch import save_file
    save_file(names, str(tmp_path / "part.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: "part.safetensors" for name in names}})
    )
    paths, total, digest = selected_weights(str(tmp_path), layer_start=1, num_layers=2)
    assert list(paths) == [
        "language_model.model.layers.1.input_layernorm.weight",
        "language_model.model.layers.2.input_layernorm.weight",
    ]
    assert total == (5 + 6) * 4
    assert len(digest) == 64


def test_partial_prefix_is_bounded_to_first_twelve_layers(tmp_path):
    with pytest.raises(ValueError, match="first 12"):
        selected_weights(str(tmp_path), layer_start=11, num_layers=2)
