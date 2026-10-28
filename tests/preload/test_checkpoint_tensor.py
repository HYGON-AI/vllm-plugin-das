# SPDX-License-Identifier: Apache-2.0
import json

import pytest
import torch
from safetensors.torch import save_file

from vllm_hcu.preload.checkpoint_tensor import read_one_tensor


def checkpoint(tmp_path):
    values = {
        "model.layers.0.weight": torch.arange(6, dtype=torch.float32),
        "model.layers.1.weight": torch.arange(5, dtype=torch.float32),
        "model.embed.weight": torch.arange(3, dtype=torch.float32),
    }
    save_file(values, str(tmp_path / "part.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: "part.safetensors" for name in values}})
    )
    return values


def test_reads_only_requested_layer_tensor(tmp_path, monkeypatch):
    import safetensors

    values = checkpoint(tmp_path)
    open_original = safetensors.safe_open
    requested = []

    class Instrumented:
        def __init__(self, *args, **kwargs):
            self.inner = open_original(*args, **kwargs)
        def __enter__(self):
            self.inner.__enter__()
            return self
        def __exit__(self, *args):
            return self.inner.__exit__(*args)
        def get_slice(self, name):
            return self.inner.get_slice(name)
        def get_tensor(self, name):
            requested.append(name)
            return self.inner.get_tensor(name)

    monkeypatch.setattr(safetensors, "safe_open", Instrumented)
    assert torch.equal(read_one_tensor(str(tmp_path), "model.layers.0.weight"),
                       values["model.layers.0.weight"])
    assert requested == ["model.layers.0.weight"]


def test_rejects_non_layer_and_oversize_before_materialization(tmp_path, monkeypatch):
    checkpoint(tmp_path)
    with pytest.raises(ValueError, match="numbered layers"):
        read_one_tensor(str(tmp_path), "model.embed.weight")
    monkeypatch.setattr("vllm_hcu.preload.checkpoint_tensor.MAX_TENSOR_BYTES", 20)
    with pytest.raises(ValueError, match="exceeds"):
        read_one_tensor(str(tmp_path), "model.layers.0.weight")


def test_rejects_index_shard_escape(tmp_path):
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.layers.0.weight": "../outside.safetensors"}})
    )
    with pytest.raises(ValueError, match="local shard"):
        read_one_tensor(str(tmp_path), "model.layers.0.weight")
