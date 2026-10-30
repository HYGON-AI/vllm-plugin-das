# SPDX-License-Identifier: Apache-2.0
import json

import pytest
import torch
from safetensors.torch import save_file

from vllm_hcu.preload.moe_expert import read_expert


def make_expert(tmp_path):
    prefix = "model.layers.1.moe.experts.0"
    tensors = {}
    for projection, shape in (("w1", (4, 3)), ("w2", (6, 2)), ("w3", (4, 3))):
        tensors[f"{prefix}.{projection}.weight"] = torch.ones(shape, dtype=torch.int8)
        tensors[f"{prefix}.{projection}.weight_scale"] = torch.ones((shape[0], 1), dtype=torch.float32)
    save_file(tensors, str(tmp_path / "part.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: "part.safetensors" for name in tensors}}))
    return prefix, tensors


def test_loads_only_one_bounded_expert(tmp_path):
    prefix, tensors = make_expert(tmp_path)
    values = read_expert(str(tmp_path), prefix)
    names = [f"{prefix}.{p}.{s}" for p in ("w1", "w2", "w3") for s in ("weight", "weight_scale")]
    assert all(torch.equal(value, tensors[name]) for name, value in zip(names, values))
    with pytest.raises(ValueError, match="one numbered layer"):
        read_expert(str(tmp_path), "model.layers.1.moe.experts.0.w1",)


def test_combined_limit_checked_before_any_tensor_read(tmp_path, monkeypatch):
    import safetensors
    prefix, _ = make_expert(tmp_path)
    monkeypatch.setattr("vllm_hcu.preload.moe_expert.MAX_RAW_BYTES", 1)
    original = safetensors.safe_open
    reads = []
    class Tracking:
        def __init__(self, *args, **kwargs):
            self.inner = original(*args, **kwargs)
        def __enter__(self):
            self.inner.__enter__()
            return self
        def __exit__(self, *args):
            return self.inner.__exit__(*args)
        def get_slice(self, name):
            return self.inner.get_slice(name)
        def get_tensor(self, name):
            reads.append(name)
            return self.inner.get_tensor(name)
    monkeypatch.setattr(safetensors, "safe_open", Tracking)
    with pytest.raises(ValueError, match="raw byte limit"):
        read_expert(str(tmp_path), prefix)
    assert reads == []
