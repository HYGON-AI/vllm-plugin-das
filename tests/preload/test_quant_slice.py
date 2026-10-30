# SPDX-License-Identifier: Apache-2.0
import json

import pytest
import torch
from safetensors.torch import save_file

from vllm_hcu.preload.quant_linear import read_slice


def test_reads_matching_rows_of_one_quantized_linear(tmp_path):
    prefix = "model.layers.0.attn.k_proj"
    weight = torch.arange(32, dtype=torch.int8).reshape(8, 4)
    scale = torch.arange(8, dtype=torch.float32).reshape(8, 1)
    save_file({prefix + ".weight": weight, prefix + ".weight_scale": scale,
               "model.layers.1.attn.k_proj.weight": weight.clone()}, str(tmp_path / "part.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
        prefix + ".weight": "part.safetensors", prefix + ".weight_scale": "part.safetensors"}}))
    sliced_weight, sliced_scale = read_slice(str(tmp_path), prefix, 3)
    assert torch.equal(sliced_weight, weight[:3])
    assert torch.equal(sliced_scale, scale[:3])
    with pytest.raises(ValueError, match="between 1 and 4096"):
        read_slice(str(tmp_path), prefix, 4097)
    with pytest.raises(ValueError, match="numbered model layer"):
        read_slice(str(tmp_path), "model.embed", 1)


@pytest.mark.parametrize("quant_format", ["w8a8", "w4a8"])
def test_fallback_reprocesses_only_selected_slice(tmp_path, monkeypatch, capsys, quant_format):
    import sys
    from vllm_hcu.preload import quant_linear

    prefix = "model.layers.0.k_proj"
    weight = torch.arange(32, dtype=torch.int8).reshape(8, 4)
    scale = torch.ones(8, 1, dtype=torch.float32)
    save_file({prefix + ".weight": weight, prefix + ".weight_scale": scale},
              str(tmp_path / "part.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
        prefix + ".weight": "part.safetensors", prefix + ".weight_scale": "part.safetensors"}}))
    monkeypatch.setattr(sys, "argv", ["quant_linear", "load", "--model-dir", str(tmp_path),
                                     "--prefix", prefix, "--socket-dir", str(tmp_path),
                                     "--rows", "3", "--format", quant_format, "--fallback"])
    quant_linear.main()
    output = capsys.readouterr().out
    assert "processed only the selected slice" in output
    assert f"verified processed {quant_format.upper()} slice from bounded fallback" in output
