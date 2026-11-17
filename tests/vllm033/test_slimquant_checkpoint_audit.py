# SPDX-License-Identifier: Apache-2.0
import json
import struct
import pytest
from hcu_vllm_das.quantization.checkpoint_audit import audit_checkpoint


def make_model(path):
    config = {"model_type":"kimi_k31", "quantization_config": {"quant_method":"slimquant_w4a8", "activation_scheme":"dynamic", "modules_to_not_convert":["self_attn.indexer"]}}
    (path / "config.json").write_text(json.dumps(config))
    names = {"language_model.model.layers.1.block_sparse_moe.experts.0.w1.weight": ("I8",[2,2],4),
             "language_model.model.layers.1.block_sparse_moe.experts.0.w1.weight_scale": ("F32",[2,1],8)}
    header, off = {}, 0
    for name,(dtype,shape,size) in names.items():
        header[name] = {"dtype":dtype,"shape":shape,"data_offsets":[off,off+size]}
        off += size
    data = json.dumps(header).encode()
    (path / "model.safetensors").write_bytes(struct.pack("<Q",len(data))+data+bytes(off))
    (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map":{k:"model.safetensors" for k in names}}))
    return header


def test_audit_reads_metadata_not_tensor_values(tmp_path):
    make_model(tmp_path)
    r = audit_checkpoint(tmp_path)
    assert r["shards"] == 1 and r["tensors"] == 2 and r["issues"] == []
    assert any(x["kind"] == "expert_packed_w4" and x["count"] == 1 for x in r["families"])
    assert "metadata only" in r["limitations"]


def test_audit_rejects_wrong_index(tmp_path):
    make_model(tmp_path)
    p = tmp_path / "model.safetensors.index.json"
    d = json.loads(p.read_text()); d["weight_map"].pop(next(iter(d["weight_map"])))
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="index/header mismatch"):
        audit_checkpoint(tmp_path)


def test_audit_reports_invalid_scale_shape(tmp_path):
    h = make_model(tmp_path)
    key = next(k for k in h if k.endswith("scale"))
    h[key]["shape"] = [1,2]
    raw = json.dumps(h).encode()
    (tmp_path / "model.safetensors").write_bytes(struct.pack("<Q",len(raw))+raw+bytes(12))
    assert len(audit_checkpoint(tmp_path)["issues"]) == 1


def test_audit_rejects_path_escape(tmp_path):
    make_model(tmp_path)
    other = tmp_path.parent / (tmp_path.name + "-outside")
    other.write_bytes(b"outside")
    p=tmp_path / "model.safetensors.index.json"
    d=json.loads(p.read_text()); d["weight_map"][next(iter(d["weight_map"]))] = "../"+other.name
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="invalid shard path"):
        audit_checkpoint(tmp_path)


def test_audit_flags_quantized_excluded_module(tmp_path):
    make_model(tmp_path)
    p = tmp_path / "config.json"
    d = json.loads(p.read_text())
    d["quantization_config"]["modules_to_not_convert"] = ["block_sparse_moe.experts"]
    p.write_text(json.dumps(d))
    assert any("conflicts" in x for x in audit_checkpoint(tmp_path)["issues"])
