# SPDX-License-Identifier: Apache-2.0
"""Read-only SlimQuant checkpoint metadata audit. No tensor data is loaded.

Metadata validation is a preflight, not a statement about model loadability,
weight values, tensor-parallel slicing, or inference correctness.
"""
from __future__ import annotations
import collections
import hashlib
import json
from pathlib import Path
import re
import struct

_MAX_HEADER = 256 * 1024 * 1024
_SIZE = {"BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
         "I16": 2, "U16": 2, "F16": 2, "BF16": 2, "I32": 4, "U32": 4,
         "F32": 4, "I64": 8, "U64": 8, "F64": 8}


def _family(name):
    name = re.sub(r"(?<=layers\.)\d+", "*", name)
    return re.sub(r"(?<=experts\.)\d+", "*", name)


def _excluded_module(name, ignores):
    # Checkpoint names only. Fused module mapping requires separate validation.
    module = name.removesuffix(".weight")
    return any(module == item or module.endswith("." + item)
               or module.startswith(item + ".") or ("." + item + ".") in module
               for item in ignores)


def audit_checkpoint(root):
    root = Path(root).resolve(strict=True)
    config_bytes = (root / "config.json").read_bytes()
    index_bytes = (root / "model.safetensors.index.json").read_bytes()
    config = json.loads(config_bytes)
    quant = config.get("quantization_config", {})
    if quant.get("quant_method") != "slimquant_w4a8" or quant.get("activation_scheme") != "dynamic":
        raise ValueError("expected dynamic slimquant_w4a8 checkpoint")
    ignores = quant.get("modules_to_not_convert")
    if not isinstance(ignores, list) or not ignores or any(not isinstance(x, str) or not x for x in ignores):
        raise ValueError("expected nonempty modules_to_not_convert list")
    index = json.loads(index_bytes).get("weight_map")
    if not isinstance(index, dict) or not index:
        raise ValueError("missing weight_map")
    grouped = collections.defaultdict(set)
    for name, shard in index.items():
        if not isinstance(name, str) or not isinstance(shard, str):
            raise ValueError("invalid index entry")
        path = (root / shard).resolve(strict=True)
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("invalid shard path")
        grouped[shard].add(name)
    specs = {}
    total_bytes = 0
    for shard, expected in grouped.items():
        path = root / shard
        size = path.stat().st_size
        with path.open("rb") as f:
            nraw = f.read(8)
            if len(nraw) != 8:
                raise ValueError(f"truncated header: {shard}")
            length = struct.unpack("<Q", nraw)[0]
            if length > _MAX_HEADER or length > size - 8:
                raise ValueError(f"invalid header length: {shard}")
            raw = f.read(length)
        header = json.loads(raw)
        names = set(header) - {"__metadata__"}
        if names != expected:
            raise ValueError(f"index/header mismatch in {shard}: missing={len(expected-names)} extra={len(names-expected)}")
        intervals = []
        for name in names:
            info = header[name]
            dtype, shape, offsets = info.get("dtype"), info.get("shape"), info.get("data_offsets")
            if dtype not in _SIZE or not isinstance(shape, list) or any(type(n) is not int or n < 0 for n in shape):
                raise ValueError(f"unsupported dtype or shape: {name}")
            if not isinstance(offsets, list) or len(offsets) != 2 or any(type(x) is not int for x in offsets):
                raise ValueError(f"invalid offsets: {name}")
            a, b = offsets
            count = 1
            for n in shape: count *= n
            if a < 0 or b < a or b > size - 8 - length or b-a != count * _SIZE[dtype]:
                raise ValueError(f"invalid byte extent: {name}")
            intervals.append((a,b,name))
            specs[name] = (dtype, tuple(shape))
        intervals.sort()
        if any(end > next_start for (_,end,_),(next_start,_,_) in zip(intervals, intervals[1:])):
            raise ValueError(f"overlapping tensors: {shard}")
        total_bytes += size
    families = collections.Counter()
    issues = []
    for name, (dtype, shape) in specs.items():
        if name.endswith(".weight_scale"):
            weight = name.removesuffix("_scale")
            if weight not in specs:
                issues.append(f"scale without weight: {name}")
                continue
            wdtype, wshape = specs[weight]
            if dtype != "F32" or wdtype != "I8" or not wshape or shape != (wshape[0], 1):
                issues.append(f"unexpected per-channel weight/scale: {name}: {wdtype}{wshape}/{dtype}{shape}")
        kind = "expert_packed_w4" if ".experts." in name and name.endswith(".weight") else "weight_with_scale" if name.endswith(".weight") and name+"_scale" in specs else "other"
        if kind in ("expert_packed_w4", "weight_with_scale"):
            if dtype != "I8" or name + "_scale" not in specs:
                issues.append(f"quantized weight missing int8 storage or scale: {name}")
            if _excluded_module(name, ignores):
                issues.append(f"quantized weight conflicts with modules_to_not_convert: {name}")
        families[(_family(name), dtype, str(list(shape)), kind)] += 1
    return {"model_type": config.get("model_type"), "architecture": config.get("architectures"),
            "quantization": quant, "index_sha256": hashlib.sha256(index_bytes).hexdigest(),
            "config_sha256": hashlib.sha256(config_bytes).hexdigest(), "shards": len(grouped),
            "tensors": len(specs), "shard_bytes": total_bytes, "issues": issues,
            "families": [{"name": n, "dtype": d, "shape": s, "kind": k, "count": c}
                         for (n,d,s,k),c in sorted(families.items())],
            "limitations": "metadata only; no tensor payload, TP loader or IPC validation"}
