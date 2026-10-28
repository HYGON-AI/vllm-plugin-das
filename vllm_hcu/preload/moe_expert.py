# SPDX-License-Identifier: Apache-2.0
"""Restricted one-expert W4A8 post-processing and cache experiment.

Reads exactly w1, w2, w3 and their scales from one numbered expert in one
numbered layer, with a combined 32 MiB raw limit. It does not construct a model
or initialize a MoE execution kernel.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
from types import SimpleNamespace

MAX_RAW_BYTES = 32 * 1024 * 1024


def read_expert(model_dir: str, expert: str):
    from safetensors import safe_open

    if (not re.fullmatch(r"[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)*\.layers\.\d+\.[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)*\.experts\.\d+", expert)
            or len(re.findall(r"(?:^|\.)layers\.\d+(?=\.|$)", expert)) != 1
            or len(re.findall(r"(?:^|\.)experts\.\d+(?=\.|$)", expert)) != 1):
        raise ValueError("expected exactly one numbered layer and one numbered expert")
    root = os.path.realpath(model_dir)
    with open(os.path.join(root, "model.safetensors.index.json"), encoding="utf-8") as f:
        mapping = json.load(f)["weight_map"]
    names = [f"{expert}.{projection}.{suffix}" for projection in ("w1", "w2", "w3")
             for suffix in ("weight", "weight_scale")]
    descriptors = []
    total = 0
    # First pass inspects all metadata; no tensor data is materialized until
    # the entire expert has passed the combined size/shape constraints.
    for name in names:
        shard = mapping[name]
        if not isinstance(shard, str) or os.path.basename(shard) != shard:
            raise ValueError("checkpoint index must name a local shard")
        path = os.path.realpath(os.path.join(root, shard))
        if os.path.commonpath((root, path)) != root or not path.endswith(".safetensors"):
            raise ValueError("checkpoint shard escapes model directory")
        with safe_open(path, framework="pt", device="cpu") as f:
            view = f.get_slice(name)
            shape, dtype = view.get_shape(), view.get_dtype()
            expected = "F32" if name.endswith("weight_scale") else "I8"
            if len(shape) != 2 or any(dim <= 0 for dim in shape) or dtype != expected:
                raise ValueError(f"unsupported tensor metadata for {name}")
            size = math.prod(shape) * (4 if dtype == "F32" else 1)
            total += size
            descriptors.append((path, name, shape))
    if total > MAX_RAW_BYTES:
        raise ValueError(f"expert exceeds {MAX_RAW_BYTES} raw byte limit")
    w1, s1, w2, s2, w3, s3 = (item[2] for item in descriptors)
    if (w1 != w3 or s1 != [w1[0], 1] or s3 != s1 or s2 != [w2[0], 1]
            or w1[1] * 2 != w2[0] or w2[1] * 2 != w1[0]):
        raise ValueError("expert projection and scale dimensions are inconsistent")
    tensors = []
    for path, name, _ in descriptors:
        with safe_open(path, framework="pt", device="cpu") as f:
            tensors.append(f.get_tensor(name))
    return tensors


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("serve", "load"))
    p.add_argument("--model-dir", required=True)
    p.add_argument("--expert", required=True)
    p.add_argument("--socket-dir", required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = p.parse_args()
    if not os.path.isabs(args.socket_dir):
        p.error("socket directory must be absolute")
    import torch
    import vllm.version
    from vllm.config.load import LoadConfig
    from vllm.model_executor.model_loader.weight_cache import daemon, ipc_loader
    from vllm.model_executor.model_loader.weight_cache.daemon import WeightCacheDaemon
    from vllm.model_executor.model_loader.weight_cache.ipc_loader import IpcModelLoader
    from vllm_hcu.runtime_compat.weight_cache import install_expert_export_metadata, install_loader_compat
    install_expert_export_metadata(daemon)
    install_loader_compat(ipc_loader)
    from vllm.model_executor.model_loader.weight_cache.protocol import WeightCacheKey, get_socket_path, send_msg
    from vllm_hcu.model_executor.layers.quantization.compressed_tensors_moe_runtime import (
        _unpack_slimquant_w4a8_tensor, prepare_vllm_w4a8_moe,
    )

    started = time.monotonic()
    w1, s1, w2, s2, w3, s3 = read_expert(args.model_dir, args.expert)
    sha = hashlib.sha256()
    for value in (w1, s1, w2, s2, w3, s3):
        sha.update(value.contiguous().view(torch.uint8).numpy().tobytes())
    digest = sha.hexdigest()
    total = sum(t.numel() * t.element_size() for t in (w1, s1, w2, s2, w3, s3))
    print(f"read one expert: {total} bytes, sha256={digest}, {time.monotonic()-started:.3f}s", flush=True)
    key = WeightCacheKey(checkpoint=f"moe-expert:{args.expert}:{digest}",
        model_arch="hcu-one-expert-lab", tp_size=1, tp_rank=0, dtype="torch.int8",
        quantization="lab-w4a8", quant_config_hash="vllm-unpacked", revision=None,
        vllm_version=vllm.version.__version__)
    path = get_socket_path(0, args.socket_dir)
    if args.device == "cuda":
        torch.cuda.set_device(0)

    def packed_model():
        model = torch.nn.Module()
        for name, tensor in (("w13_weight", torch.cat((w1, w3)).unsqueeze(0)),
                             ("w2_weight", w2.unsqueeze(0)),
                             ("w13_weight_scale", torch.cat((s1, s3)).unsqueeze(0)),
                             ("w2_weight_scale", s2.unsqueeze(0))):
            model.register_parameter(name, torch.nn.Parameter(tensor.to(args.device), requires_grad=False))
        return model

    if args.action == "serve":
        model = packed_model()
        method = SimpleNamespace(moe=SimpleNamespace(hidden_dim=w1.shape[1] * 2,
            _hcu_kimi_k3=True, activation=SimpleNamespace(value="situ")))
        prepare_vllm_w4a8_moe(method, model)
        class ExpertDaemon(WeightCacheDaemon):
            def __init__(self):
                self.model, self.cache_config, self.tp_rank, self.socket_dir = model, key, 0, args.socket_dir
                self.entries, self.aliases = {"ready": True}, {}
            def _socket_path(self):
                return path
            def _gpu_uuid(self):
                return None if args.device == "cpu" else str(torch.cuda.get_device_properties(0).uuid)
            def _handle_get_state(self, conn, request):
                other = request.get("cache_config")
                if not isinstance(other, WeightCacheKey):
                    send_msg(conn, {"status": "error", "message": "Missing cache_config"}); return
                mismatch = self.cache_config.mismatched_fields(other)
                if mismatch:
                    send_msg(conn, {"status": "mismatch", "fields": mismatch}); return
                if self.model is None:
                    send_msg(conn, {"status": "error", "message": "Weights were released"}); return
                entries, aliases = daemon.export_entries(self.model)
                send_msg(conn, {"status": "ok", "entries": entries, "aliases": aliases, "gpu_uuid": self._gpu_uuid()})
        ExpertDaemon().serve_forever()
    else:
        loader = IpcModelLoader(LoadConfig(load_format="ipc_cache", model_loader_extra_config={"socket_path": path, "fallback": False}))
        started = time.monotonic()
        entries, aliases = loader._request_state(key)
        model = torch.nn.Module()
        loader._apply_entries(model, entries, aliases, 0)
        if not (getattr(model.w13_weight, "_hcu_vllm_w4a8_unpacked", False)
                and getattr(model.w2_weight, "_hcu_vllm_w4a8_unpacked", False)):
            raise RuntimeError("unpacked expert layout markers were not restored")
        expected13 = _unpack_slimquant_w4a8_tensor(torch.cat((w1, w3)).unsqueeze(0))
        expected2 = _unpack_slimquant_w4a8_tensor(w2.unsqueeze(0))
        if (model.w13_weight.device.type != args.device or not torch.equal(model.w13_weight.cpu(), expected13)
                or not torch.equal(model.w2_weight.cpu(), expected2)
                or not torch.equal(model.w13_weight_scale.cpu(), torch.cat((s1, s3)).unsqueeze(0))
                or not torch.equal(model.w2_weight_scale.cpu(), s2.unsqueeze(0))):
            raise RuntimeError("cached expert does not match processed checkpoint weights")
        # Numerically compare bounded input slices of both expert projections.
        # This is a CPU/Torch reference, not MoE routing or an HCU kernel test.
        for actual_weight, reference_weight, actual_scale, reference_scale in (
            (model.w13_weight, expected13, model.w13_weight_scale,
             torch.cat((s1, s3)).unsqueeze(0)),
            (model.w2_weight, expected2, model.w2_weight_scale, s2.unsqueeze(0)),
        ):
            width = min(64, reference_weight.shape[-1])
            x = torch.arange(width, dtype=torch.float32).remainder(7)
            actual = (actual_weight[0, :, :width].float() @ x.to(args.device)) * actual_scale[0, :, 0]
            expected = (reference_weight[0, :, :width].float() @ x) * reference_scale[0, :, 0]
            if not torch.equal(actual.cpu(), expected):
                raise RuntimeError("cached expert projection numerical comparison failed")
        print(f"mapped and verified processed single expert in {time.monotonic()-started:.3f}s", flush=True)


if __name__ == "__main__":
    main()
