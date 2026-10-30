# SPDX-License-Identifier: Apache-2.0
"""Exercise the cache with exactly one explicitly selected checkpoint tensor.

This tool reads one tensor belonging to one numbered model layer. It never
constructs a model or loads a whole shard. It does not apply quantization
post-processing and is not a vLLM model-serving entry point.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time

MAX_TENSOR_BYTES = 16 * 1024 * 1024
_DTYPE_BYTES = {
    "BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
    "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
    "I32": 4, "U32": 4, "F32": 4, "I64": 8, "U64": 8, "F64": 8,
}


def read_one_tensor(model_dir: str, name: str):
    """Read a single bounded tensor; validate its metadata before loading it."""
    from safetensors import safe_open

    if len(re.findall(r"(?:^|\.)layers\.\d+\.", name)) != 1:
        raise ValueError("tensor name must contain exactly one numbered layers.N component")
    root = os.path.realpath(model_dir)
    with open(os.path.join(root, "model.safetensors.index.json"), encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]
    shard = weight_map[name]
    if not isinstance(shard, str) or os.path.basename(shard) != shard:
        raise ValueError("checkpoint index must name a local shard file")
    path = os.path.realpath(os.path.join(root, shard))
    if os.path.commonpath((root, path)) != root or not path.endswith(".safetensors"):
        raise ValueError("checkpoint shard escapes the model directory")
    with safe_open(path, framework="pt", device="cpu") as f:
        metadata = f.get_slice(name)
        shape, dtype = metadata.get_shape(), metadata.get_dtype()
        size = math.prod(shape) * _DTYPE_BYTES.get(dtype, MAX_TENSOR_BYTES + 1)
        if size <= 0 or size > MAX_TENSOR_BYTES:
            raise ValueError(f"tensor size {size} exceeds the {MAX_TENSOR_BYTES} byte limit")
        tensor = f.get_tensor(name)
    if tensor.numel() * tensor.element_size() != size:
        raise ValueError("loaded tensor size does not match checkpoint metadata")
    return tensor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("serve", "load"))
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--tensor", required=True)
    parser.add_argument("--socket-dir", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if not os.path.isabs(args.socket_dir):
        parser.error("--socket-dir must be absolute")

    import torch
    import vllm.version
    from vllm.config.load import LoadConfig
    from vllm.model_executor.model_loader.weight_cache.daemon import (
        WeightCacheDaemon, export_entries,
    )
    from vllm.model_executor.model_loader.weight_cache.ipc_loader import IpcModelLoader
    from vllm.model_executor.model_loader.weight_cache.protocol import (
        WeightCacheKey, get_socket_path, send_msg,
    )

    start = time.monotonic()
    tensor = read_one_tensor(args.model_dir, args.tensor)
    digest = hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    print(f"read one checkpoint tensor: {tensor.numel() * tensor.element_size()} bytes, "
          f"sha256={digest}, {time.monotonic() - start:.3f}s", flush=True)
    key = WeightCacheKey(
        checkpoint=f"single-tensor:{args.tensor}:{digest}",
        model_arch="hcu-single-checkpoint-tensor-lab", tp_size=1, tp_rank=0,
        dtype=str(tensor.dtype), quantization=None, quant_config_hash="",
        revision=None, vllm_version=vllm.version.__version__,
    )
    path = get_socket_path(0, args.socket_dir)
    if args.device == "cuda":
        torch.cuda.set_device(0)

    if args.action == "serve":
        model = torch.nn.Module()
        model.register_buffer("weight", tensor.to(args.device))

        class SingleTensorDaemon(WeightCacheDaemon):
            def __init__(self):
                self.model = model
                self.cache_config = key
                self.tp_rank = 0
                self.socket_dir = args.socket_dir
                self.entries = {"weight": True}
                self.aliases = {}

            def _socket_path(self):
                return path

            def _gpu_uuid(self):
                return None if args.device == "cpu" else str(torch.cuda.get_device_properties(0).uuid)

            def _handle_get_state(self, conn, request):
                other = request.get("cache_config")
                if not isinstance(other, WeightCacheKey):
                    send_msg(conn, {"status": "error", "message": "Missing cache_config"})
                    return
                mismatch = self.cache_config.mismatched_fields(other)
                if mismatch:
                    send_msg(conn, {"status": "mismatch", "fields": mismatch})
                    return
                if self.model is None:
                    send_msg(conn, {"status": "error", "message": "Weights were released"})
                    return
                entries, aliases = export_entries(self.model)
                send_msg(conn, {"status": "ok", "entries": entries,
                                "aliases": aliases, "gpu_uuid": self._gpu_uuid()})

        SingleTensorDaemon().serve_forever()
    else:
        loader = IpcModelLoader(LoadConfig(
            load_format="ipc_cache",
            model_loader_extra_config={"socket_path": path, "fallback": False},
        ))
        started = time.monotonic()
        entries, aliases = loader._request_state(key)
        model = torch.nn.Module()
        model.register_buffer("weight", torch.empty_like(tensor, device="meta"))
        loader._apply_entries(model, entries, aliases, 0)
        if model.weight.device.type != args.device or not torch.equal(model.weight.cpu(), tensor):
            raise RuntimeError("cached tensor does not match checkpoint and device")
        print(f"mapped and verified one tensor in {time.monotonic() - started:.3f}s", flush=True)


if __name__ == "__main__":
    main()
