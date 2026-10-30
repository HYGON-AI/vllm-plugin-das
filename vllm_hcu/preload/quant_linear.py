# SPDX-License-Identifier: Apache-2.0
"""Restricted Kimi W8A8 linear slice cache experiment, not a model server."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from types import SimpleNamespace

MAX_BYTES = 16 * 1024 * 1024


def read_slice(model_dir: str, prefix: str, rows: int):
    """Read bounded rows of one layer's quantized weight and its scales."""
    from safetensors import safe_open

    if len(re.findall(r"(?:^|\.)layers\.\d+\.", prefix)) != 1:
        raise ValueError("prefix must name exactly one numbered model layer")
    if not 1 <= rows <= 4096:
        raise ValueError("rows must be between 1 and 4096")
    root = os.path.realpath(model_dir)
    with open(os.path.join(root, "model.safetensors.index.json"), encoding="utf-8") as f:
        mapping = json.load(f)["weight_map"]
    names = [prefix + ".weight", prefix + ".weight_scale"]
    tensors = []
    shapes = []
    for name, dtype in zip(names, ("I8", "F32")):
        shard = mapping[name]
        if not isinstance(shard, str) or os.path.basename(shard) != shard:
            raise ValueError("checkpoint index must name a local shard")
        path = os.path.realpath(os.path.join(root, shard))
        if os.path.commonpath((root, path)) != root or not path.endswith(".safetensors"):
            raise ValueError("checkpoint shard escapes model directory")
        with safe_open(path, framework="pt", device="cpu") as f:
            view = f.get_slice(name)
            shape = view.get_shape()
            if (len(shape) != 2 or view.get_dtype() != dtype or shape[0] < rows
                    or shape[1] <= 0 or rows * shape[1] * (1 if dtype == "I8" else 4) > MAX_BYTES):
                raise ValueError(f"unsupported or oversized tensor slice: {name}")
            if dtype == "F32" and shape[1] != 1:
                raise ValueError("only per-channel W8A8 scales are supported")
            tensors.append(view[:rows, :])
            shapes.append(shape)
    if shapes[0][0] != shapes[1][0]:
        raise ValueError("weight and scale channel counts differ")
    return tensors


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("serve", "load"))
    p.add_argument("--model-dir", required=True)
    p.add_argument("--prefix", required=True)
    p.add_argument("--socket-dir", required=True)
    p.add_argument("--rows", type=int, default=64)
    p.add_argument("--format", choices=("w8a8", "w4a8"), default="w8a8")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--fallback", action="store_true", help="reprocess this bounded slice if cache is unavailable or mismatched")
    args = p.parse_args()
    if not os.path.isabs(args.socket_dir):
        p.error("socket directory must be absolute")
    import torch
    import vllm.version
    from vllm.config.load import LoadConfig
    from vllm.model_executor.kernels.linear.scaled_mm.aiter import AiterInt8ScaledMMLinearKernel
    from vllm.model_executor.model_loader.weight_cache.daemon import WeightCacheDaemon, export_entries
    from vllm.model_executor.model_loader.weight_cache.ipc_loader import IpcModelLoader
    from vllm.model_executor.model_loader.weight_cache.protocol import (
        CacheConfigMismatchError, WeightCacheKey, WeightCacheUnavailableError,
        get_socket_path, send_msg,
    )
    from vllm.model_executor.utils import weights_already_processed
    from vllm_hcu.model_executor.layers.quantization.slimquant_w4a8 import SlimQuantW4A8Int8LinearMethod
    from vllm_hcu.model_executor.layers.quantization.compressed_tensors_moe_runtime import _unpack_slimquant_w4a8_tensor

    started = time.monotonic()
    weight, scale = read_slice(args.model_dir, args.prefix, args.rows)
    sha = hashlib.sha256()
    for tensor in (weight, scale):
        sha.update(tensor.contiguous().view(torch.uint8).numpy().tobytes())
    digest = sha.hexdigest()
    print(f"read {args.format.upper()} slice: weight={tuple(weight.shape)}, scale={tuple(scale.shape)}, "
          f"sha256={digest}, {time.monotonic() - started:.3f}s", flush=True)
    key = WeightCacheKey(checkpoint=f"{args.format}-slice:{args.prefix}:{args.rows}:{digest}",
        model_arch="hcu-w8a8-linear-slice-lab", tp_size=1, tp_rank=0, dtype="torch.int8",
        quantization=f"lab-{args.format}", quant_config_hash="channel-dynamic-aiter",
        revision=None, vllm_version=vllm.version.__version__)
    path = get_socket_path(0, args.socket_dir)
    if args.device == "cuda":
        torch.cuda.set_device(0)

    def make_method(layer):
        kernel = object.__new__(AiterInt8ScaledMMLinearKernel)
        kernel.layer_param_names = ("weight", "weight_scale", "input_scale", "input_zero_point", "azp_adj")
        kernel.config = SimpleNamespace(is_channelwise=True, is_static_input_scheme=False, input_symmetric=True)
        class Scheme:
            def create_weights(self, **kwargs):
                self.kernel = kernel
            def process_weights_after_loading(self, layer):
                self.kernel.process_weights_after_loading(layer)
        layer.scheme = Scheme()
        layer.logical_widths = [args.rows]
        method = SlimQuantW4A8Int8LinearMethod(None)
        method.create_weights(layer, weight.shape[1], [args.rows], weight.shape[1], args.rows, torch.bfloat16)
        return method

    if args.action == "serve":
        model = torch.nn.Module()
        model.weight = torch.nn.Parameter(weight.to(args.device), requires_grad=False)
        model.weight_scale = torch.nn.Parameter(scale.to(args.device), requires_grad=False)
        if args.format == "w8a8":
            make_method(model).process_weights_after_loading(model)
        else:
            model.weight = torch.nn.Parameter(_unpack_slimquant_w4a8_tensor(model.weight), requires_grad=False)
        class SliceDaemon(WeightCacheDaemon):
            def __init__(self):
                self.model, self.cache_config, self.tp_rank, self.socket_dir = model, key, 0, args.socket_dir
                self.entries, self.aliases = {"weight": True}, {}
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
                entries, aliases = export_entries(self.model)
                send_msg(conn, {"status": "ok", "entries": entries, "aliases": aliases, "gpu_uuid": self._gpu_uuid()})
        SliceDaemon().serve_forever()
    else:
        loader = IpcModelLoader(LoadConfig(load_format="ipc_cache", model_loader_extra_config={"socket_path": path, "fallback": False}))
        started = time.monotonic()
        model = torch.nn.Module()
        source = "cache"
        try:
            entries, aliases = loader._request_state(key)
        except (CacheConfigMismatchError, WeightCacheUnavailableError):
            if not args.fallback:
                raise
            source = "bounded fallback"
            model.weight = torch.nn.Parameter(weight.to(args.device), requires_grad=False)
            model.weight_scale = torch.nn.Parameter(scale.to(args.device), requires_grad=False)
            if args.format == "w8a8":
                make_method(model).process_weights_after_loading(model)
            else:
                model.weight = torch.nn.Parameter(_unpack_slimquant_w4a8_tensor(model.weight), requires_grad=False)
            print("cache unavailable or mismatched; processed only the selected slice", flush=True)
        else:
            model.weight = torch.nn.Parameter(torch.empty_like(weight, device="meta"), requires_grad=False)
            model.weight_scale = torch.nn.Parameter(torch.empty_like(scale, device="meta"), requires_grad=False)
            method = make_method(model) if args.format == "w8a8" else None
            loader._apply_entries(model, entries, aliases, 0)
            if method is not None:
                with weights_already_processed():
                    method.process_weights_after_loading(model)
        if args.format == "w8a8":
            expected_weight = weight.T
        else:
            packed = weight.to(torch.int16) & 255
            hi, lo = packed // 16, packed % 16
            hi = torch.where(hi >= 8, hi - 16, hi)
            lo = torch.where(lo >= 8, lo - 16, lo)
            expected_weight = torch.stack((hi, lo), dim=-1).flatten(-2).to(torch.int8)
        if model.weight.device.type != args.device or not torch.equal(model.weight.cpu(), expected_weight) or not torch.equal(model.weight_scale.cpu(), scale):
            raise RuntimeError("processed weights do not match the checkpoint slice")
        matrix = model.weight if args.format == "w8a8" else model.weight.T
        reference = weight.T if args.format == "w8a8" else expected_weight.T
        x = torch.arange(matrix.shape[0], dtype=torch.float32).remainder(7).reshape(1, -1).to(args.device)
        actual = (x @ matrix.float()) * model.weight_scale.T
        expected = (x.cpu() @ reference.float()) * scale.T
        if not torch.equal(actual.cpu(), expected):
            raise RuntimeError("linear slice numerical comparison failed")
        print(f"verified processed {args.format.upper()} slice from {source} in {time.monotonic() - started:.3f}s", flush=True)


if __name__ == "__main__":
    main()
