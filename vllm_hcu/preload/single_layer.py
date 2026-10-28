# SPDX-License-Identifier: Apache-2.0
"""Exercise vLLM's weight-cache protocol with exactly one synthetic linear layer.

This is a lab tool, not a model server or a checkpoint loader. It cannot load
model weights and is deliberately bounded to one small unquantized layer.
"""
from __future__ import annotations

import argparse
import os
import time

MAX_ELEMENTS = 1024 * 1024


def dimensions(value: str) -> int:
    result = int(value)
    if not 1 <= result <= 4096:
        raise argparse.ArgumentTypeError("dimension must be in [1, 4096]")
    return result


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("serve", "load"))
    p.add_argument("--socket-dir", required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--input-size", type=dimensions, default=64)
    p.add_argument("--output-size", type=dimensions, default=64)
    p.add_argument("--seed", type=int, default=7)
    return p


def main() -> None:
    args = parser().parse_args()
    if args.input_size * args.output_size > MAX_ELEMENTS:
        raise SystemExit(f"one-layer limit exceeded ({MAX_ELEMENTS} elements)")
    if not os.path.isabs(args.socket_dir):
        raise SystemExit("--socket-dir must be an absolute path")

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

    path = get_socket_path(0, args.socket_dir)
    key = WeightCacheKey(
        checkpoint=f"synthetic-linear:{args.input_size}:{args.output_size}:{args.seed}",
        model_arch="hcu-single-linear-lab", tp_size=1, tp_rank=0,
        dtype="torch.float32", quantization=None, quant_config_hash="",
        revision=None, vllm_version=vllm.version.__version__,
    )

    if args.action == "serve":
        if args.device == "cuda":
            torch.cuda.set_device(0)
        torch.manual_seed(args.seed)
        started = time.monotonic()
        model = torch.nn.Linear(args.input_size, args.output_size, bias=False)
        model = model.to(args.device).eval()

        class SingleLayerDaemon(WeightCacheDaemon):
            def __init__(self):
                self.model = model
                self.cache_config = key
                self.tp_rank = 0
                self.socket_dir = args.socket_dir
                # Only a sentinel: IPC reductions must be made per client.
                self.entries = {"weight": True}
                self.aliases = {}

            def _socket_path(self):
                return path

            def _gpu_uuid(self):
                if args.device == "cpu":
                    return None
                return str(torch.cuda.get_device_properties(0).uuid)

            def _handle_get_state(self, conn, request):
                client = request.get("cache_config")
                if not isinstance(client, WeightCacheKey):
                    send_msg(conn, {"status": "error", "message": "Missing cache_config"})
                    return
                mismatch = self.cache_config.mismatched_fields(client)
                if mismatch:
                    send_msg(conn, {"status": "mismatch", "fields": mismatch})
                    return
                if self.model is None:
                    send_msg(conn, {"status": "error", "message": "Weights were released"})
                    return
                entries, aliases = export_entries(self.model)
                send_msg(conn, {"status": "ok", "entries": entries,
                                "aliases": aliases, "gpu_uuid": self._gpu_uuid()})

        print(f"single layer constructed in {time.monotonic() - started:.3f}s", flush=True)
        SingleLayerDaemon().serve_forever()
    else:
        if args.device == "cuda":
            torch.cuda.set_device(0)
        loader = IpcModelLoader(LoadConfig(
            load_format="ipc_cache",
            model_loader_extra_config={"socket_path": path, "fallback": False},
        ))
        # No checkpoint and no fallback: a mismatch must fail closed.
        started = time.monotonic()
        entries, aliases = loader._request_state(key)
        with torch.device("meta"):
            model = torch.nn.Linear(args.input_size, args.output_size, bias=False)
        loader._apply_entries(model, entries, aliases, 0)
        print(f"mapped {len(entries)} tensor(s) in {time.monotonic() - started:.3f}s", flush=True)
        if str(model.weight.device).split(":")[0] != args.device:
            raise RuntimeError("daemon and client device types differ")
        # Independent construction verifies the mapped layer's values.
        torch.manual_seed(args.seed)
        expected = torch.nn.Linear(args.input_size, args.output_size, bias=False)
        if not torch.equal(model.weight.cpu(), expected.weight):
            raise RuntimeError("cached layer does not match the expected values")
        print("single-layer weights match", flush=True)


if __name__ == "__main__":
    main()
