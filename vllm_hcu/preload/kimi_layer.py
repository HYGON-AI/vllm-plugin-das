# SPDX-License-Identifier: Apache-2.0
"""Bounded Kimi-K3 partial-layer preload experiment; never loads a full model."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time

# Keep this experiment bounded to the requested prefix and a finite total raw
# checkpoint budget. Kimi-K3 layers include thousands of MoE expert tensors;
# the first 12 layers are about 161 GiB in this checkpoint, so a per-layer
# 2-GiB limit rejects the requested gate before model construction. The 192-GiB
# cap still rejects a full-model scan (and the selector never leaves the first
# 12 layers).
MAX_RAW_BYTES = 192 * 1024**3
LAYER_PREFIX = "language_model.model.layers.0."


def selected_weights(model_dir, layer_start=0, num_layers=1):
    from safetensors import safe_open
    if not 0 <= layer_start < 12:
        raise ValueError("layer_start must be in [0, 11]")
    if not 1 <= num_layers <= 12 or layer_start + num_layers > 12:
        raise ValueError("the partial-layer lab supports at most the first 12 layers")
    root = os.path.realpath(model_dir)
    with open(os.path.join(root, "model.safetensors.index.json"), encoding="utf-8") as f:
        mapping = json.load(f)["weight_map"]
    prefixes = tuple(
        f"language_model.model.layers.{index}."
        for index in range(layer_start, layer_start + num_layers)
    )
    names = sorted(n for n in mapping if n.startswith(prefixes))
    if not names:
        raise ValueError("checkpoint contains no requested partial-layer tensors")
    sizes = {"I8": 1, "U8": 1, "F16": 2, "BF16": 2, "F32": 4, "I32": 4, "F64": 8, "I64": 8}
    paths = {}
    total = 0
    digest = hashlib.sha256()
    for name in names:
        shard = mapping[name]
        if not isinstance(shard, str) or os.path.basename(shard) != shard:
            raise ValueError("checkpoint index must name a local shard")
        path = os.path.realpath(os.path.join(root, shard))
        if os.path.commonpath((root, path)) != root or not path.endswith(".safetensors"):
            raise ValueError("checkpoint shard escapes model directory")
        with safe_open(path, framework="pt", device="cpu") as f:
            tensor = f.get_slice(name)
            shape, dtype = tensor.get_shape(), tensor.get_dtype()
            if dtype not in sizes or any(d < 0 for d in shape):
                raise ValueError(f"unsupported tensor metadata: {name}")
            total += math.prod(shape) * sizes[dtype]
            digest.update(json.dumps([name, shape, dtype], separators=(",", ":")).encode())
        paths[name] = path
    if total > MAX_RAW_BYTES:
        raise ValueError(f"partial-layer raw weights exceed {MAX_RAW_BYTES} bytes")
    for path in sorted(set(paths.values())):
        stat = os.stat(path)
        digest.update(json.dumps([os.path.basename(path), stat.st_size,
                                  stat.st_mtime_ns, stat.st_ctime_ns]).encode())
    digest.update(json.dumps([layer_start, num_layers]).encode())
    return paths, total, digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("serve", "load"))
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--socket-dir", required=True)
    parser.add_argument("--tp-size", type=int, choices=(1, 2, 4, 8), default=1)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--master-addr", default="127.0.0.1",
                        help="rendezvous address; use the rank-0 node for cross-node TP")
    parser.add_argument("--master-port", type=int, default=0)
    parser.add_argument("--layer-start", type=int, default=0)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--local-device", type=int, default=None,
                        help="physical device index on this node (defaults to --rank)")
    args = parser.parse_args()
    local_device = args.rank if args.local_device is None else args.local_device
    if not 0 <= args.rank < args.tp_size or local_device < 0:
        parser.error("TP rank and local device must be valid")
    if args.tp_size > 1 and not 1 <= args.master_port <= 65535:
        parser.error("TP rank and master port must be valid")
    if not os.path.isabs(args.socket_dir):
        parser.error("socket directory must be absolute")
    paths, total, digest = selected_weights(
        args.model_dir, args.layer_start, args.num_layers
    )
    layer_end = args.layer_start + args.num_layers
    print(
        f"selected layers [{args.layer_start}, {layer_end}): {len(paths)} tensors, "
        f"{total} raw bytes, fingerprint={digest}",
        flush=True,
    )

    import torch
    import vllm.version
    from safetensors import safe_open
    from vllm.plugins import load_general_plugins
    load_general_plugins()
    from vllm.engine.arg_utils import EngineArgs
    from vllm.config import set_current_vllm_config
    from vllm.distributed import init_distributed_environment, ensure_model_parallel_initialized
    from vllm.platforms import current_platform
    from vllm.utils.network_utils import get_distributed_init_method, get_open_port
    from vllm.utils.torch_utils import set_default_torch_dtype
    from vllm.model_executor.model_loader.utils import process_weights_after_loading
    from vllm.forward_context import set_forward_context
    from vllm.model_executor.model_loader.weight_cache import daemon, ipc_loader
    from vllm.model_executor.model_loader.weight_cache.protocol import WeightCacheKey, get_socket_path, recv_msg, send_msg
    from vllm.config.load import LoadConfig
    from vllm.model_executor.utils import weights_already_processed
    from vllm_hcu.patch.runtime_state import set_process_role
    from vllm_hcu.patch.worker import apply_worker_patches
    from vllm_hcu.runtime_compat.weight_cache import (
        install_per_client_export,
        install_expert_export_metadata,
        install_loader_compat,
        reconcile_router_state_devices,
    )

    if not current_platform.is_cuda_alike():
        raise RuntimeError("single-layer experiment requires HCU/CUDA")
    torch.cuda.set_device(local_device)
    device = torch.device("cuda", local_device)
    config = EngineArgs(model=args.model_dir, trust_remote_code=True,
        quantization="slimquant_w4a8", dtype="bfloat16", max_model_len=128,
        tensor_parallel_size=args.tp_size, enforce_eager=True).create_engine_config()
    set_process_role("Worker")
    apply_worker_patches(config)
    init_distributed_environment(world_size=args.tp_size, rank=args.rank, local_rank=local_device,
        distributed_init_method=get_distributed_init_method(args.master_addr, args.master_port or get_open_port()),
        backend=current_platform.dist_backend)
    with set_current_vllm_config(config):
        ensure_model_parallel_initialized(args.tp_size, 1)
        # Import only after the HCU worker callbacks have been armed.
        from vllm.models.kimi_k3.amd.linear import KimiDecoderLayer, KimiLinearModel
        # The full KimiLinearForCausalLM initializer normally installs this
        # mapping; a standalone decoder must install it explicitly.
        from vllm_hcu.model_executor.layers.kimi_k3_mixed_precision import is_mixed_slimquant
        if is_mixed_slimquant(config.quant_config):
            config.quant_config.packed_modules_mapping = {
                **config.quant_config.packed_modules_mapping,
                "fused_qkv_a_proj": ["q_a_proj", "kv_a_proj_with_mqa"],
                "gate_up_proj": ["gate_proj", "up_proj"],
            }

        def make_layer(device):
            with set_default_torch_dtype(config.model_config.dtype), torch.device(device):
                wrapper = KimiLinearModel.__new__(KimiLinearModel)
                torch.nn.Module.__init__(wrapper)
                wrapper.config = config.model_config.hf_text_config
                wrapper.layers = torch.nn.ModuleList([
                    KimiDecoderLayer(
                        wrapper.config,
                        config,
                        f"language_model.model.layers.{layer_index}",
                    )
                    for layer_index in range(
                        args.layer_start, args.layer_start + args.num_layers
                    )
                ])
                return wrapper

        key = WeightCacheKey(checkpoint=f"kimi-layers-{args.layer_start}-{layer_end}:" + digest,
            model_arch="kimi-partial-layer-lab",
            tp_size=args.tp_size, tp_rank=args.rank, dtype=str(config.model_config.dtype),
            quantization="slimquant_w4a8", quant_config_hash=f"layers-{args.layer_start}-{layer_end}", revision=None,
            vllm_version=vllm.version.__version__)
        if args.tp_size == 1:
            path = get_socket_path(0, args.socket_dir)
        else:
            gpu_uuid = str(torch.cuda.get_device_properties(local_device).uuid)
            path = get_socket_path("uuid-" + gpu_uuid, args.socket_dir)

        def sample(tensor):
            if tensor.numel() == 0:
                return (str(tensor.dtype), tuple(tensor.shape), b"")
            first = (0,) * tensor.ndim
            last = tuple(size - 1 for size in tensor.shape)
            middle = tuple(size // 2 for size in tensor.shape)
            selected = torch.stack((tensor[first], tensor[middle], tensor[last])).cpu().contiguous()
            return (str(tensor.dtype), tuple(tensor.shape), selected.view(torch.uint8).numpy().tobytes())

        def partial_request(model):
            width = config.model_config.hf_text_config.hidden_size
            x = torch.arange(width, device=device, dtype=torch.float32).remainder(13)
            x = ((x - 6) / 16).to(config.model_config.dtype).unsqueeze(0)
            with torch.inference_mode():
                # Kimi's fused MoE runner resolves the current layer through
                # vLLM's forward context.  The partial lab has no scheduler,
                # so install the smallest eager context explicitly instead of
                # treating a missing context as a preload failure.
                with set_forward_context(None, config, num_tokens=x.numel()):
                    for layer in model.layers:
                        x = layer.mlp(x)
                result = x.detach().cpu()
            if not torch.isfinite(result).all():
                raise RuntimeError("partial-layer request produced non-finite values")
            return result

        if args.action == "serve":
            started = time.monotonic()
            model = make_layer(device)
            for name, param in model.named_parameters():
                if "conv1d.weight" in name:
                    print(f"layer conv parameter {name}: {tuple(param.shape)}", flush=True)
            def weights():
                for name, shard in paths.items():
                    with safe_open(shard, framework="pt", device="cpu") as f:
                        yield name.removeprefix("language_model.model."), f.get_tensor(name)
            loaded = model.load_weights(weights())
            print(f"loaded {len(loaded)} parameters in {time.monotonic()-started:.3f}s", flush=True)
            process_weights_after_loading(model, config.model_config, device)
            print(f"processed layer in {time.monotonic()-started:.3f}s", flush=True)
            baseline = partial_request(model)
            print("disk-loaded partial-layer request baseline computed", flush=True)
            samples = {name: sample(value) for name, value in model.named_parameters(remove_duplicate=False)}
            samples.update({name: sample(value) for name, value in model.named_buffers(remove_duplicate=False)})
            install_per_client_export(daemon)
            install_expert_export_metadata(daemon)
            class LayerDaemon(daemon.WeightCacheDaemon):
                def __init__(self):
                    self.model, self.cache_config, self.tp_rank, self.socket_dir = model, key, args.rank, args.socket_dir
                    self.entries, self.aliases = {"ready": True}, {}
                def _socket_path(self):
                    return path
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
                    send_msg(conn, {"status": "ok", "entries": entries, "aliases": aliases,
                                    "gpu_uuid": self._gpu_uuid(), "samples": samples, "baseline": baseline})
            LayerDaemon().serve_forever()
        else:
            install_loader_compat(ipc_loader)
            loader = ipc_loader.IpcModelLoader(LoadConfig(load_format="ipc_cache", model_loader_extra_config={"socket_path": path, "fallback": False}))
            started = time.monotonic()
            with loader._connect(loader.state_timeout_s) as conn:
                send_msg(conn, {"cmd": "get_state", "cache_config": key})
                state = recv_msg(conn)
            if state.get("status") != "ok":
                raise RuntimeError(f"weight cache rejected the layer: {state}")
            loader._check_gpu_uuid(state.get("gpu_uuid"))
            entries, aliases = state["entries"], state.get("aliases", {})
            model = make_layer("meta")
            loader._apply_entries(model, entries, aliases, local_device)
            # IPC entries cover parameters and buffers, but a meta-created
            # router can retain a non-parameter bias on the construction
            # device. Reconcile all remaining module state before the first
            # MoE request; this is a no-op for tensors already on the target.
            model.to(device)
            with weights_already_processed():
                process_weights_after_loading(model, config.model_config, device)
            ipc_loader._materialize_remaining_meta_tensors(model, device)
            moved_router_state = reconcile_router_state_devices(model, device)
            if moved_router_state:
                print("reconciled router state devices: " + ", ".join(moved_router_state), flush=True)
            # Compare one untouched layernorm parameter to the checkpoint.
            for layer_index, layer in enumerate(model.layers, args.layer_start):
                name = f"language_model.model.layers.{layer_index}.input_layernorm.weight"
                with safe_open(paths[name], framework="pt", device="cpu") as f:
                    expected = f.get_tensor(name)
                if not torch.equal(layer.input_layernorm.weight.cpu(), expected):
                    raise RuntimeError(f"layer {layer_index} layernorm weight mismatch")
            samples = {name: sample(value) for name, value in model.named_parameters(remove_duplicate=False)}
            samples.update({name: sample(value) for name, value in model.named_buffers(remove_duplicate=False)})
            if samples != state["samples"]:
                raise RuntimeError("cached layer tensor metadata or samples differ from disk-loaded model")
            output = partial_request(model)
            if not torch.allclose(output, state["baseline"], rtol=1e-3, atol=1e-3):
                raise RuntimeError("cached partial-layer request differs from disk-loaded baseline")
            print(
                f"mapped {len(entries)} tensors, verified partial-layer request and "
                f"cache samples in {time.monotonic()-started:.3f}s",
                flush=True,
            )


if __name__ == "__main__":
    main()
