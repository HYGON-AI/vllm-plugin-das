# SPDX-License-Identifier: Apache-2.0
"""Compatibility fixes for the vLLM 0.28 weight cache daemon."""
from __future__ import annotations

from types import ModuleType
from contextlib import contextmanager
from contextvars import ContextVar

_cache_moe_backend = ContextVar("hcu_weight_cache_moe_backend", default=None)


@contextmanager
def _weight_cache_config_scope(vllm_config):
    kernel_config = getattr(vllm_config, "kernel_config", None)
    backend = getattr(kernel_config, "moe_backend", None)
    if backend is None:
        raise RuntimeError("weight cache requires kernel_config.moe_backend")
    token = _cache_moe_backend.set(str(backend))
    try:
        yield
    finally:
        _cache_moe_backend.reset(token)


def _install_cache_config_scope(module):
    import functools

    for class_name, method_name in (("WeightCacheDaemon", "__init__"),
                                    ("IpcModelLoader", "load_model")):
        cls = getattr(module, class_name, None)
        if cls is None:
            continue
        method = getattr(cls, method_name, None)
        if not callable(method):
            raise RuntimeError(f"Incompatible weight cache: missing {class_name}.{method_name}")
        if getattr(method, "_hcu_cache_config_scope", False):
            continue

        def wrap(original):
            @functools.wraps(original)
            def scoped(self, vllm_config, *args, **kwargs):
                with _weight_cache_config_scope(vllm_config):
                    return original(self, vllm_config, *args, **kwargs)

            scoped._hcu_cache_config_scope = True
            return scoped

        setattr(cls, method_name, wrap(method))


def reconcile_router_state_devices(model, target_device):
    """Move router state tensors that are not registered module state.

    Some Kimi/AMD router implementations replace the bias with a plain Tensor
    attribute (or reassign it after construction). ``Module.to`` only visits
    registered parameters and buffers, so an IPC client can otherwise invoke
    ``moe_fused_gate(router_logits, bias)`` with the activation on its local
    GPU and the bias left on CPU/meta.
    """
    import torch

    target = torch.device(target_device)
    names = {
        "e_score_correction_bias",
        "expert_bias",
        "router_bias",
    }
    # vLLM's FusedMoERouter is a plain Python object owned by MoERunner, not an
    # nn.Module. Its bias reference is therefore invisible to named_modules()
    # and can remain bound to the constructor's stale/default-device tensor
    # after the gate Parameter has been loaded onto this rank.
    modules = dict(model.named_modules())
    owners = []
    for module_name, module in modules.items():
        owners.append((module_name, module))
        router = getattr(module, "router", None)
        if router is not None and not isinstance(router, torch.nn.Module):
            owners.append((f"{module_name}.router" if module_name else "router", router))

    moved = []
    for module_name, module in owners:
        for name in names:
            if not hasattr(module, name):
                continue
            value = getattr(module, name)
            if not isinstance(value, torch.Tensor):
                continue

            # A FusedMoERouter is held by the sibling MoERunner (usually the
            # ``...experts`` module), but is not itself an nn.Module. IPC
            # loading replaces the registered gate Parameter, leaving this
            # plain router object with the original meta Parameter reference.
            # Rebind it to the loaded sibling gate tensor before device checks.
            if not isinstance(module, torch.nn.Module) and module_name.endswith(
                ".router"
            ):
                runner_name = module_name[: -len(".router")]
                parent_name = runner_name.rpartition(".")[0]
                gate = modules.get(f"{parent_name}.gate")
                gate_value = getattr(gate, name, None) if gate is not None else None
                if isinstance(gate_value, torch.Tensor):
                    value = gate_value
                    setattr(module, name, value)
                    moved.append(f"{module_name}.{name}")

            if value.device == target:
                continue
            if value.is_meta:
                raise RuntimeError(
                    f"router state {module_name}.{name} is still meta and has no "
                    "loaded sibling gate tensor"
                )
            replacement = value.to(target)
            # Preserve Parameter registration and object semantics where
            # possible; plain Tensor attributes are simply replaced.
            if isinstance(value, torch.nn.Parameter):
                if value.is_meta or target.type == "meta":
                    # Meta Parameters have no storage to rebind.
                    setattr(module, name, torch.nn.Parameter(
                        replacement, requires_grad=value.requires_grad
                    ))
                else:
                    # Keep aliases held by router objects and the owning gate
                    # attached to the same Parameter after local placement.
                    value.data = replacement
            else:
                setattr(module, name, replacement)
            moved.append(f"{module_name}.{name}" if module_name else name)
    return tuple(moved)


def _pipeline_context() -> tuple[int, int]:
    """Return ``(pipeline_size, pipeline_rank)`` without forcing PP init.

    The daemon launcher sets the environment values before constructing its
    cache object.  Engine workers resolve the same values from vLLM's model
    parallel group after distributed initialization.  Keeping this helper
    best-effort is important because the weight-cache modules are imported
    during plugin bootstrap, before model-parallel groups exist.
    """
    import os

    size = int(os.environ.get("VLLM_HCU_PRELOAD_PP_SIZE", "1"))
    rank = int(os.environ.get("VLLM_HCU_PRELOAD_PP_RANK", "0"))
    try:
        from vllm.distributed import get_pp_group

        group = get_pp_group()
        size = int(getattr(group, "world_size", size))
        rank = int(getattr(group, "rank_in_group", rank))
    except Exception:
        # The group is legitimately unavailable while the daemon constructs
        # its cache key and while the protocol modules are imported.
        pass
    if size < 1 or not 0 <= rank < size:
        raise RuntimeError(f"invalid pipeline cache context size={size} rank={rank}")
    return size, rank


def _pp_socket_identifier(identifier: str) -> str:
    """Add the PP stage to a socket identifier when PP is enabled."""
    size, rank = _pipeline_context()
    return f"{identifier}-pp{rank}" if size > 1 else identifier


def install_per_client_export(module: ModuleType) -> None:
    """Export fresh IPC reductions for each successful client request.

    Each reduction carries a PyTorch IPC reference. Sharing one reduction
    between independent clients can free it while another client still uses it.
    Keep the model itself alive and create handles only when serving a client.
    """
    cls = module.WeightCacheDaemon
    if cls.__dict__.get("_hcu_per_client_export", False):
        return
    for name in ("_export_entries", "_handle_get_state", "_handle_release"):
        if not callable(getattr(cls, name, None)):
            raise RuntimeError(f"Incompatible weight cache daemon: missing {name}")
    for name in ("WeightCacheKey", "export_entries", "send_msg"):
        if not callable(getattr(module, name, None)):
            raise RuntimeError(f"Incompatible weight cache daemon module: missing {name}")

    def mark_loaded(self):
        if self.model is None:
            raise RuntimeError("weight cache model was not loaded")
        # Preserve the original loader's tensor-count log without creating
        # IPC reductions until a matching client requests the state.
        names = {name for name, _ in self.model.named_parameters()}
        names.update(name for name, _ in self.model.named_buffers())
        self.entries = dict.fromkeys(names)
        self.aliases = {}

    def handle_get_state(self, conn, request):
        client_config = request.get("cache_config")
        if not isinstance(client_config, module.WeightCacheKey):
            module.send_msg(conn, {"status": "error", "message": "Missing cache_config"})
            return
        mismatched = self.cache_config.mismatched_fields(client_config)
        if mismatched:
            module.send_msg(conn, {"status": "mismatch", "fields": mismatched})
            return
        if self.model is None:
            module.send_msg(conn, {"status": "error", "message": "Weights were released"})
            return
        entries, aliases = module.export_entries(self.model)
        module.send_msg(conn, {
            "status": "ok", "entries": entries, "aliases": aliases,
            "gpu_uuid": self._gpu_uuid(),
        })

    cls._export_entries = mark_loaded
    cls._handle_get_state = handle_get_state
    cls._hcu_per_client_export = True


def install_expert_export_metadata(module: ModuleType) -> None:
    """Preserve the explicit unpacked MoE layout marker in exported entries."""
    original = module.export_entries
    if getattr(original, "_hcu_expert_metadata", False):
        return

    def export_entries(model):
        entries, aliases = original(model)
        metadata_names = (
            "_hcu_vllm_w4a8_unpacked",
            "_hcu_aiter_moe_solution_type",
            "_hcu_aiter_moe_weight_layout",
            "_hcu_aiter_moe_logical_shape",
            "_hcu_aiter_moe_scale_layout",
            "is_shuffled",
        )
        for name, param in model.named_parameters(remove_duplicate=False):
            if name not in entries:
                continue
            for attr in metadata_names:
                value = getattr(param, attr, None)
                if value is not None:
                    setattr(entries[name], attr, value)
        return entries, aliases

    export_entries._hcu_expert_metadata = True
    module.export_entries = export_entries


def install_ipc_quant_support(module: ModuleType) -> None:
    """Allow SlimQuant W4A8 through the vLLM IPC quantization check."""
    allowlist = getattr(module, "IPC_QUANT_ALLOWLIST", None)
    if not isinstance(allowlist, dict):
        raise RuntimeError("Incompatible weight cache protocol: missing IPC allowlist")
    existing = allowlist.get("slimquant_w4a8")
    if getattr(existing, "_hcu_slimquant_ipc", False):
        return

    def slimquant_w4a8_ipc_enabled(_quant_config):
        return True

    slimquant_w4a8_ipc_enabled._hcu_slimquant_ipc = True
    allowlist["slimquant_w4a8"] = slimquant_w4a8_ipc_enabled


def _uuid_socket_path(torch_module, get_socket_path, socket_dir):
    """Use the UUID of the current logical device, independent of visibility order."""
    import re

    index = torch_module.accelerator.current_device_index()
    props = torch_module.cuda.get_device_properties(index)
    value = getattr(props, "uuid", None)
    if value is None or not re.fullmatch(r"[A-Za-z0-9_-]+", str(value)):
        raise RuntimeError("weight cache requires a stable, filesystem-safe GPU UUID")
    return get_socket_path(_pp_socket_identifier("uuid-" + str(value)), socket_dir)


def install_uuid_daemon(module: ModuleType) -> None:
    cls = module.WeightCacheDaemon
    if cls.__dict__.get("_hcu_uuid_socket", False):
        return
    if not callable(getattr(cls, "_socket_path", None)):
        raise RuntimeError("Incompatible weight cache daemon socket interface")

    def socket_path(self):
        return _uuid_socket_path(module.torch, module.get_socket_path, self.socket_dir)

    cls._socket_path = socket_path
    cls._hcu_uuid_socket = True


def install_loader_compat(module: ModuleType) -> None:
    """Use UUID sockets and bounded startup waiting for strict cache loads."""
    import functools
    import time

    cls = module.IpcModelLoader
    if cls.__dict__.get("_hcu_weight_cache_loader", False):
        return
    original_fetch = getattr(cls, "_fetch_entries", None)
    original_connect = getattr(cls, "_connect", None)
    original_apply = getattr(cls, "_apply_entries", None)
    original_check = getattr(cls, "_check_supported", None)
    original_materialize = getattr(module, "_materialize_remaining_meta_tensors", None)
    if (not callable(original_fetch)
            or not callable(original_connect)
            or not callable(original_apply)
            or not callable(original_check)
            or not callable(getattr(cls, "_resolve_socket_path", None))
            or not callable(original_materialize)):
        raise RuntimeError("Incompatible weight cache loader interface")

    def materialize_remaining(model, target_device):
        # The daemon exports parameters and buffers, including non-persistent
        # buffers. A remaining meta tensor can otherwise become uninitialized
        # storage and silently corrupt model outputs.
        missing = [name for name, tensor in model.named_parameters(remove_duplicate=False)
                   if tensor.is_meta]
        missing += [name for name, tensor in model.named_buffers(remove_duplicate=False)
                    if tensor.is_meta]
        if missing:
            # A generic failure routes copy-mode loads through the original
            # loader's release path before any disk fallback.
            raise RuntimeError(
                "Weight cache is missing model tensors: " + ", ".join(missing[:8])
            )
        return original_materialize(model, target_device)

    def apply_entries(self, model, entries, aliases, device_index):
        result = original_apply(self, model, entries, aliases, device_index)
        params = dict(model.named_parameters(remove_duplicate=False))
        for name, entry in entries.items():
            for attr in (
                "_hcu_vllm_w4a8_unpacked",
                "_hcu_aiter_moe_solution_type",
                "_hcu_aiter_moe_weight_layout",
                "_hcu_aiter_moe_logical_shape",
                "_hcu_aiter_moe_scale_layout",
                "is_shuffled",
            ):
                value = getattr(entry, attr, None)
                if value is not None:
                    param = params.get(name)
                    if param is None:
                        raise RuntimeError(f"Expert parameter {name!r} is missing")
                    setattr(param, attr, value)
            if getattr(entry, "_hcu_vllm_w4a8_unpacked", False) is True:
                param = params.get(name)
                if param is None:
                    raise RuntimeError(f"Unpacked expert parameter {name!r} is missing")
                param._hcu_vllm_w4a8_unpacked = True
        return result

    def check_supported(vllm_config, model_config):
        from vllm.platforms import current_platform
        if not current_platform.is_cuda_alike():
            raise RuntimeError("weight-cache IPC requires a CUDA/ROCm-compatible platform")
        return original_check(vllm_config, model_config)

    def resolve_socket_path(self):
        if self.socket_path is not None:
            return self.socket_path
        try:
            return _uuid_socket_path(module.torch, module.get_socket_path, self.socket_dir)
        except (RuntimeError, AttributeError) as exc:
            raise module.WeightCacheUnavailableError(
                f"Cannot identify weight cache GPU UUID: {exc}"
            ) from exc

    @functools.wraps(original_connect)
    def connect(self, timeout):
        # _request_state passes state_timeout_s to _connect. In strict mode,
        # cap that socket timeout to the remaining outer deadline so one
        # unavailable daemon cannot overrun the advertised startup timeout.
        deadline = getattr(self, "_hcu_fetch_deadline", None)
        if deadline is not None:
            timeout = min(timeout, max(0.0, deadline - time.monotonic()))
        return original_connect(self, timeout)

    @functools.wraps(original_fetch)
    def fetch_entries(self, model_config):
        if self.fallback:
            return original_fetch(self, model_config)
        deadline = time.monotonic() + max(0.0, self.state_timeout_s)
        self._hcu_fetch_deadline = deadline
        try:
            while True:
                try:
                    return original_fetch(self, model_config)
                except module.WeightCacheUnavailableError as exc:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise module.WeightCacheUnavailableError(
                            "Weight cache daemon did not become ready within "
                            f"{self.state_timeout_s:.1f}s"
                        ) from exc
                    time.sleep(min(0.5, remaining))
        finally:
            delattr(self, "_hcu_fetch_deadline")

    module._materialize_remaining_meta_tensors = materialize_remaining
    cls._apply_entries = apply_entries
    cls._check_supported = staticmethod(check_supported)
    cls._connect = connect
    cls._resolve_socket_path = resolve_socket_path
    cls._fetch_entries = fetch_entries
    cls._hcu_weight_cache_loader = True


def install_key_fingerprint(module: ModuleType) -> None:
    """Invalidate local checkpoint keys when shard metadata or plugin version changes.

    No checkpoint tensor is read here. Stat timestamps are a conservative
    change detector, not a cryptographic hash of checkpoint contents.
    """
    import dataclasses
    import glob
    import hashlib
    import json
    import os

    _install_cache_config_scope(module)
    cls = module.WeightCacheKey
    if cls.__dict__.get("_hcu_fingerprint", False):
        return
    original = cls.from_model_config
    if not callable(original):
        raise RuntimeError("Incompatible weight cache key interface")

    @classmethod
    def from_model_config(key_cls, model_config, tp_size, tp_rank):
        from vllm_hcu.version import __hcu_version__

        moe_backend = _cache_moe_backend.get()
        if moe_backend is None:
            raise RuntimeError("weight cache key requires vLLM kernel configuration")
        key = original(model_config, tp_size=tp_size, tp_rank=tp_rank)
        pp_size, pp_rank = _pipeline_context()
        model_path = model_config.model
        local_shards = sorted(glob.glob(os.path.join(model_path, "*.safetensors"))) if os.path.isdir(model_path) else []
        hasher = hashlib.sha256()
        hasher.update(key.checkpoint.encode())
        for path in local_shards:
            info = os.stat(path)
            record = [os.path.basename(path), info.st_size, info.st_mtime_ns, info.st_ctime_ns]
            hasher.update(json.dumps(record, separators=(",", ":")).encode())

        # Header hashes alone do not encode model overrides or nested Kimi
        # compression settings. Include the resolved HF config and the model
        # options that affect weight interpretation, before post-load can
        # mutate the quantization config. Runtime capacity settings such as
        # max_model_len must not participate: the preload daemon and serving
        # engine can use different KV-cache budgets for the same weights.
        hf_config = getattr(model_config, "hf_config", None)
        hf_dict = hf_config.to_dict() if hf_config is not None else None
        options = {
            name: getattr(model_config, name, None)
            for name in (
                "code_revision", "model_weights", "model_impl", "convert",
                "model_class_overrides", "quantization_config",
            )
        }
        # HCU switches alter post-processed layouts and must match in both
        # daemon and engine. Record effective values rather than raw spelling.
        from vllm_hcu.platforms import envs as henvs
        from vllm_hcu.platforms.envs import (
            custom_ops_enabled,
            custom_quantization_gemm_enabled,
            resolve_aiter_moe_shuffle,
        )
        options["hcu_runtime"] = {
            "custom_ops": custom_ops_enabled(),
            "custom_quantization_gemm": custom_quantization_gemm_enabled(),
            "aiter_moe_shuffle": resolve_aiter_moe_shuffle(),
            "moe_backend": moe_backend,
            "use_nn": bool(henvs.VLLM_USE_NN),
        }
        try:
            config_bytes = json.dumps(
                {"hf_config": hf_dict, "model_options": options},
                sort_keys=True, separators=(",", ":"), default=str,
            ).encode()
        except (TypeError, ValueError) as exc:
            raise RuntimeError("Cannot fingerprint model configuration") from exc
        hasher.update(config_bytes)
        scope = f"|hcu-pp:{pp_size}:{pp_rank}" if pp_size > 1 else ""
        key = dataclasses.replace(
            key, checkpoint="hcu-config-v3:" + hasher.hexdigest() + scope
        )
        return dataclasses.replace(key, vllm_version=f"{key.vllm_version}|hcu:{__hcu_version__}")

    cls.from_model_config = from_model_config
    cls._hcu_fingerprint = True
