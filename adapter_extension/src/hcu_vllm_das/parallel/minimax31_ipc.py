# SPDX-License-Identifier: Apache-2.0
"""Explicit PCP8/EP8 strict IPC transport on top of guarded vLLM primitives.

The PCP command cannot be served by an ordinary native daemon: both the
supplemental rank identity and the unchanged native WeightCacheKey must
match before the native exporter is invoked.
"""
from dataclasses import asdict
import gc
import hashlib
import inspect
import json

from vllm.config import set_current_vllm_config
from vllm.distributed.parallel_state import (
    ensure_model_parallel_initialized, get_ep_group, get_pcp_group,
    get_world_group, init_distributed_environment,
)
from vllm.model_executor.model_loader import register_model_loader
from vllm.model_executor.model_loader.weight_cache.daemon import (
    WeightCacheDaemon, exported_weight_bytes,
)
from vllm.model_executor.model_loader.weight_cache.ipc_loader import IpcModelLoader
from vllm.model_executor.model_loader.weight_cache.protocol import (
    CacheConfigMismatchError, WeightCacheState, WeightCacheUnavailableError,
    get_current_device_uuid, hash_rope_parameters, recv_msg, send_msg,
)
from vllm.platforms import current_platform
import torch

from .minimax31_preload import (
    mismatched_partition_fields, partition_identity, validate_parallel_config, require_extensions,
)

LOAD_FORMAT = "minimax31_pcp_ep_ipc"
COMMAND = "get_minimax31_pcp_ep_state"
_GUARDS = {
    (WeightCacheDaemon, "__init__"): "c6a69a065d1bb6d3e74db00d36563c2d042efbb3ad3c2405427b221aad834b82",
    (WeightCacheDaemon, "load_model"): "66e4182a03ff17f6494dec58f8ee506bcdaa6fa868e2050696d63ed07927ce28",
    (WeightCacheDaemon, "_handle_get_state"): "fdc32a17791d489efe311fe1fcb5c04fbd2ad4efd4a199aa924050d2f5247549",
    (WeightCacheDaemon, "serve_forever"): "34d66b795d80e657c124adfa9698dbb4954434d7b20a82871d8e401bc6c5dfaf",
    (IpcModelLoader, "__init__"): "83c52a09f03d7c56a223972e363a6e20f0021dc23ead46eb3f5b70563891893e",
    (IpcModelLoader, "load_model"): "6aef627d4f915f37dec8aae68bf1f538886272c3b1884158a7e7052f9a697e25",
    (IpcModelLoader, "_fetch_entries"): "29624d9f665827f331e393029450cc0cc0141aac2d3bc9cb24b13bde0e4d2852",
    (IpcModelLoader, "_request_state"): "dd3a9a79d5a13ee1a52e0601051e1c5ae194a42dc83ed6ac39c91290ce3ec52b",
    (IpcModelLoader, "_build_model"): "fbe680e4271ceab2a91835f0e94afa66d71b8120fb57bdcaf4e547e8e80e592d",
}


def require_core_contract():
    for (cls, name), expected in _GUARDS.items():
        try:
            actual = hashlib.sha256(inspect.getsource(getattr(cls, name)).encode()).hexdigest()
        except (OSError, TypeError) as exc:
            raise RuntimeError("cannot inspect guarded core IPC seam") from exc
        if actual != expected:
            raise RuntimeError("unsupported core IPC seam: %s.%s" % (cls.__name__, name))


def model_weight_contract_sha256(model_config):
    quant = getattr(model_config.hf_config, "quantization_config", None)
    if not isinstance(quant, dict):
        raise ValueError("explicit MiniMax quantization identity required")
    identity = quant.get("hcu_weight_cache_identity")
    if not isinstance(identity, dict):
        raise ValueError("PCP/EP requires an audited weight identity")
    if identity.get("schema") == "minimax31-wheel-weight-contract-v3":
        # Task-provided verifier checks the live installed wheel, source,
        # producer config and sidecar before a PCP/EP request is made.
        from wheel_runtime_guard import verify_wheel_producer_identity
        verify_wheel_producer_identity(model_config, identity)
    elif identity.get("schema") != "minimax31-weight-contract-v2":
        raise ValueError("PCP/EP requires an audited v2 or wheel-v3 identity")
    data = json.dumps(identity, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def live_partition(config, model_config):
    validate_parallel_config(config)
    world, pcp, ep = get_world_group(), get_pcp_group(), get_ep_group()
    if (world.world_size != 8 or pcp.world_size != 8 or ep.world_size != 8
            or pcp.rank_in_group != ep.rank_in_group
            or world.rank != pcp.rank_in_group
            or list(pcp.ranks) != list(range(8))
            or list(ep.ranks) != list(range(8))):
        raise ValueError("actual world/PCP/EP group members or rank differ")
    from .minimax31_static_contract import from_model_config
    return partition_identity(pcp.rank_in_group, get_current_device_uuid(),
                              model_weight_contract_sha256(model_config), from_model_config(model_config))


class MiniMax31PcpEpIpcLoader(IpcModelLoader):
    """Strict no-fallback loader using both PCP partition and native keys."""
    def __init__(self, load_config):
        require_core_contract()
        super().__init__(load_config)
        if self.fallback or self.mode != "zero_copy" or self.is_draft or self.socket_path:
            raise ValueError("PCP/EP IPC requires target zero_copy, fallback=false and per-device sockets")
        self._partition = None

    def load_model(self, vllm_config, model_config, prefix=""):
        from .minimax31_pcp import validate_scope
        require_extensions()
        validate_scope(vllm_config)
        model = super().load_model(vllm_config, model_config, prefix=prefix)
        if hasattr(self, "_static_manifest"):
            from .minimax31_static_ipc import verify_mapped_model
            verify_mapped_model(model, self._static_manifest)
        return model

    def _fetch_entries(self, model_config, cache_config, parallel_config,
                       kernel_config, *, is_reload=False):
        if is_reload:
            raise ValueError("PCP/EP in-place reload is not supported")
        self._partition = live_partition(parallel_config, model_config)
        return super()._fetch_entries(model_config, cache_config, parallel_config,
                                      kernel_config, is_reload=False)

    def _request_state(self, cache_key, *, is_reload=False):
        if is_reload or self._partition is None:
            raise ValueError("PCP/EP identity must be established before IPC")
        with self._connect(self.state_timeout_s) as conn:
            request={"cmd": COMMAND, "cache_key": cache_key,
                     "is_reload": False, "partition": asdict(self._partition)}
            static=hasattr(self._partition,"map_sha256")
            if static:
                from .minimax31_static_ipc import COMMAND as STATIC_COMMAND, expected_publication
                expected=expected_publication(self._partition)
                request.update(cmd=STATIC_COMMAND,publication=expected)
            send_msg(conn, request)
            response = recv_msg(conn)
        status = response.get("status")
        if status == "mismatch":
            raise CacheConfigMismatchError("strict PCP/EP or native key mismatch: %s" % response.get("fields"))
        if status != "ok":
            raise WeightCacheUnavailableError("PCP/EP daemon refused: %s" % response.get("message"))
        if not response.get("gpu_uuid"):
            raise CacheConfigMismatchError("PCP/EP daemon omitted its GPU UUID")
        self._check_gpu_uuid(response["gpu_uuid"])
        if static:
            from .minimax31_static_ipc import verify_response
            self._static_manifest=verify_response(response,expected,self._partition)
        return WeightCacheState(entries=response["entries"], aliases=response.get("aliases", {}),
                                attrs=response.get("attrs", {}))

    def load_weights(self, model, model_config):
        raise ValueError("PCP/EP in-place weight reload is unsupported")


class MiniMax31PcpEpDaemon(WeightCacheDaemon):
    """Single-rank producer; native export follows strict PCP partition check."""
    def __init__(self, vllm_config, global_rank, local_rank, distributed_init_method, socket_dir):
        require_core_contract()
        validate_parallel_config(vllm_config.parallel_config)
        if type(global_rank) is not int or not 0 <= global_rank < 8 or local_rank != global_rank:
            raise ValueError("single-node PCP global/local rank must match in [0, 8)")
        super().__init__(vllm_config, 0, local_rank, distributed_init_method, socket_dir)
        self.global_rank = global_rank
        self.world_size = 8
        self.vllm_config.parallel_config.rank = global_rank
        self._partition = None

    def load_model(self):
        torch.accelerator.set_device_index(self.local_rank)
        init_distributed_environment(world_size=8, rank=self.global_rank,
             distributed_init_method=self.distributed_init_method, local_rank=self.local_rank,
             backend=current_platform.dist_backend)
        with set_current_vllm_config(self.vllm_config):
            ensure_model_parallel_initialized(1, 1, 8, 1)
            self._partition = live_partition(self.vllm_config.parallel_config, self.model_config)
            if self._partition.pcp_rank != self.global_rank:
                raise ValueError("producer global rank disagrees with PCP group")
            self.model = self.get_model()
        self.loaded_rope_parameters_hash = hash_rope_parameters(self.model_config)
        gc.collect()
        torch.accelerator.empty_cache()

    def _handle_connection(self, conn):
        request = recv_msg(conn)
        cmd = request.get("cmd") if isinstance(request, dict) else None
        from .minimax31_static_ipc import COMMAND as STATIC_COMMAND
        if cmd in (COMMAND, STATIC_COMMAND):
            if self._partition is None or request.get("is_reload") is not False:
                send_msg(conn, {"status": "error", "message": "PCP/EP partition not ready or reload unsupported"})
                return
            try:
                mismatch = mismatched_partition_fields(self._partition, request.get("partition"))
            except ValueError as exc:
                send_msg(conn, {"status": "error", "message": str(exc)})
                return
            if mismatch:
                send_msg(conn, {"status": "mismatch", "fields": list(mismatch)})
                return
            static=hasattr(self._partition,"map_sha256")
            if static != (cmd == STATIC_COMMAND):
                send_msg(conn,{"status":"mismatch","fields":["static_protocol"]})
                return
            if static:
                from .minimax31_static_ipc import serve_static
                serve_static(self,conn,request)
                return
            # The unchanged native method checks every native WeightCacheKey
            # field before it constructs or sends any IPC entries.
            self._handle_get_state(conn, request)
        elif cmd == "get_memory":
            send_msg(conn, {"status": "ok", "memory_bytes": exported_weight_bytes(self.model)})
        elif cmd == "watch":
            self._handle_watch(conn)
        else:
            send_msg(conn, {"status": "error", "message": "unsupported strict PCP/EP command"})


_installed = False

def install_pcp_ep_ipc_loader():
    """Called by the explicit required minimax31_pcp_ep extension."""
    global _installed
    require_core_contract()
    if _installed:
        check_pcp_ep_ipc_health()
        return False
    from vllm.model_executor import model_loader
    if LOAD_FORMAT in model_loader._LOAD_FORMAT_TO_MODEL_LOADER:
        raise RuntimeError("PCP/EP loader name already owned")
    register_model_loader(LOAD_FORMAT)(MiniMax31PcpEpIpcLoader)
    _installed = True
    return True


def check_pcp_ep_ipc_health():
    """Fail closed if the registered loader is no longer ours."""
    from vllm.model_executor import model_loader
    require_core_contract()
    if not _installed or model_loader._LOAD_FORMAT_TO_MODEL_LOADER.get(LOAD_FORMAT) is not MiniMax31PcpEpIpcLoader:
        raise RuntimeError("PCP/EP IPC loader ownership changed")
