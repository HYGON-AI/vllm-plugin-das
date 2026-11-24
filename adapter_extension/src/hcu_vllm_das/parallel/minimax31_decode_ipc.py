# SPDX-License-Identifier: Apache-2.0
"""Strict DP8 target/draft publications over native vLLM weight-cache IPC."""
from __future__ import annotations
from dataclasses import asdict
import hashlib
import inspect
import importlib.util
import json
from pathlib import Path

from vllm.model_executor.model_loader.weight_cache.daemon import WeightCacheDaemon
from vllm.model_executor.model_loader.weight_cache.ipc_loader import IpcModelLoader
from vllm.model_executor.model_loader.weight_cache.protocol import (
    CacheConfigMismatchError, WeightCacheUnavailableError, WeightCacheState,
    get_current_device_uuid, recv_msg, send_msg)
from .minimax31_decode_contract import digest, partition, require_partition, validate_parallel
from .minimax31_decode_placement import placement_for_model

LOAD_FORMAT = "minimax31_decode_ipc"
COMMAND = "get_minimax31_decode_state_v1"


def verify_imported_weight_source(filename,expected):
    """Bind the declaration to actual Python resolution, including wheels."""
    source = '/' + filename.lstrip('/')
    legacy_slimquant = '/vllm_hcu/model_executor/layers/quantization/slimquant_w4a8.py'
    if source.endswith(legacy_slimquant):
        relative = 'hcu_vllm_das/quantization/slimquant_w4a8.py'
    else:
        for namespace in ('hcu_vllm_das', 'triton_tmp_dir', 'vllm_hcu'):
            marker = '/' + namespace + '/'
            if marker in source:
                relative = namespace + '/' + source.split(marker, 1)[1]
                break
        else:
            raise ValueError('unknown weight source import contract: ' + filename)
    if not relative.endswith('.py'):raise ValueError('weight source must be Python')
    module=relative[:-3].replace('/','.')
    spec=importlib.util.find_spec(module)
    if (spec is None or not spec.origin or
            hashlib.sha256(Path(spec.origin).read_bytes()).hexdigest()!=expected):
        raise ValueError('actual imported weight source differs: '+module)


def weight_identity(model_config):
    config = model_config.hf_config
    declaration = getattr(config, "hcu_minimax_decode_weight_identity", None)
    if not isinstance(declaration, dict) or declaration.get("schema") != "minimax31-decode-weight-v1":
        raise ValueError("explicit decode weight identity is required before IPC")
    placement = placement_for_model(model_config)
    if declaration.get("placement_sha256") != (placement.map_sha256 if placement else None):
        raise ValueError("decode weight identity does not bind its physical expert placement")
    # Bind all native IPC seams to the actual supported implementation. A
    # consumer may not silently reinterpret entries after a wheel refresh.
    methods = {"daemon_init": WeightCacheDaemon.__init__,
               "daemon_load": WeightCacheDaemon.load_model,
               "daemon_export": WeightCacheDaemon._handle_get_state,
               "ipc_build": IpcModelLoader._build_model,
               "ipc_fetch": IpcModelLoader._fetch_entries}
    actual = {name: hashlib.sha256(inspect.getsource(fn).encode()).hexdigest()
              for name, fn in methods.items()}
    if declaration.get("native_ipc") != actual:
        raise ValueError("decode native IPC ABI/source changed")
    sources=declaration.get("weight_sources")
    if not isinstance(sources,dict) or not sources:
        raise ValueError("decode weight-processing source manifest is required")
    for filename,expected in sources.items():
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest()!=expected:
            raise ValueError("weight-processing source changed: "+filename)
        verify_imported_weight_source(filename,expected)
    import torch
    if (declaration.get("torch")!=torch.__version__ or
            declaration.get("cxx11_abi")!=bool(torch._C._GLIBCXX_USE_CXX11_ABI)):
        raise ValueError("decode Torch storage ABI changed")
    return digest(declaration)


def live_partition(vllm_config, model_config, role):
    from vllm.distributed.parallel_state import get_dp_group, get_ep_group
    pc = vllm_config.parallel_config
    validate_parallel(pc)
    dp = get_dp_group()
    if dp.world_size != 8 or dp.rank_in_group != pc.data_parallel_rank:
        raise ValueError("actual DP group/rank differs from decode configuration")
    if role == "target":
        ep = get_ep_group()
        if not pc.enable_expert_parallel or ep.world_size != 8 or ep.rank_in_group != dp.rank_in_group:
            raise ValueError("actual EP group/rank differs from decode DP ownership")
    return partition(role, dp.rank_in_group, get_current_device_uuid(),
                     weight_identity(model_config))


class DecodeDaemon(WeightCacheDaemon):
    def load_model(self):
        super().load_model()
        self.decode_partition = live_partition(self.vllm_config, self.model_config,
                                               "draft" if self.is_draft else "target")
        placement = placement_for_model(self.model_config)
        if placement is not None and self.vllm_config.load_config.load_format == "safetensors":
            from .minimax31_decode import verify_permuted_model
            verify_permuted_model(self.model, placement, self.decode_partition.dp_rank, checkpoint=True)

    def _handle_connection(self, conn):
        request = recv_msg(conn)
        cmd = request.get("cmd") if isinstance(request, dict) else None
        if cmd == COMMAND:
            try:
                require_partition(self.decode_partition, request.get("partition"))
                if request.get("generation") != self.decode_generation or request.get("is_reload") is not False:
                    raise ValueError("wrong decode generation or in-place reload request")
            except ValueError as exc:
                send_msg(conn, {"status": "mismatch", "fields": str(exc)})
                return
            # The unchanged native handler validates every WeightCacheKey field
            # BEFORE creating or sending IPC handles.
            self._handle_get_state(conn, request)
        elif cmd == "get_memory":
            from vllm.model_executor.model_loader.weight_cache.daemon import exported_weight_bytes
            send_msg(conn, {"status": "ok", "memory_bytes": exported_weight_bytes(self.model)})
        elif cmd == "watch":
            self._handle_watch(conn)
        else:
            send_msg(conn, {"status": "error", "message": "unsupported decode IPC command"})


class DecodeIpcLoader(IpcModelLoader):
    def __init__(self, load_config):
        from vllm.config import replace
        extra = dict(load_config.model_loader_extra_config or {})
        self.manifest_path = extra.pop("decode_ready_manifest", None)
        self.dynamic_eplb = extra.pop("decode_dynamic_eplb", False)
        if type(self.dynamic_eplb) is not bool:
            raise ValueError("decode_dynamic_eplb must be a boolean")
        eplb = extra.pop("decode_eplb", {})
        if not isinstance(eplb, dict) or set(eplb) - {"window", "interval"}:
            raise ValueError("decode_eplb accepts only window and interval")
        if eplb and not self.dynamic_eplb:
            raise ValueError("decode_eplb options require decode_dynamic_eplb")
        self.eplb_window = eplb.get("window", 16)
        self.eplb_interval = eplb.get("interval", 96)
        if (type(self.eplb_window) is not int or type(self.eplb_interval) is not int
                or not 1 <= self.eplb_window <= self.eplb_interval):
            raise ValueError("decode_eplb requires integers with 1 <= window <= interval")
        super().__init__(replace(load_config, model_loader_extra_config=extra))
        if self.fallback or self.mode != "zero_copy" or self.socket_path or not self.manifest_path:
            raise ValueError("decode IPC requires zero_copy, fallback=false, role sockets and ready manifest")

    def load_model(self, vllm_config, model_config, prefix=""):
        self.decode_partition = live_partition(vllm_config, model_config,
                                               "draft" if self.is_draft else "target")
        ready = json.loads(Path(self.manifest_path).read_text())
        from .minimax31_decode_contract import ready_roles
        ready_roles(ready["ranks"], with_draft=ready["with_draft"])
        row = next((x for x in ready["ranks"] if x["role"] == self.decode_partition.role
                    and x["dp_rank"] == self.decode_partition.dp_rank), None)
        if row is None:
            raise CacheConfigMismatchError("draft/target publication is missing")
        require_partition(self.decode_partition, row["partition"])
        self.decode_generation = row.get("generation",ready["generation"])
        model = super().load_model(vllm_config, model_config, prefix=prefix)
        placement = placement_for_model(model_config)
        if placement is not None:
            from .minimax31_decode import verify_permuted_model
            verify_permuted_model(model, placement, self.decode_partition.dp_rank, checkpoint=False)
        if self.dynamic_eplb:
            if placement is None or self.is_draft:
                raise ValueError("dynamic decode EPLB requires a permuted target")
            from .minimax31_decode_eplb import attach_decode_eplb
            attach_decode_eplb(model, window=self.eplb_window, interval=self.eplb_interval)
        return model

    def _request_state(self, cache_key, *, is_reload=False):
        if is_reload:
            raise ValueError("decode in-place weight reload is unsupported; restart consumer")
        with self._connect(self.state_timeout_s) as conn:
            send_msg(conn, {"cmd": COMMAND, "cache_key": cache_key,
                           "partition": asdict(self.decode_partition),
                           "generation": self.decode_generation, "is_reload": False})
            response = recv_msg(conn)
        if response.get("status") == "mismatch":
            raise CacheConfigMismatchError(str(response.get("fields")))
        if response.get("status") != "ok" or not response.get("gpu_uuid"):
            raise WeightCacheUnavailableError("decode daemon refused strict publication")
        self._check_gpu_uuid(response["gpu_uuid"])
        print('MINIMAX_DECODE_IPC_HIT '+json.dumps({
            'role':self.decode_partition.role,'dp_rank':self.decode_partition.dp_rank,
            'generation':self.decode_generation,'weight_identity':self.decode_partition.weight_identity,
            'entries':len(response['entries']),'attrs':response.get('attrs',{}),
            'mode':'zero_copy','fallback':False}),flush=True)
        return WeightCacheState(entries=response["entries"], aliases=response.get("aliases", {}),
                                attrs=response.get("attrs", {}))

    def load_weights(self, model, model_config):
        raise ValueError("decode in-place reload is unsupported")

    def get_external_weight_memory(self, vllm_config):
        total = self._daemon_memory()
        spec = vllm_config.speculative_config
        if not self.is_draft and spec is not None and spec.use_dspark():
            draft = spec.draft_load_config
            if draft is None or draft.load_format != LOAD_FORMAT:
                raise ValueError("DSpark decode requires explicit strict draft IPC load config")
            other = DecodeIpcLoader(draft)
            if not other.is_draft:
                raise ValueError("DSpark draft IPC role was lost")
            total += other._daemon_memory()
        return total


def register_loader():
    from vllm.model_executor import model_loader
    existing = model_loader._LOAD_FORMAT_TO_MODEL_LOADER.get(LOAD_FORMAT)
    if existing is None:
        model_loader.register_model_loader(LOAD_FORMAT)(DecodeIpcLoader)
    elif existing is not DecodeIpcLoader:
        raise RuntimeError("foreign decode IPC loader owner")
