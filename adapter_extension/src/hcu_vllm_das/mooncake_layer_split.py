# SPDX-License-Identifier: Apache-2.0
"""Opt-in V41 PCP Mooncake connector; Core remains installed and unmodified.

TP and PP retain their real meaning. A versioned bootstrap advertises PCP
owners separately. Every PCP worker acknowledges every pull (including workers
with no payload), so Core's worker completion aggregation can release P blocks.
Only persistent owner pages are sent; scratch, row padding and wire buffers
never appear in a transfer descriptor.
"""
from __future__ import annotations

import asyncio
import copy
import secrets
import threading
import time
from dataclasses import replace
from types import SimpleNamespace

import httpx
from fastapi import HTTPException
from pydantic import BaseModel

from vllm import envs
from vllm.distributed import get_pcp_group
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake import mooncake_connector as core
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_utils import (
    EngineEntry, MooncakeBootstrapServer,
)
from vllm.model_executor.models.utils import extract_layer_index
from vllm.utils.network_utils import make_zmq_path
from vllm.v1.kv_cache_interface import CircularBufferSpec

from .layer_split_config import layer_split_enabled
from .dsa_cache_layout import _owner, _specs

PROTOCOL = 'hcu-v41-pcp-owners-v3'
CONNECTOR = 'HcuLayerSplitMooncakeConnector'
MODULE = 'hcu_vllm_das.mooncake_layer_split'
OWNER_QUERY_PATH = '/hcu/pcp-owners/v3/query'


def validate_config(config):
    p = config.parallel_config
    transfer = config.kv_transfer_config
    if transfer is None or transfer.kv_role not in ('kv_producer', 'kv_consumer'):
        raise ValueError('PCP Mooncake requires an explicit producer or consumer role')
    if p.tensor_parallel_size != 1 or p.pipeline_parallel_size != 1 or p.decode_context_parallel_size != 1:
        raise ValueError('PCP Mooncake currently requires TP/PP/DCP=1')
    if not config.model_config.use_mla:
        raise ValueError('PCP Mooncake requires V41 MLA')
    ids = tuple(getattr(config.model_config.hf_config, 'kv_source_layer_ids', ()) or ())
    if not ids or ids != tuple(sorted(set(ids))):
        raise ValueError('PCP Mooncake requires ordered V41 KV source layers')
    if config.speculative_config is not None:
        raise ValueError('PCP Mooncake PD does not yet transfer DSpark draft context; disable speculation on both P and D')
    if transfer.kv_role == 'kv_producer':
        if p.data_parallel_size != 1 or config.speculative_config is not None:
            raise ValueError('PCP Mooncake producer requires DP=1 and no drafter')
    elif p.prefill_context_parallel_size != 1 or layer_split_enabled(config):
        raise ValueError('PCP Mooncake consumer requires PCP=1 and layer split off')
    return ids


def transfer_owner(name, ids, width, split):
    owner = _owner(name, ids, width) if split else None
    # SWA follows the same cyclic owner as the planner and attention. Only
    # replicated compressor state (and split-off caches) defaults to rank 0.
    return 0 if owner is None else owner


def describe_caches(kv_config, caches):
    """Keep a descriptor for each named page, even when storages alias.

    The span is the actual dense row, not spec.page_size_bytes (padding),
    block_stride (other layers), or storage.nbytes (all groups plus scratch).
    """
    regions, manifest = [], {}
    tensors = {n: t for t in kv_config.kv_cache_tensors for n in t.layers}
    for group_id, group in enumerate(kv_config.transfer_groups):
        for name, spec in _specs(group):
            cache = caches[name]
            count = kv_config.num_blocks_of(tensors[name])
            if cache.ndim < 2 or cache.shape[0] != count or not cache[0].is_contiguous():
                raise ValueError(f'PCP Mooncake requires dense per-block views: {name}')
            size = cache[0].nbytes
            stride = cache.stride(0) * cache.element_size()
            offset = cache.data_ptr() - cache.untyped_storage().data_ptr()
            if not 0 < size <= stride or offset + (cache.shape[0] - 1) * stride + size > cache.untyped_storage().nbytes():
                raise ValueError(f'PCP Mooncake invalid page bounds: {name}')
            regions.append(core.TransferRegion(name, extract_layer_index(name),
                cache.data_ptr(), stride, size, group_id, (), -1))
            manifest[name] = dict(group=group_id, kind=type(spec).__name__,
                block_size=spec.block_size, tokens_per_state=getattr(spec, 'tokens_per_state', 1),
                shape=list(cache.shape[1:]), dtype=str(cache.dtype), bytes=size,
                sliding_window=getattr(spec, 'sliding_window', None),
                bounded_replay=getattr(spec, 'bounded_replay', False))
    if not regions:
        raise ValueError('PCP Mooncake has no transferable cache pages')
    return regions, manifest


def validate_manifest(manifest):
    groups = {}
    for entry in manifest.values():
        group = entry.get('group')
        if type(group) is not int or group < 0:
            raise ValueError('Invalid KV manifest group index')
        semantics = tuple(entry.get(k) for k in
            ('kind', 'block_size', 'sliding_window', 'bounded_replay'))
        if group in groups and groups[group] != semantics:
            raise ValueError('Inconsistent semantics within KV manifest group')
        groups[group] = semantics
    if not groups or set(groups) != set(range(len(groups))):
        raise ValueError('KV manifest groups must be dense and nonempty')


def semantic_manifest(manifest):
    validate_manifest(manifest)
    return {n: {k: v for k, v in entry.items() if k != 'group'}
            for n, entry in manifest.items()}


class PeerRegions(list):
    """Keep the complete group count through Core's abbreviated-layout cache."""
    def __init__(self, regions, group_count):
        super().__init__(regions)
        self.group_count = group_count


class OwnerRegistration(BaseModel):
    protocol: str
    engine_id: str
    pcp_rank: int
    pcp_size: int
    split: bool
    source_ids: list[int]
    addr: str
    manifest: dict[str, dict]
    sends: list[str]


class OwnerBootstrap(MooncakeBootstrapServer):
    def _register_routes(self):
        self.app.post('/register')(self.register_worker)
        self.app.get('/query', response_model=dict[int, EngineEntry])(self.query)
        self.app.get(OWNER_QUERY_PATH)(self.query_owners)

    async def register_worker(self, payload: OwnerRegistration):
        data = payload.model_dump()
        try:
            validate_manifest(payload.manifest)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        if (payload.protocol != PROTOCOL or payload.pcp_size < 1
                or not 0 <= payload.pcp_rank < payload.pcp_size):
            raise HTTPException(400, 'Invalid PCP owner protocol/rank')
        expected = sorted(name for name in payload.manifest if transfer_owner(
            name, tuple(payload.source_ids), payload.pcp_size, payload.split) == payload.pcp_rank)
        if sorted(payload.sends) != expected or len(set(payload.sends)) != len(payload.sends):
            raise HTTPException(400, 'Invalid PCP owner coverage')
        if self.workers:
            first = next(iter(self.workers.values()))
            for key in ('protocol', 'engine_id', 'pcp_size', 'split', 'source_ids', 'manifest'):
                if first[key] != data[key]:
                    raise HTTPException(400, f'PCP workers disagree on {key}')
        old = self.workers.get(payload.pcp_rank)
        if old is not None and old != data:
            raise HTTPException(409, 'PCP rank already registered with different state')
        if any(row['addr'] == payload.addr and rank != payload.pcp_rank for rank, row in self.workers.items()):
            raise HTTPException(400, 'PCP worker addresses must be distinct')
        self.workers[payload.pcp_rank] = data
        return {'status': 'ok'}

    def _require_all_owners(self):
        if not self.workers or len(self.workers) != next(iter(self.workers.values()))['pcp_size']:
            raise HTTPException(503, 'Waiting for all PCP owners')

    async def query(self) -> dict[int, EngineEntry]:
        # Router discovery uses the actual DP=TP=PP=1 producer topology.
        # PCP ranks are not TP ranks: the consumer discovers every owner via
        # the versioned endpoint below, never this routing-only leader address.
        self._require_all_owners()
        leader = self.workers[0]
        return {0: EngineEntry(engine_id=leader['engine_id'],
                               worker_addr={0: {0: leader['addr']}})}

    async def query_owners(self):
        self._require_all_owners()
        return {'protocol': PROTOCOL,
                'workers': [self.workers[rank] for rank in sorted(self.workers)]}


def validate_peers(data, manifest):
    if data.get('protocol') != PROTOCOL:
        raise ValueError('Peer is not an owner-aware PCP Mooncake endpoint')
    rows = data.get('workers', [])
    if not rows:
        raise ValueError('Empty PCP peer set')
    first = rows[0]
    width = first['pcp_size']
    if len(rows) != width or {row['pcp_rank'] for row in rows} != set(range(width)):
        raise ValueError('Incomplete PCP peer set')
    addresses, covered = set(), []
    for row in rows:
        for key in ('protocol', 'engine_id', 'pcp_size', 'split', 'source_ids', 'manifest'):
            if row[key] != first[key]:
                raise ValueError(f'PCP peers disagree on {key}')
        if semantic_manifest(row['manifest']) != semantic_manifest(manifest):
            raise ValueError('P/D KV schemas differ (layers/groups/ring capacity/dtype); no bytes transferred')
        expected = sorted(name for name in manifest if transfer_owner(
            name, tuple(row['source_ids']), width, row['split']) == row['pcp_rank'])
        if sorted(row['sends']) != expected:
            raise ValueError('Missing or duplicate PCP layer owner')
        addresses.add(row['addr'])
        covered.extend(row['sends'])
    if len(addresses) != width or sorted(covered) != sorted(manifest):
        raise ValueError('PCP coverage is not exactly once per layer')
    return first['engine_id'], [row['addr'] for row in sorted(rows, key=lambda row: row['pcp_rank'])]


class OwnerWorker(core.MooncakeConnectorWorker):
    def __init__(self, vllm_config, engine_id, kv_cache_config):
        self.source_ids = validate_config(vllm_config)
        self.pcp_size = vllm_config.parallel_config.prefill_context_parallel_size
        self.pcp_rank = get_pcp_group().rank_in_group if self.pcp_size > 1 else 0
        self.split = layer_split_enabled(vllm_config)
        self._owner_bootstrap = None
        self._owner_peers = {}
        # Core's bootstrap election sees TP=0 on every PCP worker. Point a
        # private config copy at our explicitly elected rank-0 bootstrap.
        private = copy.copy(vllm_config)
        private.kv_transfer_config = copy.copy(vllm_config.kv_transfer_config)
        private.kv_transfer_config.kv_connector_extra_config = dict(
            vllm_config.kv_transfer_config.kv_connector_extra_config)
        self._bootstrap_host, self._bootstrap_port = core.get_mooncake_bootstrap_addr(vllm_config)
        private.kv_transfer_config.kv_connector_extra_config['bootstrap_server_address'] = (
            f'{self._bootstrap_host}:{self._bootstrap_port}')
        super().__init__(private, engine_id, kv_cache_config)
        if self._physical_blocks_per_logical_kv_block != 1:
            raise ValueError('PCP Mooncake does not support kernel block splitting')
        if self.is_kv_producer and self.pcp_rank == 0:
            if vllm_config.kv_transfer_config.get_from_extra_config('bootstrap_server_address', None):
                raise ValueError('External PCP bootstrap deployment is not supported')
            self._owner_bootstrap = OwnerBootstrap('0.0.0.0', self._bootstrap_port)
            self._owner_bootstrap.start()

    def shutdown(self):
        if getattr(self, '_owner_bootstrap', None) is not None:
            self._owner_bootstrap.shutdown()
            self._owner_bootstrap = None
        if hasattr(self, 'async_zmq_ctx'):
            super().shutdown()

    def register_kv_caches(self, kv_caches):
        all_regions, self._manifest = describe_caches(self.kv_cache_config, kv_caches)
        self._regions = [r for r in all_regions if not self.is_kv_producer or transfer_owner(
            r.layer_name, self.source_ids, self.pcp_size, self.split) == self.pcp_rank]
        self.device_kv_caches = kv_caches
        storages = {kv_caches[r.layer_name].untyped_storage().data_ptr():
                    kv_caches[r.layer_name].untyped_storage().nbytes() for r in self._regions}
        self.seen_base_addresses, self._kv_data_lens = list(storages), list(storages.values())
        if storages and self.engine.batch_register_memory(self.seen_base_addresses, self._kv_data_lens) != 0:
            raise RuntimeError('PCP Mooncake memory registration failed')
        self._kv_unregistered = False
        self._kv_layout_id = secrets.randbits(63) | 1
        self._prepared_transfer_regions.clear()
        self._layout_by_peer.clear()
        self._acked_layout_peers.clear()
        self.kv_caches_base_addr = [r.base_addr for r in self._regions]
        self.block_len_per_layer = [r.block_len for r in self._regions]
        self.kv_block_len_per_layer = [r.kv_block_len for r in self._regions]
        self.registered_layer_names = [r.layer_name for r in self._regions]
        self.registered_layer_indices = [r.layer_index for r in self._regions]
        self.registered_group_indices = [r.group_index for r in self._regions]
        self.region_shared_groups = [() for _ in self._regions]
        self.region_row_offsets = [-1 for _ in self._regions]
        self.opaque_packed_storages = set()
        core.logger.info('PCP Mooncake rank %d/%d split=%s sends %s',
            self.pcp_rank, self.pcp_size, self.split, self.registered_layer_names)
        if self.is_kv_consumer:
            return
        ready = threading.Event()
        future = asyncio.run_coroutine_threadsafe(self._mooncake_sender_listener(ready), self.sender_loop)
        deadline = time.monotonic() + 4 * core._BOOTSTRAP_MAX_ATTEMPTS * envs.VLLM_MOONCAKE_CONNECTOR_TIMEOUT + 120
        while not ready.wait(1):
            if future.done():
                future.result()
                raise RuntimeError('PCP Mooncake listener exited before ready')
            if time.monotonic() > deadline:
                future.cancel()
                raise TimeoutError('PCP Mooncake registration timed out')

    async def register_worker_with_bootstrap(self):
        payload = OwnerRegistration(protocol=PROTOCOL, engine_id=self.engine_id,
            pcp_rank=self.pcp_rank, pcp_size=self.pcp_size, split=self.split,
            source_ids=list(self.source_ids), manifest=self._manifest,
            sends=self.registered_layer_names,
            addr=make_zmq_path('tcp', self.hostname, self.side_channel_port)).model_dump()
        url = make_zmq_path('http', self._bootstrap_host, self._bootstrap_port) + '/register'
        async with httpx.AsyncClient(timeout=envs.VLLM_MOONCAKE_CONNECTOR_TIMEOUT) as client:
            for attempt in range(core._BOOTSTRAP_MAX_ATTEMPTS):
                try:
                    response = await client.post(url, json=payload)
                    response.raise_for_status()
                    return
                except (httpx.ConnectError, httpx.TimeoutException):
                    if attempt + 1 == core._BOOTSTRAP_MAX_ATTEMPTS:
                        raise
                    await asyncio.sleep(min(2 ** attempt, 10))

    async def _connect_to_prefiller_bootstrap(self, remote_bootstrap_addr):
        try:
            async with httpx.AsyncClient(timeout=envs.VLLM_MOONCAKE_CONNECTOR_TIMEOUT) as client:
                response = await client.get(remote_bootstrap_addr + OWNER_QUERY_PATH)
                response.raise_for_status()
                engine_id, peers = validate_peers(response.json(), self._manifest)
                self._owner_peers[engine_id] = peers
                self._remote_agents[engine_id] = {}  # Core's discovery-ready marker.
        except Exception:
            core.logger.exception('PCP Mooncake bootstrap/schema validation failed')
        finally:
            self._pending_bootstrap_queries.pop(remote_bootstrap_addr).set()

    def receive_kv(self, remote_engine_id, pull_metas):
        peers = self._owner_peers[remote_engine_id]
        for meta in pull_metas.values():
            self._validate_destination_ids(meta.local_block_ids)
            meta.pull_tasks_count = len(peers)
        # Shared PullReqMeta instances preserve Core's sticky failure flag and
        # counter. Each peer gets its own dict in the inherited receiver.
        for peer in peers:
            asyncio.create_task(self.receive_kv_from_single_worker(peer, pull_metas))

    def _validate_destination_ids(self, block_ids):
        groups = self.kv_cache_config.transfer_groups
        if len(block_ids) != len(groups):
            raise ValueError('Destination KV group count mismatch')
        for group, ids in zip(groups, block_ids):
            count = self._cache_page_count(group.layer_names[0])
            if any(type(i) is not int or not 0 <= i < count for i in ids):
                raise ValueError('Destination KV block ID exceeds its pool')

    def _prepare_transfer_regions(self, meta):
        if meta.remote_tp_size != 1 or meta.remote_pp_size != 1:
            return [], [], 'PCP Mooncake requires TP/PP=1 on D'
        remote = self._get_transfer_regions(meta.kv_caches_base_addr, meta.block_lens,
            meta.kv_block_lens, meta.registered_layer_names, meta.registered_layer_indices,
            meta.registered_group_indices, [tuple(x) for x in meta.registered_shared_group_ids],
            meta.registered_row_offsets)
        by_name = {r.layer_name: r for r in remote}
        if len(by_name) != len(remote) or set(by_name) != set(self._manifest):
            return [], [], 'D layout must identify every transferable layer exactly once'
        for region in remote:
            entry = self._manifest[region.layer_name]
            if (region.group_index < 0 or region.kv_block_len != entry['bytes']
                    or region.shared_group_ids or region.block_len < region.kv_block_len):
                return [], [], 'D page length/group mismatch'
        remote_manifest = {n: dict(self._manifest[n], group=r.group_index)
                           for n, r in by_name.items()}
        try:
            validate_manifest(remote_manifest)
        except ValueError as exc:
            return [], [], str(exc)
        aligned = PeerRegions([by_name[r.layer_name] for r in self._regions],
                              len({r.group_index for r in remote}))
        prepared = (self._regions, aligned, None)
        # No row promotion/coalescing across layers: P's groups alias scratch.
        self._remember_peer_layout(meta, prepared)
        return prepared

    async def _build_transfer_params(self, ready_reqs, agent_meta, local_regions, remote_regions):
        """Project named layers into matched virtual groups for native transfer.

        P may merge SWA groups while D keeps seven separate block tables. Each
        layer takes IDs from its own side's group; Core still performs the
        native suffix alignment and descriptor generation. No shared worker
        state is mutated while concurrent async transfers are active.
        """
        if len(local_regions) != len(remote_regions) or not isinstance(remote_regions, PeerRegions):
            return [], [], [], [r for r, _ in ready_reqs], 'Region count mismatch'
        virtual_groups = [self.kv_cache_config.transfer_groups[r.group_index]
                          for r in local_regions]
        facade = SimpleNamespace(kv_cache_config=SimpleNamespace(transfer_groups=virtual_groups),
            _logical_to_kernel_block_ids=lambda ids: ids,
            _get_sender_transfer_plan=self._get_sender_transfer_plan)
        left = [replace(r, group_index=i) for i, r in enumerate(local_regions)]
        right = [replace(r, group_index=i) for i, r in enumerate(remote_regions)]
        prepared, rejected, projected = [], [], {}
        remote_group_count = remote_regions.group_count
        for req_id, send in ready_reqs:
            transfer_id, remote_ids = agent_meta.req_blocks[req_id]
            local_ids = send.local_block_ids
            if (len(local_ids) != len(self.kv_cache_config.transfer_groups)
                    or len(remote_ids) != remote_group_count
                    or any(a.layer_name != b.layer_name for a, b in zip(local_regions, remote_regions))):
                rejected.append(req_id)
                continue
            if any(type(i) is not int or i < 0 for ids in remote_ids for i in ids):
                rejected.append(req_id)
                continue
            if any(type(i) is not int or i < 0 or i >= self._cache_page_count(r.layer_name)
                   for r in local_regions for i in local_ids[r.group_index]):
                rejected.append(req_id)
                continue
            private_send = copy.copy(send)
            private_send.local_block_ids = [local_ids[r.group_index] for r in local_regions]
            prepared.append((req_id, private_send))
            projected[req_id] = (transfer_id, [remote_ids[r.group_index] for r in remote_regions])
        private_meta = copy.copy(agent_meta)
        private_meta.req_blocks = projected
        src, dst, sizes, errors, error = await core.MooncakeConnectorWorker._build_transfer_params(
            facade, prepared, private_meta, left, right)
        return src, dst, sizes, rejected + errors, error or ('Invalid pool block table' if rejected else None)

    def _cache_page_count(self, name):
        tensor = next(t for t in self.kv_cache_config.kv_cache_tensors if name in t.layers)
        return self.kv_cache_config.num_blocks_of(tensor)


class OwnerScheduler(core.MooncakeConnectorScheduler):
    """Transfer the state before the token D recomputes for its first logits.

    PCP only retains the open ratio-2 compressor row. Computing all N tokens
    on P can close/discard the row needed when D recomputes token N-1. The
    native recurrent-state truncation helper already handles this lifecycle;
    apply it to CircularBufferSpec as well as Core's Mamba/bounded-SWA cases.
    """
    def __init__(self, vllm_config, engine_id, kv_cache_config):
        super().__init__(vllm_config, engine_id, kv_cache_config)
        self._has_compressor_state = any(isinstance(spec, CircularBufferSpec)
            for group in kv_cache_config.transfer_groups for _, spec in _specs(group))

    def on_new_request(self, request):
        super().on_new_request(request)
        params = request.kv_transfer_params
        if self._has_compressor_state and params is not None and params.get('do_remote_decode'):
            self._truncate_mamba_request_for_prefill(request)

    def request_finished(self, request, block_ids):
        params = request.kv_transfer_params
        if (self._has_compressor_state and params
                and params.get('_p_side_truncated')
                and params.get('do_remote_decode') and params.get('transfer_id')
                and request.status == core.RequestStatus.FINISHED_STOPPED):
            # P samples at N-1 only to finish its internal handoff. That token
            # is discarded: D recomputes N-1 and applies the user's stopping
            # rules. An EOS/stop here must not strand D waiting for these pages.
            # Abort/error and untruncated requests keep Core's cleanup below.
            assert not self.is_kv_consumer
            delay_free_blocks = any(block_ids)
            if delay_free_blocks:
                self._reqs_need_send[request.request_id] = (
                    request, self.get_sw_clipped_blocks(block_ids))
            return delay_free_blocks, None
        return super().request_finished(request, block_ids)

    def _get_remote_prefill_token_count(self, num_prompt_tokens):
        if self._has_compressor_state and num_prompt_tokens > 1:
            return num_prompt_tokens - 1
        return super()._get_remote_prefill_token_count(num_prompt_tokens)


class HcuLayerSplitMooncakeConnector(core.MooncakeConnector):
    def __init__(self, vllm_config, role, kv_cache_config):
        validate_config(vllm_config)
        KVConnectorBase_V1.__init__(self, vllm_config, role, kv_cache_config)
        self.engine_id = vllm_config.kv_transfer_config.engine_id
        self.connector_scheduler = self.connector_worker = None
        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = OwnerScheduler(
                vllm_config, self.engine_id, kv_cache_config)
        elif role == KVConnectorRole.WORKER:
            self.connector_worker = OwnerWorker(vllm_config, self.engine_id, kv_cache_config)
        else:
            raise ValueError(f'Unknown connector role {role}')
