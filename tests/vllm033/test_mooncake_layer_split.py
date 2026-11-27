"""CPU-only owner discovery, native byte transfer and completion regressions."""
import asyncio
import copy
import ctypes
from concurrent.futures import ThreadPoolExecutor
import queue
from collections import OrderedDict
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest
import torch
import msgspec
import zmq
import zmq.asyncio
from fastapi import HTTPException

from hcu_vllm_das import mooncake_layer_split as m
from hcu_vllm_das import dsa_cache_layout as placement
from hcu_vllm_das.dsa_cache_layer_split import validate_layer_split_transfer
from test_dsa_cache_layout import config, groups
from vllm.v1.kv_cache_interface import CircularBufferSpec, KVCacheGroupSpec, SlidingWindowMLASpec
from vllm.v1.core.kv_cache_planning import DefaultKVCacheConfigBuilder
from vllm.v1.kv_cache_layout import KVCacheLayout
from vllm.v1.worker import utils
from vllm.config import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory


def hybrid_groups():
    result = groups()
    for name, spec in (
        ('model.layers.0.attn.swa_cache', SlidingWindowMLASpec(
            block_size=64, num_kv_heads=1, head_size=32, dtype=torch.uint8, sliding_window=64)),
        ('model.layers.2.attn.compressor', CircularBufferSpec(
            block_size=8, num_kv_heads=1, head_size=12, head_size_v=0, dtype=torch.float32)),
    ):
        result.append(KVCacheGroupSpec([name], spec))
    result[1].layer_names.append('model.layers.7.attn.swa_cache')
    return result


def allocated(rank, split):
    cfg, gs = config(), hybrid_groups()
    with patch.object(cfg, 'additional_config', {'hcu_layer_split': {'enabled': split}}, create=True):
        plan = (placement._materialize(cfg, gs, 7, rank) if split else
                DefaultKVCacheConfigBuilder().get_kv_cache_config_from_groups(cfg, gs, 7))
    with patch.object(utils.current_platform, 'is_rocm', return_value=False):
        caches = utils.allocate_kv_cache(plan, torch.device('cpu'), KVCacheLayout.BLHNC)
    return plan, caches


def bare_worker():
    worker = m.OwnerWorker.__new__(m.OwnerWorker)
    worker._draft_context = False
    worker._draft_groups = {}
    worker._owner_bootstrap = None
    worker._layout_by_peer = OrderedDict()
    worker._prepared_transfer_regions = {}
    worker.finished_recving_reqs = set()
    worker._failed_recv_reqs = queue.Queue()
    worker._invalid_block_ids = queue.Queue()
    worker._is_hma_required = True
    worker.xfer_stats = m.core.MooncakeKVConnectorStats()
    worker.tp_size = worker.pp_size = 1
    worker.tp_rank = 0
    worker.use_mla = True
    worker._physical_blocks_per_logical_kv_block = 1
    return worker


def bare_scheduler():
    scheduler = m.OwnerScheduler.__new__(m.OwnerScheduler)
    scheduler._draft_context = False
    scheduler._draft_groups = {}
    return scheduler


def registrations(split=True):
    plan, caches = allocated(0, split)
    _, manifest = m.describe_caches(plan, caches)
    return [m.OwnerRegistration(protocol=m.PROTOCOL, engine_id='p',
        pcp_rank=rank, pcp_size=8, split=split, source_ids=[2, 8, 14, 20],
        addr=f'tcp://127.0.0.1:{19000 + rank}', manifest=manifest,
        sends=[name for name in manifest if m.transfer_owner(name, (2, 8, 14, 20), 8, split) == rank]
    ) for rank in range(8)]


@pytest.mark.parametrize('split', [False, True])
def test_bootstrap_waits_for_all_eight_and_covers_every_layer_once(split):
    async def run():
        server = m.OwnerBootstrap('127.0.0.1', 0)
        regs = registrations(split)
        for payload in regs[:-1]:
            await server.register_worker(payload)
        with pytest.raises(HTTPException) as exc:
            await server.query()
        assert exc.value.status_code == 503
        await server.register_worker(regs[-1])
        await server.register_worker(regs[-1])  # idempotent retry
        engine, peers = m.validate_peers(await server.query_owners(), regs[0].manifest)
        assert engine == 'p' and len(peers) == 8
        assert all(not regs[rank].sends for rank in range(4, 7))
        assert regs[7].sends == (['model.layers.7.attn.swa_cache'] if split else [])
        assert sum(len(p.sends) for p in regs) == len(regs[0].manifest)
    asyncio.run(run())


@pytest.mark.parametrize('fault', ['rank', 'owner', 'schema', 'addr', 'engine'])
def test_bootstrap_rejects_conflicting_registration(fault):
    async def run():
        server = m.OwnerBootstrap('127.0.0.1', 0)
        regs = registrations()
        await server.register_worker(regs[0])
        payload = regs[1].model_copy(deep=True)
        if fault == 'rank': payload.pcp_rank = 8
        elif fault == 'owner': payload.sends = regs[0].sends
        elif fault == 'schema': payload.manifest[next(iter(payload.manifest))]['bytes'] += 1
        elif fault == 'addr': payload.addr = regs[0].addr
        else: payload.engine_id = 'wrong'
        with pytest.raises(HTTPException):
            await server.register_worker(payload)
    asyncio.run(run())


@pytest.mark.parametrize('split', [False, True])
def test_native_transfer_descriptors_copy_only_owned_live_pages(split):
    async def run():
        dst_plan, dst = allocated(0, False)
        dst_regions, dst_manifest = m.describe_caches(dst_plan, dst)
        # Reserve disjoint physical block IDs for the three native groups.
        src_blocks, dst_blocks = [[1, 3], [2], [4]], [[2, 5], [3], [6]]
        expected = {}
        for cache in dst.values():
            cache.untyped_storage().fill_(173)
        before = bytes(dst[next(iter(dst))].untyped_storage())
        touched = set()
        for rank in range(8):
            plan, caches = allocated(rank, split)
            regions, manifest = m.describe_caches(plan, caches)
            assert manifest == dst_manifest
            worker = bare_worker()
            worker.kv_cache_config = plan
            worker._manifest = manifest
            worker._regions = [r for r in regions if m.transfer_owner(r.layer_name, (2, 8, 14, 20), 8, split) == rank]
            worker._get_sender_transfer_plan = lambda **kw: (True, 0, 0, kw['local_kv_block_len'])
            for idx, region in enumerate(worker._regions):
                for block in src_blocks[region.group_index]:
                    caches[region.layer_name][block].fill_(11 + rank * 7 + idx)
                    expected[region.layer_name, block] = caches[region.layer_name][block].clone()
            meta = m.core.MooncakeXferMetadata(remote_hostname='localhost', remote_port=1,
                remote_tp_size=1, remote_tp_rank=0, req_blocks={'d': ('x', dst_blocks)},
                kv_caches_base_addr=[r.base_addr for r in dst_regions],
                block_lens=[r.block_len for r in dst_regions], kv_block_lens=[r.kv_block_len for r in dst_regions],
                registered_layer_names=[r.layer_name for r in dst_regions],
                registered_layer_indices=[r.layer_index for r in dst_regions],
                registered_group_indices=[r.group_index for r in dst_regions],
                registered_shared_group_ids=[[] for _ in dst_regions],
                registered_row_offsets=[-1 for _ in dst_regions], layout_id=1)
            local, remote, error = worker._prepare_transfer_regions(meta)
            assert error is None
            assert worker._lookup_peer_layout(meta) == (local, remote)
            send = m.core.SendBlockMeta('p', 'x', src_blocks, asyncio.Event())
            srcs, dsts, sizes, errors, error = await worker._build_transfer_params([('d', send)], meta, local, remote)
            assert not errors and error is None
            base = dst[next(iter(dst))].untyped_storage().data_ptr()
            for src, target, size in zip(srcs, dsts, sizes):
                span = set(range(target - base, target - base + size))
                assert not span & touched  # each page has exactly one writer
                touched.update(span)
                ctypes.memmove(target, src, size)
        for region in dst_regions:
            for src_block, dst_block in zip(src_blocks[region.group_index], dst_blocks[region.group_index]):
                assert torch.equal(dst[region.layer_name][dst_block], expected[region.layer_name, src_block])
        after = bytes(dst[next(iter(dst))].untyped_storage())
        assert all(a == b for i, (a, b) in enumerate(zip(before, after)) if i not in touched)
        assert not torch.cuda.is_initialized()
    asyncio.run(run())


@pytest.mark.parametrize('failure', [False, True])
def test_completion_waits_for_every_owner_even_after_failure(failure):
    worker = bare_worker()
    pull = m.core.PullReqMeta('d', 'x', [[2], [3]], 'p', 'http://localhost', pull_tasks_count=8)
    response = m.core.MooncakeXferResponse(m.core.MooncakeXferResponseStatus.FINISH, ok_reqs=['d'])
    peers = [{'d': pull} for _ in range(8)]
    for i in range(7):
        if failure and i == 2:
            worker._handle_failed_recv(peers[i], ['d'], 'injected owner failure')
        else:
            worker.process_pulling_result(response, peers[i])
            worker.process_pulling_result(response, peers[i])  # duplicate ACK cannot decrement again
        assert not worker.finished_recving_reqs
    worker.process_pulling_result(response, peers[-1])
    assert worker.finished_recving_reqs == {'d'} and pull.pull_tasks_count == 0
    assert pull.failed == failure
    assert worker._failed_recv_reqs.qsize() == int(failure)


def test_consumer_rejects_missing_peer_or_schema_before_any_write():
    regs = registrations()
    data = dict(protocol=m.PROTOCOL, workers=[p.model_dump() for p in regs])
    with pytest.raises(ValueError, match='Incomplete'):
        m.validate_peers(dict(data, workers=data['workers'][:-1]), regs[0].manifest)
    wrong = copy.deepcopy(regs[0].manifest)
    wrong['model.layers.2.attn.compressor']['block_size'] = 16
    with pytest.raises(ValueError, match='schemas differ'):
        m.validate_peers(data, wrong)


@pytest.mark.parametrize('blocks,error', [
    ([[1]], 'group count'),
    ([[2], [3], [7]], 'exceeds its pool'),
])
def test_consumer_rejects_nonempty_invalid_destination_block_tables(blocks, error):
    worker = bare_worker()
    worker.kv_cache_config, _ = allocated(0, False)
    with pytest.raises(ValueError, match=error):
        worker._validate_destination_ids(blocks)


def test_guard_requires_explicit_plugin_connector_and_producer():
    cfg = config()
    for transfer in (object(), NS(kv_connector='MooncakeConnector'),
                     NS(kv_connector=m.CONNECTOR, kv_connector_module_path=m.MODULE, kv_role='kv_consumer')):
        cfg.kv_transfer_config = transfer
        with pytest.raises(ValueError, match='owner-aware'):
            validate_layer_split_transfer(cfg)
    cfg.kv_transfer_config = NS(kv_connector=m.CONNECTOR, kv_connector_module_path=m.MODULE, kv_role='kv_producer')
    validate_layer_split_transfer(cfg)


def test_public_connector_factory_resolves_hma_plugin():
    transfer = KVTransferConfig(kv_connector=m.CONNECTOR,
        kv_connector_module_path=m.MODULE, kv_role='kv_producer')
    assert KVConnectorFactory.get_connector_class(transfer) is m.HcuLayerSplitMooncakeConnector
    assert KVConnectorFactory.supports_hma_config(transfer)


def test_pcp_bootstrap_elects_only_rank_zero_without_mutating_config(monkeypatch):
    cfg = config()
    cfg.model_config.use_mla = True
    cfg.kv_transfer_config = KVTransferConfig(kv_connector=m.CONNECTOR,
        kv_connector_module_path=m.MODULE, kv_role='kv_producer')
    plan = DefaultKVCacheConfigBuilder().get_kv_cache_config_from_groups(
        cfg, hybrid_groups(), 7)
    started = []
    def base_init(worker, private, engine, plan):
        assert private is not cfg and private.kv_transfer_config is not cfg.kv_transfer_config
        assert private.kv_transfer_config.get_from_extra_config('bootstrap_server_address', None) == '127.0.0.1:18998'
        worker.is_kv_producer = True
        worker._physical_blocks_per_logical_kv_block = 1
    monkeypatch.setattr(m.core.MooncakeConnectorWorker, '__init__', base_init)
    monkeypatch.setattr(m.core, 'get_mooncake_bootstrap_addr', lambda _: ('127.0.0.1', 18998))
    monkeypatch.setattr(m.OwnerBootstrap, 'start', lambda server: started.append(server))
    for rank in range(8):
        monkeypatch.setattr(m, 'get_pcp_group', lambda: NS(rank_in_group=rank))
        worker = m.OwnerWorker(cfg, 'p', plan)
        assert worker.pcp_rank == rank and worker.pcp_size == 8
        assert cfg.kv_transfer_config.kv_connector_extra_config == {}
        worker.shutdown()
    assert len(started) == 1


def test_draft_context_is_not_silently_omitted():
    cfg = config(1)
    cfg.model_config.use_mla = True
    cfg.kv_transfer_config = KVTransferConfig(kv_connector=m.CONNECTOR,
        kv_connector_module_path=m.MODULE, kv_role='kv_consumer')
    cfg.speculative_config = NS(method='dspark', num_speculative_tokens=5)
    with pytest.raises(ValueError, match='requires explicit DSpark context transfer'):
        m.validate_config(cfg)


@pytest.mark.parametrize('length', [1, 2, 127, 128, 129, 4096])
def test_compressor_handoff_stops_before_decoder_recomputed_token(length):
    scheduler = bare_scheduler()
    scheduler._has_compressor_state = True
    scheduler._has_mamba = scheduler._bounded_replay = False
    request = NS(kv_transfer_params={'do_remote_decode': True},
        num_prompt_tokens=length, prompt_token_ids=list(range(length)),
        _all_token_ids=list(range(length)), max_tokens=7,
        get_skip_reading_prefix_cache=lambda: False)
    scheduler.on_new_request(request)
    scheduler.on_new_request(request)  # preemption/re-entry must be idempotent
    expected = length - 1 if length > 1 else 1
    assert request.num_prompt_tokens == expected
    assert len(request._all_token_ids) == len(request.prompt_token_ids) == expected
    assert scheduler._get_remote_prefill_token_count(length) == expected
    if length > 1:
        assert request.max_tokens == 1


@pytest.mark.parametrize('fail_rank,with_releases', [(None, False), (2, False), (None, True)])
def test_real_zmq_eight_peer_pull_with_cpu_transfer_engine(fail_rank, with_releases):
    """Exercise Core's actual serialized pull/send/ACK path, no RDMA or GPU."""
    from vllm.sampling_params import SamplingParams
    from vllm.v1.request import Request, RequestStatus

    async def run():
        ctx = zmq.asyncio.Context()
        pool = ThreadPoolExecutor(max_workers=4)
        decoder = bare_worker()
        decoder.async_zmq_ctx = ctx
        decoder.hostname, decoder.rpc_port = '127.0.0.1', 1234
        decoder._encoder = msgspec.msgpack.Encoder()
        decoder._xfer_resp_decoder = msgspec.msgpack.Decoder(m.core.MooncakeXferResponse)
        decoder._acked_layout_peers = set()
        decoder._kv_layout_id = 77
        plan, dest = allocated(0, False)
        decoder.kv_cache_config = plan
        regions, _ = m.describe_caches(plan, dest)
        for attr, field in [('kv_caches_base_addr','base_addr'), ('block_len_per_layer','block_len'),
                            ('kv_block_len_per_layer','kv_block_len'), ('registered_layer_names','layer_name'),
                            ('registered_layer_indices','layer_index'), ('registered_group_indices','group_index')]:
            setattr(decoder, attr, [getattr(r, field) for r in regions])
        decoder.region_shared_groups = [() for _ in regions]
        decoder.region_row_offsets = [-1 for _ in regions]
        scheduler = bare_scheduler()
        scheduler.is_kv_producer, scheduler.is_kv_consumer = False, True
        scheduler._has_compressor_state = True
        scheduler._is_hma_required = True
        scheduler.kv_cache_config = plan
        scheduler.blocks_per_sw = [0] * len(plan.transfer_groups)
        scheduler._reqs_need_recv = {}

        def request(request_id, transfer_id):
            req = Request(request_id, list(range(8)), SamplingParams(max_tokens=4), None)
            req.kv_transfer_params = dict(do_remote_prefill=True, transfer_id=transfer_id,
                remote_engine_id='p', remote_bootstrap_addr='http://unused')
            return req

        scheduler.update_state_after_alloc(request('d', 'x'),
            NS(get_unhashed_block_ids_all_groups=lambda: ([4], [5], [6])), 7)
        senders = {'x': 'p-request'}
        if with_releases:
            abort = request('d-abort', 'x-abort')
            abort.status = RequestStatus.FINISHED_ABORTED
            assert scheduler.request_finished(abort, ()) == (False, None)
            scheduler.update_state_after_alloc(request('d-prefix', 'x-prefix'), None, 0)
            senders.update({'x-abort': 'p-abort', 'x-prefix': 'p-prefix'})
        metadata = scheduler.build_connector_meta(None)
        pulls = metadata.reqs_to_recv['p']
        assert pulls['d'].local_block_ids == [[4], [5], [6]]
        assert all(pull.local_block_ids == [] for req_id, pull in pulls.items() if req_id != 'd')
        counts = [0] * 8
        producers, sockets, tasks, addresses, keepalive = [], [], [], [], []
        last_release = asyncio.Event()

        async def serve(worker, sock, rank):
            identity, encoded = await sock.recv_multipart()
            if rank == 7:
                await last_release.wait()
            meta = msgspec.msgpack.decode(encoded, type=m.core.MooncakeXferMetadata)
            assert meta.req_blocks == {req_id: (pull.transfer_id, pull.local_block_ids)
                                       for req_id, pull in pulls.items()}
            await worker.send_kv_to_decode(identity, sock, meta)

        receive = decoder.receive_kv_from_single_worker
        receivers = {}

        async def track_receive(addr, metas):
            receivers[addr] = asyncio.current_task()
            tasks.append(receivers[addr])
            await receive(addr, metas)

        decoder.receive_kv_from_single_worker = track_receive
        try:
            for rank in range(8):
                worker = bare_worker()
                worker.kv_cache_config, caches = allocated(rank, True)
                keepalive.append(caches)
                all_regions, worker._manifest = m.describe_caches(worker.kv_cache_config, caches)
                worker._regions = [r for r in all_regions if m.transfer_owner(r.layer_name, (2,8,14,20), 8, True) == rank]
                for region in worker._regions:
                    block = [1,2,3][region.group_index]
                    caches[region.layer_name][block].fill_(rank + 21)
                def transfer(remote, srcs, dsts, sizes, rank=rank):
                    counts[rank] += len(sizes)
                    if rank == fail_rank:
                        return 1
                    for src, dst, size in zip(srcs, dsts, sizes):
                        ctypes.memmove(dst, src, size)
                    return 0
                worker.engine = NS(batch_transfer_sync_write=transfer)
                worker.transfer_topo = NS(handshake_target_ranks=lambda _: [0], local_replicates_kv_cache=True)
                worker.opaque_packed_storages = set()
                worker._encoder = msgspec.msgpack.Encoder()
                worker.sender_loop = asyncio.get_running_loop()
                worker._sender_executor = pool
                worker.finished_sending_reqs = set()
                ready = asyncio.Event(); ready.set()
                worker.reqs_need_send = {tid: m.core.SendBlockMeta(pid, tid, [[1], [2], [3]], ready)
                                         for tid, pid in senders.items()}
                sock = ctx.socket(zmq.ROUTER)
                port = sock.bind_to_random_port('tcp://127.0.0.1')
                addresses.append(f'tcp://127.0.0.1:{port}')
                producers.append(worker); sockets.append(sock)
                tasks.append(asyncio.create_task(serve(worker, sock, rank)))
            decoder._owner_peers = {'p': addresses}
            decoder._remote_agents = {'p': {}}
            await decoder._start_load_kv(metadata.reqs_to_recv)
            assert all(pull.pull_tasks_count == 8 for pull in pulls.values())
            await asyncio.sleep(0)  # Register the tasks created by OwnerWorker.receive_kv.
            await asyncio.wait_for(asyncio.gather(*(receivers[addr] for addr in addresses[:-1])), timeout=15)
            assert all(pull.pull_tasks_count == 1 for pull in pulls.values())
            assert not decoder.finished_recving_reqs
            last_release.set()
            await asyncio.wait_for(asyncio.gather(*tasks), timeout=15)
            assert decoder.finished_recving_reqs == {'d'}
            assert all(pull.pull_tasks_count == 0 for pull in pulls.values())
            assert pulls['d'].failed == (fail_rank is not None)
            assert not any(pull.failed for req_id, pull in pulls.items() if req_id != 'd')
            for rank, worker in enumerate(producers):
                assert counts[rank] == len(worker._regions)
                if rank != fail_rank:
                    assert worker.finished_sending_reqs == set(senders.values())
                    assert not worker.reqs_need_send
                    for region in worker._regions:
                        assert torch.all(dest[region.layer_name][[4,5,6][region.group_index]] == rank+21)
                else:
                    assert not worker.finished_sending_reqs
                    assert worker.reqs_need_send['x'].sending == 0
            assert not torch.cuda.is_initialized()
        finally:
            last_release.set()
            for task in tasks:
                if not task.done(): task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            for sock in sockets: sock.close(0)
            pool.shutdown(wait=True)
            del decoder.async_zmq_ctx
            ctx.term()
    asyncio.run(run())


@pytest.mark.parametrize('split', [False, True])
def test_live_http_bootstrap_preserves_router_schema_and_discovers_all_owners(split):
    """Real HTTP discovery: native router topology and independent PCP manifest."""
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_utils import (
        MooncakeBootstrapServer, RegisterWorkerPayload,
    )

    async def run():
        server = m.OwnerBootstrap('127.0.0.1', 0)
        server.start()
        url = f'http://127.0.0.1:{server.port}'
        regs = registrations(split)
        try:
            async with m.httpx.AsyncClient(trust_env=False) as client:
                for endpoint in ('/query', m.OWNER_QUERY_PATH):
                    assert (await client.get(url + endpoint)).status_code == 503
                for reg in regs[:-1]:
                    response = await client.post(url + '/register', json=reg.model_dump())
                    response.raise_for_status()
                for endpoint in ('/query', m.OWNER_QUERY_PATH):
                    assert (await client.get(url + endpoint)).status_code == 503
                response = await client.post(url + '/register', json=regs[-1].model_dump())
                response.raise_for_status()
                topology = (await client.get(url + '/query')).json()
                # Compare with Core's real standard HTTP response, not a duplicate
                # implementation of the Router's expected serialization.
                native = MooncakeBootstrapServer('127.0.0.1', 0)
                native.start()
                try:
                    response = await client.post(
                        f'http://127.0.0.1:{native.port}/register',
                        json=RegisterWorkerPayload(engine_id='p', dp_rank=0,
                            tp_rank=0, pp_rank=0, addr=regs[0].addr).model_dump())
                    response.raise_for_status()
                    expected = (await client.get(
                        f'http://127.0.0.1:{native.port}/query')).json()
                finally:
                    native.shutdown()
                assert topology == expected
                assert list(map(int, topology)) == [0]
                owners = (await client.get(url + m.OWNER_QUERY_PATH)).json()
                assert m.validate_peers(owners, regs[0].manifest) == (
                    'p', [reg.addr for reg in regs])

            worker = bare_worker()
            worker._manifest = regs[0].manifest
            worker._owner_peers = {}
            worker._remote_agents = {}
            ready = asyncio.Event()
            worker._pending_bootstrap_queries = {url: ready}
            await worker._connect_to_prefiller_bootstrap(url)
            assert ready.is_set()
            assert url not in worker._pending_bootstrap_queries
            assert worker._owner_peers == {'p': [reg.addr for reg in regs]}
            assert worker._remote_agents == {'p': {}}
        finally:
            server.shutdown()
        assert not torch.cuda.is_initialized()

    asyncio.run(run())


@pytest.mark.parametrize('token', [2, 3, 9])
def test_truncated_prefill_stop_token_still_hands_off_real_state(token):
    """The token sampled at N-1 is discarded; even EOS must release KV to D."""
    from vllm.sampling_params import SamplingParams
    from vllm.v1.request import Request, RequestStatus
    from vllm.v1.core.sched.utils import check_stop
    scheduler = bare_scheduler()
    scheduler._has_compressor_state = True
    scheduler._has_mamba = scheduler._bounded_replay = False
    scheduler.is_kv_consumer = False
    scheduler._reqs_need_send = {}
    scheduler._reqs_not_processed = set()
    scheduler.get_sw_clipped_blocks = lambda blocks: [group[-2:] for group in blocks]
    sampling = SamplingParams(max_tokens=16, stop_token_ids=[3])
    sampling._eos_token_id = 2
    request = Request('p', list(range(8)), sampling, None)
    request.kv_transfer_params = {'do_remote_decode': True, 'transfer_id': 'x'}
    scheduler.on_new_request(request)
    assert request.num_prompt_tokens == 7
    request.append_output_token_ids(token)
    assert check_stop(request, 16384)
    status = request.status
    assert status == (RequestStatus.FINISHED_LENGTH_CAPPED if token == 9
                      else RequestStatus.FINISHED_STOPPED)
    delayed, response = scheduler.request_finished(request, ([1, 2, 3], [4]))
    assert delayed and response is None
    assert scheduler._reqs_need_send['p'] == (request, [[2, 3], [4]])
    assert not scheduler._reqs_not_processed
    assert request.status == status
    assert request.sampling_params is sampling
    assert sampling.eos_token_id == 2 and sampling.stop_token_ids == [3]
    assert not sampling.ignore_eos
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('status_name,truncated', [
    ('FINISHED_STOPPED', False), ('FINISHED_ABORTED', True),
    ('FINISHED_ERROR', True),
])
def test_non_handoff_stops_and_aborts_keep_native_cleanup(status_name, truncated):
    from vllm.v1.request import RequestStatus
    scheduler = bare_scheduler()
    scheduler._has_compressor_state = True
    scheduler.is_kv_consumer = False
    scheduler._reqs_need_send = {}
    scheduler._reqs_not_processed = set()
    request = NS(request_id='p', status=getattr(RequestStatus, status_name),
                 kv_transfer_params={'do_remote_decode': True, 'transfer_id': 'x',
                                     '_p_side_truncated': truncated})
    assert scheduler.request_finished(request, ([1],)) == (False, None)
    assert scheduler._reqs_not_processed == {'x'}
    assert not scheduler._reqs_need_send
