# SPDX-License-Identifier: Apache-2.0
"""Real Core connector and completion aggregation, without engine/GPU setup."""
import asyncio
import threading
from types import SimpleNamespace as NS

import pytest

from hcu_vllm_das import hy4_pcp_mooncake as owned
from vllm.distributed.kv_transfer.kv_connector.v1 import base
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake import mooncake_connector as native
from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput


def config(pcp=2):
    return NS(model_config=NS(hf_config=NS(model_type='hy_v4')),
        parallel_config=NS(prefill_context_parallel_size=pcp, data_parallel_size=1,
            tensor_parallel_size=1, pipeline_parallel_size=1, decode_context_parallel_size=1),
        kv_transfer_config=NS(engine_id='probe', kv_role='kv_producer'))


def worker(canonical=False):
    obj = object.__new__(owned.HY4PCPMooncakeWorker)
    obj._canonical_pcp_sender = canonical
    obj._replica_pending = {}
    obj.reqs_need_send = {}
    obj.finished_sending_reqs = set()
    return obj


def test_worker_binding_retains_native_constructor_and_globals(monkeypatch):
    original = native.MooncakeConnector.__init__
    calls = []
    monkeypatch.setattr(base.KVConnectorBase_V1, '__init__', lambda *args: None)
    monkeypatch.setattr(owned, '_init_native',
        lambda self, *args: calls.append(type(self)))
    monkeypatch.setattr(native.MooncakeConnectorWorker, 'shutdown', lambda self: None)
    monkeypatch.setattr(owned, 'get_pcp_group', lambda: NS(world_size=2, rank_in_group=1))
    connector = owned.MooncakeConnector(config(), base.KVConnectorRole.WORKER, NS())
    assert type(connector.connector_worker) is owned.HY4PCPMooncakeWorker
    assert calls == [owned.HY4PCPMooncakeWorker]
    assert native.MooncakeConnector.__init__ is original
    assert original.__globals__['MooncakeConnectorWorker'] is native.MooncakeConnectorWorker


def test_replica_does_not_bind_or_register_but_primary_delegates(monkeypatch):
    calls = []
    async def listener(self, event):
        calls.append(self)
        event.set()
    monkeypatch.setattr(native.MooncakeConnectorWorker, '_mooncake_sender_listener', listener)
    replica, primary = worker(), worker(True)
    for obj in (replica, primary):
        ready = threading.Event()
        asyncio.run(obj._mooncake_sender_listener(ready))
        assert ready.is_set()
    assert calls == [primary]


def test_replica_placeholders_finish_and_cancel_follow_native_completion_path():
    obj = worker()
    async def replay():
        await obj.record_send_reqs(NS(reqs_to_send={'r': ('t', [])}, reqs_not_processed=()))
        assert not await obj.fetch_finished_sending_reqs()
        await obj.record_send_reqs(NS(reqs_to_send={'r': ('t', [[3], [5]])}, reqs_not_processed=()))
        assert await obj.fetch_finished_sending_reqs() == {'r'}
        assert not obj._replica_pending
        await obj.record_send_reqs(NS(reqs_to_send={'cancelled': ('u', [])}, reqs_not_processed=()))
        await obj.record_send_reqs(NS(reqs_to_send={}, reqs_not_processed=('u',)))
        # Core frees an aborted request immediately; it never waits for ACKs.
        # Emitting a replica-only completion would strand an aggregator entry.
        assert not await obj.fetch_finished_sending_reqs()
        assert not obj._replica_pending and not obj.reqs_need_send
    asyncio.run(replay())


def test_native_aggregator_keeps_kv_lease_until_primary_ack():
    aggregator = KVOutputAggregator(expected_finished_count=2)
    def output(sent=()):
        return ModelRunnerOutput.with_kv_conn_output_only(
            KVConnectorOutput(finished_sending=set(sent)))
    partial = aggregator.aggregate([output(), output(('r',))])
    assert not partial.kv_connector_output.finished_sending
    finished = aggregator.aggregate([output(('r',)), output()])
    assert finished.kv_connector_output.finished_sending == {'r'}


def test_unsupported_pcp_topology_fails_before_native_engine(monkeypatch):
    monkeypatch.setattr(owned, '_init_native',
        lambda *args: pytest.fail('native engine initialized'))
    cfg = config()
    cfg.parallel_config.tensor_parallel_size = 2
    with pytest.raises(ValueError, match='TP1/PP1/DP1/DCP1'):
        owned.HY4PCPMooncakeWorker(cfg, 'probe', NS())


def test_failed_prevalidation_has_no_native_resources_to_destroy(monkeypatch):
    calls = []
    monkeypatch.setattr(native.MooncakeConnectorWorker, 'shutdown', lambda self: calls.append(self))
    obj = object.__new__(owned.HY4PCPMooncakeWorker)
    cfg = config()
    cfg.parallel_config.decode_context_parallel_size = 2
    with pytest.raises(ValueError):
        obj.__init__(cfg, 'probe', NS())
    obj.__del__()
    assert not calls


def test_public_external_connector_resolution_leaves_registry_unchanged():
    registry = KVConnectorFactory._registry.copy()
    cfg = NS(kv_connector='MooncakeConnector',
             kv_connector_module_path='hcu_vllm_das.hy4_pcp_mooncake')
    assert KVConnectorFactory.get_connector_class(cfg) is owned.MooncakeConnector
    assert KVConnectorFactory._registry == registry


def test_native_connector_name_preserves_core_compatibility_guards():
    from vllm.config.kv_transfer import KVTransferConfig
    cfg = KVTransferConfig(kv_connector='MooncakeConnector', kv_role='kv_producer',
        kv_buffer_device='cuda', kv_connector_module_path='hcu_vllm_das.hy4_pcp_mooncake')
    assert cfg.has_connector('MooncakeConnector')
    assert KVConnectorFactory.get_connector_class(cfg) is owned.MooncakeConnector


def test_non_pcp_consumer_uses_native_worker_without_pcp_group(monkeypatch):
    calls = []
    monkeypatch.setattr(owned, 'get_pcp_group',
        lambda: pytest.fail('non-PCP consumer queried the PCP group'))
    monkeypatch.setattr(owned, '_init_native',
        lambda self, *args: calls.append(args))
    monkeypatch.setattr(native.MooncakeConnectorWorker, 'shutdown', lambda self: None)
    cfg = config(pcp=1)
    cfg.parallel_config.data_parallel_size = 16
    cfg.kv_transfer_config.kv_role = 'kv_consumer'
    obj = owned.HY4PCPMooncakeWorker(cfg, 'consumer', NS())
    assert obj._canonical_pcp_sender and len(calls) == 1


def test_primary_records_ready_and_cancel_with_native_metadata():
    obj = worker(canonical=True)
    async def replay():
        await obj.record_send_reqs(NS(reqs_to_send={'r': ('t', [])}, reqs_not_processed=()))
        meta = obj.reqs_need_send['t']
        assert not meta.ready.is_set()
        await obj.record_send_reqs(NS(reqs_to_send={'r': ('t', [[3], [5]])}, reqs_not_processed=()))
        assert meta.ready.is_set() and meta.local_block_ids == [[3], [5]]
        assert not await obj.fetch_finished_sending_reqs()
        await obj.record_send_reqs(NS(reqs_to_send={'cancelled': ('u', [])}, reqs_not_processed=()))
        await obj.record_send_reqs(NS(reqs_to_send={}, reqs_not_processed=('u',)))
        assert 'u' not in obj.reqs_need_send
        assert 't' in obj.reqs_need_send
    asyncio.run(replay())


@pytest.fixture
def native_constructor(monkeypatch):
    """Execute Core's complete constructor; stub device/transport/thread leaves."""
    calls = []
    class Engine:
        def initialize(self, *args):
            return 0
        def get_rpc_port(self):
            return 19000
    class Loop:
        def is_running(self):
            return False
        def stop(self):
            pass
        def call_soon_threadsafe(self, function):
            calls.append('loop stop')
    class Thread:
        def __init__(self, **kwargs):
            pass
        def start(self):
            pass
        def join(self, **kwargs):
            calls.append('thread join')
    class Executor:
        def __init__(self, **kwargs):
            pass
        def shutdown(self, **kwargs):
            calls.append('executor shutdown')
    class Bootstrap:
        def __init__(self, *args):
            pass
        def start(self):
            calls.append('bootstrap start')
        def shutdown(self):
            calls.append('bootstrap shutdown')
    monkeypatch.setattr(native, 'TransferEngine', Engine)
    monkeypatch.setattr(native.torch.accelerator, 'current_device_index', lambda: 0)
    monkeypatch.setattr(native, 'current_platform', NS(set_device=lambda x: None))
    monkeypatch.setattr(native, 'get_ip', lambda: '127.0.0.1')
    monkeypatch.setattr(native, 'get_tensor_model_parallel_rank', lambda: 0)
    monkeypatch.setattr(native, 'get_tensor_model_parallel_world_size', lambda: 1)
    monkeypatch.setattr(native, 'get_pp_group', lambda: NS(rank_in_group=0))
    monkeypatch.setattr(native.asyncio, 'new_event_loop', Loop)
    monkeypatch.setattr(native.threading, 'Thread', Thread)
    monkeypatch.setattr(native, 'ThreadPoolExecutor', Executor)
    monkeypatch.setattr(native, 'MooncakeBootstrapServer', Bootstrap)
    monkeypatch.setattr(native, 'get_mooncake_bootstrap_addr', lambda cfg: ('localhost', 19033))
    monkeypatch.setattr(native.MooncakeConnectorWorker, '_sync_block_size_with_kernel', lambda self: None)
    monkeypatch.setattr(native, 'get_current_attn_backends', lambda cfg: [])
    monkeypatch.setattr(native, 'TransferTopology', lambda **kwargs: NS(**kwargs))
    monkeypatch.setattr(native.zmq.asyncio, 'Context', lambda: NS(term=lambda: None))
    cfg = config(pcp=16)
    cfg.parallel_config.data_parallel_index = 0
    cfg.parallel_config.data_parallel_rank_local = 0
    cfg.parallel_config.local_engines_only = False
    cfg.kv_transfer_config.kv_connector_extra_config = {}
    cfg.kv_transfer_config.get_from_extra_config = lambda name, default: cfg.kv_transfer_config.kv_connector_extra_config.get(name, default)
    cfg.cache_config = NS(block_size=64)
    cfg.model_config.use_mla = True
    cfg.model_config.get_total_num_kv_heads = lambda: 1
    cache = NS(kv_cache_groups=[], transfer_groups=[], has_mamba_layers=False)
    return cfg, cache, calls


def test_real_native_constructor_starts_one_bootstrap_for_sixteen_pcp_ranks(monkeypatch, native_constructor):
    cfg, cache, calls = native_constructor
    original = native.MooncakeConnectorWorker.__init__
    workers = []
    try:
        for rank in range(16):
            monkeypatch.setattr(owned, 'get_pcp_group', lambda rank=rank: NS(world_size=16, rank_in_group=rank))
            workers.append(owned.HY4PCPMooncakeWorker(cfg, 'probe', cache))
        assert calls.count('bootstrap start') == 1
        assert native.MooncakeConnectorWorker.__init__ is original
        assert original.__globals__['should_launch_bootstrap_server'] is native.should_launch_bootstrap_server
    finally:
        for obj in workers:
            obj.__del__()
            obj._native_entered = False


def test_partial_native_constructor_cleans_sender_without_masking_error(monkeypatch, native_constructor):
    cfg, cache, calls = native_constructor
    monkeypatch.setattr(owned, 'get_pcp_group', lambda: NS(world_size=16, rank_in_group=0))
    def fail(self):
        raise OSError('bootstrap bind failed')
    monkeypatch.setattr(native.MooncakeBootstrapServer, 'start', fail)
    obj = object.__new__(owned.HY4PCPMooncakeWorker)
    with pytest.raises(OSError, match='bootstrap bind failed'):
        obj.__init__(cfg, 'probe', cache)
    obj.__del__()
    assert calls.count('executor shutdown') == 1
    assert calls.count('bootstrap shutdown') == 1
    assert calls.count('loop stop') == calls.count('thread join') == 1


def test_external_bootstrap_remains_external_with_pcp(monkeypatch, native_constructor):
    cfg, cache, calls = native_constructor
    cfg.kv_transfer_config.kv_connector_extra_config['bootstrap_server_address'] = 'localhost:19100'
    monkeypatch.setattr(owned, 'get_pcp_group', lambda: NS(world_size=16, rank_in_group=0))
    obj = owned.HY4PCPMooncakeWorker(cfg, 'probe', cache)
    obj.__del__()
    obj._native_entered = False
    assert 'bootstrap start' not in calls


def test_real_non_pcp_consumer_retains_receiver_constructor(monkeypatch, native_constructor):
    cfg, cache, calls = native_constructor
    cfg.parallel_config.prefill_context_parallel_size = 1
    cfg.kv_transfer_config.kv_role = 'kv_consumer'
    monkeypatch.setattr(owned, 'get_pcp_group', lambda: pytest.fail('PCP queried'))
    obj = owned.HY4PCPMooncakeWorker(cfg, 'consumer', cache)
    assert obj.is_kv_consumer and hasattr(obj, 'receiver_loop')
    obj.__del__()
    obj._native_entered = False
    assert 'bootstrap start' not in calls
