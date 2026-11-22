# SPDX-License-Identifier: Apache-2.0
"""Real MRV2 lifecycle and producer event loop; no model or device setup."""
import asyncio
from contextlib import contextmanager
import queue
import threading
from types import SimpleNamespace as NS

import pytest

from hcu_vllm_das import hy4_pcp_mooncake as owned
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake import mooncake_connector as native
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.stats import MooncakeKVConnectorStats
from vllm.v1.worker.gpu import kv_connector as runner_kv


def metadata(*, completed=False, cancelled=False):
    result = native.MooncakeConnectorMetadata()
    if cancelled:
        result.reqs_not_processed.add('transfer')
    else:
        result.reqs_to_send['request'] = ('transfer', [[3], [5]] if completed else [])
    return result


@pytest.fixture
def lifecycle(monkeypatch):
    # Use complete native pre/post/no-forward methods; only replace the device
    # forward-context leaf, since producer submission does not consume it.
    monkeypatch.setattr(runner_kv, 'is_forward_context_available', lambda: True)
    monkeypatch.setattr(runner_kv, 'get_forward_context', lambda: NS())

    @contextmanager
    def create(*, canonical=True, producer=True, consumer=False, disabled=False):
        loop = asyncio.new_event_loop()
        thread = threading.Thread(target=loop.run_forever)
        thread.start()
        worker = object.__new__(owned.HY4PCPMooncakeWorker)
        worker._native_entered = False
        worker._canonical_pcp_sender = canonical
        worker._replica_pending = {}
        worker.is_kv_producer = producer
        worker.is_kv_consumer = consumer
        worker.sender_loop = worker.receiver_loop = loop
        worker.tp_rank = 0
        worker.reqs_need_send = {}
        worker.finished_sending_reqs = set()
        worker.finished_recving_reqs = set()
        worker._failed_recv_reqs = queue.Queue()
        worker._invalid_block_ids = queue.Queue()
        worker.xfer_stats = MooncakeKVConnectorStats()
        connector = object.__new__(owned.MooncakeConnector)
        connector.connector_worker = worker
        connector.connector_scheduler = None
        connector._connector_metadata = None
        runner = object.__new__(runner_kv.ActiveKVConnector)
        runner.kv_connector = connector
        runner._disabled = disabled
        runner._pending_load_kwargs = None
        calls = []
        original = worker.start_load_kv

        def start(data):
            calls.append(data)
            return original(data)

        worker.start_load_kv = start

        async def checkpoint():
            pass

        def settle():
            asyncio.run_coroutine_threadsafe(checkpoint(), loop).result(timeout=5)

        def record(data):
            asyncio.run_coroutine_threadsafe(worker.record_send_reqs(data), loop).result(timeout=5)

        try:
            yield NS(worker=worker, connector=connector, runner=runner, calls=calls,
                     settle=settle, record=record)
        finally:
            settle()
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=5)
            assert not thread.is_alive()
            loop.close()

    return create


def schedule(data, *, sync=False):
    return NS(kv_connector_metadata=data, has_sync_kv_loads=sync, finished_req_ids=set())


@pytest.mark.parametrize('canonical', [True, False])
def test_producer_ready_before_next_model_forward(lifecycle, canonical):
    with lifecycle(canonical=canonical) as state:
        state.record(metadata())
        data = metadata(completed=True)
        state.runner.pre_forward(schedule(data))
        state.settle()
        assert state.calls == [data], 'Finished P metadata waited until after next model forward'
        if canonical:
            send = state.worker.reqs_need_send['transfer']
            assert send.ready.is_set() and send.local_block_ids == [[3], [5]]
            assert not state.worker.finished_sending_reqs  # Real sender ACK still required.
        else:
            assert state.worker.finished_sending_reqs == {'request'}


@pytest.mark.parametrize('sync', [False, True])
def test_completed_send_is_not_resubmitted_after_forward(lifecycle, sync):
    with lifecycle() as state:
        state.record(metadata())
        data = metadata(completed=True)
        state.runner.pre_forward(schedule(data, sync=sync))
        state.settle()
        assert state.calls == [data]
        # Simulate native sender consuming this entry while next model runs.
        async def acknowledge():
            state.worker.reqs_need_send.pop('transfer')
            state.worker.finished_sending_reqs.add('request')
        asyncio.run_coroutine_threadsafe(acknowledge(), state.worker.sender_loop).result(timeout=5)
        state.runner.finish_forward()
        result = state.runner.post_forward(set())
        state.settle()
        assert state.calls == [data]
        assert result.finished_sending == {'request'}
        assert state.connector._connector_metadata is None


@pytest.mark.parametrize('canonical', [True, False])
def test_cancel_is_submitted_before_forward_without_false_completion(lifecycle, canonical):
    with lifecycle(canonical=canonical) as state:
        state.record(metadata())
        data = metadata(cancelled=True)
        state.runner.pre_forward(schedule(data))
        state.settle()
        assert state.calls == [data]
        assert not state.worker.reqs_need_send and not state.worker._replica_pending
        assert not state.worker.finished_sending_reqs
        state.runner.post_forward(set())
        assert state.calls == [data]


def test_no_forward_submits_once_and_reused_metadata_submits_next_step(lifecycle):
    with lifecycle() as state:
        data = metadata()
        for _ in range(2):
            state.runner.no_forward(schedule(data))
            state.settle()
        assert state.calls == [data, data]
        assert state.connector._connector_metadata is None


@pytest.mark.parametrize('producer,consumer', [(False, True), (False, False)])
def test_consumer_and_both_roles_retain_native_post_forward_submission(lifecycle, producer, consumer):
    with lifecycle(producer=producer, consumer=consumer) as state:
        data = native.MooncakeConnectorMetadata()
        state.runner.pre_forward(schedule(data))
        state.settle()
        assert not state.calls
        state.runner.post_forward(set())
        assert state.calls == [data]


def test_disabled_lifecycle_does_not_publish(lifecycle):
    with lifecycle(disabled=True) as state:
        state.runner.pre_forward(schedule(metadata()))
        state.runner.post_forward(set())
        assert not state.calls and state.connector._connector_metadata is None


def test_scheduler_binding_has_no_worker_submission():
    connector = object.__new__(owned.MooncakeConnector)
    connector.connector_worker = None
    data = native.MooncakeConnectorMetadata()
    connector.bind_connector_metadata(data)
    assert connector._connector_metadata is data
    connector.clear_connector_metadata()
    assert connector._connector_metadata is None


def test_missing_metadata_keeps_native_assertion(lifecycle):
    with lifecycle() as state:
        with pytest.raises(AssertionError):
            state.connector.start_load_kv(NS())


def test_submission_failure_propagates_without_marking_as_done(lifecycle, monkeypatch):
    with lifecycle() as state:
        def fail(data):
            raise OSError('submission failed')
        monkeypatch.setattr(state.worker, 'start_load_kv', fail)
        with pytest.raises(OSError, match='submission failed'):
            state.connector.bind_connector_metadata(metadata())
        state.connector.clear_connector_metadata()
