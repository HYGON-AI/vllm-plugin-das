# SPDX-License-Identifier: Apache-2.0
"""Exercise Core's actual sender queue/readiness/ACK flow on CPU."""

import asyncio
from types import SimpleNamespace as NS

import pytest

from hcu_vllm_das import hy4_pcp_mooncake as owned
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake import (
    mooncake_connector as native,
)


class ReadyEvent(asyncio.Event):
    def __init__(self):
        super().__init__()
        self.wait_started = asyncio.Event()

    async def wait(self):
        self.wait_started.set()
        return await super().wait()


@pytest.mark.parametrize("producer", [True, False])
def test_progress_policy_is_published_before_native_threads_start(
    monkeypatch, producer
):
    cfg = NS(
        model_config=NS(
            hf_config=NS(model_type="hy_v4", hy4_standard_vl=True),
            enforce_eager=True,
        ),
        parallel_config=NS(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=16 if producer else 1,
            data_parallel_size=1 if producer else 16,
            data_parallel_rank=0,
            enable_expert_parallel=True,
            all2all_backend="deepep_high_throughput",
        ),
        kv_transfer_config=NS(
            kv_connector="MooncakeConnector",
            kv_connector_module_path="hcu_vllm_das.hy4_pcp_mooncake",
            kv_role="kv_producer" if producer else "kv_consumer",
        ),
    )
    seen = []

    def initialize(self, *args):
        seen.append((self._ready_sends_enabled, len(self._native_send_tasks)))
        assert self._ready_send_slots is None

    monkeypatch.setattr(owned, "_init_native", initialize)
    monkeypatch.setattr(
        owned, "get_pcp_group", lambda: NS(world_size=16, rank_in_group=0)
    )
    monkeypatch.setattr(native.MooncakeConnectorWorker, "shutdown", lambda _: None)
    obj = owned.HY4PCPMooncakeWorker(cfg, "e", NS())
    assert seen == [(producer, 0)] and obj._native_entered


def make_worker():
    obj = object.__new__(owned.HY4PCPMooncakeWorker)
    obj._native_entered = False
    obj._canonical_pcp_sender = True
    obj._ready_sends_enabled = True
    obj._native_send_tasks = set()
    obj.num_sender_tasks = 2
    obj._ready_send_slots = asyncio.Semaphore(obj.num_sender_tasks)
    obj.sender_worker_queue = asyncio.Queue()
    obj._xfer_meta_decoder = NS(decode=lambda value: value)
    obj._encoder = NS(encode=lambda value: value)
    obj.transfer_topo = NS(handshake_target_ranks=lambda _: [0])
    obj.tp_rank = 0
    obj.tp_size = obj.pp_size = 1
    obj.use_mla = True
    obj.opaque_packed_storages = ()
    obj.kv_cache_config = NS(has_mamba_layers=False)
    obj._lookup_peer_layout = lambda _: ([], [])
    obj._producer_cache_is_replicated = lambda: True
    obj.reqs_need_send = {}
    obj.finished_sending_reqs = set()
    obj.sender_loop = asyncio.get_running_loop()
    obj.built = []

    async def build(ready, *args):
        obj.built.extend(key for key, _ in ready)
        # Memory movement is a leaf double; queue/readiness/native ACKs are real.
        return [], [], [], [], None

    obj._build_transfer_params = build
    return obj


def envelope(obj, key, *, ready):
    event = ReadyEvent()
    if ready:
        event.set()
    obj.reqs_need_send[key] = native.SendBlockMeta(
        p_req_id=key,
        transfer_id=key,
        local_block_ids=[[1]],
        ready=event,
    )
    return NS(
        remote_tp_rank=0,
        remote_tp_size=1,
        remote_pp_size=1,
        layout_id=1,
        kv_caches_base_addr=[],
        registered_row_offsets=[],
        kv_block_lens=[],
        block_lens=[],
        req_blocks={key: (key, [[1]])},
    )


def test_ready_transfer_progresses_when_all_native_dispatchers_wait_for_other_prompts():
    async def replay():
        obj = make_worker()
        responses = []
        ack = asyncio.Event()

        async def send(parts):
            responses.append(parts[1])
            if parts[1].ok_reqs == ["ready"]:
                ack.set()

        sock = NS(send_multipart=send)
        dispatchers = [
            asyncio.create_task(obj._sender_worker(sock))
            for _ in range(obj.num_sender_tasks)
        ]
        try:
            for key in ("waiting0", "waiting1"):
                await obj.sender_worker_queue.put(
                    (b"peer", envelope(obj, key, ready=False))
                )
            for key in ("waiting0", "waiting1"):
                await asyncio.wait_for(
                    obj.reqs_need_send[key].ready.wait_started.wait(), 1
                )
            # P has computed this later prompt and holds its KV. It must be
            # sent/ACKed without waiting for the earlier uncomputed prompts.
            await obj.sender_worker_queue.put(
                (b"peer", envelope(obj, "ready", ready=True))
            )
            await asyncio.wait_for(ack.wait(), 1)
            assert obj.built == ["ready"]
            assert obj.finished_sending_reqs == {"ready"}
            assert set(obj.reqs_need_send) == {"waiting0", "waiting1"}
            assert responses[-1].status == native.MooncakeXferResponseStatus.FINISH
        finally:
            # The isolated event loop also owns Core's readiness child tasks.
            for task in dispatchers:
                task.cancel()
            pending = asyncio.all_tasks() - {asyncio.current_task()}
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

    asyncio.run(replay())


def test_ready_work_retains_native_concurrency_bound():
    async def replay():
        obj = make_worker()
        release = asyncio.Event()
        saturated = asyncio.Event()
        active = peak = 0

        async def build(ready, *args):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            if active == obj.num_sender_tasks:
                saturated.set()
            await release.wait()
            active -= 1
            return [], [], [], [], None

        async def send(parts):
            pass

        obj._build_transfer_params = build
        sock = NS(send_multipart=send)
        try:
            for i in range(5):
                await obj.send_kv_to_decode(
                    b"peer", sock, envelope(obj, str(i), ready=True)
                )
            await asyncio.wait_for(saturated.wait(), 1)
            for _ in range(10):
                await asyncio.sleep(0)
            assert active == peak == obj.num_sender_tasks
            release.set()
            await asyncio.wait_for(asyncio.gather(*obj._native_send_tasks), 1)
            assert peak == 2 and obj.finished_sending_reqs == set(map(str, range(5)))
            assert obj._ready_send_slots._value == 2
        finally:
            pending = asyncio.all_tasks() - {asyncio.current_task()}
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

    asyncio.run(replay())


def test_listener_shutdown_cancels_native_readiness_children(monkeypatch):
    async def listener(self, ready):
        meta = envelope(self, "waiting", ready=False)
        await self.send_kv_to_decode(b"peer", NS(), meta)
        await self.reqs_need_send["waiting"].ready.wait_started.wait()
        ready.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(
        native.MooncakeConnectorWorker, "_mooncake_sender_listener", listener
    )

    async def replay():
        obj = make_worker()
        ready = asyncio.Event()
        task = asyncio.create_task(obj._mooncake_sender_listener(ready))
        await asyncio.wait_for(ready.wait(), 1)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await asyncio.sleep(0)
        assert not obj._native_send_tasks and not obj.finished_sending_reqs
        assert asyncio.all_tasks() == {asyncio.current_task()}

    asyncio.run(replay())


def test_other_roles_keep_native_send(monkeypatch):
    calls = []

    async def original(self, *args):
        calls.append(args)

    async def replay():
        obj = make_worker()
        obj._ready_sends_enabled = False
        monkeypatch.setattr(
            native.MooncakeConnectorWorker, "send_kv_to_decode", original
        )
        await obj.send_kv_to_decode("peer", "sock", "meta")
        assert calls == [("peer", "sock", "meta")] and not obj._native_send_tasks

    asyncio.run(replay())


def test_native_send_error_is_not_reported_as_ack():
    async def replay():
        obj = make_worker()
        responses = []

        async def build(*args):
            raise ValueError("transfer preparation failed")

        async def send(parts):
            responses.append(parts[1])

        obj._build_transfer_params = build
        await obj.send_kv_to_decode(
            b"peer", NS(send_multipart=send), envelope(obj, "r", ready=True)
        )
        await asyncio.wait_for(asyncio.gather(*obj._native_send_tasks), 1)
        assert len(responses) == 1
        assert responses[0].status == native.MooncakeXferResponseStatus.ERROR
        assert responses[0].err_msg == "transfer preparation failed"
        assert not obj.finished_sending_reqs
        assert obj._ready_send_slots._value == 2

    asyncio.run(replay())
