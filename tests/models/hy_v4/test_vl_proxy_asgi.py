# SPDX-License-Identifier: Apache-2.0
"""Real ASGI response lifecycle with controlled upstreams, without sockets."""

import asyncio
import importlib.util
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[3] / "tools/hy_v4/vl_pd_validation"
HAS_DEPS = all(importlib.util.find_spec(name) for name in ("fastapi", "httpx"))


@unittest.skipUnless(HAS_DEPS, "FastAPI/httpx required for ASGI boundary tests")
class ManagedProxyASGITests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        spec = importlib.util.spec_from_file_location(
            "managed_proxy_test", HERE / "managed_proxy_ka1.py"
        )
        self.module = importlib.util.module_from_spec(spec)
        sys.path.insert(0, str(HERE))
        try:
            spec.loader.exec_module(self.module)
        finally:
            sys.path.pop(0)
        self.module.proxy.app.state.ready = asyncio.Event()
        self.module.proxy.app.state.ready.set()
        self.module.proxy.get_next_client = lambda app, role: (
            ({}, 0) if role == "prefill" else {}
        )
        self.started = asyncio.Event()
        self.prefill_closed = asyncio.Event()
        self.decode_started = asyncio.Event()
        self.decode_closed = asyncio.Event()

        async def prefill(*args):
            self.started.set()
            try:
                await asyncio.Event().wait()
            finally:
                self.prefill_closed.set()

        async def decode(*args):
            self.decode_started.set()
            try:
                yield b"first"
                await asyncio.Event().wait()
            finally:
                self.decode_closed.set()

        self.module.proxy.send_request_to_service = prefill
        self.module.proxy.stream_service_response = decode

    async def response(self):
        async def json():
            return {"stream": True}

        request = SimpleNamespace(app=self.module.proxy.app, json=json)
        return await self.module.managed_handle("/v1/chat/completions", request)

    async def test_asgi_send_disconnect_closes_both_upstreams(self):
        from starlette.requests import ClientDisconnect

        response = await self.response()

        async def receive():
            self.fail("ASGI2.4 must report disconnect through send")

        async def send(message):
            if message["type"] == "http.response.body":
                raise OSError("client disconnected while sending")

        with self.assertRaises(ClientDisconnect):
            await response(
                {"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send
            )
        self.assertTrue(self.started.is_set())
        self.assertTrue(self.prefill_closed.is_set())
        self.assertTrue(self.decode_closed.is_set())

    async def test_asgi_receive_disconnect_closes_both_upstreams(self):
        response = await self.response()

        async def receive():
            await self.started.wait()
            await self.decode_started.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            pass

        await response({"type": "http", "asgi": {"spec_version": "2.0"}}, receive, send)
        self.assertTrue(self.prefill_closed.is_set())
        self.assertTrue(self.decode_closed.is_set())

    async def test_invalid_deadline_fails_at_startup(self):
        for value in ("0", "-1", "inf", "nan"):
            spec = importlib.util.spec_from_file_location(
                "invalid_proxy_test", HERE / "managed_proxy_ka1.py"
            )
            candidate = importlib.util.module_from_spec(spec)
            with patch.dict(os.environ, HY4_PD_PROXY_TIMEOUT_SECONDS=value):
                sys.path.insert(0, str(HERE))
                try:
                    with self.assertRaises(ValueError):
                        spec.loader.exec_module(candidate)
                finally:
                    sys.path.pop(0)

    async def _bootstrap_recovers(self, failure):
        import httpx

        original_client = httpx.AsyncClient
        queries = 0
        clients = []

        async def upstream(request):
            nonlocal queries
            if request.url.path == "/query":
                queries += 1
                if queries == 1:
                    if failure == "503":
                        return httpx.Response(503)
                    if failure == "connect":
                        raise httpx.ConnectError("bootstrap starting", request=request)
                    if failure == "json":
                        return httpx.Response(200, content=b"not json")
                    if failure == "ranks":
                        return httpx.Response(
                            200, json={"1": {"engine_id": "producer-1"}}
                        )
                    if failure == "id":
                        return httpx.Response(200, json={"0": {"engine_id": ""}})
                    return httpx.Response(200, json={})
                return httpx.Response(200, json={"0": {"engine_id": "producer-1"}})
            return httpx.Response(200)

        def client_factory(**kwargs):
            client = original_client(**kwargs, transport=httpx.MockTransport(upstream))
            clients.append(client)
            return client

        self.module.proxy.global_args = SimpleNamespace(
            prefill=[("http://prefill:8000", 8998)], decode=["http://decode:8000"]
        )
        loop = asyncio.get_running_loop()
        old_handler = loop.get_exception_handler()
        errors = []
        loop.set_exception_handler(lambda loop, context: errors.append(context))
        try:
            with patch.object(httpx, "AsyncClient", client_factory):
                async with self.module.proxy.app.router.lifespan_context(
                    self.module.proxy.app
                ):
                    try:
                        await asyncio.wait_for(
                            self.module.proxy.app.state.ready.wait(), timeout=2
                        )
                    except asyncio.TimeoutError:  # noqa: UP041 - distinct on Python 3.10
                        self.fail(
                            "proxy never recovered after bootstrap became healthy"
                        )
                    info = self.module.proxy.app.state.prefill_clients[0]
                    self.assertEqual(info["dp_engine_id"], {0: "producer-1"})
                    self.assertEqual(info["dp_size"], 1)
                    async with original_client(
                        transport=httpx.ASGITransport(app=self.module.proxy.app),
                        base_url="http://proxy",
                    ) as caller:
                        self.assertEqual((await caller.get("/health")).status_code, 200)
            self.assertFalse(errors, "initializer must not fail as an unowned task")
            self.assertTrue(all(client.is_closed for client in clients))
        finally:
            loop.set_exception_handler(old_handler)

    async def test_bootstrap_recovers_after_http503(self):
        await self._bootstrap_recovers("503")

    async def test_bootstrap_recovers_after_connect_failure(self):
        await self._bootstrap_recovers("connect")

    async def test_bootstrap_recovers_after_invalid_json(self):
        await self._bootstrap_recovers("json")

    async def test_bootstrap_waits_for_nonempty_engine_map(self):
        await self._bootstrap_recovers("empty")

    async def test_bootstrap_waits_for_contiguous_rank_map(self):
        await self._bootstrap_recovers("ranks")

    async def test_bootstrap_waits_for_engine_id(self):
        await self._bootstrap_recovers("id")

    async def test_shutdown_cancels_and_joins_pending_bootstrap(self):
        import httpx

        original_client = httpx.AsyncClient
        started, closed = asyncio.Event(), asyncio.Event()

        async def upstream(request):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                closed.set()

        def client_factory(**kwargs):
            return original_client(**kwargs, transport=httpx.MockTransport(upstream))

        self.module.proxy.global_args = SimpleNamespace(
            prefill=[("http://prefill:8000", 8998)], decode=["http://decode:8000"]
        )
        existing = asyncio.all_tasks()
        try:
            with patch.object(httpx, "AsyncClient", client_factory):
                async with self.module.proxy.app.router.lifespan_context(
                    self.module.proxy.app
                ):
                    await asyncio.wait_for(started.wait(), timeout=1)
            self.assertTrue(
                closed.is_set(), "bootstrap request survived proxy shutdown"
            )
            self.assertFalse(self.module.proxy.app.state.ready.is_set())
            self.assertFalse(asyncio.all_tasks() - existing)
        finally:
            leaked = asyncio.all_tasks() - existing
            for task in leaked:
                task.cancel()
            await asyncio.gather(*leaked, return_exceptions=True)
