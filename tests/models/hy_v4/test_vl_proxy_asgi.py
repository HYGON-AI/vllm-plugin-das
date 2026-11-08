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
