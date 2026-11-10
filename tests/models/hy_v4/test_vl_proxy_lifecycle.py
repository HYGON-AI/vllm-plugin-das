# SPDX-License-Identifier: Apache-2.0
"""No sockets: exercise ownership of concurrent P and D operations."""

import asyncio
import importlib.util
import unittest
from pathlib import Path

PATH = (
    Path(__file__).resolve().parents[3]
    / "tools/hy_v4/vl_pd_validation/proxy_task_lifecycle.py"
)
module = None
if PATH.exists():
    spec = importlib.util.spec_from_file_location("proxy_task_lifecycle", PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


class ProxyLifecycleTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.assertIsNotNone(module, "owned P/D stream implementation required")

    async def test_prefill_failure_closes_decode_and_propagates(self):
        closed = asyncio.Event()

        async def prefill():
            raise RuntimeError("prefill HTTP503")

        async def decode():
            try:
                await asyncio.Event().wait()
                yield b"unreachable"
            finally:
                closed.set()

        with self.assertRaisesRegex(RuntimeError, "prefill HTTP503"):
            async for _ in module.owned_pd_stream(prefill, decode, timeout=1):
                self.fail("decode must not produce data after P fails")
        self.assertTrue(closed.is_set())

    async def test_decode_failure_cancels_prefill(self):
        cancelled = asyncio.Event()

        async def prefill():
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        async def decode():
            raise RuntimeError("decode failed")
            yield b"unreachable"

        with self.assertRaisesRegex(RuntimeError, "decode failed"):
            async for _ in module.owned_pd_stream(prefill, decode, timeout=1):
                pass
        self.assertTrue(cancelled.is_set())

    async def test_closing_downstream_cancels_pending_prefill(self):
        cancelled = asyncio.Event()
        closed = asyncio.Event()

        async def prefill():
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        async def decode():
            try:
                yield b"first"
                await asyncio.Event().wait()
            finally:
                closed.set()

        stream = module.owned_pd_stream(prefill, decode, timeout=1)
        self.assertEqual(await stream.__anext__(), b"first")
        await stream.aclose()
        self.assertTrue(cancelled.is_set())
        self.assertTrue(closed.is_set())

    async def test_timeout_cancels_both_upstreams(self):
        closed = []

        async def prefill():
            try:
                await asyncio.Event().wait()
            finally:
                closed.append("P")

        async def decode():
            try:
                await asyncio.Event().wait()
                yield b"unreachable"
            finally:
                closed.append("D")

        with self.assertRaises(asyncio.TimeoutError):
            async for _ in module.owned_pd_stream(prefill, decode, timeout=0.02):
                pass
        self.assertEqual(set(closed), {"P", "D"})

    async def test_decode_finishing_first_still_waits_for_prefill(self):
        complete = asyncio.Event()

        async def prefill():
            await asyncio.sleep(0)
            complete.set()

        async def decode():
            yield b"first"
            yield b"last"

        chunks = [
            chunk async for chunk in module.owned_pd_stream(prefill, decode, timeout=1)
        ]
        self.assertEqual(chunks, [b"first", b"last"])
        self.assertTrue(complete.is_set())

    async def test_prefill_success_does_not_interrupt_decode(self):
        async def prefill():
            return None

        async def decode():
            await asyncio.sleep(0)
            yield b"answer"

        chunks = [
            chunk async for chunk in module.owned_pd_stream(prefill, decode, timeout=1)
        ]
        self.assertEqual(chunks, [b"answer"])

    async def test_invalid_timeout_rejected_before_starting_upstreams(self):
        def unexpected():
            self.fail("upstream must not start")

        for timeout in [0, -1, float("inf"), float("nan")]:
            with self.assertRaises(ValueError):
                async for _ in module.owned_pd_stream(
                    unexpected, unexpected, timeout=timeout
                ):
                    pass


if __name__ == "__main__":
    unittest.main()
