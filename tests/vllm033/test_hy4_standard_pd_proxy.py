# SPDX-License-Identifier: Apache-2.0
"""Replay the unchanged canonical proxy with actual HTTPX request serialization."""

import asyncio
import importlib.util
import json
import os
from pathlib import Path

import httpx
import pytest


def test_canonical_proxy_preserves_mm_metadata_and_original_prompt():
    path = os.environ.get("HY4_PD_PROXY_PATH")
    if not path:
        pytest.skip("Set HY4_PD_PROXY_PATH to the frozen canonical proxy.py")
    spec = importlib.util.spec_from_file_location("canonical_hy4_pd_proxy", Path(path))
    proxy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(proxy)
    data = {
        "model": "/models/hy4",
        "max_tokens": 1200,
        "max_completion_tokens": 1200,
        "stream": True,
        "stream_options": {"include_usage": True},
        "cache_salt": "same-mm-salt",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Which side is red?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,aW1hZ2U="},
                    },
                ],
            }
        ],
        "mm_processor_kwargs": {"min_pixels": 256, "max_pixels": 4096},
        "multi_modal_uuids": {"image": ["stable-image-uuid"]},
    }
    captured = {}

    def transport(req):
        captured[req.url.host] = (json.loads(req.content), dict(req.headers))
        return httpx.Response(200, content=b"data: [DONE]\n\n")

    async def replay():
        async with (
            httpx.AsyncClient(
                base_url="http://producer", transport=httpx.MockTransport(transport)
            ) as p,
            httpx.AsyncClient(
                base_url="http://decoder", transport=httpx.MockTransport(transport)
            ) as d,
        ):
            producer = {
                "client": p,
                "bootstrap_addr": "http://producer:19033",
                "dp_engine_id": {0: "p-engine"},
            }
            decoder = {"client": d}
            chunks = [
                item
                async for item in proxy.owned_pd_stream(
                    lambda: proxy.send_request_to_service(
                        producer, 0, "/v1/chat/completions", data, "shared-uuid"
                    ),
                    lambda: proxy.stream_service_response(
                        producer,
                        0,
                        decoder,
                        "/v1/chat/completions",
                        data.copy(),
                        "shared-uuid",
                    ),
                )
            ]
            assert chunks == [b"data: [DONE]\n\n"]

    asyncio.run(replay())
    p, ph = captured["producer"]
    d, dh = captured["decoder"]
    for field in ("messages", "cache_salt", "mm_processor_kwargs", "multi_modal_uuids"):
        assert p[field] == d[field] == data[field]
    assert p["max_tokens"] == p["max_completion_tokens"] == 1
    assert not p["stream"] and "stream_options" not in p
    assert d["max_tokens"] == d["max_completion_tokens"] == 1200 and d["stream"]
    assert d["stream_options"] == data["stream_options"]
    assert ph["x-request-id"] == dh["x-request-id"] == "shared-uuid"
    assert p["kv_transfer_params"] == {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "transfer_id": "xfer-shared-uuid",
    }
    assert d["kv_transfer_params"] == {
        "do_remote_decode": False,
        "do_remote_prefill": True,
        "transfer_id": "xfer-shared-uuid",
        "remote_bootstrap_addr": "http://producer:19033",
        "remote_engine_id": "p-engine",
    }
    assert "kv_transfer_params" not in data and data["max_tokens"] == 1200
