# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""OpenAI and Anthropic protocol feature coverage on one Qwen3 server."""

from __future__ import annotations

import json
from collections.abc import Iterator

import pytest

from tests.fixtures.resources import TestResources as HcuTestResources
from tests.integration.model_runtime import require_model_runtime
from tests.integration.server.openai_server import (
    OpenAIServer,
    serve_openai_protocol_model,
)


QWEN3_4B = "qwen3/Qwen3-4B"

pytestmark = [
    pytest.mark.hcu,
    pytest.mark.model,
    pytest.mark.hcu_count(1),
    pytest.mark.slow,
]


@pytest.fixture(scope="module")
def qwen3_protocol_server(
    hcu_test_resources: HcuTestResources,
) -> Iterator[OpenAIServer]:
    model_path = require_model_runtime(
        hcu_test_resources,
        env_name="VLLM_HCU_PROTOCOL_MODEL",
        relative_path=QWEN3_4B,
        label="Qwen3-4B protocol",
    )
    with serve_openai_protocol_model(
        model_path,
        enable_qwen3_parsers=True,
    ) as server:
        yield server


def _assert_success(response, server: OpenAIServer) -> dict:
    assert response.status == 200, (
        f"request failed: {response.body}; server_log={server.log_path}\n"
        f"server log tail:\n{server.log_tail()}"
    )
    return response.body


def test_openai_completion_request_controls(
    qwen3_protocol_server: OpenAIServer,
) -> None:
    server = qwen3_protocol_server
    body = _assert_success(
        server.post(
            "/v1/completions",
            {
                "model": server.model_name,
                "prompt": "Answer with one number: 2 + 2 =",
                "temperature": 0,
                "max_tokens": 4,
                "ignore_eos": True,
                "min_tokens": 4,
                "logprobs": 2,
            },
        ),
        server,
    )
    assert body["object"] == "text_completion"
    assert body["usage"]["completion_tokens"] == 4
    assert body["choices"][0]["finish_reason"] == "length"
    assert len(body["choices"][0]["logprobs"]["token_logprobs"]) == 4


def test_openai_chat_jinja_reasoning_and_logprobs(
    qwen3_protocol_server: OpenAIServer,
) -> None:
    server = qwen3_protocol_server
    body = _assert_success(
        server.post(
            "/v1/chat/completions",
            {
                "model": server.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": "Think briefly, then answer: what is 2 + 2?",
                    }
                ],
                "temperature": 0,
                "max_completion_tokens": 32,
                "logprobs": True,
                "top_logprobs": 2,
            },
        ),
        server,
    )
    choice = body["choices"][0]
    message = choice["message"]
    assert choice["logprobs"]["content"]
    assert "reasoning" in message
    assert isinstance(message.get("reasoning"), (str, type(None)))
    assert isinstance(message.get("content"), (str, type(None)))


def test_openai_seeded_top_k_top_p_sampling(
    qwen3_protocol_server: OpenAIServer,
) -> None:
    server = qwen3_protocol_server
    request = {
        "model": server.model_name,
        "messages": [
            {"role": "user", "content": "Name one color in one word."}
        ],
        "enable_thinking": False,
        "temperature": 0.7,
        "top_k": 8,
        "top_p": 0.8,
        "seed": 2026,
        "max_completion_tokens": 8,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    first = _assert_success(server.post("/v1/chat/completions", request), server)
    second = _assert_success(server.post("/v1/chat/completions", request), server)

    first_message = first["choices"][0]["message"]
    second_message = second["choices"][0]["message"]
    assert isinstance(first_message["content"], str)
    assert first_message["content"]
    assert first_message["content"] == second_message["content"]


def test_openai_metrics_endpoint_reports_request_observability(
    qwen3_protocol_server: OpenAIServer,
) -> None:
    server = qwen3_protocol_server
    response = server.get_text("/metrics")

    assert response.status == 200, (
        f"metrics request failed: {response.text[:1000]!r}; "
        f"server_log={server.log_path}"
    )
    assert "vllm:num_requests_running" in response.text
    assert "vllm:request_success" in response.text


def test_openai_chat_streaming_with_usage(
    qwen3_protocol_server: OpenAIServer,
) -> None:
    server = qwen3_protocol_server
    response = server.post_sse(
        "/v1/chat/completions",
        {
            "model": server.model_name,
            "messages": [
                {"role": "user", "content": "Answer with the number 4."}
            ],
            "temperature": 0,
            "max_completion_tokens": 16,
            "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )

    assert response.status == 200
    assert response.events[-1].data == "[DONE]"
    chunks = [event.data for event in response.events if isinstance(event.data, dict)]
    assert chunks
    assert all(chunk["object"] == "chat.completion.chunk" for chunk in chunks)
    assert any(
        choice.get("delta", {}).get("content")
        for chunk in chunks
        for choice in chunk.get("choices", [])
    )
    usage = [chunk.get("usage") for chunk in chunks if chunk.get("usage")]
    assert usage
    assert usage[-1]["prompt_tokens"] > 0
    assert usage[-1]["completion_tokens"] > 0


def test_openai_json_mode_and_json_schema(
    qwen3_protocol_server: OpenAIServer,
) -> None:
    server = qwen3_protocol_server
    json_object = _assert_success(
        server.post(
            "/v1/chat/completions",
            {
                "model": server.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            'Return JSON with the integer field "answer" '
                            "set to 4."
                        ),
                    }
                ],
                "temperature": 0,
                "max_completion_tokens": 32,
                "response_format": {"type": "json_object"},
                "chat_template_kwargs": {"enable_thinking": False},
            },
        ),
        server,
    )
    parsed_object = json.loads(json_object["choices"][0]["message"]["content"])
    assert isinstance(parsed_object, dict)

    schema = {
        "name": "arithmetic_answer",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"answer": {"type": "integer", "const": 4}},
            "required": ["answer"],
            "additionalProperties": False,
        },
    }
    json_schema = _assert_success(
        server.post(
            "/v1/chat/completions",
            {
                "model": server.model_name,
                "messages": [
                    {"role": "user", "content": "Return the answer to 2 + 2."}
                ],
                "temperature": 0,
                "max_completion_tokens": 32,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": schema,
                },
                "chat_template_kwargs": {"enable_thinking": False},
            },
        ),
        server,
    )
    parsed_schema = json.loads(json_schema["choices"][0]["message"]["content"])
    assert parsed_schema == {"answer": 4}


def test_openai_ebnf_structured_output(
    qwen3_protocol_server: OpenAIServer,
) -> None:
    server = qwen3_protocol_server
    body = _assert_success(
        server.post(
            "/v1/completions",
            {
                "model": server.model_name,
                "prompt": "Output exactly the three letters HCU:",
                "temperature": 0,
                "max_tokens": 3,
                "structured_outputs": {"grammar": 'root ::= "HCU"'},
            },
        ),
        server,
    )
    assert body["choices"][0]["text"].strip() == "HCU"


def test_openai_named_tool_choice(
    qwen3_protocol_server: OpenAIServer,
) -> None:
    server = qwen3_protocol_server
    tool = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
    body = _assert_success(
        server.post(
            "/v1/chat/completions",
            {
                "model": server.model_name,
                "messages": [
                    {"role": "user", "content": "What is the weather in Chengdu?"}
                ],
                "tools": [tool],
                "tool_choice": {
                    "type": "function",
                    "function": {"name": "get_weather"},
                },
                "temperature": 0,
                "max_completion_tokens": 64,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        ),
        server,
    )
    message = body["choices"][0]["message"]
    tool_calls = message.get("tool_calls")
    assert tool_calls, f"named tool call was not returned: {message}"
    assert len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == "get_weather"
    arguments = json.loads(tool_calls[0]["function"]["arguments"])
    assert isinstance(arguments.get("city"), str)


def test_anthropic_messages_protocol(
    qwen3_protocol_server: OpenAIServer,
) -> None:
    server = qwen3_protocol_server
    body = _assert_success(
        server.post(
            "/v1/messages",
            {
                "model": server.model_name,
                "max_tokens": 8,
                "temperature": 0,
                "messages": [
                    {"role": "user", "content": "Answer briefly: 2 + 2 = ?"}
                ],
            },
            headers={
                "x-api-key": "EMPTY",
                "anthropic-version": "2023-06-01",
            },
        ),
        server,
    )
    assert body["type"] == "message"
    assert body["role"] == "assistant"
    assert body["content"]
    assert body["usage"]["input_tokens"] > 0


def test_anthropic_messages_streaming(
    qwen3_protocol_server: OpenAIServer,
) -> None:
    server = qwen3_protocol_server
    response = server.post_sse(
        "/v1/messages",
        {
            "model": server.model_name,
            "max_tokens": 8,
            "temperature": 0,
            "stream": True,
            "messages": [
                {"role": "user", "content": "Answer with one number: 2 + 2."}
            ],
        },
        headers={
            "x-api-key": "EMPTY",
            "anthropic-version": "2023-06-01",
        },
    )

    assert response.status == 200
    event_types = [event.event for event in response.events]
    assert event_types[0] == "message_start"
    assert "content_block_delta" in event_types
    assert "message_delta" in event_types
    assert event_types[-1] == "message_stop"


def test_openai_request_length_rejection_and_truncation(
    qwen3_protocol_server: OpenAIServer,
) -> None:
    server = qwen3_protocol_server
    long_prompt = "long request token " * 800
    rejected = server.post(
        "/v1/completions",
        {
            "model": server.model_name,
            "prompt": long_prompt,
            "temperature": 0,
            "max_tokens": 1,
        },
    )
    assert rejected.status in {400, 413}
    assert "error" in rejected.body

    truncated = _assert_success(
        server.post(
            "/v1/completions",
            {
                "model": server.model_name,
                "prompt": long_prompt,
                "temperature": 0,
                "max_tokens": 1,
                "truncate_prompt_tokens": 128,
            },
        ),
        server,
    )
    assert truncated["usage"]["prompt_tokens"] <= 128
