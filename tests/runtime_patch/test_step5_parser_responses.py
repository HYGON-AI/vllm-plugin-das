# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""P2 regressions for the Step5 Responses tool-call path.

P2-1: a Responses named tool choice (``ToolChoiceFunction``) used to raise
``AttributeError`` because the parser read ``.function.name`` from the Chat
Completions schema.

P2-2: Responses ``FunctionTool`` entries are flat (``name``/``parameters`` on
the tool), while the inherited XML parser reads ``tool.function.parameters``.
Without normalization every argument was typed as a string.

The tokenizer is a stub: the Step5 parsers only read ``get_vocab()`` for the
``<think>`` markers. Chat Completions cases stay in this file as guards so a
future change cannot fix Responses by regressing the existing schema.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

pytest.importorskip("vllm", reason="requires the audited vLLM runtime")

from openai.types.responses import FunctionTool  # noqa: E402
from openai.types.responses.response import ToolChoiceFunction  # noqa: E402
from vllm.entrypoints.openai.chat_completion.protocol import (  # noqa: E402
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.responses.protocol import (  # noqa: E402
    ResponsesRequest,
)
from vllm.parser.parser_manager import ParserManager  # noqa: E402

from vllm_hcu.patch.platform.core_fix import (  # noqa: E402
    patch_step5_reasoning_parser,
    patch_step5_tool_parser,
)

THINK_START_ID = 128798
THINK_END_ID = 128799

PARAMETERS = {
    "type": "object",
    "properties": {
        "city": {"type": "string"},
        "days": {"type": "integer"},
        "metric": {"type": "boolean"},
        "tags": {"type": "array", "items": {"type": "string"}},
        "options": {"type": "object"},
    },
    "required": ["city"],
}

CHAT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city",
            "parameters": PARAMETERS,
        },
    }
]

RESP_TOOLS = [
    FunctionTool(
        type="function",
        name="get_weather",
        description="Get the current weather for a city",
        parameters=PARAMETERS,
    )
]

NAMED_CHAT = {"type": "function", "function": {"name": "get_weather"}}
NAMED_RESP = ToolChoiceFunction(type="function", name="get_weather")
MISMATCHED_RESP = ToolChoiceFunction(type="function", name="get_time")

OUT_TYPED = "I should call a tool\n</think>" + (
    "<tool_call>\n<function=get_weather>\n"
    "<parameter=city>\nBeijing\n</parameter>\n"
    "<parameter=days>\n3\n</parameter>\n"
    "<parameter=metric>\ntrue\n</parameter>\n"
    "<parameter=tags>\n[\"a\", \"b\"]\n</parameter>\n"
    "<parameter=options>\n{\"unit\": \"C\"}\n</parameter>\n"
    "</function>\n</tool_call>"
)

OUT_NO_TOOL = "I should call a tool\n</think>The weather is nice today."

TYPED_ARGUMENTS = {
    "city": "Beijing",
    "days": 3,
    "metric": True,
    "tags": ["a", "b"],
    "options": {"unit": "C"},
}

# Omit tool_choice entirely for the Responses API: its schema rejects None,
# and the server default ("auto") is the case we want to cover.
OMIT_TOOL_CHOICE = object()


@pytest.fixture(scope="module", autouse=True)
def _apply_step5_registrations():
    assert patch_step5_reasoning_parser.apply() in (True, False)
    assert patch_step5_tool_parser.apply() in (True, False)
    yield


def _tokenizer() -> SimpleNamespace:
    return SimpleNamespace(
        get_vocab=lambda: {"<think>": THINK_START_ID, "</think>": THINK_END_ID}
    )


def _build(tools, tool_choice, request_cls):
    """Compose the parser exactly as the OpenAI serving layer does."""
    kwargs = {} if tool_choice is OMIT_TOOL_CHOICE else {"tool_choice": tool_choice}
    if request_cls is ChatCompletionRequest:
        request = ChatCompletionRequest(
            model="step5-test",
            messages=[{"role": "user", "content": "weather in Beijing?"}],
            tools=tools,
            **kwargs,
        )
    else:
        request = ResponsesRequest(
            model="step5-test",
            input="weather in Beijing?",
            tools=tools,
            **kwargs,
        )
    parser_cls = ParserManager.get_parser(
        tool_parser_name="step5",
        reasoning_parser_name="step5",
        enable_auto_tools=True,
        model_name="step5",
    )
    assert parser_cls is not None
    return parser_cls(_tokenizer(), request.tools, chat_template_kwargs={}), request


def _parse(parser, request, model_output: str = OUT_TYPED):
    return parser.parse(model_output, request, enable_auto_tools=True)


def _single_call(result):
    _, content, tool_calls = result
    assert content is None
    assert tool_calls is not None
    assert len(tool_calls) == 1
    return tool_calls[0]


def _typed_call(result):
    call = _single_call(result)
    return call.name, json.loads(call.arguments)


# --- P2-1: Responses named tool choice ---------------------------------------


def test_responses_named_tool_choice_returns_typed_call() -> None:
    parser, request = _build(RESP_TOOLS, NAMED_RESP, ResponsesRequest)

    name, arguments = _typed_call(_parse(parser, request))

    assert name == "get_weather"
    assert arguments == TYPED_ARGUMENTS


def test_responses_named_tool_choice_mismatch_returns_generated_call(caplog) -> None:
    parser, request = _build(RESP_TOOLS, MISMATCHED_RESP, ResponsesRequest)

    name, arguments = _typed_call(_parse(parser, request))

    assert name == "get_weather"
    assert arguments == TYPED_ARGUMENTS
    assert "asked for function 'get_time'" in caplog.text


# --- P2-2: Responses flat FunctionTool schema --------------------------------


@pytest.mark.parametrize(
    "tool_choice",
    [
        pytest.param("auto", id="auto"),
        pytest.param("required", id="required"),
        pytest.param(OMIT_TOOL_CHOICE, id="default"),
    ],
)
def test_responses_flat_tools_keep_schema_types(tool_choice) -> None:
    parser, request = _build(RESP_TOOLS, tool_choice, ResponsesRequest)

    _, arguments = _typed_call(_parse(parser, request))

    assert arguments == TYPED_ARGUMENTS
    assert type(arguments["days"]) is int
    assert arguments["metric"] is True
    assert isinstance(arguments["tags"], list)
    assert isinstance(arguments["options"], dict)


def test_responses_streaming_keeps_schema_types() -> None:
    parser, request = _build(RESP_TOOLS, NAMED_RESP, ResponsesRequest)
    tool_text = OUT_TYPED.split("</think>", 1)[1]

    delta = parser.tool_parser.extract_tool_calls_streaming(
        previous_text="",
        current_text=tool_text,
        delta_text=tool_text,
        previous_token_ids=[],
        current_token_ids=[],
        delta_token_ids=[],
        request=request,
    )

    assert delta is not None
    assert [call.function.name for call in delta.tool_calls] == ["get_weather"]
    assert json.loads(delta.tool_calls[0].function.arguments) == TYPED_ARGUMENTS


# --- Chat Completions compatibility ------------------------------------------


@pytest.mark.parametrize(
    "tool_choice",
    [
        pytest.param("auto", id="auto"),
        pytest.param("required", id="required"),
        pytest.param(NAMED_CHAT, id="named"),
        pytest.param(None, id="none"),
    ],
)
def test_chat_completions_keep_schema_types(tool_choice) -> None:
    parser, request = _build(CHAT_TOOLS, tool_choice, ChatCompletionRequest)

    name, arguments = _typed_call(_parse(parser, request))

    assert name == "get_weather"
    assert arguments == TYPED_ARGUMENTS


# --- generation-side request adjustment --------------------------------------


@pytest.mark.parametrize(
    "tool_choice",
    [
        pytest.param("required", id="required"),
        pytest.param(NAMED_RESP, id="named"),
    ],
)
def test_responses_required_and_named_do_not_install_json_grammar(tool_choice) -> None:
    parser, request = _build(RESP_TOOLS, tool_choice, ResponsesRequest)

    parser.adjust_request(request)

    assert request.text is None
    assert getattr(request, "structured_outputs", None) is None


def test_responses_auto_does_not_install_json_grammar() -> None:
    parser, request = _build(RESP_TOOLS, "auto", ResponsesRequest)

    parser.adjust_request(request)

    assert request.text is None
    assert getattr(request, "structured_outputs", None) is None


# --- malformed and unsupported inputs stay safe ------------------------------


@pytest.mark.parametrize(
    "tool_choice",
    [pytest.param("auto", id="auto"), pytest.param("required", id="required")],
)
def test_responses_plain_text_is_not_turned_into_a_call(tool_choice) -> None:
    parser, request = _build(RESP_TOOLS, tool_choice, ResponsesRequest)

    _, content, tool_calls = _parse(parser, request, OUT_NO_TOOL)

    assert not tool_calls
    assert content == "The weather is nice today."


def test_responses_without_tools_is_safe() -> None:
    parser, request = _build([], "auto", ResponsesRequest)

    _, content, tool_calls = _parse(parser, request)

    assert not tool_calls
    assert content and "<tool_call>" in content


def test_responses_malformed_integer_does_not_raise() -> None:
    parser, request = _build(RESP_TOOLS, "auto", ResponsesRequest)
    output = OUT_TYPED.replace(
        "<parameter=days>\n3\n</parameter>",
        "<parameter=days>\nnot-a-number\n</parameter>",
    )

    call = _single_call(_parse(parser, request, output))

    assert call.name == "get_weather"
    assert "not-a-number" in call.arguments


def test_responses_unclosed_tool_call_does_not_raise() -> None:
    parser, request = _build(RESP_TOOLS, "auto", ResponsesRequest)

    call = _single_call(
        _parse(
            parser,
            request,
            "I should call a tool\n</think>"
            "<tool_call><function=get_weather><parameter=city>Beijing",
        )
    )

    assert call.name == "get_weather"
    assert "Beijing" in call.arguments


def test_responses_unknown_parameter_does_not_raise() -> None:
    parser, request = _build(RESP_TOOLS, "auto", ResponsesRequest)
    output = OUT_TYPED.replace("<parameter=city>", "<parameter=unknown>")

    call = _single_call(_parse(parser, request, output))

    assert call.name == "get_weather"
    assert "unknown" in call.arguments


def test_responses_schema_without_properties_falls_back_to_string() -> None:
    tools = [
        FunctionTool(
            type="function",
            name="get_weather",
            parameters={"type": "object"},
        )
    ]
    parser, request = _build(tools, "auto", ResponsesRequest)

    _, arguments = _typed_call(_parse(parser, request))

    assert arguments["days"] == "3"
    assert arguments["metric"] == "true"
    assert arguments["tags"] == '["a", "b"]'
    assert arguments["options"] == '{"unit": "C"}'
