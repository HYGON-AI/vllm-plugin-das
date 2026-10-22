# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Parser non-regression checks that must pass before and after the Step5 patch.

This module deliberately does not import ``vllm_hcu.parsers`` or the Step5
runtime adapters. It pins the pre-existing Step3.5 and generic parser paths so
the same file can be executed against a clean target tree and the patched tree.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

pytest.importorskip("vllm", reason="requires the audited vLLM runtime")

from vllm.entrypoints.openai.chat_completion.protocol import (  # noqa: E402
    ChatCompletionRequest,
)
from vllm.parser.parser_manager import ParserManager  # noqa: E402
from vllm.reasoning import ReasoningParserManager  # noqa: E402
from vllm.tool_parsers import ToolParserManager  # noqa: E402
from vllm.tool_parsers.hermes_tool_parser import (  # noqa: E402
    Hermes2ProToolParser,
)
from vllm.tool_parsers.step3p5_tool_parser import (  # noqa: E402
    Step3p5ToolParser,
)

THINK_START_ID = 128798
THINK_END_ID = 128799

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "days": {"type": "integer"},
                },
                "required": ["city"],
            },
        },
    }
]

OUT_XML = "I should call a tool\n</think>" + (
    "<tool_call>\n<function=get_weather>\n<parameter=city>\nBeijing\n</parameter>\n"
    "</function>\n</tool_call>"
)
OUT_JSON = (
    "<tool_call>"
    '{"name": "get_weather", "arguments": {"city": "Beijing"}}'
    "</tool_call>"
)


def _tokenizer() -> SimpleNamespace:
    return SimpleNamespace(
        get_vocab=lambda: {"<think>": THINK_START_ID, "</think>": THINK_END_ID}
    )


def _request(tool_choice="auto") -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="step5-test",
        messages=[{"role": "user", "content": "what is the weather in Beijing"}],
        tools=TOOLS,
        tool_choice=tool_choice,
    )


def test_step3p5_tool_parser_is_registered() -> None:
    assert ToolParserManager.get_tool_parser("step3p5") is Step3p5ToolParser


def test_step3p5_reasoning_parser_is_registered() -> None:
    from vllm.reasoning.step3p5_reasoning_parser import Step3p5ReasoningParser

    assert (
        ReasoningParserManager.get_reasoning_parser("step3p5")
        is Step3p5ReasoningParser
    )


def test_step3p5_direct_extraction_is_unchanged() -> None:
    parser = Step3p5ToolParser(_tokenizer(), TOOLS)

    info = parser.extract_tool_calls(OUT_XML, _request())

    assert info.tools_called
    assert [call.function.name for call in info.tool_calls] == ["get_weather"]
    assert json.loads(info.tool_calls[0].function.arguments) == {"city": "Beijing"}


def test_step3p5_factory_path_is_unchanged() -> None:
    request = _request("auto")
    parser_cls = ParserManager.get_parser(
        tool_parser_name="step3p5",
        reasoning_parser_name="step3p5",
        enable_auto_tools=True,
        model_name="step3p5",
    )
    assert parser_cls is not None
    parser = parser_cls(_tokenizer(), request.tools, chat_template_kwargs={})

    reasoning, content, tool_calls = parser.parse(
        OUT_XML, request, enable_auto_tools=True
    )

    assert reasoning == "I should call a tool"
    assert "<tool_call>" not in (content or "")
    assert [call.name for call in tool_calls or []] == ["get_weather"]


def test_generic_hermes_parser_extraction_is_unchanged() -> None:
    parser = Hermes2ProToolParser(_tokenizer(), TOOLS)

    info = parser.extract_tool_calls(OUT_JSON, _request())

    assert info.tools_called
    assert [call.function.name for call in info.tool_calls] == ["get_weather"]
    assert json.loads(info.tool_calls[0].function.arguments) == {"city": "Beijing"}


def test_generic_hermes_factory_path_is_unchanged() -> None:
    request = _request("auto")
    parser_cls = ParserManager.get_parser(
        tool_parser_name="hermes",
        reasoning_parser_name=None,
        enable_auto_tools=True,
        model_name="hermes",
    )
    assert parser_cls is not None
    parser = parser_cls(_tokenizer(), request.tools, chat_template_kwargs={})

    _, _, tool_calls = parser.parse(OUT_JSON, request, enable_auto_tools=True)

    assert [call.name for call in tool_calls or []] == ["get_weather"]
