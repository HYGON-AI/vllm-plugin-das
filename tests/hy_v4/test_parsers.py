# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

import json

from xgrammar import Grammar
from xgrammar.testing import _is_grammar_accept_string
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm_hcu.reasoning.hy_v4_reasoning_parser import (
    HYV4ReasoningExtractor,
    HYV4ReasoningParser,
    detect_token_suffix as detect_reasoning_suffix,
)
from vllm_hcu.tool_parsers.hy_v4_tool_parser import (
    HYV4ToolParser,
    HYV4ToolExtractor,
    detect_token_suffix as detect_tool_suffix,
)


SUFFIX = ":hcu"
THINK_START = f"<think{SUFFIX}>"
THINK_END = f"</think{SUFFIX}>"
THINK_START_ID = 10
THINK_END_ID = 11


class FakeTokenizer:
    def __init__(self, vocabulary: dict[str, int]):
        self._vocabulary = vocabulary
        self.init_kwargs: dict[str, object] = {}

    def get_vocab(self) -> dict[str, int]:
        return self._vocabulary


def _tool_vocabulary() -> dict[str, int]:
    names = ("think", "tool_calls", "tool_call", "arg_key", "arg_value")
    tokens = [f"<{name}{SUFFIX}>" for name in names]
    tokens += [f"</{name}{SUFFIX}>" for name in names]
    return {token: index for index, token in enumerate(tokens)}


def _arg(key: str, value: str) -> str:
    return (
        f"<arg_key{SUFFIX}>{key}</arg_key{SUFFIX}>"
        f"<arg_value{SUFFIX}>{value}</arg_value{SUFFIX}>"
    )


def _call(name: str, arguments: str = "") -> str:
    return f"<tool_call{SUFFIX}>{name}{arguments}</tool_call{SUFFIX}>"


def _block(body: str) -> str:
    return f"<tool_calls{SUFFIX}>{body}</tool_calls{SUFFIX}>"


def test_reasoning_parser_respects_thinking_mode_and_suffix() -> None:
    vocabulary = {
        THINK_START: THINK_START_ID,
        THINK_END: THINK_END_ID,
    }
    tokenizer = FakeTokenizer(vocabulary)

    assert detect_reasoning_suffix(tokenizer) == SUFFIX
    thinking = HYV4ReasoningParser(tokenizer, reasoning_effort="high")
    no_think = HYV4ReasoningParser(tokenizer, reasoning_effort="no_think")

    assert thinking.extract_reasoning(f"分析{THINK_END}答案", None) == (
        "分析",
        "答案",
    )
    assert no_think.extract_reasoning("直接答案", None) == (None, "直接答案")
    assert thinking.reasoning_start_str == THINK_START
    assert thinking.reasoning_end_str == THINK_END


def test_reasoning_streaming_splits_end_marker_from_content() -> None:
    extractor = HYV4ReasoningExtractor(
        {THINK_START: THINK_START_ID, THINK_END: THINK_END_ID},
        SUFFIX,
        thinking=True,
    )
    delta = extractor.extract_reasoning_streaming(
        "分析",
        f"分析{THINK_END}答案",
        f"{THINK_END}答案",
        [1],
        [1, THINK_END_ID, 2],
        [THINK_END_ID, 2],
    )
    assert delta == {"reasoning": "", "content": "答案"}


def test_tool_parser_coerces_schema_values_and_preserves_content() -> None:
    vocabulary = _tool_vocabulary()
    tokenizer = FakeTokenizer(vocabulary)
    assert detect_tool_suffix(tokenizer) == SUFFIX
    extractor = HYV4ToolExtractor(vocabulary, SUFFIX, strict=True)
    tools = [
        {
            "name": "weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "days": {"type": "integer"},
                    "detail": {"type": "boolean"},
                },
            },
        }
    ]
    output = "调用天气工具：" + _block(
        _call(
            "weather",
            _arg("city", "北京") + _arg("days", "3") + _arg("detail", "true"),
        )
    )

    result = extractor.extract_tool_calls(output, tools)

    assert result["tools_called"] is True
    assert result["content"] == "调用天气工具："
    assert result["tool_calls"] == [
        {
            "name": "weather",
            "arguments": json.dumps(
                {"city": "北京", "days": 3, "detail": True},
                ensure_ascii=False,
            ),
        }
    ]


def test_tool_parser_preserves_content_after_tool_block() -> None:
    extractor = HYV4ToolExtractor(_tool_vocabulary(), SUFFIX, strict=True)
    output = "before" + _block(_call("date")) + "after"

    result = extractor.extract_tool_calls(output, None)

    assert result["tools_called"] is True
    assert result["content"] == "beforeafter"
    assert result["tool_calls"] == [{"name": "date", "arguments": "{}"}]


def test_tool_parser_streams_multiple_calls_without_losing_arguments() -> None:
    extractor = HYV4ToolExtractor(_tool_vocabulary(), SUFFIX, strict=True)
    tools = [
        {
            "name": "weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
        },
        {"name": "date", "parameters": {"type": "object", "properties": {}}},
    ]
    output = _block(_call("weather", _arg("city", "武汉")) + _call("date"))
    previous = ""
    names: dict[int, str] = {}
    arguments: dict[int, str] = {}

    for chunk in output:
        current = previous + chunk
        delta = extractor.extract_tool_calls_streaming(
            previous,
            current,
            chunk,
            [],
            [],
            [],
            tools,
        )
        if delta is not None:
            for tool_call in delta["tool_calls"]:
                index = tool_call["index"]
                if tool_call["name"] is not None:
                    names[index] = tool_call["name"]
                if tool_call["arguments"] is not None:
                    arguments[index] = (
                        arguments.get(index, "") + tool_call["arguments"]
                    )
        previous = current

    assert names == {0: "weather", 1: "date"}
    assert arguments == {0: '{"city": "武汉"}', 1: "{}"}


def test_tool_parser_streaming_is_chunk_invariant_for_complete_delta() -> None:
    extractor = HYV4ToolExtractor(_tool_vocabulary(), SUFFIX, strict=True)
    output = (
        "before"
        + _block(
            _call("weather", _arg("city", "武汉"))
            + _call("date")
        )
        + "after"
    )

    delta = extractor.extract_tool_calls_streaming(
        "", output, output, [], [], [], None
    )

    assert delta is not None
    assert delta["content"] == "beforeafter"
    assert [tool["name"] for tool in delta["tool_calls"]] == ["weather", "date"]
    assert [tool["arguments"] for tool in delta["tool_calls"]] == [
        '{"city": "武汉"}',
        "{}",
    ]


def test_tool_parser_streams_content_after_tool_block() -> None:
    extractor = HYV4ToolExtractor(_tool_vocabulary(), SUFFIX, strict=True)
    output = _block(_call("date")) + "after"
    previous = ""
    content = ""

    for chunk in output:
        current = previous + chunk
        delta = extractor.extract_tool_calls_streaming(
            previous, current, chunk, [], [], [], None
        )
        if delta is not None:
            content += delta["content"] or ""
        previous = current

    assert content == "after"


def test_tool_parser_ignores_responses_builtin_tools() -> None:
    request = ResponsesRequest(
        model="hy4",
        input="北京天气如何？",
        tools=[
            {"type": "web_search"},
            {
                "type": "function",
                "name": "weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            },
        ],
        tool_choice="auto",
    )

    parser = HYV4ToolParser(FakeTokenizer(_tool_vocabulary()), request.tools)

    assert parser._plain_tools == [
        {
            "name": "weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
        }
    ]


def test_required_tool_choice_preserves_hyv4_native_output_format() -> None:
    tokenizer = FakeTokenizer(_tool_vocabulary())
    parser = HYV4ToolParser(tokenizer)
    request = ChatCompletionRequest(
        model="hy4",
        messages=[{"role": "user", "content": "北京天气如何？"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "weather",
                    "description": "查询天气",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ],
        tool_choice="required",
    )

    adjusted = parser.adjust_request(request)

    assert adjusted.structured_outputs is None
    assert adjusted.skip_special_tokens is False

    structural_tag = parser.get_structural_tag(request)
    assert structural_tag is not None
    dumped_tag = structural_tag.model_dump_json()
    assert f"<tool_calls{SUFFIX}>" in dumped_tag
    assert f"<tool_call{SUFFIX}>weather" in dumped_tag
    assert f"<arg_key{SUFFIX}>city</arg_key{SUFFIX}>" in dumped_tag

    grammar = Grammar.from_structural_tag(structural_tag)
    valid = _block(_call("weather", _arg("city", "北京")))
    malformed = (
        f"<tool_calls{SUFFIX}>weather"
        f"<arg_key{SUFFIX}>city</arg_value{SUFFIX}>"
        f"<arg_value{SUFFIX}>北京</arg_value{SUFFIX}>"
        f"</tool_calls{SUFFIX}>"
    )
    assert _is_grammar_accept_string(grammar, valid)
    assert not _is_grammar_accept_string(grammar, malformed)


def test_required_tool_choice_falls_back_to_vllm_json_guidance() -> None:
    parser = HYV4ToolParser(FakeTokenizer(_tool_vocabulary()))
    parser.supports_required_and_named = True
    request = ChatCompletionRequest(
        model="hy4",
        messages=[{"role": "user", "content": "北京天气如何？"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ],
        tool_choice="required",
    )

    adjusted = parser.adjust_request(request)

    assert adjusted.structured_outputs is not None
    assert adjusted.structured_outputs.json is not None


def test_structural_tag_enforces_required_key_missing_from_properties() -> None:
    parser = HYV4ToolParser(FakeTokenizer(_tool_vocabulary()))
    request = ChatCompletionRequest(
        model="hy4",
        messages=[{"role": "user", "content": "call odd"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "odd",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": ["ghost"],
                    },
                },
            }
        ],
        tool_choice="required",
    )
    structural_tag = parser.get_structural_tag(request)
    assert structural_tag is not None
    grammar = Grammar.from_structural_tag(structural_tag)

    assert _is_grammar_accept_string(
        grammar, _block(_call("odd", _arg("ghost", "1")))
    )
    assert not _is_grammar_accept_string(grammar, _block(_call("odd")))


def test_structural_tag_allows_optional_arguments_in_any_order() -> None:
    parser = HYV4ToolParser(FakeTokenizer(_tool_vocabulary()))
    request = ChatCompletionRequest(
        model="hy4",
        messages=[{"role": "user", "content": "call optional"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "optional",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "string"},
                            "b": {"type": "string"},
                        },
                    },
                },
            }
        ],
        tool_choice="required",
    )
    structural_tag = parser.get_structural_tag(request)
    assert structural_tag is not None
    grammar = Grammar.from_structural_tag(structural_tag)

    assert _is_grammar_accept_string(
        grammar,
        _block(_call("optional", _arg("b", "2") + _arg("a", "1"))),
    )
