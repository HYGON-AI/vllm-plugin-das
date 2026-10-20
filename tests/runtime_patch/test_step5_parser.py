# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""CPU contracts for the Step5 parser registration and required/named choice.

Step5 answers with the same XML tool-call format as Step3.5, so these tests pin
the behaviour that the runtime registration changes rather than the extraction
itself: the ``step5`` registry entries, that required/named tool choice reaches
the XML parser instead of the generic JSON path, and that ``auto`` is untouched.

The tokenizer is a stub -- the Step5 parsers only read ``get_vocab()`` for the
``<think>`` markers, so no checkpoint, device or network access is involved.
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

from vllm_hcu.patch.platform.core_fix import (  # noqa: E402
    patch_step5_reasoning_parser,
    patch_step5_tool_parser,
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
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get the current time",
            "parameters": {
                "type": "object",
                "properties": {"tz": {"type": "string"}},
                "required": ["tz"],
            },
        },
    },
]

NAMED_GET_WEATHER = {"type": "function", "function": {"name": "get_weather"}}
NAMED_GET_TIME = {"type": "function", "function": {"name": "get_time"}}

# Step5 is generated after the chat template has already emitted "<think>", so
# the model only produces the closing marker.
OUT_WEATHER = "I should call a tool\n</think>" + (
    "<tool_call>\n<function=get_weather>\n<parameter=city>\nBeijing\n</parameter>\n"
    "</function>\n</tool_call>"
)
OUT_WEATHER_MULTIARG = "I should call a tool\n</think>" + (
    "<tool_call>\n<function=get_weather>\n<parameter=city>\nBeijing\n</parameter>\n"
    "<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>"
)
OUT_NO_TOOL = "I should call a tool\n</think>The weather is nice today."


@pytest.fixture(scope="module", autouse=True)
def _apply_step5_registrations():
    """Apply the production adapters once, through their real entry point."""

    assert patch_step5_reasoning_parser.apply() in (True, False)
    assert patch_step5_tool_parser.apply() in (True, False)
    yield


def _tokenizer() -> SimpleNamespace:
    return SimpleNamespace(
        get_vocab=lambda: {"<think>": THINK_START_ID, "</think>": THINK_END_ID}
    )


def _request(tool_choice) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="step5-test",
        messages=[{"role": "user", "content": "what is the weather in Beijing"}],
        tools=TOOLS,
        tool_choice=tool_choice,
    )


def _build(tool_choice):
    """Compose the parser exactly as the OpenAI serving layer does."""
    request = _request(tool_choice)
    parser_cls = ParserManager.get_parser(
        tool_parser_name="step5",
        reasoning_parser_name="step5",
        enable_auto_tools=True,
        model_name="step5",
    )
    assert parser_cls is not None
    return parser_cls(_tokenizer(), request.tools, chat_template_kwargs={}), request


def _calls(result) -> list[dict]:
    _, _, tool_calls = result
    return [
        {"name": call.name, "arguments": json.loads(call.arguments)}
        for call in tool_calls or []
    ]


# --- registry ---------------------------------------------------------------


def test_step5_reasoning_name_resolves_to_the_step3p5_parser() -> None:
    from vllm.reasoning.step3p5_reasoning_parser import Step3p5ReasoningParser

    assert "step5" in ReasoningParserManager.list_registered()
    assert (
        ReasoningParserManager.get_reasoning_parser("step5")
        is Step3p5ReasoningParser
    )


def test_step5_tool_name_resolves_to_the_step5_parser() -> None:
    from vllm.tool_parsers.step3p5_tool_parser import Step3p5ToolParser

    from vllm_hcu.parsers.step5_tool_parser import Step5ToolParser

    assert "step5" in ToolParserManager.list_registered()
    assert ToolParserManager.get_tool_parser("step5") is Step5ToolParser
    assert issubclass(Step5ToolParser, Step3p5ToolParser)


def test_step5_registry_entries_are_plugin_owned_lazy_modules() -> None:
    assert ToolParserManager.lazy_parsers["step5"] == (
        "vllm_hcu.parsers.step5_tool_parser",
        "Step5ToolParser",
    )
    assert ReasoningParserManager.lazy_parsers["step5"] == (
        "vllm.reasoning.step3p5_reasoning_parser",
        "Step3p5ReasoningParser",
    )


def test_runtime_registration_is_idempotent() -> None:
    assert patch_step5_reasoning_parser.apply() is False
    assert patch_step5_tool_parser.apply() is False


def test_conflicting_step5_tool_registration_fails_closed() -> None:
    import vllm.tool_parsers as tool_parsers_module

    from vllm_hcu.patch.platform.core_fix._common import PatchCompatibilityError

    manager = ToolParserManager
    saved_lazy = manager.lazy_parsers.get("step5")
    saved_eager = manager.tool_parsers.get("step5")
    saved_marker = getattr(manager, patch_step5_tool_parser._MARKER, None)
    try:
        manager.lazy_parsers["step5"] = ("example.conflict", "ConflictParser")
        manager.tool_parsers.pop("step5", None)
        if hasattr(manager, patch_step5_tool_parser._MARKER):
            delattr(manager, patch_step5_tool_parser._MARKER)
        with pytest.raises(PatchCompatibilityError):
            patch_step5_tool_parser.apply_to_module(tool_parsers_module)
    finally:
        if saved_lazy is None:
            manager.lazy_parsers.pop("step5", None)
        else:
            manager.lazy_parsers["step5"] = saved_lazy
        if saved_eager is None:
            manager.tool_parsers.pop("step5", None)
        else:
            manager.tool_parsers["step5"] = saved_eager
        if saved_marker is not None:
            setattr(manager, patch_step5_tool_parser._MARKER, saved_marker)


# --- extraction --------------------------------------------------------------


@pytest.mark.parametrize(
    "tool_choice",
    ["auto", "required", NAMED_GET_WEATHER],
    ids=["auto", "required", "named"],
)
def test_xml_tool_call_is_returned_for_every_tool_choice_mode(tool_choice) -> None:
    """auto, required and named must all yield the model's own call."""
    parser, request = _build(tool_choice)

    result = parser.parse(OUT_WEATHER, request, enable_auto_tools=True)

    reasoning, content, _ = result
    assert _calls(result) == [
        {"name": "get_weather", "arguments": {"city": "Beijing"}}
    ]
    assert reasoning == "I should call a tool"
    assert "<tool_call>" not in (content or "")


def test_required_keeps_the_schema_typed_arguments() -> None:
    parser, request = _build("required")

    result = parser.parse(OUT_WEATHER_MULTIARG, request, enable_auto_tools=True)

    assert _calls(result) == [
        {"name": "get_weather", "arguments": {"city": "Beijing", "days": 3}}
    ]


def test_required_without_a_call_does_not_invent_one() -> None:
    parser, request = _build("required")

    _, content, tool_calls = parser.parse(
        OUT_NO_TOOL, request, enable_auto_tools=True
    )

    assert not tool_calls
    assert content == "The weather is nice today."


def test_named_keeps_the_function_the_model_actually_called() -> None:
    """get_time was requested; the parser reports the model's get_weather."""
    parser, request = _build(NAMED_GET_TIME)

    result = parser.parse(OUT_WEATHER, request, enable_auto_tools=True)

    assert _calls(result) == [
        {"name": "get_weather", "arguments": {"city": "Beijing"}}
    ]


def test_named_without_a_call_does_not_invent_one() -> None:
    parser, request = _build(NAMED_GET_TIME)

    _, content, tool_calls = parser.parse(
        OUT_NO_TOOL, request, enable_auto_tools=True
    )

    assert not tool_calls
    assert content == "The weather is nice today."


def test_required_constraint_violation_is_logged(caplog) -> None:
    parser, request = _build("required")

    parser.parse(OUT_NO_TOOL, request, enable_auto_tools=True)

    assert "tool_choice='required' but the model produced no tool call." in (
        caplog.text
    )


def test_named_constraint_violation_is_logged(caplog) -> None:
    parser, request = _build(NAMED_GET_TIME)

    parser.parse(OUT_WEATHER, request, enable_auto_tools=True)

    assert "asked for function 'get_time'" in caplog.text


# --- generation side ---------------------------------------------------------


@pytest.mark.parametrize(
    "tool_choice",
    ["auto", "none"],
    ids=["auto", "none"],
)
def test_request_is_left_to_the_inherited_behaviour_for_auto_and_none(
    tool_choice,
) -> None:
    parser, request = _build(tool_choice)

    assert parser.adjust_request(request) is not None


@pytest.mark.parametrize(
    "tool_choice",
    ["required", NAMED_GET_WEATHER],
    ids=["required", "named"],
)
def test_no_json_schema_grammar_is_installed_for_required_or_named(
    tool_choice,
) -> None:
    """required/named must not be forced into JSON by guided decoding."""
    parser, request = _build(tool_choice)

    parser.adjust_request(request)

    assert getattr(request, "structured_outputs", None) is None
    assert getattr(request, "response_format", None) is None


def test_auto_still_derives_no_schema() -> None:
    from vllm.tool_parsers.utils import get_json_schema_from_tools

    assert get_json_schema_from_tools("auto", TOOLS) is None


# --- streaming ---------------------------------------------------------------


def test_streaming_uses_the_model_parser_for_required_and_named() -> None:
    """The flag is what routes required/named streaming to the model's parser.

    ``DelegatingParser._extract_tool_calls_streaming`` sends required and named
    tool choice to ``extract_named_tool_call_streaming`` /
    ``extract_required_tool_call_streaming`` while
    ``supports_required_and_named`` is True. Those helpers expect JSON, so for
    Step5 they would never surface a ``<tool_call>``. This pins the dispatch
    precondition; the stream plumbing itself is upstream code the adapter does
    not touch, and end-to-end streaming is covered by the GPU gate.
    """
    from vllm.tool_parsers.step3p5_tool_parser import Step3p5ToolParser

    assert Step3p5ToolParser.supports_required_and_named is True
    assert (
        ToolParserManager.get_tool_parser("step5").supports_required_and_named
        is False
    )


@pytest.mark.parametrize(
    "tool_choice",
    ["required", NAMED_GET_WEATHER],
    ids=["required", "named"],
)
def test_streaming_extraction_yields_the_call_for_required_and_named(
    tool_choice,
) -> None:
    """The parser's own streaming extraction is reached and returns the call."""
    parser, request = _build(tool_choice)
    tool_text = OUT_WEATHER.split("</think>")[1]

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
    assert json.loads(delta.tool_calls[0].function.arguments) == {
        "city": "Beijing"
    }
