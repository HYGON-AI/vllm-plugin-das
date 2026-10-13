# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Review 2026-10-12 fixes #4/#5: content passthrough and stream-end flush.

#4  Non-streaming ``tool_choice="auto"`` with NO tool call: upstream
    DelegatingParser discards the parser's cleaned content and the raw XTML
    end markers reach the user (repro
    ``Hi<|close|>response<|sep|><|close|>message<|sep|>``).  The platform
    wrapper restores fork parity: the cleaned body wins.
#5  Streaming: a trailing partial tag ('1 ' + '<') must flush at the
    finished=True turn, matching the non-streaming output ('1 <').  The
    delegating parse_delta wrapper calls the parser's finish_streaming()
    only when the wheel carries the ``finished`` protocol.

Review 2026-10-13 additions: unique tool call ids (#6), whitespace-
tolerant holdback (#4), and the message-close streaming boundary (#7).
"""

from __future__ import annotations

import inspect
from types import ModuleType, SimpleNamespace

import pytest

from vllm_hcu.patch.platform.core_fix import patch_delegating_tool_content
from vllm_hcu.tool_parsers.kimi_k3_tool_parser import KimiK3ToolParser

_RAW = "Hi<|close|>response<|sep|><|close|>message<|sep|>"
_CLEAN = "Hi"


def _make_delegator(result, tool_parser, *, with_parse_delta: bool = False):
    """A fresh delegator class per test: no cross-test wrapper state."""

    class DelegatingParser:
        def __init__(self):
            self.result = result
            self._tool_parser = tool_parser

        def _extract_tool_calls(self, content, request, *args, **kwargs):
            return self.result

        if with_parse_delta:

            def parse_delta(
                self, delta_text, delta_token_ids, request, *, finished=False
            ):
                self.parsed = (delta_text, finished)
                return SimpleNamespace(content=delta_text, tool_calls=[])

    module = ModuleType(patch_delegating_tool_content.TARGET_MODULE)
    module.DelegatingParser = DelegatingParser
    return module, DelegatingParser()


def _stub_parser(raw: str, content: str | None) -> KimiK3ToolParser:
    parser = object.__new__(KimiK3ToolParser)
    parser._hcu_last_extract = (
        raw,
        SimpleNamespace(tools_called=False, tool_calls=[], content=content),
    )
    return parser


def _flushing_stub_parser(held: str) -> KimiK3ToolParser:
    parser = object.__new__(KimiK3ToolParser)
    parser._hcu_last_extract = None

    def finish_streaming():
        return SimpleNamespace(content=held, tool_calls=None)

    parser.finish_streaming = finish_streaming  # type: ignore[method-assign]
    return parser


def test_no_tool_call_passes_cleaned_content_through() -> None:
    module, delegator = _make_delegator(([], _RAW), _stub_parser(_RAW, _CLEAN))

    assert patch_delegating_tool_content.apply_to_module(module) is True
    tool_calls, out = delegator._extract_tool_calls(_RAW, object())
    assert tool_calls == []
    assert out == _CLEAN


def test_tool_call_branch_keeps_upstream_contract() -> None:
    module, delegator = _make_delegator(
        ([object()], _CLEAN), _stub_parser(_RAW, _CLEAN)
    )

    assert patch_delegating_tool_content.apply_to_module(module) is True
    tool_calls, out = delegator._extract_tool_calls(_RAW, object())
    assert len(tool_calls) == 1
    assert out == _CLEAN


def test_other_parser_types_untouched() -> None:
    module, delegator = _make_delegator(([], _RAW), object())

    assert patch_delegating_tool_content.apply_to_module(module) is True
    _, out = delegator._extract_tool_calls(_RAW, object())
    assert out == _RAW


def test_mismatched_record_input_untouched() -> None:
    module, delegator = _make_delegator(([], _RAW), _stub_parser("other", _CLEAN))

    assert patch_delegating_tool_content.apply_to_module(module) is True
    _, out = delegator._extract_tool_calls(_RAW, object())
    assert out == _RAW


def test_apply_is_idempotent_per_target() -> None:
    module, _ = _make_delegator(([], _RAW), _stub_parser(_RAW, _CLEAN))

    assert patch_delegating_tool_content.apply_to_module(module) is True
    assert patch_delegating_tool_content.apply_to_module(module) is False


def test_parse_delta_flushes_the_k3_parser_at_finished() -> None:
    module, delegator = _make_delegator(
        ([], _RAW), _flushing_stub_parser("<"), with_parse_delta=True
    )

    assert patch_delegating_tool_content.apply_to_module(module) is True

    unfinished_delta = delegator.parse_delta("<", [9], object(), finished=False)
    assert unfinished_delta.content == "<"  # passthrough baseline, no extra flush

    delegator._tool_parser = _flushing_stub_parser("<")
    wrapped_delta = object.__getattribute__(type(delegator), "parse_delta")
    assert getattr(
        wrapped_delta, patch_delegating_tool_content._WRAPPER, False
    ), "finished-capable parse_delta must be wrapped"

    result = delegator.parse_delta("<", [9], object(), finished=True)
    assert result.content == "<<"  # baseline '<' + flush '<'


def test_parse_delta_flushes_a_live_k3_parser_end_to_end() -> None:
    """P2-8: real parser, real '1 ' + '<' holdback, through the wrapper."""

    module, delegator = _make_delegator(
        ([], ""), _live_parser(), with_parse_delta=True
    )
    assert patch_delegating_tool_content.apply_to_module(module) is True

    request = SimpleNamespace(tool_choice="auto")
    delegator._tool_parser.extract_tool_calls_streaming(
        "", "1 ", "1 ", [], [], [], request
    )
    delegator._tool_parser.extract_tool_calls_streaming(
        "1 ", "1 <", "<", [], [], [], request
    )

    result = delegator.parse_delta("", [9], request, finished=True)
    assert result.content == "<", "the held '<' must reach the (empty) final delta"


def test_parse_delta_flushes_with_positional_finished() -> None:
    """P2-3 regression: finished passed POSITIONALLY must still be bound."""

    class DelegatingParser:
        def __init__(self):
            self._tool_parser = _flushing_stub_parser("<")

        def _extract_tool_calls(self, content, request, *args, **kwargs):
            return ([], "")

        def parse_delta(
            self, delta_text, delta_token_ids, request, finished=False
        ):
            return SimpleNamespace(content=delta_text, tool_calls=[])

    module = ModuleType(patch_delegating_tool_content.TARGET_MODULE)
    module.DelegatingParser = DelegatingParser
    assert patch_delegating_tool_content.apply_to_module(module) is True

    delegator = DelegatingParser()
    flushed = delegator.parse_delta("", [9], object(), True)
    assert flushed.content == "<"
    unflushed = delegator.parse_delta("", [9], object(), False)
    assert unflushed.content == ""


def test_parse_delta_without_finished_protocol_is_left_alone() -> None:
    class DelegatingParser:
        def __init__(self):
            self._tool_parser = _flushing_stub_parser("<")

        def _extract_tool_calls(self, content, request, *args, **kwargs):
            return ([], content)

        def parse_delta(self, delta_text, delta_token_ids, request):
            return SimpleNamespace(content=delta_text)

    module = ModuleType(patch_delegating_tool_content.TARGET_MODULE)
    module.DelegatingParser = DelegatingParser

    assert patch_delegating_tool_content.apply_to_module(module) is True
    parameters = inspect.signature(DelegatingParser.parse_delta).parameters
    assert "finished" not in parameters  # soft skip: signature untouched
    assert not getattr(
        DelegatingParser.parse_delta,
        patch_delegating_tool_content._WRAPPER,
        False,
    )


def _live_parser() -> KimiK3ToolParser:
    return KimiK3ToolParser(tokenizer=object())


def _stream(parser: KimiK3ToolParser, *currents: str) -> None:
    """Feed cumulative texts; delta = current minus previous."""

    request = SimpleNamespace(tool_choice="auto")
    previous = ""
    for current in currents:
        parser.extract_tool_calls_streaming(
            previous,
            current,
            current[len(previous) :],
            [],
            [],
            [],
            request,
        )
        previous = current


def test_stream_end_flushes_partial_tag_tail() -> None:
    """'1 ' + '<' + finished -> flushed '<' (total streamed '1 <')."""

    parser = _live_parser()
    _stream(parser, "1 ", "1 <")
    assert parser._hcu_last_text == "1 <"
    flush = parser.finish_streaming()
    assert flush is not None
    assert flush.content == "<"


def test_stream_flush_silent_for_marker_terminated_reply() -> None:
    """Plain reply ending at the response close flushes NOTHING beyond it."""

    parser = _live_parser()
    _stream(parser, _RAW)  # 'Hi' streamed; the rest is markup
    flushed = parser._sent_content_idx == 2 and parser._hcu_last_text == _RAW
    assert flushed, "baseline: only 'Hi' sent, text fully accumulated"
    assert parser.finish_streaming() is None


def test_stream_flush_silent_for_generated_open_marker() -> None:
    """N3: a model-GENERATED response open marker is markup, not content."""

    parser = _live_parser()
    generated = "<|open|>response<|sep|><|close|>response<|sep|><|open|>tools<|sep|>"
    _stream(parser, generated)
    assert parser.finish_streaming() is None

    parser = _live_parser()
    _stream(parser, "<|open|>response<|sep|><")
    flush = parser.finish_streaming()
    assert flush is not None and flush.content == "<"


def test_stream_flush_idempotent_without_holdback() -> None:
    parser = _live_parser()
    _stream(parser, "1 ")
    assert parser.finish_streaming() is None
    assert parser.finish_streaming() is None


def test_stream_flush_keeps_tool_markup_silent() -> None:
    parser = _live_parser()
    block = (
        "<|open|>tools<|sep|>"
        '<|open|>call tool="py" index="1"<|sep|>'
        '<|open|>argument key="code" type="string"<|sep|>'
        "print(1)<|close|>argument<|sep|>"
        "<|close|>call<|sep|>"
        "<|close|>tools<|sep|>"
    )
    _stream(parser, block)
    assert parser.finish_streaming() is None


def test_live_parser_records_extractions_for_the_wrapper() -> None:
    parser = _live_parser()
    request = SimpleNamespace(tool_choice="auto")

    info = parser.extract_tool_calls(_RAW, request)

    assert info.tools_called is False
    assert info.content == _CLEAN
    assert parser._hcu_last_extract is not None
    assert parser._hcu_last_extract[0] == _RAW
    assert parser._hcu_last_extract[1].content == _CLEAN


# --- review 2026-10-13: #6 unique call ids, #4 whitespace-tolerant
# holdback, #7 message-close streaming boundary ---------------------------


def test_tool_call_ids_stay_unique_across_turns() -> None:
    """P1 #6: K3 restarts the XTML index every turn; ids must not collide."""

    parser = _live_parser()
    block = (
        '<|open|>tools<|sep|>'
        '<|open|>call tool="py" index="1"<|sep|>'
        '<|open|>argument key="code" type="string"<|sep|>'
        "print(1)<|close|>argument<|sep|>"
        "<|close|>call<|sep|>"
        "<|close|>tools<|sep|>"
    )
    first = parser._decode_call('tool="py" index="1"', "")
    info = parser.extract_tool_calls(block, SimpleNamespace(tool_choice="auto"))
    assert info.tools_called is True
    repeat = parser.extract_tool_calls(block, SimpleNamespace(tool_choice="auto"))
    a, b = info.tool_calls[0], repeat.tool_calls[0]
    assert a.id.startswith("py:") and b.id.startswith("py:")
    assert a.id != b.id, f"ids collide across turns: {a.id} vs {b.id}"
    assert first.id.startswith("py:")


def test_streaming_holds_whitespace_tolerant_marker_prefix() -> None:
    """P2 #4: '<|close|> ' chunks must not leak the partial marker; the
    whitespace-tolerant shapes the regexes accept are held back too."""

    parser = _live_parser()
    _stream(parser, "Hi<|close|> ")
    # 'Hi' went out, the partial close marker is held
    assert parser._sent_content_idx == 2

    parser = _live_parser()
    _stream(parser, "Hi<|close|> resp")
    assert parser._sent_content_idx == 2

    # completing the marker never re-emits any of it
    parser = _live_parser()
    _stream(parser, "Hi<|close|> response <|sep|>")
    assert parser._sent_content_idx == 2


def test_streaming_message_close_is_a_content_boundary() -> None:
    """P2 #7: non-strict auto output may terminate with
    <|close|>message<|sep|> WITHOUT a response close; neither the live
    stream nor the end flush may leak that marker."""

    parser = _live_parser()
    _stream(parser, "Hi<|close|>message<|sep|>")
    assert parser._sent_content_idx == 2
    assert parser.finish_streaming() is None

    # chunked arrival, marker split across deltas
    parser = _live_parser()
    _stream(
        parser,
        "Hi<|close|> ",
        "Hi<|close|>message<|sep|>",
        "Hi<|close|>message<|sep|><|open|>tools<|sep|>",
    )
    assert parser._sent_content_idx == 2
    assert parser.finish_streaming() is None


def test_marker_holdback_crosses_the_pipe_prefix() -> None:
    """R2 #4: a chunk landing INSIDE the tail prefix ('<|', '<|se') must be
    held; the old ``<[^|]*`` group could never cross the pipe and leaked
    those suffixes into the content channel."""

    from vllm_hcu.tool_parsers.kimi_k3_tool_parser import _marker_holdback

    head, word, tail = "<|close|>", "response", "<|sep|>"
    # the hold is the whole suffix ANCHORED AT head (branch shapes inherit
    # the head), so the returned length covers head + word + tail-prefix
    assert _marker_holdback("Hi<|close|> response <", head, word, tail) == len("<|close|> response <")
    assert _marker_holdback("Hi<|close|> response <|", head, word, tail) == len("<|close|> response <|")
    assert _marker_holdback("Hi<|close|> response <|se", head, word, tail) == len("<|close|> response <|se")
    assert _marker_holdback("plain text done", head, word, tail) == 0
    # R11 P2: the whitespace runs are UNBOUNDED (the old +8 window dropped
    # the head once the combined length exceeded it and leaked the prefix).
    assert _marker_holdback("x" * 20 + head + " " * 11, head, word, tail) == len(head) + 11
    assert (
        _marker_holdback("Hi" + head + " " * 12 + "respo", head, word, tail)
        == len("Hi" + head + " " * 12 + "respo") - 2
    )
    assert (
        _marker_holdback(head + " " * 30 + word + " " * 9 + "<|se", head, word, tail)
        == len(head) + 30 + len(word) + 9 + 4
    )
    # a head followed by junk stays untouched (no hold without a shape)
    assert _marker_holdback("a<|close|>b c", head, word, tail) == 0


def test_streaming_holds_a_long_space_marker_prefix() -> None:
    """R11 P2 parser-level twin: >8 spaces between head and word must not
    stream the control prefix as content before <|sep|> arrives."""

    parser = _live_parser()
    _stream(parser, "Hi<|close|>" + " " * 12)
    assert parser._sent_content_idx == 2

    parser = _live_parser()
    _stream(parser, "Hi<|close|>", "Hi<|close|>" + " " * 10)
    assert parser._sent_content_idx == 2


def test_streaming_holds_tail_prefixes_inside_a_marker() -> None:
    """Parser-level twin of the pipe-prefix holdback unit test."""

    parser = _live_parser()
    _stream(parser, "Hi<|close|> response <|se")
    assert parser._sent_content_idx == 2

    parser = _live_parser()
    _stream(parser, "Hi<|close|> response <|", "Hi<|close|> response <|sep|>")
    assert parser._sent_content_idx == 2
    assert parser.finish_streaming() is None


# --- review 2026-10-15: logging-policy / required-contract / empty-body ---


def test_empty_k3_response_propagates_none_over_raw_markers() -> None:
    """P2 #3: an EMPTY response body strips to ``content=None``; the
    delegator must pass that through instead of leaking the raw XTML
    delimiters (``<|close|>response<|sep|>...``) to the client."""

    module, delegator = _make_delegator(([], _RAW), _stub_parser(_RAW, None))
    assert patch_delegating_tool_content.apply_to_module(module) is True
    tool_calls, out = delegator._extract_tool_calls(_RAW, object())
    assert tool_calls == []
    assert out is None


# --- review 2026-10-15 round 11: tool_choice="none" framing parse ---


def _none_extract_parser(
    content: str | None, tools_called: bool = False
) -> KimiK3ToolParser:
    parser = object.__new__(KimiK3ToolParser)
    parser._hcu_last_extract = None  # none short-circuits: no record

    def extract_tool_calls(model_output, request=None):
        return SimpleNamespace(
            tools_called=tools_called, tool_calls=[], content=content
        )

    parser.extract_tool_calls = extract_tool_calls  # type: ignore[method-assign]
    return parser


def test_none_choice_parses_k3_framing_without_a_record() -> None:
    """R11 P2: with tool_choice="none" the wheel never calls the tool
    parser, so no ``_hcu_last_extract`` record exists; the wrapper must
    still strip the XTML framing from the raw content channel."""

    module, delegator = _make_delegator(
        ([], _RAW), _none_extract_parser(_CLEAN)
    )
    assert patch_delegating_tool_content.apply_to_module(module) is True
    request = SimpleNamespace(tool_choice="none")
    tool_calls, out = delegator._extract_tool_calls(_RAW, request)
    assert tool_calls == []
    assert out == _CLEAN


def test_none_choice_suppresses_emitted_tool_calls() -> None:
    """R11 P2: a model that emitted calls despite tool_choice="none" keeps
    its pre-tools text as content; the calls are dropped (API contract)."""

    module, delegator = _make_delegator(
        ([], _RAW), _none_extract_parser("pre", tools_called=True)
    )
    assert patch_delegating_tool_content.apply_to_module(module) is True
    tool_calls, out = delegator._extract_tool_calls(
        _RAW, SimpleNamespace(tool_choice="none")
    )
    assert tool_calls == []
    assert out == "pre"


def test_none_choice_uses_the_live_k3_parser() -> None:
    """Live-parser variant: real XTML extraction strips the framing, and a
    response channel BEFORE the tools channel survives as content."""

    module, delegator = _make_delegator(([], _RAW), _live_parser())
    assert patch_delegating_tool_content.apply_to_module(module) is True
    tool_calls, out = delegator._extract_tool_calls(
        _RAW, SimpleNamespace(tool_choice="none")
    )
    assert tool_calls == []
    assert out == _CLEAN


def test_none_choice_beats_a_stale_exact_record() -> None:
    """R11 P3-1: the none check runs BEFORE the exact-record branch, so a
    leftover record from a previous request (same input, tools_called) on a
    reused parser instance cannot bypass the framing parse."""

    parser = object.__new__(KimiK3ToolParser)
    parser._hcu_last_extract = (
        _RAW,
        SimpleNamespace(tools_called=True, tool_calls=[], content=None),
    )

    def extract_tool_calls(model_output, request=None):
        return SimpleNamespace(
            tools_called=True, tool_calls=[], content="text"
        )

    parser.extract_tool_calls = extract_tool_calls  # type: ignore[method-assign]
    module, delegator = _make_delegator(([], _RAW), parser)
    assert patch_delegating_tool_content.apply_to_module(module) is True
    tool_calls, out = delegator._extract_tool_calls(
        _RAW, SimpleNamespace(tool_choice="none")
    )
    assert tool_calls == []
    assert out == "text"


def test_none_choice_with_a_tools_channel_keeps_only_pre_text() -> None:
    """R11 P3-6: live parser, raw output carrying a full tools channel --
    calls suppressed, the pre-tools text passes as content."""

    raw = (
        "Hi<|close|>response<|sep|><|open|>tools<|sep|>"
        '<|open|>call tool="py" index="1"<|sep|>'
        '<|open|>argument key="code" type="string"<|sep|>print(1)'
        "<|close|>argument<|sep|><|close|>call<|sep|>"
        "<|close|>tools<|sep|>"
    )
    module, delegator = _make_delegator(([], raw), _live_parser())
    assert patch_delegating_tool_content.apply_to_module(module) is True
    tool_calls, out = delegator._extract_tool_calls(
        raw, SimpleNamespace(tool_choice="none")
    )
    assert tool_calls == []
    assert out == "Hi"


def test_required_without_structural_tag_fails_loudly() -> None:
    """P2 #2 (+ round-5/6): `required` without an attached XTML tag has no
    constraint at all. Strict OFF tags nothing; strict ON normally attaches
    the tag BEFORE adjust_request (wheel adjust_request order), so a missing
    tag there means the attach was skipped (request already carries
    structured_outputs with no structural_tag) -- still unconstrained.
    Falling through to the generic JSON guidance would constrain output the
    K3 extractor can never decode, and the wheel routes such required output
    through the AUTO path (supports_required_and_named=False): raise, like
    the named guard. NOTE: intentional behavior change -- strict-OFF
    `required` used to pass through unconstrained, it now returns 400."""

    from vllm.exceptions import VLLMValidationError

    request = SimpleNamespace(
        tools=[{"type": "function", "function": {"name": "py"}}],
        tool_choice="required",
        structured_outputs=None,
        skip_special_tokens=True,
    )

    # no tag at all (strict calling disabled, nothing ever attaches)
    with pytest.raises(VLLMValidationError):
        _live_parser().adjust_request(request)

    # strict ON but the attach was SKIPPED: the request already carries
    # structured_outputs with structural_tag=None (round-6: the guard no
    # longer reads the env flag; this namespace shape is the real hole)
    request.structured_outputs = SimpleNamespace(structural_tag=None)
    with pytest.raises(VLLMValidationError):
        _live_parser().adjust_request(request)

    # a tag attached by the delegating side keeps the constrained XTML path
    request.structured_outputs = SimpleNamespace(structural_tag="x")
    adjusted = _live_parser().adjust_request(request)
    assert adjusted.skip_special_tokens is False
