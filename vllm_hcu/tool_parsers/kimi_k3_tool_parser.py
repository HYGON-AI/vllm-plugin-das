# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# Modified by Hygon Information Technology Co., Ltd., 2026.
"""Tool-call parser for the Kimi K3 (XTML) chat format.

This turns the generated XTML ``response`` and ``tools`` channels back into
OpenAI-compatible ``content`` and ``tool_calls``.

K3 assistant tool calls live in a nested ``tools`` channel::

    <|open|>tools<|sep|>
      <|open|>call tool="python" index="1"<|sep|>
        <|open|>argument key="code" type="string"<|sep|>print(1)<|close|>argument<|sep|>
        <|open|>argument key="opts" type="object"<|sep|>{"a":1}<|close|>argument<|sep|>
      <|close|>call<|sep|>
    <|close|>tools<|sep|>

where ``<|open|>``, ``<|close|>``, ``<|sep|>`` are dedicated special tokens. The plain
reply lives in a sibling ``response`` channel which is unwrapped into content.
``response`` always precedes ``tools`` (see the protocol channel grammar).

Argument decoding mirrors the template's type tagging (inverse encoding):
  * type="string"  -> value is the RAW text, no unescaping (template emits it raw)
  * other types    -> value is JSON-decoded (number/boolean/null/object/array)

Attribute values (``tool=``, ``index=``, ``key=``, ``type=``) ARE escaped on the
encode side (``&`` -> ``&amp;``, ``"`` -> ``&quot;``), so ``_attrs`` reverses
them on decode (``&quot;`` before ``&amp;`` -- reverse of encode order).

Known limitation: because string argument and response bodies are emitted raw,
a value that literally contains ``<|close|>argument<|sep|>`` or
``<|close|>response<|sep|>`` is indistinguishable from a real closing marker.
"""

import json
from collections.abc import Sequence

import regex as re
from openai.types.responses import ToolChoiceFunction
from vllm.entrypoints.chat_utils import make_tool_call_id

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.exceptions import VLLMValidationError
from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import Tool, ToolParser

logger = init_logger(__name__)

_O, _C, _S = r"<\|open\|>", r"<\|close\|>", r"<\|sep\|>"
_TEXT_UNTIL_SEP = r"(?:(?!" + _S + r").)*?"
# Literal marker fragments for the whitespace-tolerant holdback helper.
_OO, _CO, _SS = "<|open|>", "<|close|>", "<|sep|>"


def _partial_tag_overlap(text: str, tag: str) -> int:
    max_len = min(len(text), len(tag) - 1)
    for n in range(max_len, 0, -1):
        if text.endswith(tag[:n]):
            return n
    return 0


def _marker_holdback(text: str, head: str, word: str, tail: str) -> int:
    """Longest suffix that may still grow into a WHITESPACE-TOLERANT marker.

    The compiled marker regexes accept optional whitespace around the word
    (``<|close|> response <|sep|>``); a chunk boundary can land inside that
    shape (e.g. current ends at ``<|close|> ``), and the canonical-string
    overlap then sees nothing -- the partial marker would leak as content
    (review 2026-10-13 P2).  This check mirrors the same whitespace
    tolerance for suffixes: head (+spaces), head + word PREFIX, head + word
    (+spaces), head + word + TAIL PREFIX.  Canonical no-space prefixes are
    covered by :func:`_partial_tag_overlap` first.

    The whitespace runs are UNBOUNDED (review 2026-10-15 round 11 P2): the
    old fixed `+8`-slack window truncated long runs, which dropped the
    marker head and streamed the control prefix as content.  Instead of
    scanning a bounded window, anchor at every occurrence of ``head`` --
    every holdable suffix STARTS with ``head``, and an occurrence at index
    ``i`` yields a hold of ``len(text) - i`` when the remainder past the
    head fits one of the shapes -- and keep the earliest (longest) hold.
    """

    esc_head, esc_word = re.escape(head), re.escape(word)
    prefix_re = re.compile(esc_head + r"(\s*)([A-Za-z]*)$")
    word_re = re.compile(esc_head + r"\s*" + esc_word + r"\s*([<|a-z]*)")
    best = 0
    head_len = len(head)
    # Only the LAST head occurrence can anchor a holdable suffix: a suffix
    # that starts at an earlier head would have to cross a later ``head``
    # literal, and no holdable shape contains one (whitespace / letters /
    # word / the ``<|s`` tail prefix cannot produce it).  One anchored check
    # replaces the per-anchor scans of the first draft (review R12 P3).
    start = text.rfind(head)
    if start != -1 and len(text) - start > best:
        rest_re = re.compile(r"\s*")
        rest = text[start + head_len :]
        if rest_re.fullmatch(rest):
            best = len(text) - start
        else:
            pm = prefix_re.fullmatch(text, start)
            if (
                pm is not None
                and len(pm[2]) < len(word)
                and word.startswith(pm[2])
            ):
                best = len(text) - start
            else:
                wm = word_re.fullmatch(text, start)
                if wm is not None:
                    tail_prefix = wm[1]
                    if tail_prefix == "" or (
                        len(tail_prefix) < len(tail)
                        and tail.startswith(tail_prefix)
                    ):
                        best = len(text) - start
    return best


class KimiK3ToolParser(ToolParser):
    supports_required_and_named = False
    # Enables the vLLM-side XTML structural tag builder
    # (``get_kimi_k3_structural_tag`` in ``structural_tag_registry``). With
    # ``VLLM_ENFORCE_STRICT_TOOL_CALLING`` on (default), ``_apply_structural_tag``
    # constrains generation to K3's ``<|open|>tools<|sep|>`` channel for
    # ``required`` (and ``auto`` when a tool sets ``strict``) instead of the
    # generic JSON guided decoding, which conflicts with the XTML format.
    structural_tag_model = "kimi_k3"

    def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] | None = None):
        super().__init__(tokenizer, tools)

        self.tools_open = "<|open|>tools<|sep|>"
        self.tools_close = "<|close|>tools<|sep|>"
        self.response_open = "<|open|>response<|sep|>"
        self.response_close = "<|close|>response<|sep|>"
        self.message_close = "<|close|>message<|sep|>"

        # Regexes operate on detokenized text. The XTML markers reach us as the
        # literal strings <|open|>/<|close|>/<|sep|>. adjust_request keeps them
        # from being stripped and suppresses spacing between adjacent token
        # pieces. As DEFENSE-IN-DEPTH we still tolerate optional whitespace
        # WITHIN a marker (e.g. "<|open|> tools <|sep|>") in case some serving
        # path leaves vLLM's added-token spacing on -- this `\s*` is a no-op on
        # clean input, so the normal byte-exact path is unaffected. We do NOT
        # strip body whitespace, so clean content stays byte-exact.
        # All bodies use non-greedy (.*?) so each block stops at its own first
        # closing marker -- see the module-level KNOWN LIMITATION about
        # literal-marker values.
        self._tools_open_re = re.compile(_O + r"\s*tools\s*" + _S)
        self._tools_close_re = re.compile(_C + r"\s*tools\s*" + _S)
        self._response_open_re = re.compile(_O + r"\s*response\s*" + _S)
        self._response_close_re = re.compile(_C + r"\s*response\s*" + _S)
        self._message_close_re = re.compile(_C + r"\s*message\s*" + _S)
        self._call_re = re.compile(
            _O
            + r"\s*call\s+(?P<attrs>"
            + _TEXT_UNTIL_SEP
            + r")"
            + _S
            + r"(?P<body>.*?)"
            + _C
            + r"\s*call\s*"
            + _S,
            re.DOTALL,
        )
        self._arg_re = re.compile(
            _O
            + r"\s*argument\s+(?P<attrs>"
            + _TEXT_UNTIL_SEP
            + r")"
            + _S
            + r"(?P<val>.*?)"
            + _C
            + r"\s*argument\s*"
            + _S,
            re.DOTALL,
        )
        # R13 P2: streaming emits call headers and argument payloads while
        # the call is still open; the closed ``_call_re`` stays reserved
        # for finalization of the whole call.
        self._call_open_re = re.compile(
            _O + r"\s*call\s+(?P<attrs>" + _TEXT_UNTIL_SEP + r")" + _S
        )
        self._arg_open_re = re.compile(
            _O + r"\s*argument\s+(?P<attrs>" + _TEXT_UNTIL_SEP + r")" + _S
        )
        self._call_close_re = re.compile(_C + r"\s*call\s*" + _S)
        # attr segment: key="value" (value already escaped on the encode side)
        self._attr_re = re.compile(r'(?P<k>\w+)="(?P<v>[^"]*)"')
        self._response_re = re.compile(
            _O + r"\s*response\s*" + _S + r"(?P<c>.*?)" + _C + r"\s*response\s*" + _S,
            re.DOTALL,
        )

        # streaming state
        self._sent_content_idx = 0
        self._sent_tool_call_count = 0
        # R13 P2: the ONE partially generated call being streamed
        # incrementally (models emit calls sequentially -- the grammar
        # allows at most one open call at any prefix):
        # {header_start, index, id, name, emitted}; ``emitted`` is the
        # length of the JSON arguments prefix already sent.  Recompute
        # everything else from the accumulated text each delta.
        self._streaming_call: dict | None = None
        # Stream-end flush + delegation bookkeeping (review 2026-10-12):
        # _hcu_last_text  -> accumulated text for finish_streaming();
        # _hcu_last_extract -> (raw model_output, info) so the delegating
        # no-tool-call branch can pass through the CLEANED content (upstream
        # discards it; fork 877e6996 keeps it).
        self._hcu_last_text: str | None = None
        self._hcu_last_extract: tuple[str, ExtractedToolCallInformation] | None = None

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ToolParser "
                "constructor during construction."
            )

    def adjust_request(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> ChatCompletionRequest | ResponsesRequest:
        named = isinstance(
            request.tool_choice,
            (ChatCompletionNamedToolChoiceParam, ToolChoiceFunction),
        )
        structured_outputs = getattr(request, "structured_outputs", None)
        has_structural_tag = (
            structured_outputs is not None
            and structured_outputs.structural_tag is not None
        )
        if named and not has_structural_tag:
            # Without the XTML structural tag there is no way to force the
            # named call (the generic JSON guided-decoding path conflicts
            # with the XTML channel format).
            raise VLLMValidationError(
                "Named tool choice for Kimi K3 requires strict tool calling "
                "(VLLM_ENFORCE_STRICT_TOOL_CALLING) so the XTML structural "
                "tag can force the call. Otherwise use `tool_choice` \"auto\" "
                'or "none".',
                parameter="tool_choice",
                value=request.tool_choice,
            )

        if request.tools and (request.tool_choice == "required" or named):
            # K3 emits tool calls in XTML. When strict tool calling is enabled,
            # DelegatingParser._apply_structural_tag has already attached the K3
            # XTML structural tag (structural_tag_model = "kimi_k3"). We return
            # early to skip the generic parent path, which would otherwise attach
            # JSON guided decoding when strict calling is off -- that JSON
            # constraint conflicts with the <|open|>tools<|sep|> channel.
            #
            # Review 2026-10-15 P2 (+round-5 follow-up): `required` reaches
            # here unconstrained whenever no XTML tag is attached.  The
            # generic parent's JSON guidance is NOT an escape hatch: the
            # wheel routes required-with-``supports_required_and_named=False``
            # through the AUTO parse path, so JSON-constrained output would
            # never decode as K3 XTML -- the user would get raw JSON as
            # content with no tool calls.  And with strict calling ON the tag
            # is normally attached BEFORE this call (wheel adjust_request
            # order), so a missing tag means the attach was skipped (request
            # already carries structured_outputs) -- still unconstrained.
            # Both shapes violate the `required` contract: raise, mirroring
            # the named guard above.
            if (
                request.tool_choice == "required"
                and not named
                and not has_structural_tag
            ):
                raise VLLMValidationError(
                    '`required` tool choice for Kimi K3 arrived without the '
                    "XTML structural tag, so the call is unconstrained: "
                    "either strict tool calling "
                    "(VLLM_ENFORCE_STRICT_TOOL_CALLING) is disabled, or tag "
                    "attachment was skipped because the request already "
                    "carries structured_outputs/response_format. Enable "
                    "strict tool calling and drop the request-level "
                    'structured override, or use `tool_choice` "auto" or '
                    '"none".',
                    parameter="tool_choice",
                    value=request.tool_choice,
                )
            request.skip_special_tokens = False
            if hasattr(request, "spaces_between_special_tokens"):
                request.spaces_between_special_tokens = False
            return request

        request = super().adjust_request(request)
        # The XTML markers (<|open|>/<|close|>/<|sep|>,
        # <|open|>response<|sep|> ...) must
        # reach this parser as CONTIGUOUS literal text. Two request flags govern
        # that, and vLLM's detokenizer couples them:
        #   effective_spaces_between =
        #       skip_special_tokens OR spaces_between_special_tokens
        #   (vllm/v1/engine/detokenizer.py).
        # The detokenizer treats these control tokens as separate sub-texts and,
        # when the effective flag is True, joins them with spaces ->
        # "<|open|> response <|sep|>", which the regexes below would NOT match.
        # Forcing BOTH flags off is the only way to suppress that spacing. We
        # set both unconditionally: the response channel is unwrapped from these
        # markers even with no tools.
        request.skip_special_tokens = False
        if hasattr(request, "spaces_between_special_tokens"):
            request.spaces_between_special_tokens = False
        return request

    def _attrs(self, s: str) -> dict[str, str]:
        return {
            m["k"]: m["v"].replace("&quot;", '"').replace("&amp;", "&")
            for m in self._attr_re.finditer(s)
        }

    def _decode_call(self, attrs: str, body: str) -> ToolCall | None:
        """Decode one ``call`` block into a :class:`ToolCall`.

        ``attrs`` is the text between ``<|open|>call `` and ``<|sep|>`` (carries
        ``tool=`` / ``index=``); ``body`` is the sequence of ``argument`` blocks.
        Each argument is re-typed per its ``type=`` tag: strings pass through
        raw, everything else is JSON-decoded (falling back to raw text if the
        JSON is malformed, so a partial stream never raises). Returns ``None``
        when no tool name is present (an empty/garbage block is dropped).
        """
        call_attrs = self._attrs(attrs)
        tool_name = call_attrs.get("tool", "")
        arguments: dict = {}
        for arg_match in self._arg_re.finditer(body):
            arg_attrs = self._attrs(arg_match["attrs"])
            key = arg_attrs.get("key", "")
            arg_type = arg_attrs.get("type", "string")
            raw_value = arg_match["val"]
            if arg_type == "string":
                arguments[key] = raw_value
            else:
                try:
                    arguments[key] = json.loads(raw_value)
                except json.JSONDecodeError:
                    arguments[key] = raw_value
        if not tool_name:
            return None
        # Review 2026-10-13 P1: K3 restarts the XTML index at 1 in EVERY
        # assistant turn, so ``tool:index`` collides across turns
        # (get_weather:0 twice).  Consumers key results/traces by call id ->
        # wrong-turn attachment.  Keep the readable name prefix but make the
        # suffix unique per call with the wheel's generated-id fallback.
        tool_call_id = f"{tool_name}:{make_tool_call_id()}"
        return ToolCall(
            id=tool_call_id,
            type="function",
            function=FunctionCall(
                name=tool_name,
                arguments=json.dumps(arguments, ensure_ascii=False),
            ),
        )

    def _strip_response_content(self, text: str) -> str | None:
        """Strip XTML response/message markers from generated response text.

        In chat serving, ``<|open|>response<|sep|>`` is often part of the prompt
        generation prefix, so the model output may only contain the body plus
        ``<|close|>response<|sep|>``. Handle both that consumed-prefix shape and a
        complete ``<|open|>response<|sep|>... <|close|>response<|sep|>`` wrapper.
        """
        m_open = self._response_open_re.search(text)
        if m_open is not None:
            m_close = self._response_close_re.search(text, m_open.end())
            if m_close is not None:
                text = text[m_open.end() : m_close.start()]
            else:
                text = text[m_open.end() :]
        else:
            text = self._response_close_re.sub("", text)
        text = self._message_close_re.sub("", text)
        return text or None

    def _content(self, model_output: str, before: str) -> str | None:
        # prefer the unwrapped response channel; else the text before the tools
        m = self._response_re.search(model_output)
        if m is not None:
            return m["c"] or None
        return self._strip_response_content(before)

    def _extract_response_content(self, current_text: str) -> str | None:
        # Streaming response text is computed from the accumulated text. This is
        # what keeps split markers from leaking:
        #   <|open|> / response / <|sep|>Hi     -> emit only "Hi" after open closes
        #   Hi<|open|> / tools / <|sep|>...     -> emit "Hi", hold the tools marker
        #   Hi<|close|> / response / <|sep|>... -> emit "Hi", hold the close marker
        # Tool calls are even simpler: they are not emitted until a full
        # <|close|>call<|sep|> is present in current_text.
        m_open = self._response_open_re.search(current_text)
        # In the normal chat path, the response-open marker may be consumed as
        # the generation prefix. Then generated text starts directly with the
        # response body or with <|close|>response<|sep|> before a tools channel.
        body_start = m_open.end() if m_open is not None else 0
        m_tools = self._tools_open_re.search(current_text, body_start)
        m_rclose = self._response_close_re.search(current_text, body_start)
        # Review 2026-10-13 P2: non-strict auto output may end with
        # <|close|>message<|sep|> WITHOUT a response-close first (the message
        # close is the model's turn terminator).  The non-streaming path
        # strips it (_strip_response_content -> _message_close_re), so the
        # streaming boundary set must include it too or the raw control
        # marker streams to the client.
        m_mclose = self._message_close_re.search(current_text, body_start)
        tools_start = m_tools.start() if m_tools else -1
        response_end = m_rclose.start() if m_rclose else -1
        message_end = m_mclose.start() if m_mclose else -1

        candidates = [
            i for i in (tools_start, response_end, message_end) if i != -1
        ]
        if candidates:
            sendable_idx = min(candidates)
        else:
            # Holdback must consider BOTH the canonical markers and the
            # whitespace-tolerant forms the regexes accept
            # (review 2026-10-13 P2: '<|close|> ' / '<|close|> resp' chunks).
            overlap = max(
                _partial_tag_overlap(current_text, self.response_open),
                _partial_tag_overlap(current_text, self.response_close),
                _partial_tag_overlap(current_text, self.tools_open),
                _partial_tag_overlap(current_text, self.message_close),
                _marker_holdback(current_text, _OO, "response", _SS),
                _marker_holdback(current_text, _CO, "response", _SS),
                _marker_holdback(current_text, _OO, "tools", _SS),
                _marker_holdback(current_text, _CO, "message", _SS),
            )
            sendable_idx = len(current_text) - overlap

        if sendable_idx <= body_start:
            return None
        if self._sent_content_idx < body_start:
            self._sent_content_idx = body_start
        if sendable_idx <= self._sent_content_idx:
            return None

        content = current_text[self._sent_content_idx : sendable_idx]
        self._sent_content_idx = sendable_idx
        return content or None

    def extract_tool_calls(
        self, model_output: str, request: ChatCompletionRequest
    ) -> ExtractedToolCallInformation:
        info = self._extract_tool_calls_impl(model_output, request)
        self._hcu_last_extract = (model_output, info)
        return info

    def _extract_tool_calls_impl(
        self, model_output: str, request: ChatCompletionRequest
    ) -> ExtractedToolCallInformation:
        m_open = self._tools_open_re.search(model_output)
        if m_open is None:
            # no tools channel -> content is the response channel (unwrapped)
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=self._content(model_output, model_output),
            )
        try:
            before = model_output[: m_open.start()]
            start = m_open.end()
            m_close = self._tools_close_re.search(model_output, start)
            section = (
                model_output[start:]
                if m_close is None
                else model_output[start : m_close.start()]
            )

            tool_calls = [
                tc
                for m in self._call_re.finditer(section)
                if (tc := self._decode_call(m["attrs"], m["body"])) is not None
            ]
            if not tool_calls:
                return ExtractedToolCallInformation(
                    tools_called=False,
                    tool_calls=[],
                    content=self._content(model_output, before),
                )
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=self._content(model_output, before),
            )
        except Exception:
            logger.exception("Error extracting K3 tool calls.")
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        # Track the accumulated text; the platform patch wraps the
        # delegating parse_delta to call finish_streaming() on the
        # finished=True turn (review 2026-10-12 P2 #5).
        self._hcu_last_text = current_text

        # Stream unwrapped response-channel text as content; tool calls are
        # streamed incrementally: the call header (id + name) as soon as it
        # completes, argument payloads while they generate, and the closing
        # ``}`` when the call closes (review R13 P2 -- the old closed-only
        # path left clients silent for minutes during long arguments).
        content = self._extract_response_content(current_text)

        m_tools = self._tools_open_re.search(current_text)
        if m_tools is None:
            return DeltaMessage(content=content) if content else None

        section = current_text[m_tools.end() :]
        # Nothing after the tools close marker is a call.
        m_tools_close = self._tools_close_re.search(section)
        if m_tools_close is not None:
            section = section[: m_tools_close.start()]

        closed_matches = list(self._call_re.finditer(section))
        calls = [
            tc
            for m in closed_matches
            if (tc := self._decode_call(m["attrs"], m["body"])) is not None
        ]

        # R14 P1: retirement is independent of the open scan.  A delta can
        # close call k AND open call k+1 (speculative decoding, stream
        # interval > 1); the elif below ran the retirement only when NO
        # header followed, which dropped k's tail and emitted k+1's header
        # reusing k's index.
        closed_by_start = {m.start(): m for m in closed_matches}
        deltas: list[DeltaToolCall] = []
        st = self._streaming_call
        if st is not None and st["header_start"] in closed_by_start:
            deltas.extend(
                self._retire_streaming_call(
                    closed_by_start[st["header_start"]], st
                )
            )
            self._streaming_call = None
            st = None

        # Order follows POSITION (review R15 P1): every CLOSED call that
        # was never streamed emits BEFORE the next header opens, so one
        # delta delivering "short call k closes + call k+1 header" keeps
        # indices dense (k at its index, k+1 at the next) instead of
        # dropping k and reusing its slot for k+1.
        pending = calls[self._sent_tool_call_count :]
        if pending:
            deltas.extend(
                DeltaToolCall(
                    index=self._sent_tool_call_count + i,
                    id=tc.id,
                    type="function",
                    function=DeltaFunctionCall(
                        name=tc.function.name, arguments=tc.function.arguments
                    ).model_dump(exclude_none=True),
                )
                for i, tc in enumerate(pending)
            )
            self._sent_tool_call_count += len(pending)

        open_match = self._call_open_re.search(
            section, closed_matches[-1].end() if closed_matches else 0
        )
        if open_match is not None and not self._attrs(open_match["attrs"]).get(
            "tool"
        ):
            open_match = None

        if open_match is not None:
            if st is not None and st["header_start"] != open_match.start():
                # Defensive: a record that matches neither the closed body
                # nor the new header cannot be trusted.  Retire its index
                # so the restart cannot alias a client-side accumulator
                # (review R16 P3-1); unreachable under sequential output.
                self._sent_tool_call_count = st["index"] + 1
                self._streaming_call = None
                st = None
            if st is None:
                meta = self._attrs(open_match["attrs"])
                name = meta.get("tool", "")
                st = {
                    "header_start": open_match.start(),
                    "index": self._sent_tool_call_count,
                    "id": f"{name}:{make_tool_call_id()}",
                    "name": name,
                    "emitted": 0,
                }
                self._streaming_call = st
                deltas.append(
                    DeltaToolCall(
                        index=st["index"],
                        id=st["id"],
                        type="function",
                        function=DeltaFunctionCall(
                            name=name, arguments=""
                        ).model_dump(exclude_none=True),
                    )
                )
            body = section[open_match.end() :]
            prefix = self._streaming_arguments_prefix(body, final=False)
            if len(prefix) > st["emitted"]:
                deltas.append(
                    DeltaToolCall(
                        index=st["index"],
                        function=DeltaFunctionCall(
                            arguments=prefix[st["emitted"] :]
                        ).model_dump(exclude_none=True),
                    )
                )
                st["emitted"] = len(prefix)

        if deltas:
            return DeltaMessage(content=content, tool_calls=deltas)
        return DeltaMessage(content=content) if content else None

    def _retire_streaming_call(self, match, st: dict) -> list[DeltaToolCall]:
        """Emit the remaining arguments of a closed streamed call.

        ``match`` is the closed call located by its header start (R14 P1:
        index-based lookup can drift past unnamed blocks).  The streamed
        non-string values are raw model text; when their concatenation is
        not valid JSON the canonical closed-path dump takes over -- but
        only if it extends the already-streamed bytes, otherwise the
        incremental output stands as-is (documented limitation).
        """

        final_prefix = self._streaming_arguments_prefix(
            match["body"], final=True
        )
        # The incremental algorithm is deterministic: what was streamed is
        # exactly the final prefix truncated at ``emitted``.
        streamed = final_prefix[: st["emitted"]]
        tail = final_prefix[st["emitted"] :]
        if tail:
            try:
                json.loads(final_prefix)
            except ValueError:
                # Raw non-string junk (Python literals, truncation): prefer
                # the canonical closed-path dump -- but only when it extends
                # the already-streamed bytes; otherwise the incremental
                # output stands as-is (documented limitation).
                decode = self._decode_call(match["attrs"], match["body"])
                canonical = (
                    decode.function.arguments if decode is not None else None
                )
                if canonical is not None and canonical.startswith(streamed):
                    tail = canonical[st["emitted"] :]
        self._sent_tool_call_count = st["index"] + 1
        if not tail:
            return []
        return [
            DeltaToolCall(
                index=st["index"],
                function=DeltaFunctionCall(arguments=tail).model_dump(
                    exclude_none=True
                ),
            )
        ]

    @staticmethod
    def _json_escape(value: str) -> str:
        # Per-character (context-free) JSON string escaping -- identical to
        # the closed-path ``json.dumps(..., ensure_ascii=False)`` for values
        # and prefix-stable across delta boundaries.
        return json.dumps(value, ensure_ascii=False)[1:-1]

    def _streaming_arguments_prefix(self, body: str, final: bool) -> str:
        """JSON-so-far for the OPEN call's argument sequence.

        Recomputed from scratch each delta (same stateless pattern as the
        other paths); the only persistent state is how much of the prefix
        was already emitted, so fragments can be diffed out.  The output
        shape mirrors the closed path's ``json.dumps(arguments,
        ensure_ascii=False)``: ``{"k1": v1, "k2": v2}``.  Duplicate keys
        render EVERY occurrence (review 2026-10-15 round 17 P2): clients
        assembling with ``json.loads`` then keep the last value -- the
        same last-wins semantics as the closed path's dict extraction.
        Clients that patch key-by-key under a single-key assumption will
        see the same key twice (only when the model actually repeats a
        key, which the renderer never does).  Remaining known limitation:
        a truncated non-string (object/array)
        value cannot be repaired into valid JSON at end-of-stream.
        """

        parts: list[str] = []
        pos = 0
        for m in self._arg_re.finditer(body):
            attrs = self._attrs(m["attrs"])
            key = attrs.get("key", "")
            is_string = attrs.get("type", "string") == "string"
            rendered = (
                f'"{self._json_escape(m["val"])}"'
                if is_string
                else m["val"]
            )
            parts.append(
                f'{", " if parts else ""}'
                f"{json.dumps(key, ensure_ascii=False)}: {rendered}"
            )
            pos = m.end()

        m_open = self._arg_open_re.search(body, pos)
        trailing_string = False
        if m_open is not None:
            attrs = self._attrs(m_open["attrs"])
            key = attrs.get("key", "")
            trailing_string = attrs.get("type", "string") == "string"
            raw = body[m_open.end() :]
            # Hold back any suffix that may still grow into the
            # argument close marker -- canonical prefix plus the
            # whitespace-tolerant shapes (same pairing as the content
            # channel and review R11's holdback fix).  At end-of-stream
            # nothing more can arrive, so the hold MUST drop -- a
            # literal trailing ``<`` is data, not a prefix (R12 P2 #5
            # precedent on the content channel; review R14 P3).
            hold = (
                0
                if final
                else max(
                    _partial_tag_overlap(raw, _CO + "argument" + _SS),
                    _marker_holdback(raw, _CO, "argument", _SS),
                )
            )
            streamable = raw[: len(raw) - hold] if hold else raw
            joiner = ", " if parts else ""
            if trailing_string:
                parts.append(
                    f'{joiner}{json.dumps(key, ensure_ascii=False)}: '
                    f'"{self._json_escape(streamable)}'
                )
            else:
                parts.append(
                    f'{joiner}{json.dumps(key, ensure_ascii=False)}: '
                    f"{streamable}"
                )

        prefix = "{" + "".join(parts)
        if final:
            if m_open is not None and trailing_string:
                # repair the truncated string value into valid JSON
                prefix += '"'
            prefix += "}"
        return prefix

    def _finalize_streaming_call(self) -> list[DeltaToolCall] | None:
        """Flush an unfinished open call at end-of-stream.

        Emits whatever remains of the arguments prefix (with repaired
        closing quotes, where the value shape allows it) exactly once, and
        clears the streaming state.
        """

        st = self._streaming_call
        if st is None:
            return None
        self._streaming_call = None
        text = self._hcu_last_text
        if text is None:
            return None
        m_tools = self._tools_open_re.search(text)
        if m_tools is None:
            return None
        section = text[m_tools.end() :]
        m_tools_close = self._tools_close_re.search(section)
        if m_tools_close is not None:
            section = section[: m_tools_close.start()]
        last_closed_end = 0
        for m in self._call_re.finditer(section):
            last_closed_end = m.end()
        open_match = self._call_open_re.search(section, last_closed_end)
        if open_match is None or open_match.start() != st["header_start"]:
            # The call closed already, or the record cannot be re-derived;
            # the regular path handled it.
            return None
        body = section[open_match.end() :]
        if self._call_close_re.search(body) is not None:
            return None
        prefix = self._streaming_arguments_prefix(body, final=True)
        if len(prefix) <= st["emitted"]:
            return None
        self._sent_tool_call_count = st["index"] + 1
        return [
            DeltaToolCall(
                index=st["index"],
                function=DeltaFunctionCall(
                    arguments=prefix[st["emitted"] :]
                ).model_dump(exclude_none=True),
            )
        ]

    def finish_streaming(self) -> DeltaMessage | None:
        """Release response text held back as a partial-tag prefix.

        Called by the delegating parser at stream end (``finished=True``
        routes through the wrapped ``parse_delta``).  A literal trailing
        ``<`` is held only because it may grow into a channel marker; at
        end-of-stream it is literal and must be emitted (review 2026-10-12,
        P2 #5: ``'1 '`` + ``finished=True`` dropped the ``<``).  The flush
        window is EXACTLY the streaming window: it starts at the response
        body (the ``<|open|>response<|sep|>`` open marker is markup when the
        model generated it, comment rounds 2/N3) and stops at the first
        structural marker (``<|close|>response<|sep|>``,
        ``<|open|>tools<|sep|>``) -- everything from there on is markup,
        never content.
        """

        text = self._hcu_last_text
        if text is None:
            return None
        m_open = self._response_open_re.search(text)
        body_start = m_open.end() if m_open is not None else 0
        start = max(self._sent_content_idx, body_start)
        m_tools = self._tools_open_re.search(text, body_start)
        m_rclose = self._response_close_re.search(text, body_start)
        m_mclose = self._message_close_re.search(text, body_start)
        candidates = [
            position
            for position in (
                m_tools.start() if m_tools is not None else -1,
                m_rclose.start() if m_rclose is not None else -1,
                m_mclose.start() if m_mclose is not None else -1,
            )
            if position != -1
        ]
        limit = min(candidates) if candidates else len(text)
        flush: DeltaMessage | None = None
        if limit > start:
            held = text[start:limit]
            self._sent_content_idx = limit
            flush = DeltaMessage(content=held) if held else None
        # R13 P2: a call truncated mid-generation releases its remaining
        # arguments (closing quote repaired for string values) exactly once.
        final_deltas = self._finalize_streaming_call()
        if final_deltas:
            if flush is None:
                return DeltaMessage(tool_calls=final_deltas)
            flush.tool_calls = final_deltas
        return flush
