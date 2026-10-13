# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Review 2026-10-12 fixes #4 and #5 on the delegating parser.

#4  Upstream ``DelegatingParser._extract_tool_calls`` (0.25.1+das) discards
    the tool parser's unwrapped content when it finds no tool calls: with
    ``tool_choice="auto"`` and a plain reply, the raw XTML end markers
    (``...<|close|>response<|sep|><|close|>message<|sep|>``) reach the user
    even though the parser returned the cleaned body.  The evaluated fork
    keeps the parser content in that branch ("Preserve any content the
    parser unwrapped even when it found no tool calls").  The wrapper below
    restores that for the K3 XTML parser only, bound by exact raw-input
    match on the extraction the parser itself recorded.

#5  A streaming reply ending inside a channel marker holds the partial tag
    back ('1 ' + '<'); nothing on the wheel flushes it at stream end, so
    the final '<' is dropped (non-streaming keeps it: '1 <').  The parser
    exposes ``finish_streaming()`` (fork flush protocol); this patch wraps
    ``DelegatingParser.parse_delta`` to call it exactly on the
    ``finished=True`` turn and merge the flush delta.  The wrap is SOFT: if
    the installed wheel's ``parse_delta`` has no ``finished`` parameter, it
    is left untouched (upstream behavior, no regression) instead of failing
    the boot.

Routing note: neither wrapper touches ``engine_based_streaming`` -- the
composite ``_engine_based`` is computed once at delegator construction and
drives engine-side plumbing; flipping it would reroute more than the fix
allows.  Both behaviors ride the token-side call path unchanged.
"""

from __future__ import annotations

import functools
import inspect
import sys
from types import ModuleType

from ._common import (
    PatchCompatibilityError,
    apply_once,
    load_exact_module,
    require_callable,
)

# NOTE: NO module-level ``vllm`` imports here. This patch module is
# imported during plugin registration (patch/platform/core_fix/__init__),
# and anything pulling vllm eagerly breaks the clean-import lifecycle
# contract (tests/patch/test_plugin_lifecycle.py: heavy == []). The logger
# is resolved lazily inside the failure branch, where the serving process
# has vllm (and torch) imported anyway.

TARGET_MODULE = "vllm.parser.abstract_parser"
PATCH_ID = "platform.core_fix.kimi_k3_delegating_content_pass"
TARGETS = (
    f"{TARGET_MODULE}.DelegatingParser._extract_tool_calls",
    f"{TARGET_MODULE}.DelegatingParser.parse_delta",
)
_MARKER = "_vllm_hcu_kimi_k3_content_pass_applied"
_WRAPPER = "_vllm_hcu_kimi_k3_content_pass_wrapper"
_PARSER_MODULE = "vllm_hcu.tool_parsers.kimi_k3_tool_parser"
_PARSER_CLASS = "KimiK3ToolParser"


def _parser_cls():
    loaded = sys.modules.get(_PARSER_MODULE)
    cls = getattr(loaded, _PARSER_CLASS, None) if loaded is not None else None
    if cls is None:
        import importlib

        cls = getattr(importlib.import_module(_PARSER_MODULE), _PARSER_CLASS)
    return cls


def _is_k3_parser(tool_parser) -> bool:
    return type(tool_parser) is _parser_cls() if tool_parser is not None else False


def _wrap_extract(cls: type, module: ModuleType) -> None:
    original = require_callable(cls, "_extract_tool_calls", TARGETS[0])

    @functools.wraps(original)
    def hcu_extract_tool_calls(self, content, request, *args, **kwargs):
        tool_calls, out = original(self, content, request, *args, **kwargs)
        if tool_calls or not _is_k3_parser(getattr(self, "_tool_parser", None)):
            # Calls were made: upstream already adopted info.content. Other
            # parsers keep the upstream contract untouched.
            return tool_calls, out
        if getattr(request, "tool_choice", None) == "none":
            # Review 2026-10-15 round 11 P2: the wheel's none-branch
            # short-circuits BEFORE any tool-parser extraction, so no
            # ``_hcu_last_extract`` record exists -- and because
            # adjust_request deliberately KEEPS the XTML special tokens, the
            # raw ``<|close|>response<|sep|>`` / message-close framing would
            # reach the client.  Run the K3 content extractor for this
            # disabled-tools path: parse the framing for the content channel
            # and SUPPRESS any tool calls the model emitted anyway (the API
            # contract for "none" is content only).  Checked BEFORE the
            # exact-record branch (review R11 P3-1): a stale exact-input
            # record left over from a previous request must not bypass the
            # framing parse.
            try:
                info = self._tool_parser.extract_tool_calls(
                    content if isinstance(content, str) else "",
                    request=request,
                )
            except Exception as exc:  # noqa: BLE001 - keep upstream result
                # Review R11 P3-2: never silent -- a framing leak here is
                # otherwise undebuggable in production.
                from vllm.logger import init_logger

                init_logger(__name__).warning(
                    "kimi_k3 tool_choice=none framing extraction failed; "
                    "returning the raw wheel output unchanged: %r",
                    exc,
                )
                return tool_calls, out
            return [], getattr(info, "content", None)
        record = self._tool_parser._hcu_last_extract
        if record is None or record[0] != content:
            # Only rewrite when the parser ran on exactly this input.
            return tool_calls, out
        info = record[1]
        if info is not None and not getattr(info, "tools_called", False):
            # Propagate the parser's cleaned result EVEN WHEN IT IS NONE
            # (review 2026-10-15 P2): the exact-input record above proves
            # this text WAS handled by the K3 parser, so a
            # ``content=None`` means the parser deliberately withheld
            # everything user-facing.  That covers BOTH the empty
            # response body (``_strip_response_content`` folds "" into
            # None) and a present-but-unparseable tools channel (e.g.
            # max_tokens truncation mid-call), where the alternative is
            # leaking the raw XTML delimiters; None is also the wheel's
            # own "no content" representation in this branch.
            out = getattr(info, "content", None)
        return tool_calls, out

    setattr(hcu_extract_tool_calls, _WRAPPER, True)
    cls._extract_tool_calls = hcu_extract_tool_calls


def _merge_flush(delta_message, flush):
    if flush is None:
        return delta_message
    if delta_message is None:
        return flush
    if getattr(flush, "content", None):
        delta_message.content = (delta_message.content or "") + flush.content
    if getattr(flush, "tool_calls", None):
        delta_message.tool_calls = (delta_message.tool_calls or []) + flush.tool_calls
    return delta_message


def _none_stream_delta(tool_parser, request, result, finished):
    """Streamed ``tool_choice="none"`` framing strip (review round 17 P2).

    The wheel short-circuits ``parse_delta`` for disabled tools, so the
    K3 XTML response/message framing streams to the client unfiltered and
    the finish-time flush cannot retract what earlier chunks already
    sent.  Drive the parser's cumulative content extractor over the
    accumulated VISIBLE text -- the original call's ``result.content``
    (review R17 P1-2: raw ``delta_text`` would also fold in reasoning
    segments that upstream already routed elsewhere).  Markers are
    stripped, split markers are held back across chunk boundaries, and
    any tool calls the model emitted anyway are suppressed (the "none"
    contract is content only).  State is keyed by request identity; a
    parser instance reused across requests restarts its buffer on the
    first mismatch.
    """

    if result is None:
        # Review R17 P1-1: an empty delta (or a reasoning boundary) makes
        # the wheel return None -- keep None semantics; nothing visible
        # means nothing to accumulate or emit.
        return None

    rid = getattr(request, "request_id", None) or id(request)
    state = getattr(tool_parser, "_hcu_none_state", None)
    if state is None or state[0] != rid:
        # A stale streaming-content index from a previous request on the
        # same instance must not clip this request's head.
        tool_parser._sent_content_idx = 0
        previous = ""
    else:
        previous = state[1]
    # Only text the wheel itself made visible belongs in the buffer.
    visible = getattr(result, "content", None) or ""
    current = previous + visible
    tool_parser._hcu_none_state = (rid, current)

    content = tool_parser._extract_response_content(current) or ""
    if finished:
        tail_start = tool_parser._sent_content_idx
        tail = current[tail_start:]
        # Review R17 P2-1: if the model emitted a tools channel despite
        # tool_choice="none", that channel must never leak as content --
        # cut it like the non-streaming none path does, THEN strip the
        # remaining framing (the holdback releases as data).
        m_tools = tool_parser._tools_open_re.search(tail)
        if m_tools is not None:
            tail = tail[: m_tools.start()]
        content += tool_parser._strip_response_content(tail) or ""
        tool_parser._hcu_none_state = None
        tool_parser._sent_content_idx = 0

    # In-place mutation keeps every other field (e.g. reasoning content)
    # untouched and works for pydantic models and test namespaces alike.
    result.content = content or None
    result.tool_calls = None
    return result


def _wrap_parse_delta_soft(cls: type, module: ModuleType) -> bool:
    """Wrap parse_delta only when the wheel carries the finished protocol."""

    original = getattr(cls, "parse_delta", None)
    if not callable(original):
        return False
    try:
        signature = inspect.signature(original)
    except (TypeError, ValueError):
        return False
    if "finished" not in signature.parameters:
        return False

    @functools.wraps(original)
    def hcu_parse_delta(self, delta_text, delta_token_ids, request, *args, **kwargs):
        finished = bool(kwargs.get("finished", False)) or _bound_finished(
            signature, self, delta_text, delta_token_ids, request, args, kwargs
        )
        result = original(self, delta_text, delta_token_ids, request, *args, **kwargs)
        tool_parser = getattr(self, "_tool_parser", None)
        if (
            not _is_k3_parser(tool_parser)
            or getattr(request, "tool_choice", None) != "none"
        ):
            if not finished:
                return result
            if _is_k3_parser(tool_parser):
                finish = getattr(tool_parser, "finish_streaming", None)
                if callable(finish):
                    result = _merge_flush(result, finish())
            return result
        return _none_stream_delta(tool_parser, request, result, finished)

    setattr(hcu_parse_delta, _WRAPPER, True)
    cls.parse_delta = hcu_parse_delta
    return True


def _bound_finished(signature, self, delta_text, delta_token_ids, request, args, kwargs) -> bool:
    """finished passed positionally (wheel variants without kw-only)."""

    try:
        binding = signature.bind(self, delta_text, delta_token_ids, request, *args, **kwargs)
    except TypeError:
        return False
    return bool(binding.arguments.get("finished", False))


def apply_to_module(module: ModuleType) -> bool:
    delegating_module = load_exact_module(TARGET_MODULE, module)
    cls = getattr(delegating_module, "DelegatingParser", None)
    if not isinstance(cls, type):
        raise PatchCompatibilityError(
            "required HCU patch target "
            "vllm.parser.abstract_parser.DelegatingParser is missing"
        )
    original = getattr(cls, "_extract_tool_calls", None)
    if getattr(original, _WRAPPER, False):
        return False
    if not callable(original):
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGETS[0]} is missing"
        )
    _parser_cls()  # fail fast if the HCU parser module is unavailable

    _wrap_extract(cls, delegating_module)
    _wrap_parse_delta_soft(cls, delegating_module)
    setattr(delegating_module, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    delegating_module = load_exact_module(TARGET_MODULE, module)
    return apply_once(
        patch_id=PATCH_ID,
        targets=TARGETS,
        marker_owner=delegating_module,
        marker=_MARKER,
        callback=lambda: apply_to_module(delegating_module),
    )


__all__ = [
    "PATCH_ID",
    "TARGETS",
    "TARGET_MODULE",
    "PatchCompatibilityError",
    "apply",
    "apply_to_module",
]
