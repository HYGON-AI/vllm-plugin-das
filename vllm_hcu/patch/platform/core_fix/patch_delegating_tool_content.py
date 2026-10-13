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
        record = self._tool_parser._hcu_last_extract
        if record is None or record[0] != content:
            # Only rewrite when the parser ran on exactly this input.
            return tool_calls, out
        info = record[1]
        if info is not None and not getattr(info, "tools_called", False):
            # Propagate the parser's cleaned result EVEN WHEN IT IS NONE
            # (review 2026-10-15 P2): the exact-input record above proves
            # this text WAS handled by the K3 parser, so a ``content=None``
            # means the parser deliberately withheld everything user-facing.
            # That covers BOTH the empty response body
            # (``_strip_response_content`` folds "" into None) and a
            # present-but-unparseable tools channel (e.g. max_tokens
            # truncation mid-call), where the alternative is leaking the raw
            # XTML delimiters; None is also the wheel's own "no content"
            # representation in this branch.
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
        result = original(self, delta_text, delta_token_ids, request, *args, **kwargs)
        if not kwargs.get("finished", False) and not (
            _bound_finished(signature, self, delta_text, delta_token_ids, request, args, kwargs)
        ):
            return result
        if _is_k3_parser(getattr(self, "_tool_parser", None)):
            finish = getattr(self._tool_parser, "finish_streaming", None)
            if callable(finish):
                result = _merge_flush(result, finish())
        return result

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
