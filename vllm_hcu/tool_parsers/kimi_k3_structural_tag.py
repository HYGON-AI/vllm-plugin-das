# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""K3 XTML structural-tag builder for strict tool calling.

Ported verbatim from the evaluated fork (877e6996,
``vllm/tool_parsers/structural_tag_registry.py`` :366-608) because the
wheel's (0.25.1+das) ``structural_tag_registry`` predates the kimi_k3
builder: ``ToolParserManager`` accepted our parser, but every strict
enforcement request died in ``get_model_structural_tag`` with
``ValueError: Unknown format type: kimi_k3`` (review 2026-10-12, P1 #2).

The registration side lives in
``patch/platform/core_fix/patch_kimi_k3_structural_tag.py``; keep this
module pure (builder only) so it can be unit-tested without the patch
framework. All XTML byte-level constants mirror
``vllm_hcu tool_parsers/kimi_k3_tool_parser.py`` exactly.
"""

from __future__ import annotations

from typing import Any

from xgrammar.openai_tool_call_schema import FunctionToolParam
from xgrammar.structural_tag import (
    AnyTextFormat,
    ConstStringFormat,
    JSONSchemaFormat,
    OptionalFormat,
    OrFormat,
    PlusFormat,
    RegexFormat,
    SequenceFormat,
    StarFormat,
    TagFormat,
    TagsWithSeparatorFormat,
)

__all__ = [
    "get_kimi_k3_structural_tag",
    "KIMI_K3_STRUCTURAL_TAG_MODEL",
]

KIMI_K3_STRUCTURAL_TAG_MODEL = "kimi_k3"

_K3_OPEN = "<|open|>"
_K3_CLOSE = "<|close|>"
_K3_SEP = "<|sep|>"
_K3_RESPONSE_OPEN = f"{_K3_OPEN}response{_K3_SEP}"
_K3_RESPONSE_CLOSE = f"{_K3_CLOSE}response{_K3_SEP}"
_K3_TOOLS_OPEN = f"{_K3_OPEN}tools{_K3_SEP}"
_K3_TOOLS_CLOSE = f"{_K3_CLOSE}tools{_K3_SEP}"
_K3_CALL_CLOSE = f"{_K3_CLOSE}call{_K3_SEP}"
_K3_ARG_CLOSE = f"{_K3_CLOSE}argument{_K3_SEP}"
# The model closes the assistant turn with <|close|>message<|sep|> right
# before the end-of-message token. It is generated (not part of the prompt
# prefix), so the tag must permit it or the FSM would mask the model's
# natural terminator.
_K3_MESSAGE_CLOSE = f"{_K3_CLOSE}message{_K3_SEP}"

# JSON-schema type -> K3 XTML ``type=`` attribute value (integer collapses
# onto number).
_K3_JSON_TO_XTML_TYPE = {
    "string": "string",
    "integer": "number",
    "number": "number",
    "boolean": "boolean",
    "null": "null",
    "object": "object",
    "array": "array",
}


def _k3_escape_attr(value: str) -> str:
    """Mirror the K3 renderer's attribute escaping (``&`` then ``"``)."""

    return str(value).replace("&", "&amp;").replace('"', "&quot;")


_K3_STRING_ATOM = r"(?:[^<]|<[^|])"
"""One raw-string character: anything but the ambiguous "<|" marker prefix."""


def _k3_bounded_string_regex(prop: dict[str, Any]) -> str | None:
    """Length/pattern constraint for the raw string channel, if expressible.

    The XTML string channel emits values raw (not JSON-quoted), so
    JSONSchemaFormat cannot enforce string constraints there; xgrammar's
    regex engine has no lookahead, so the close marker is kept unambiguous
    by excluding the "<|" prefix from value characters.

    Returns a regex for the value, or None to keep permissive AnyText.
    """

    max_len = prop.get("maxLength")
    min_len = prop.get("minLength", 0)
    if not isinstance(max_len, int) or max_len < 0 or max_len > 4096:
        return None
    if not isinstance(min_len, int) or min_len < 0 or min_len > max_len:
        min_len = 0
    return _K3_STRING_ATOM + f"{{{min_len},{max_len}}}"


def _k3_argument_tag(
    key: str,
    schema: dict[str, Any],
    root_defs: dict[str, Any] | None = None,
) -> TagFormat | None:
    """Build one ``argument`` XTML tag for property ``key``.

    ``string`` values are emitted raw (bounded by the close marker); every
    other JSON type is emitted as JSON and validated against the property
    schema. A property whose type is a union / missing is left permissive so
    a valid call is never rejected. ``root_defs`` re-attaches the tool's
    root-level ``$defs`` / ``definitions`` to keep embedded schemas
    self-contained.
    """

    prop = schema if isinstance(schema, dict) else {}
    json_type = prop.get("type")
    xtml_type = (
        _K3_JSON_TO_XTML_TYPE.get(json_type) if isinstance(json_type, str) else None
    )
    if xtml_type is None:
        # Unknown / union type: constrain the key but keep the value permissive.
        return None
    begin = (
        f'{_K3_OPEN}argument key="{_k3_escape_attr(key)}" type="{xtml_type}"{_K3_SEP}'
    )
    if xtml_type == "string":
        # Raw string channel: an enum/const of strings is a finite set that
        # can be enforced exactly with const-string alternation (exclusive,
        # never over-rejects); a bounded open string uses the non-lookahead
        # regex; anything else stays permissive AnyText.
        enum_values = prop.get("enum")
        if enum_values is None and isinstance(prop.get("const"), str):
            enum_values = [prop["const"]]
        if (
            isinstance(enum_values, list)
            and enum_values
            and len(enum_values) <= 256
            and all(isinstance(v, str) for v in enum_values)
            and not any("<|" in v for v in enum_values)
        ):
            branches = [ConstStringFormat(value=value) for value in enum_values]
            content: Any = (
                branches[0] if len(branches) == 1 else OrFormat(elements=branches)
            )
        elif (bounded := _k3_bounded_string_regex(prop)) is not None:
            content = RegexFormat(pattern=bounded)
        else:
            content = AnyTextFormat(excludes=[_K3_CLOSE])
    else:
        embedded = prop
        if root_defs:
            embedded = dict(prop)
            for defs_key, defs_value in root_defs.items():
                embedded.setdefault(defs_key, defs_value)
        content = JSONSchemaFormat(json_schema=embedded)
    return TagFormat(begin=begin, content=content, end=_K3_ARG_CLOSE)


def _k3_permissive_argument_tag() -> TagFormat:
    """A key/type-agnostic ``argument`` tag: any attributes, raw value."""

    return TagFormat(
        begin=_K3_OPEN + "argument ",
        content=SequenceFormat(
            elements=[
                RegexFormat(pattern=r"[^<]*" + _K3_SEP.replace("|", r"\|")),
                AnyTextFormat(excludes=[_K3_CLOSE]),
            ]
        ),
        end=_K3_ARG_CLOSE,
    )


def _k3_arguments_block(parameters: dict[str, Any] | bool) -> Any:
    """Build ``argument`` tags for a tool's parameter schema.

    Require at least one tag when the root schema declares required
    properties; otherwise accept zero-or-more. Arguments stay order-agnostic
    and non-unique.
    """

    if not isinstance(parameters, dict):
        return StarFormat(content=_k3_permissive_argument_tag())
    props = parameters.get("properties")
    if not isinstance(props, dict) or not props:
        return StarFormat(content=_k3_permissive_argument_tag())
    root_defs = {
        defs_key: parameters[defs_key]
        for defs_key in ("$defs", "definitions")
        if isinstance(parameters.get(defs_key), dict)
    }
    tags: list[TagFormat] = []
    for key, prop in props.items():
        tag = _k3_argument_tag(key, prop, root_defs)
        tags.append(tag if tag is not None else _k3_permissive_argument_tag())
    inner = tags[0] if len(tags) == 1 else OrFormat(elements=list(tags))
    required = parameters.get("required")
    if isinstance(required, list) and required:
        return PlusFormat(content=inner)
    return StarFormat(content=inner)


def _k3_call_tag(tool: FunctionToolParam) -> TagFormat:
    """One ``call`` tag: ``<|open|>call tool="N" index="<digits>"<|sep|> args``."""

    function = tool.function
    parameters = _get_function_parameters(function)
    begin = f'{_K3_OPEN}call tool="{_k3_escape_attr(function.name)}" index="'
    return TagFormat(
        begin=begin,
        content=SequenceFormat(
            elements=[
                RegexFormat(pattern=r"[0-9]+"),
                ConstStringFormat(value=f'"{_K3_SEP}'),
                _k3_arguments_block(parameters),
            ]
        ),
        end=_K3_CALL_CLOSE,
    )


def _k3_response_prefix() -> list[Any]:
    """The response channel that always precedes the tools channel.

    ``response`` is generated in thinking mode but is part of the generation
    prefix in non-thinking mode, so its open marker is optional; the body is
    bounded by the response close marker.
    """

    return [
        OptionalFormat(content=ConstStringFormat(value=_K3_RESPONSE_OPEN)),
        TagFormat(begin="", content=AnyTextFormat(), end=_K3_RESPONSE_CLOSE),
    ]


def _k3_tools_channel(tools: list[FunctionToolParam]) -> TagFormat:
    return TagFormat(
        begin=_K3_TOOLS_OPEN,
        content=TagsWithSeparatorFormat(
            tags=[_k3_call_tag(tool) for tool in tools],
            separator="",
            at_least_one=True,
        ),
        end=_K3_TOOLS_CLOSE,
    )


def _get_function_parameters(function) -> dict[str, Any] | bool:
    if getattr(function, "strict", None) is False:
        return True
    return function.parameters if function.parameters is not None else True


def get_kimi_k3_structural_tag(
    tools: list[FunctionToolParam],
    builtin_tools: list,
    tool_choice: Any,
    reasoning: bool,
) -> "StructuralTag":  # noqa: F821 - xgrammar type, resolved at call time
    """Build the K3 XTML structural tag for strict tool calling (fork)."""

    del builtin_tools, reasoning

    from xgrammar import StructuralTag

    trailer = OptionalFormat(content=ConstStringFormat(value=_K3_MESSAGE_CLOSE))

    if not tools:
        return StructuralTag(
            format=SequenceFormat(elements=[*_k3_response_prefix(), trailer])
        )

    if tool_choice == "auto":
        tools_part: Any = OptionalFormat(content=_k3_tools_channel(tools))
    elif tool_choice == "forced":
        # K3 rejects named tool choice upstream; treat defensively as a
        # single mandatory call of the first tool.
        tools_part = _k3_tools_channel(tools[:1])
    else:  # required
        tools_part = _k3_tools_channel(tools)

    return StructuralTag(
        format=SequenceFormat(
            elements=[*_k3_response_prefix(), tools_part, trailer]
        )
    )
