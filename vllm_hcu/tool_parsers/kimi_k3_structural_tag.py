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

from itertools import permutations
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
"""One raw-string ATOM: anything but the ambiguous "<|" marker prefix.

Known boundary effects, inherited deliberately from the fork
(structural_tag_registry.py:395-401 -- "xgrammar's regex engine has no
lookahead", so a 1-char consuming alternative for '<' is not expressible;
review 2026-10-13 P2 #3 kept this behavior with this documentation):

- The ``<[^|]`` branch consumes TWO characters, so the ``{min,max}``
  quantifier counts atoms, not characters.  A value with k embedded
  "<x" pairs (2 chars per atom) passes maxLength when
  ``len(value) <= max + k``; worst case, ``maxLength: 1`` accepts
  ``"<a"``.  This over-acceptance is bounded (k <= max, because the k
  two-char atoms must still all fit) and was judged preferable to
  dropping the length bound -- the fork added the regex because
  unconstrained AnyText let over-long junk through.
- A value whose run of TRAILING ``<`` characters has ODD length is not
  representable (one ``<`` stays unpaired and the atom cannot consume a
  lone ``<``); an even trailing run -- ``a<<``, ``x<<<<`` -- IS
  representable, the two-char branch consumes the pairs.  The grammar
  steers generation away from odd-trailing values instead of truncating.
- A ``<<|`` run smuggles a literal ``<|`` into the value: the second
  ``<`` pairs with the char after it, so ``a<<|close|>`` matches with a
  real ``<|`` at value positions 2-3.  Inherited fork behavior.  This
  regex can still not TERMINATE a channel: a close needs the exact
  ``argument<|sep|>`` sequence and its ``t<|`` junction cannot be
  produced by the atom (its ``<`` only pairs with a non-``|`` char) --
  RegexFormat has no ``excludes`` parameter; the ``excludes`` lists on
  the AnyText channels are a separate, additional guard.
"""


def _k3_bounded_string_regex(prop: dict[str, Any]) -> str | None:
    """Length/pattern constraint for the raw string channel, if expressible.

    The XTML string channel emits values raw (not JSON-quoted), so
    JSONSchemaFormat cannot enforce string constraints there; xgrammar's
    regex engine has no lookahead, so the close marker is kept unambiguous
    by excluding the "<|" prefix from value characters.  The quantifier
    counts ATOMS, not characters -- the two resulting boundary effects are
    documented on :data:`_K3_STRING_ATOM` and intentionally ported (review
    2026-10-13 P2 #3).

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
        # Review 2026-10-15 P1 (round 8): an empty CLOSED schema ("properties"
        # empty/absent + additionalProperties: false) permits NO arguments --
        # the permissive star let the model invent arbitrary argument names
        # that reach the tool executor.  ``ConstStringFormat("")`` matches
        # only the empty string (verified against xgrammar on this wheel:
        # a zero-argument call compiles and completes; an invented
        # ``argument`` tag fails the matcher), and the block is positionally
        # optional between the call header and the call close marker.
        if (
            parameters.get("additionalProperties") is False
            # R9 P3-a: patternProperties makes the schema closed-but-patterned;
            # the builder has no pattern channel, so fall back to permissive
            # (pre-fix behavior) instead of forbidding legal arguments.
            and not parameters.get("patternProperties")
        ):
            return ConstStringFormat(value="")
        return StarFormat(content=_k3_permissive_argument_tag())
    root_defs = {
        defs_key: parameters[defs_key]
        for defs_key in ("$defs", "definitions")
        if isinstance(parameters.get(defs_key), dict)
    }
    tags: list[TagFormat] = []
    expressible: dict[str, TagFormat] = {}
    for key, prop in props.items():
        tag = _k3_argument_tag(key, prop, root_defs)
        if tag is not None:
            expressible[key] = tag
            tags.append(tag)
        else:
            tags.append(_k3_permissive_argument_tag())
    inner = tags[0] if len(tags) == 1 else OrFormat(elements=list(tags))
    required = parameters.get("required")
    if isinstance(required, list) and required:
        strict = _k3_required_coverage(list(required), expressible, inner)
        if strict is not None:
            return strict
        return PlusFormat(content=inner)
    return StarFormat(content=inner)


# Order-agnostic coverage of required properties explodes as K! branches,
# K <= 3 (<= 6 permutations) stays compileable, larger schemas keep the
# fork's one-of shape.  Cost driver is the number of ``inner`` (the full
# Or-of-all-props) copies: K! branches x (K+1) stars each, capped here so
# future raises of the K cap cannot silently blow up compile size
# (review R3 P2 #2).
_K3_MAX_STRICT_REQUIRED = 3
_K3_MAX_COVERAGE_INNER_COPIES = 24


def _k3_required_coverage(
    required: list[str],
    expressible: dict[str, TagFormat],
    inner: Any,
) -> Any | None:
    """Demand at least one ``argument`` tag per schema-required property.

    Review 2026-10-13 P1 #2: the portable fork shape
    ``PlusFormat(content=OrFormat(tags))`` only demands >=1 argument tag
    overall, so a call omitting a required parameter compiles fine and
    surfaces downstream as a validation error.  K3 arguments are
    order-agnostic (the renderer never fixes arg order), so true coverage
    must allow required tags in any order AND interleaved with optional
    extras: each permutation branch is
    ``SequenceFormat(StarFormat(inner), <tag> per required-prop with
    ``StarFormat(inner)`` between and around them)`` -- every required
    property appears exactly once with the bare tag (the trailing stars of
    ``inner`` already carry duplicates and optional extras, so a Plus
    quantifier would be redundant ambiguity -- review R3 P2 #2), optional
    extras may appear before, between and after them (review follow-up R2:
    the earlier requireds-first compaction ``Seq(Plus, Plus, Star)``
    wrongly rejected legal interleavings like ``a c b``).  Returns None
    (caller keeps the fork shape) when coverage is not expressible: any
    required key missing from ``properties``, mapped to a permissive tag
    (union/unknown type -- a quantifier over it would be vacuous), when K
    exceeds :data:`_K3_MAX_STRICT_REQUIRED`, or when the inner-copy count
    would exceed :data:`_K3_MAX_COVERAGE_INNER_COPIES`.  Diverges from the
    fork (structural_tag_registry.py:533) on purpose.
    """

    required = list(dict.fromkeys(required))  # dedupe, keep order
    if not 0 < len(required) <= _K3_MAX_STRICT_REQUIRED:
        return None
    picked = [expressible.get(key) for key in required]
    if any(tag is None for tag in picked):
        return None
    perms = list(permutations(picked))
    if len(perms) * (len(picked) + 1) > _K3_MAX_COVERAGE_INNER_COPIES:
        return None
    branches = []
    for perm in perms:
        elements: list[Any] = [StarFormat(content=inner)]
        for tag in perm:
            elements.append(tag)
            elements.append(StarFormat(content=inner))
        branches.append(SequenceFormat(elements=elements))
    return branches[0] if len(branches) == 1 else OrFormat(elements=branches)


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
        # Review 2026-10-13 P2 #8: exclude the reserved "<|close|>" prefix
        # from the response body -- deliberately diverging from the fork,
        # whose body is an unbounded AnyTextFormat at
        # structural_tag_registry.py:565 while the SAME file already uses
        # excludes=[_K3_CLOSE] for every other raw channel (:479/:501).
        # Unbounded AnyText let the model pre-terminate the response
        # channel with a bare close marker (open-channel markers remain
        # possible -- the exclusion covers "<|close|>" only; the fork
        # leaves both open).
        TagFormat(
            begin="",
            content=AnyTextFormat(excludes=[_K3_CLOSE]),
            end=_K3_RESPONSE_CLOSE,
        ),
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
