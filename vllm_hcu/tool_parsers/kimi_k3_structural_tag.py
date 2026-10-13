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

import json
import re
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


_K3_PATTERN_CEILING = 4096

# ``^CLASS<quantifier>$`` shapes enforceable exactly by merging with the
# min/maxLength bounds: one character class plus one bounded quantifier.
# The braces form REQUIRES a leading digit (review R17 P3-1): ``{}`` and
# ``{,5}`` are quantifier-free literals in ECMA Annex B and must NOT be
# reinterpreted here (an empty ``{}`` previously crashed ``int("")`` at
# build time instead of failing loudly at grammar compilation).
_K3_CLASS_QUANT_RE = re.compile(
    r"^(\[(?:\\.|[^\\\]])*\])"        # one character class (escapes ok)
    r"(\{(\d+)(?:(,)(\d+)?)?\}|[+*?])?$"  # counted/plus/star/opt quantifier
)


def _k3_simple_pattern_bounds(
    pattern: str,
) -> tuple[str, int, int | None] | None:
    """``(class_source, lo, hi_or_None)`` for ``^CLASS<quant>$`` patterns."""

    if len(pattern) < 3 or pattern[0] != "^" or pattern[-1] != "$":
        return None
    body = pattern[1:-1]
    m = _K3_CLASS_QUANT_RE.match(body)
    if m is None:
        return None
    cls, quant = m.group(1), m.group(2)
    if quant is None:
        return cls, 1, 1
    if quant in ("+", "*", "?"):
        return {
            "+": (cls, 1, None),
            "*": (cls, 0, None),
        }[quant]
    inner = quant[1:-1]
    if "," not in inner:
        lo = hi = int(inner)
    else:
        lo_s, hi_s = inner.split(",", 1)
        lo = int(lo_s)
        hi = int(hi_s) if hi_s else None
    return cls, lo, hi


def _k3_bounded_string_regex(prop: dict[str, Any]) -> str | None:
    """Length/pattern constraints for the raw string channel, if expressible.

    The XTML string channel emits values raw (not JSON-quoted), so
    JSONSchemaFormat cannot enforce string constraints there; xgrammar's
    regex engine has no lookahead, so the close marker is kept unambiguous
    by excluding the "<|" prefix from value characters.  The quantifier
    counts ATOMS, not characters -- the two resulting boundary effects are
    documented on :data:`_K3_STRING_ATOM` and intentionally ported (review
    2026-10-13 P2 #3).

    Pattern regimes (review 2026-10-15 round 17 P1: a declared ``pattern``
    must be enforced, never silently dropped):

    - ``^CLASS<quant>$`` simple forms merge EXACTLY with minLength and
      maxLength into one quantified class regex (e.g. ``^[0-9]+$`` +
      ``maxLength: 5`` -> ``[0-9]{1,5}``); unbounded simple quantifiers
      ride the standing 4096-atom ceiling, exactly like the length-only
      regime;
    - any other ANCHORED pattern is enforced verbatim as the whole
      value; its atoms lack the ``<|`` exclusion and the declared
      length bounds are NOT folded in (documented on the branch);
    - an UNANCHORED pattern cannot be enforced on this engine (prefix
      completion, documented on the branch): the permissive atom stands
      unless a maxLength is declared, in which case the length-only
      regime applies so a declared bound is never silently dropped;
    - constructs outside the engine's regex subset fail grammar
      compilation LOUDLY, which mirrors the ``$defs`` merge verdict: a
      loud request error beats a silently relaxed schema.

    Returns a regex for the value, or None to keep permissive AnyText
    (only when neither an enforceable pattern nor a valid maxLength is
    declared).
    """

    max_len = prop.get("maxLength")
    min_len = prop.get("minLength", 0)
    if (
        not isinstance(max_len, int)
        or max_len < 0
        or max_len > _K3_PATTERN_CEILING
    ):
        max_len = None
    if (
        not isinstance(min_len, int)
        or min_len < 0
        or (max_len is not None and min_len > max_len)
    ):
        min_len = 0

    pattern = prop.get("pattern")
    if isinstance(pattern, str) and pattern:
        simple = _k3_simple_pattern_bounds(pattern)
        if simple is not None:
            cls, plo, phi = simple
            lo = max(min_len, plo)
            hi = phi if max_len is None else (
                max_len if phi is None else min(phi, max_len)
            )
            if hi is None:
                hi = _K3_PATTERN_CEILING
            if lo > hi:
                # The bounds cannot co-satisfy the pattern (e.g.
                # minLength 5 vs a {0,4} quantifier).  No grammar is a
                # correct merge here: keep the PATTERN bounds alone so
                # the declared shape still holds -- but note this DOES
                # relax an infeasible minLength (the schema itself was
                # contradictory; RFC 6901-style validators would reject
                # the whole schema before sampling).
                lo = plo
                hi = phi if phi is not None else _K3_PATTERN_CEILING
            return f"{cls}{{{lo},{hi}}}"
        if len(pattern) >= 2 and pattern[0] == "^" and pattern[-1] == "$":
            # Whole-value enforcement.  NOTE (review R17 P2-2): a
            # non-simple anchored pattern does NOT fold minLength /
            # maxLength into the regex, and a general regex
            # intersection is not expressible here -- length bounds
            # are unenforced under this regime.  Its ATOMS also lack
            # the ``<|`` exclusion of _K3_STRING_ATOM, so a pattern
            # like ``^.+$`` can consume framing markers; declare such
            # patterns carefully (review R17 P3-2).
            return f"(?:{pattern[1:-1]})"
        if max_len is None:
            # PROVEN engine limit (probe 2026-10-15, xgrammar 0.2.3):
            # matching is prefix-completing, and ANY value can be
            # completed to a containment witness by appending a match
            # ("urrgent no" is a prefix of "urrgent nourgent").
            # Containment ("search") semantics is unfalsifiable at the
            # value end and CANNOT be enforced; stay permissive rather
            # than build a toothless grammar (anchored regimes above --
            # the realistic shapes -- are enforced exactly).
            return None
        # fall through: an unanchored pattern cannot be enforced, but a
        # declared maxLength still is (no behavior regression vs the
        # length-only regime, review R17 P2-2).
    if max_len is None:
        return None
    return _K3_STRING_ATOM + f"{{{min_len},{max_len}}}"


def _k3_pinned_key_tag(key: str) -> TagFormat:
    """A key-pinned ``argument`` tag with permissive type and raw value.

    Review 2026-10-15 round 11 P1: union (anyOf/oneOf/type-array) or
    missing-type properties keep the declared key enforced by the grammar
    (the key-agnostic permissive tag let closed schemas accept invented
    argument names while omitting the declared property).  The quoted type
    is restricted to the renderer's XTML type vocabulary (review R11 P3-3)
    and the raw value stays AnyText so a valid call is never rejected.
    """

    return TagFormat(
        begin=f'{_K3_OPEN}argument key="{_k3_escape_attr(key)}" ',
        content=SequenceFormat(
            elements=[
                RegexFormat(
                    pattern=r'type="(?:'
                    + "|".join(_K3_JSON_TO_XTML_TYPE.values())
                    + r')"'
                    + _K3_SEP.replace("|", r"\|")
                ),
                AnyTextFormat(excludes=[_K3_CLOSE]),
            ]
        ),
        end=_K3_ARG_CLOSE,
    )


def _k3_union_branches(prop: dict[str, Any]) -> list[Any] | None:
    """The declared alternatives of a union property, in declaration order.

    Covers ``anyOf``, ``oneOf`` and the JSON type array; returns None when
    the property declares no union shape.
    """

    for combo_key in ("anyOf", "oneOf"):
        branches = prop.get(combo_key)
        if isinstance(branches, list) and branches:
            return branches
    json_type = prop.get("type")
    if isinstance(json_type, list) and json_type:
        return [{"type": branch} for branch in json_type]
    return None


def _merge_same_begin(group: list[Any]) -> Any:
    """Merge union branch tags that share one ``begin`` (R14 P1).

    Equal begins share the XTML type channel, so the branch constraints
    must be UNIONED, not dropped: integer|number both render
    ``type="number"`` (and ``3.5`` Legal only via the branch union). The
    string channel merges its content shapes (const sets / bounded regex /
    absorbing AnyText); the JSON channel merges the embedded schemas under
    one ``anyOf`` (nested anyOf is ordinary JSON Schema).
    """

    first = group[0]
    contents = [getattr(tag, "content", None) for tag in group]
    names = [type(c).__name__ for c in contents]

    def str_signature(c: Any) -> str:
        kind = type(c).__name__
        if kind == "RegexFormat":
            return f"re:{getattr(c, 'pattern', '')}"
        if kind == "ConstStringFormat":
            return f"const:{getattr(c, 'value', '')!r}"
        if kind == "OrFormat":
            inner = "|".join(
                repr(getattr(e, "value", "")) for e in getattr(c, "elements", [])
            )
            return f"or:{inner}"
        return "anytext"

    if "JSONSchemaFormat" in names:
        schemas: list[Any] = []
        for c, kind in zip(contents, names):
            if kind != "JSONSchemaFormat":
                continue
            schema = getattr(c, "json_schema", {})
            key_dump = json.dumps(schema, sort_keys=True)
            for existing in schemas:
                if json.dumps(existing, sort_keys=True) == key_dump:
                    break
            else:
                schemas.append(schema)
        if len(schemas) == 1:
            return TagFormat(
                begin=first.begin, content=contents[0], end=first.end
            )
        # Review R15 P2: every branch carried the tool's root ``$defs`` via
        # setdefault, but ``#/$defs/X`` resolves from the DOCUMENT root --
        # i.e. the anyOf wrapper that loses the merge.  Strip the defs
        # tables from the branches and hoist them onto the merged root so
        # embedded ``$ref``s keep resolving.
        hoisted: dict[str, Any] = {}
        stripped: list[Any] = []
        for schema in schemas:
            slim = {
                k: v
                for k, v in schema.items()
                if k not in ("$defs", "definitions")
            }
            for defs_key in ("$defs", "definitions"):
                table = schema.get(defs_key)
                if isinstance(table, dict) and defs_key not in hoisted:
                    # First table wins (review R16 P3-2): two branches with
                    # the SAME defs key but different tables lose the later
                    # ones -- loudly (a $ref fails compation), never by
                    # silently relaxing a constraint.  Such schemas are
                    # already malformed downstream of R15 (a branch-local
                    # defs table was never reachable by "#/$defs"... refs).
                    hoisted[defs_key] = table
            stripped.append(slim)
        merged_root: dict[str, Any] = {"anyOf": stripped}
        merged_root.update(hoisted)
        return TagFormat(
            begin=first.begin,
            content=JSONSchemaFormat(json_schema=merged_root),
            end=first.end,
        )

    # string channel
    if any(kind == "AnyTextFormat" for kind in names):
        return TagFormat(
            begin=first.begin,
            content=AnyTextFormat(excludes=[_K3_CLOSE]),
            end=first.end,
        )
    seen: list[str] = []
    shapes: list[Any] = []
    for c in contents:
        sig = str_signature(c)
        if sig in seen:
            continue
        seen.append(sig)
        shapes.append(c)
    if len(shapes) == 1:
        return TagFormat(begin=first.begin, content=shapes[0], end=first.end)
    return TagFormat(
        begin=first.begin, content=OrFormat(elements=shapes), end=first.end
    )


def _k3_argument_tag(
    key: str,
    schema: dict[str, Any],
    root_defs: dict[str, Any] | None = None,
) -> Any:
    """Build one ``argument`` XTML tag for property ``key``.

    ``string`` values are emitted raw (bounded by the close marker); every
    other JSON type is emitted as JSON and validated against the property
    schema. A union property (anyOf/oneOf/type-array) alternates one full
    branch tag per declared alternative, each with the SAME pinned key, so
    the rendered value is constrained to the declared branches (review
    2026-10-13 round 13 P1, value axis; R11 pinned the key axis). Only
    when no usable union is declared does it fall back to the key-pinned
    wildcard tag. ``root_defs`` re-attaches the tool's root-level ``$defs``
    / ``definitions`` to keep embedded schemas self-contained.
    """

    prop = schema if isinstance(schema, dict) else {}
    json_type = prop.get("type")
    xtml_type = (
        _K3_JSON_TO_XTML_TYPE.get(json_type) if isinstance(json_type, str) else None
    )
    if xtml_type is None:
        branches = _k3_union_branches(prop)
        if branches and len(branches) <= _K3_MAX_UNION_BRANCHES:
            built = [
                _k3_argument_tag(key, branch, root_defs)
                for branch in branches
                if isinstance(branch, dict)
            ]
            # R14 P1: branches sharing a begin (integer|number both render
            # ``type="number"``; two object schemas; enum|maxLen strings)
            # must have their content MERGED, not dropped -- keeping only
            # the first rejected legal values like 3.5. Merge within each
            # begin bucket:
            #   string channel -> OrFormat of the distinct content shapes
            #   (any AnyText absorbs all);
            #   JSON channel    -> one JSONSchemaFormat over the combined
            #   ``anyOf`` (nested anyOf is legal JSON Schema).
            buckets: list[list[Any]] = []
            for tag in built:
                begin = getattr(tag, "begin", "")
                placed = False
                for group in buckets:
                    if begin and getattr(group[0], "begin", "") == begin:
                        group.append(tag)
                        placed = True
                        break
                if not placed:
                    buckets.append([tag])
            if buckets:
                merged = [
                    group[0] if len(group) == 1 else _merge_same_begin(group)
                    for group in buckets
                ]
                return (
                    merged[0] if len(merged) == 1 else OrFormat(elements=merged)
                )
        return _k3_pinned_key_tag(key)
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

    if parameters is False:
        # Review 2026-10-15 round 17 P3: JSON Schema ``false`` rejects any
        # value -- a no-argument tool.  The permissive star let the model
        # invent argument tags here while the empty-CLOSED-schema branch
        # below correctly returns ``ConstStringFormat("")``; keep the two
        # zero-parameter shapes consistent (same wheel-verified matching
        # property: the zero-argument call completes, an invented
        # ``argument`` tag fails the matcher).
        return ConstStringFormat(value="")
    if not isinstance(parameters, dict):
        # ``True`` (or any non-dict truthy config) keeps the permissive
        # fallback -- no constraint was declared.
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
    tags: list[Any] = []
    expressible: dict[str, Any] = {}
    for key, prop in props.items():
        tag = _k3_argument_tag(key, prop, root_defs)
        if tag is not None:
            expressible[key] = tag
            tags.append(tag)
        else:
            # Defensive: _k3_argument_tag currently returns a key-pinned tag
            # for every property, so this fallback no longer fires; keep it
            # as the documented None-contract handling.
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
# Grammar bloat guard for union alternation (review R13 P1): beyond this
# many declared branches the key-pinned wildcard tag is cheaper and the
# value axis stays (documented) permissive.
_K3_MAX_UNION_BRANCHES = 8


def _k3_required_coverage(
    required: list[str],
    expressible: dict[str, Any],
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
    required key missing from ``properties``, when K exceeds
    :data:`_K3_MAX_STRICT_REQUIRED`, or when the inner-copy count would
    exceed :data:`_K3_MAX_COVERAGE_INNER_COPIES`.  Diverges from the fork
    (structural_tag_registry.py:533) on purpose.  NOTE (review R11 P3-4):
    the permissive-tag clause is gone -- union/unknown-type properties are
    key-pinned, not mapped to the key-agnostic tag.
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


def _k3_call_tag(tool: FunctionToolParam, *, strict_gate: bool) -> TagFormat:
    """One ``call`` tag: ``<|open|>call tool="N" index="<digits>"<|sep|> args``.

    ``strict_gate`` is True ONLY for ``tool_choice="auto"`` (the wheel
    probes strict flags for auto at structural_tag_registry.py:109 and
    installs one shared tag for the whole list).  For required / forced /
    named the wheel never consults strict, so the gate is off there and a
    merely-omitted flag stays schema-constrained (review R13/R14 #2).
    """

    function = tool.function
    parameters = _get_function_parameters(function, strict_gate=strict_gate)
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


def _k3_tools_channel(
    tools: list[FunctionToolParam], *, strict_gate: bool
) -> TagFormat:
    return TagFormat(
        begin=_K3_TOOLS_OPEN,
        content=TagsWithSeparatorFormat(
            tags=[
                _k3_call_tag(tool, strict_gate=strict_gate) for tool in tools
            ],
            separator="",
            at_least_one=True,
        ),
        end=_K3_TOOLS_CLOSE,
    )


def _get_function_parameters(
    function, *, strict_gate: bool
) -> dict[str, Any] | bool:
    """Parameter channel for one tool under the current ``tool_choice``.

    ``strict_gate=True`` (auto only, review R13 P2): schema constraint
    requires an EXPLICIT ``strict=True``; a sibling that merely OMITS the
    flag (None, the API default) stays unconstrained so valid best-effort
    calls are not masked.  ``strict_gate=False`` (required / forced /
    named): the wheel constrains unless the tool explicitly DECLINED
    (``strict is False``) -- identical to structural_tag_registry.py:208,
    which never sees these paths' tool lists.
    """

    flag = getattr(function, "strict", None)
    if strict_gate:
        if flag is not True:
            return True
    elif flag is False:
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
        tools_part: Any = OptionalFormat(
            content=_k3_tools_channel(tools, strict_gate=True)
        )
    elif tool_choice == "forced":
        # K3 rejects named tool choice upstream; treat defensively as a
        # single mandatory call of the first tool.
        tools_part = _k3_tools_channel(tools[:1], strict_gate=False)
    else:  # required
        tools_part = _k3_tools_channel(tools, strict_gate=False)

    return StructuralTag(
        format=SequenceFormat(
            elements=[*_k3_response_prefix(), tools_part, trailer]
        )
    )
