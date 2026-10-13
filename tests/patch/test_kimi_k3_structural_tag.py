# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""K3 XTML structural tag: builder units and registration patch paths.

Review 2026-10-12 P1 #2: the wheel registry predates the kimi_k3 builder,
so strict enforcement died with ``Unknown format type: kimi_k3``.  The
builder is ported verbatim from the evaluated fork (877e6996); these tests
drive the REAL ``get_model_structural_tag``-equivalent entry (the builder
directly) plus both wheel-adaptation paths of the platform patch.
"""

from __future__ import annotations

from types import ModuleType

import pytest

xgrammar = pytest.importorskip("xgrammar")
from xgrammar.openai_tool_call_schema import FunctionToolParam  # noqa: E402

from vllm_hcu.patch.platform.core_fix import patch_kimi_k3_structural_tag  # noqa: E402
from vllm_hcu.tool_parsers.kimi_k3_structural_tag import (  # noqa: E402
    KIMI_K3_STRUCTURAL_TAG_MODEL,
    get_kimi_k3_structural_tag,
)


def _tool(name: str = "python") -> FunctionToolParam:
    # Plain dict on purpose: xgrammar's FunctionToolParam.field type for
    # ``function`` can resolve to a DIFFERENT FunctionDefinition class than
    # ``openai.types.shared`` in some wheel combos (container xgrammar
    # 0.2.3 vs openai sdk) -- pydantic then rejects the instance but always
    # accepts the dict form.
    return FunctionToolParam(
        function={
            "name": name,
            "description": "run",
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string"}},
                "required": ["code"],
            },
            "strict": True,
        }
    )


def test_builder_root_has_response_channel_and_message_trailer() -> None:
    tag = get_kimi_k3_structural_tag(
        [_tool()],
        builtin_tools=[],
        tool_choice="required",
        reasoning=False,
    )
    rendered = str(tag)
    assert "open" in rendered and "tools" in rendered
    assert KIMI_K3_STRUCTURAL_TAG_MODEL == "kimi_k3"


def test_builder_no_tools_falls_back_to_response_only() -> None:
    tag = get_kimi_k3_structural_tag(
        [],
        builtin_tools=[],
        tool_choice="auto",
        reasoning=False,
    )
    assert tag is not None
    assert "tools" not in str(tag)


def test_builder_forced_takes_first_tool_only() -> None:
    tag = get_kimi_k3_structural_tag(
        [_tool("a"), _tool("b")],
        builtin_tools=[],
        tool_choice="forced",
        reasoning=False,
    )
    assert tag is not None
    assert '"a"' in str(tag)
    assert '"b"' not in str(tag)


def _fake_registry_module(with_registry: bool, strict: bool = True) -> ModuleType:
    module = ModuleType(patch_kimi_k3_structural_tag.TARGET_MODULE)

    if with_registry:
        registry: dict = {}
        module._VLLM_STRUCTURAL_TAG_REGISTRY = registry

        def register(model):
            def decorator(func):
                registry[model] = func
                return func

            return decorator

        module.register_vllm_structural_tag = register
    else:
        # Signatures mirror the fork dispatch: dump per SINGLE tool.
        module.normalize_tool_choice = lambda tools, choice: (tools, [], choice)
        module._dump_tool_for_xgrammar = lambda tool: tool
        module._dump_tool_choice_for_xgrammar = lambda choice: choice
        module._any_tool_strict = lambda tools: strict

        def _original(model, tools, tool_choice, reasoning):
            raise AssertionError("original dispatch must not run for kimi_k3")

        module.get_model_structural_tag = _original
    return module


def test_preferred_path_registers_into_the_wheel_vllm_registry() -> None:
    module = _fake_registry_module(with_registry=True)

    assert patch_kimi_k3_structural_tag.apply_to_module(module) is True
    builder = module._VLLM_STRUCTURAL_TAG_REGISTRY.get("kimi_k3")
    assert builder is not None
    assert builder.__module__ == "vllm_hcu.tool_parsers.kimi_k3_structural_tag"
    assert patch_kimi_k3_structural_tag.apply_to_module(module) is False


def test_fallback_path_pre_dispatches_kimi_k3() -> None:
    module = _fake_registry_module(with_registry=False)

    assert patch_kimi_k3_structural_tag.apply_to_module(module) is True
    # served from the ported builder, never from the wheel lookup
    tag = module.get_model_structural_tag("kimi_k3", [_tool()], "required", False)
    assert tag is not None
    # other models fall through untouched (stub raises)
    with pytest.raises(AssertionError):
        module.get_model_structural_tag("hermes", [_tool()], "required", False)


def test_fallback_requires_the_dispatch_helpers() -> None:
    """A wheel missing normalize_tool_choice fails before any wrap."""

    module = ModuleType(patch_kimi_k3_structural_tag.TARGET_MODULE)
    module.get_model_structural_tag = lambda *a, **k: None

    from vllm_hcu.patch.platform.core_fix._common import PatchCompatibilityError

    with pytest.raises(PatchCompatibilityError, match="normalize_tool_choice"):
        patch_kimi_k3_structural_tag.apply_to_module(module)


def test_fallback_requires_the_per_tool_dump_helper() -> None:
    """N6: with normalize present, a missing dump helper must still fail."""

    module = ModuleType(patch_kimi_k3_structural_tag.TARGET_MODULE)
    module.normalize_tool_choice = lambda tools, choice: (tools, [], choice)
    module.get_model_structural_tag = lambda *a, **k: None

    from vllm_hcu.patch.platform.core_fix._common import PatchCompatibilityError

    with pytest.raises(PatchCompatibilityError, match="_dump_tool_for_xgrammar"):
        patch_kimi_k3_structural_tag.apply_to_module(module)


def test_fallback_requires_the_strict_probe() -> None:
    module = ModuleType(patch_kimi_k3_structural_tag.TARGET_MODULE)
    module.normalize_tool_choice = lambda tools, choice: (tools, [], choice)
    module._dump_tool_for_xgrammar = lambda tool: tool
    module._dump_tool_choice_for_xgrammar = lambda choice: choice
    module.get_model_structural_tag = lambda *a, **k: None

    from vllm_hcu.patch.platform.core_fix._common import PatchCompatibilityError

    with pytest.raises(PatchCompatibilityError, match="_any_tool_strict"):
        patch_kimi_k3_structural_tag.apply_to_module(module)


def test_wrapper_falls_back_for_non_strict_auto() -> None:
    module = _fake_registry_module(with_registry=False, strict=False)

    assert patch_kimi_k3_structural_tag.apply_to_module(module) is True
    # non-strict auto must delegate to the original (stub raises -> proof)
    with pytest.raises(AssertionError):
        module.get_model_structural_tag("kimi_k3", [_tool()], "auto", False)


def test_wrapper_dumps_each_tool_individually() -> None:
    """P1-4 regression: _dump_tool_for_xgrammar takes ONE tool."""

    module = _fake_registry_module(with_registry=False)
    seen: list = []

    def spy_dump(tool):
        assert not isinstance(tool, list), "must be called per single tool"
        seen.append(tool)
        return tool

    module._dump_tool_for_xgrammar = spy_dump

    assert patch_kimi_k3_structural_tag.apply_to_module(module) is True
    tag = module.get_model_structural_tag(
        "kimi_k3", [_tool("a"), _tool("b")], "required", False
    )
    assert [t.function.name for t in seen] == ["a", "b"]
    assert tag is not None


# --- review 2026-10-13 fixes #2 (strict required coverage) / #8 (body
# excludes) ---------------------------------------------------------------


def _walk_formats(fmt, out: list | None = None) -> list:
    out = [] if out is None else out
    out.append(fmt)
    for elems in (
        getattr(fmt, "elements", None),
        getattr(fmt, "tags", None),
    ):  # both attrs exist only on their own formats; None skips its loop
        for child in elems or []:
            _walk_formats(child, out)
    content = getattr(fmt, "content", None)
    if content is not None and not isinstance(content, (str, list, dict)):
        _walk_formats(content, out)
    return out


def _two_required_tool(name: str = "python") -> FunctionToolParam:
    return FunctionToolParam(
        function={
            "name": name,
            "parameters": {
                "type": "object",
                "properties": {
                    "alpha": {"type": "string"},
                    "beta": {"type": "integer"},
                },
                "required": ["alpha", "beta"],
            },
            "strict": True,
        }
    )


def _mixed_required_tool(name: str = "python") -> FunctionToolParam:
    """alpha/beta required, gamma optional: interleave coverage target."""

    return FunctionToolParam(
        function={
            "name": name,
            "parameters": {
                "type": "object",
                "properties": {
                    "alpha": {"type": "string"},
                    "beta": {"type": "integer"},
                    "gamma": {"type": "string"},
                },
                "required": ["alpha", "beta"],
            },        }
    )


def _union_tool(name: str = "fuzzy") -> FunctionToolParam:
    return FunctionToolParam(
        function={
            "name": name,
            "parameters": {
                "type": "object",
                "properties": {"mystery": {"type": ["string", "integer"]}},
                "required": ["mystery"],
            },        }
    )


def _k_required_tools(k: int) -> list[FunctionToolParam]:
    props = {f"p{i}": {"type": "string"} for i in range(k)}
    return [
        FunctionToolParam(
            function={
                "name": "wide",
                "parameters": {
                    "type": "object",
                    "properties": props,
                    "required": sorted(props),
                },
            }
        )
    ]


def test_required_properties_are_individually_covered() -> None:
    """P1 #2: >=1 argument tag per REQUIRED property, any order."""

    tag = get_kimi_k3_structural_tag(
        [_two_required_tool()], [], "required", False
    )
    shapes = [type(f).__name__ for f in _walk_formats(tag.format)]
    # R3 #2: bare-permutation tags, no Plus quantifiers in the branches
    assert shapes.count("PlusFormat") == 0, shapes
    # one SequenceFormat branch per permutation of the required tags
    assert shapes.count("SequenceFormat") >= 2, shapes


def test_union_required_pins_the_key() -> None:
    """R11 P1: a union-valued required property keeps its declared KEY
    pinned in the grammar (the old None-contract substituted the
    key-agnostic permissive tag, letting closed schemas accept invented
    argument names).  The pinned prop participates in the strict
    per-required coverage: no Plus fallback shape remains."""

    tag = get_kimi_k3_structural_tag([_union_tool()], [], "required", False)
    shapes = [type(f).__name__ for f in _walk_formats(tag.format)]
    assert shapes.count("PlusFormat") == 0, shapes
    begins = [
        getattr(f, "begin", "")
        for f in _walk_formats(tag.format)
        if type(f).__name__ == "TagFormat"
    ]
    assert any('key="mystery"' in b for b in begins), begins


def _int_bool_union_tool(name: str = "i32") -> FunctionToolParam:
    """anyOf(integer|boolean) -- the bot's counterexample shape."""

    return FunctionToolParam(
        function={
            "name": name,
            "parameters": {
                "type": "object",
                "properties": {"payload": {"anyOf": [
                    {"type": "integer"}, {"type": "boolean"},
                ]}},
                "required": ["payload"],
                "additionalProperties": False,
            },        }
    )


def test_union_alternation_constrains_the_value_axis() -> None:
    """R13 P1: a union property must alternate DECLARED branches only --
    same pinned key, one branch per alternative, value per branch type.
    The old key-pinned wildcard accepted type="object" with arbitrary
    content for an integer|boolean schema."""

    tag = get_kimi_k3_structural_tag(
        [_int_bool_union_tool()], [], "required", False
    )
    begins = [
        getattr(f, "begin", "")
        for f in _walk_formats(tag.format)
        if type(f).__name__ == "TagFormat" and 'key="payload"' in getattr(f, "begin", "")
    ]
    # two XTML type renderings for the two branches (int -> number);
    # object/array/string/null/boolean-all must NOT appear
    assert any('type="number"' in b for b in begins), begins
    assert any('type="boolean"' in b for b in begins), begins
    assert not any(
        'type="string"' in b or 'type="object"' in b or 'type="array"' in b
        for b in begins
    ), begins

    # End-to-end FSM: number and boolean accepted-complete, object with
    # arbitrary content dead, payload omitted dead.
    xg = pytest.importorskip("xgrammar")
    compiler_cls = getattr(xg, "GrammarCompiler", None)
    if compiler_cls is None or not hasattr(compiler_cls, "compile_structural_tag"):
        pytest.skip("xgrammar lacks GrammarCompiler.compile_structural_tag")
    vocab = [
        "a", "b", "c", "d", "e", "f", "g", "v", "3", "<", "|", ">", '"', "=",
        "open", "close", "sep", "argument", "call", "tools", "response",
        "message", "number", "boolean", "true", "key", "i32", "admin",
        "payload", "index", "_", "s", "p", "l", "u", "t", "o", "n",
        "r", "i", "x", " ", "1",
    ]
    compiled = compiler_cls(xg.TokenizerInfo(vocab)).compile_structural_tag(tag)

    pre = '<|close|>response<|sep|><|open|>tools<|sep|><|open|>call tool="i32" index="1"<|sep|>'
    post = "<|close|>call<|sep|><|close|>tools<|sep|>"
    end = "<|close|>message<|sep|>"

    def arg(t: str, val: str) -> str:
        return (
            f'<|open|>argument key="payload" type="{t}"<|sep|>{val}'
            "<|close|>argument<|sep|>"
        )

    def accepts(sample: str) -> tuple[bool, bool]:
        matcher = xg.GrammarMatcher(compiled)
        return bool(matcher.accept_string(sample)), matcher.is_completed()

    assert accepts(pre + arg("number", "3") + post + end) == (True, True)
    assert accepts(pre + arg("boolean", "true") + post + end) == (True, True)
    # the declining case: object + arbitrary JSON -- dead BEFORE sampling
    assert accepts(pre + arg("object", '{"a": 1}') + post + end)[0] is False
    assert accepts(pre + post + end)[0] is False


def test_union_same_begin_branches_merge_not_drop() -> None:
    """R14 P1: integer|number both render type="number"; the merge must
    UNION the branch constraints (one JSONSchemaFormat anyOf) so 3.5 is
    legal -- keeping only the first branch silently rejected 3.5."""

    def int_number_tool(name: str = "floaty") -> FunctionToolParam:
        return FunctionToolParam(
            function={
                "name": name,
                "parameters": {
                    "type": "object",
                    "properties": {"v": {"anyOf": [
                        {"type": "integer"}, {"type": "number"},
                    ]}},
                    "required": ["v"],
                    "additionalProperties": False,
                },
            }
        )

    tag = get_kimi_k3_structural_tag([int_number_tool()], [], "required", False)
    walked = [
        f
        for f in _walk_formats(tag.format)
        if type(f).__name__ == "TagFormat" and 'key="v"' in getattr(f, "begin", "")
    ]
    # required coverage repeats the SAME merged tag across permutations:
    # one distinct begin, one merged content shape per occurrence
    assert len({getattr(f, "begin", "") for f in walked}) == 1, walked
    assert walked, "no argument tag for key='v'"
    contents = {id(getattr(f, "content", None)) for f in walked}
    if len(contents) == 1:
        merged = walked[0].content
    else:
        # fall back: same serialized schema
        import json as _json

        assert (
            len(
                {
                    _json.dumps(getattr(f, "content", None), default=repr)
                    for f in walked
                }
            )
            == 1
        ), walked
        merged = walked[0].content
    assert type(merged).__name__ == "JSONSchemaFormat", type(merged).__name__
    assert len(merged.json_schema.get("anyOf", [])) == 2, merged.json_schema

    xg = pytest.importorskip("xgrammar")
    compiler_cls = getattr(xg, "GrammarCompiler", None)
    if compiler_cls is None or not hasattr(compiler_cls, "compile_structural_tag"):
        pytest.skip("xgrammar lacks GrammarCompiler.compile_structural_tag")
    vocab = [
        "a", "b", "c", "d", "e", "f", "g", "v", "3", "5", ".", "<", "|", ">",
        '"', "=", "open", "close", "sep", "argument", "call", "tools",
        "response", "message", "number", "key", "floaty", "payload",
        "index", "_", "s", "p", "l", "u", "t", "o", "n", "r", "i", "x", " ",
        "1",
    ]
    compiled = compiler_cls(xg.TokenizerInfo(vocab)).compile_structural_tag(tag)

    pre = '<|close|>response<|sep|><|open|>tools<|sep|><|open|>call tool="floaty" index="1"<|sep|>'
    post = "<|close|>call<|sep|><|close|>tools<|sep|>"
    end = "<|close|>message<|sep|>"

    def arg(val: str) -> str:
        return (
            f'<|open|>argument key="v" type="number"<|sep|>{val}'
            "<|close|>argument<|sep|>"
        )

    def accepts(sample: str) -> tuple[bool, bool]:
        matcher = xg.GrammarMatcher(compiled)
        return bool(matcher.accept_string(sample)), matcher.is_completed()

    assert accepts(pre + arg("3") + post + end) == (True, True)
    assert accepts(pre + arg("3.5") + post + end) == (True, True)


def _closed_union_tool(name: str = "py") -> FunctionToolParam:
    return FunctionToolParam(
        function={
            "name": name,
            "parameters": {
                "type": "object",
                "properties": {"payload": {"anyOf": [
                    {"type": "string"}, {"type": "integer"},
                ]}},
                "required": ["payload"],
                "additionalProperties": False,
            },        }
    )


def test_closed_union_schema_rejects_invented_keys_via_xgrammar() -> None:
    """R11 P1 end-to-end: the closed union tool must complete with the
    required ``payload`` key (any concrete type rendering, permissive raw
    value) and must dead-end on an invented ``key="admin"`` argument with
    ``payload`` omitted."""

    xg = pytest.importorskip("xgrammar")
    compiler_cls = getattr(xg, "GrammarCompiler", None)
    if compiler_cls is None or not hasattr(compiler_cls, "compile_structural_tag"):
        pytest.skip("xgrammar lacks GrammarCompiler.compile_structural_tag")

    tag = get_kimi_k3_structural_tag(
        [_closed_union_tool()], [], "required", False
    )
    vocab = [
        "a", "b", "c", "d", "e", "f", "g", "v", "3", "<", "|", ">", '"', "=",
        "open", "close", "sep", "argument", "call", "tools", "response",
        "message", "number", "string", "integer", "key", "py", "admin",
        "payload", "index", "_", "s", "p", "l", "u", "t", "o", "n",
        "r", "i", "x", " ", "1",
    ]
    compiled = compiler_cls(xg.TokenizerInfo(vocab)).compile_structural_tag(tag)

    pre = '<|close|>response<|sep|><|open|>tools<|sep|><|open|>call tool="py" index="1"<|sep|>'
    post = "<|close|>call<|sep|><|close|>tools<|sep|>"
    end = "<|close|>message<|sep|>"
    payload = '<|open|>argument key="payload" type="string"<|sep|>v<|close|>argument<|sep|>'
    payload_num = '<|open|>argument key="payload" type="number"<|sep|>3<|close|>argument<|sep|>'
    invented = '<|open|>argument key="admin" type="string"<|sep|>v<|close|>argument<|sep|>'

    def accepts(sample: str) -> tuple[bool, bool]:
        matcher = xg.GrammarMatcher(compiled)
        return bool(matcher.accept_string(sample)), matcher.is_completed()

    assert accepts(pre + payload + post + end) == (True, True)
    assert accepts(pre + payload_num + post + end) == (True, True)
    assert accepts(pre + invented + post + end)[0] is False
    assert accepts(pre + invented + payload + post + end)[0] is False


def test_required_branches_stagger_required_and_optional_args() -> None:
    """R2 #2: each coverage branch is Seq(Star, tag, Star, ..., Star) so
    optional args may appear before/between/after the required ones (the
    earlier requireds-first compaction rejected legal interleavings);
    R3 #2: the required entries are the bare per-prop tags -- the trailing
    stars of ``inner`` already carry duplicates and extras, so Plus would
    be redundant ambiguity."""

    tag = get_kimi_k3_structural_tag(
        [_mixed_required_tool()], [], "required", False
    )
    branches = [
        f
        for f in _walk_formats(tag.format)
        if type(f).__name__ == "SequenceFormat"
        and len(getattr(f, "elements", [])) == 5
    ]
    assert branches, "no 5-element required-coverage branch found"
    assert len(branches) == 2  # 2! permutations of {alpha, beta}
    for branch in branches:
        names = [type(e).__name__ for e in branch.elements]
        assert names == [
            "StarFormat",
            "TagFormat",
            "StarFormat",
            "TagFormat",
            "StarFormat",
        ], names
        tags = [
            getattr(e, "begin", "") for e in branch.elements
            if type(e).__name__ == "TagFormat"
        ]
        assert sum('key="alpha"' in t for t in tags) == 1
        assert sum('key="beta"' in t for t in tags) == 1


def test_required_coverage_accepts_interleaved_args_via_xgrammar() -> None:
    """Compile the tag and let the real FSM judge: optional args may sit
    before/between/after required ones, and a missing required prop is
    still rejected.  Counting Format nodes alone cannot see this (R2 #2
    regression guard; guards the xgrammar compile API behind skips)."""

    xg = pytest.importorskip("xgrammar")
    compiler_cls = getattr(xg, "GrammarCompiler", None)
    if compiler_cls is None or not hasattr(compiler_cls, "compile_structural_tag"):
        pytest.skip("xgrammar lacks GrammarCompiler.compile_structural_tag")

    tag = get_kimi_k3_structural_tag(
        [_mixed_required_tool()], [], "required", False
    )
    vocab = [
        "a", "b", "c", "d", "e", "f", "g", "v", "3", "<", "|", ">", '"', "=",
        "open", "close", "sep", "argument", "call", "tools", "response",
        "message", "alpha", "beta", "gamma", "number", "string", "key",
        "type", "python", "index", "1", "_", "s", "p", "l", "u", "t", "o",
        "n", "e", "r", "i", "x", " ",
    ]
    compiled = compiler_cls(xg.TokenizerInfo(vocab)).compile_structural_tag(tag)

    arg_alpha = '<|open|>argument key="alpha" type="string"<|sep|>val<|close|>argument<|sep|>'
    arg_beta = '<|open|>argument key="beta" type="number"<|sep|>3<|close|>argument<|sep|>'
    arg_gamma = '<|open|>argument key="gamma" type="string"<|sep|>g<|close|>argument<|sep|>'
    pre = '<|close|>response<|sep|><|open|>tools<|sep|><|open|>call tool="python" index="1"<|sep|>'
    post = "<|close|>call<|sep|><|close|>tools<|sep|>"

    def accepts(sample: str) -> tuple[bool, bool]:
        matcher = xg.GrammarMatcher(compiled)
        # is_completed == the grammar is fully consumed (empty jump-forward);
        # is_terminated additionally needs stop tokens, which these tags do
        # not carry -- completed is the strong signal here (R3 #5:
        # accept_string alone only proves a prefix is reachable).
        return bool(matcher.accept_string(sample)), matcher.is_completed()

    # the grammar tail is the optional message close (P2 #7); complete
    # samples end with it, and positives must fully consume the grammar
    end = "<|close|>message<|sep|>"
    assert accepts(pre + arg_alpha + arg_gamma + arg_beta + post + end) == (True, True)
    assert accepts(pre + arg_beta + arg_alpha + post + end) == (True, True)
    # without beta anywhere the FSM dead-ends: not accepted
    assert accepts(pre + arg_alpha + arg_gamma + post + end)[0] is False


def test_wide_required_schema_stays_fork_shaped() -> None:
    """K > 3 permutations would explode: fall back to the fork shape."""

    tag = get_kimi_k3_structural_tag(_k_required_tools(4), [], "required", False)
    shapes = [type(f).__name__ for f in _walk_formats(tag.format, [])]
    assert shapes.count("PlusFormat") == 1, shapes


def test_response_body_excludes_reserved_close_prefix() -> None:
    """P2 #8: the response channel cannot emit a bare <|close|> prefix."""

    tag = get_kimi_k3_structural_tag([_tool()], [], "auto", False)
    bodies = [
        f
        for f in _walk_formats(tag.format, [])
        if type(f).__name__ == "AnyTextFormat"
    ]
    assert bodies, "response body AnyTextFormat missing"
    assert all(
        "<|close|>" in (getattr(f, "excludes", None) or ()) for f in bodies
    ), [getattr(f, "excludes", None) for f in bodies]


# --- review 2026-10-15 round 8: closed empty parameter schemas ---


def _closed_empty_tool(name: str = "python") -> FunctionToolParam:
    return FunctionToolParam(
        function={
            "name": name,
            "description": "accepts no arguments",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },        }
    )


def test_closed_empty_schema_forbids_argument_tags() -> None:
    """R8 P1: an empty CLOSED schema permits NO arguments.  The branch
    returned the permissive star, letting the model invent argument names
    that reach the tool executor."""

    from vllm_hcu.tool_parsers.kimi_k3_structural_tag import (
        _k3_arguments_block,
    )

    schema = {"type": "object", "properties": {}, "additionalProperties": False}
    block = _k3_arguments_block(schema)
    # an exact-empty match: ConstStringFormat("") (verified against the
    # container xgrammar: SequenceFormat([]) is rejected by the compiler)
    assert type(block).__name__ == "ConstStringFormat"
    assert block.value == ""

    tag = get_kimi_k3_structural_tag(
        [_closed_empty_tool()], [], "required", False
    )
    shapes = [type(f).__name__ for f in _walk_formats(tag.format, [])]
    assert shapes.count("StarFormat") == 0, shapes


def test_open_empty_schema_stays_permissive() -> None:
    """Without additionalProperties: false, generic arguments remain legal
    per JSON Schema semantics -- unchanged behavior.  A closed-BUT-
    patterned schema (R9 P3-a) also stays permissive: the builder has no
    pattern channel, and forbidding would reject legal pattern-matching
    arguments."""

    from vllm_hcu.tool_parsers.kimi_k3_structural_tag import (
        _k3_arguments_block,
    )

    block = _k3_arguments_block({"type": "object", "properties": {}})
    assert type(block).__name__ == "StarFormat"
    patterned = _k3_arguments_block(
        {
            "type": "object",
            "properties": {},
            "patternProperties": {"^x": {"type": "string"}},
            "additionalProperties": False,
        }
    )
    assert type(patterned).__name__ == "StarFormat"


def test_closed_empty_call_rejects_invented_args_via_xgrammar() -> None:
    """Compile the closed-empty tag and let the real FSM judge: the
    zero-argument call must complete the grammar; an invented argument
    tag must dead-end it."""

    xg = pytest.importorskip("xgrammar")
    compiler_cls = getattr(xg, "GrammarCompiler", None)
    if compiler_cls is None or not hasattr(compiler_cls, "compile_structural_tag"):
        pytest.skip("xgrammar lacks GrammarCompiler.compile_structural_tag")

    tag = get_kimi_k3_structural_tag(
        [_closed_empty_tool()], [], "required", False
    )
    vocab = [
        "a", "b", "c", "d", "e", "f", "g", "v", "3", "<", "|", ">", '"', "=",
        "open", "close", "sep", "argument", "call", "tools", "response",
        "message", "number", "string", "key", "python", "index", "x",
        "_", "s", "p", "l", "u", "t", "o", "n", "e", "r", "i", " ", "1",
    ]
    compiled = compiler_cls(xg.TokenizerInfo(vocab)).compile_structural_tag(tag)

    pre = '<|close|>response<|sep|><|open|>tools<|sep|><|open|>call tool="python" index="1"<|sep|>'
    post = "<|close|>call<|sep|><|close|>tools<|sep|>"
    end = "<|close|>message<|sep|>"
    invented = '<|open|>argument key="x" type="string"<|sep|>v<|close|>argument<|sep|>'

    def accepts(sample: str) -> tuple[bool, bool]:
        matcher = xg.GrammarMatcher(compiled)
        return bool(matcher.accept_string(sample)), matcher.is_completed()

    assert accepts(pre + post + end) == (True, True)
    assert accepts(pre + invented + post + end)[0] is False


# --- review 2026-10-15 round 13 #2: explicit strict opt-in only ---


def _plain_sibling_tool(name: str = "plain_lookup") -> FunctionToolParam:
    """No ``strict`` key at all (the API default)."""

    return FunctionToolParam(
        function={
            "name": name,
            "parameters": {
                "type": "object",
                "properties": {"q": {"type": "string"}},
                "required": ["q"],
            },
        }
    )


def test_union_merge_hoists_defs_to_the_wrapper_root() -> None:
    """R15 P2: ``#/$defs/T`` resolves from the document root -- the anyOf
    wrapper created by same-begin merging.  Branches must shed their
    (identical) defs tables and the wrapper carries them, or references
    inside merged object branches stop resolving."""

    schema = {
        "type": "object",
        "properties": {
            "pick": {
                "anyOf": [
                    {
                        "type": "object",
                        "properties": {"r": {"$ref": "#/$defs/T"}},
                    },
                    {
                        "type": "object",
                        "properties": {"s": {"$ref": "#/$defs/T"}},
                    },
                ]
            }
        },
        "required": ["pick"],
        "additionalProperties": False,
        "$defs": {"T": {"type": "string"}},
    }
    tool = FunctionToolParam(
        function={"name": "picker", "parameters": schema}
    )
    tag = get_kimi_k3_structural_tag([tool], [], "required", False)
    merged_tags = [
        f
        for f in _walk_formats(tag.format)
        if type(f).__name__ == "JSONSchemaFormat"
        and isinstance(getattr(f, "json_schema", None), dict)
        and "anyOf" in f.json_schema
    ]
    assert merged_tags, "merged anyOf wrapper not found"
    wrapper = merged_tags[0].json_schema
    assert isinstance(wrapper["$defs"], dict) and "T" in wrapper["$defs"], wrapper
    for branch in wrapper["anyOf"]:
        assert "$defs" not in branch, branch
    # every $ref keeps resolving against the hoisted table
    for branch in wrapper["anyOf"]:
        props = branch.get("properties", {})
        for prop in props.values():
            if "$ref" in prop:
                name = prop["$ref"].split("/")[-1]
                assert name in wrapper["$defs"], (prop, wrapper)

    # compile the tag: a broken $ref would fail the whole request here
    xg = pytest.importorskip("xgrammar")
    compiler_cls = getattr(xg, "GrammarCompiler", None)
    if compiler_cls is None or not hasattr(
        compiler_cls, "compile_structural_tag"
    ):
        pytest.skip("xgrammar lacks GrammarCompiler.compile_structural_tag")
    vocab = [
        "a", "b", "c", "d", "e", "f", "g", "v", "3", "<", "|", ">", '"', "=",
        "open", "close", "sep", "argument", "call", "tools", "response",
        "message", "object", "string", "key", "picker", "pick", "r", "s",
        "index", "{", "}", ":", " ", "_",
    ]
    compiled = compiler_cls(xg.TokenizerInfo(vocab)).compile_structural_tag(tag)
    pre = '<|close|>response<|sep|><|open|>tools<|sep|><|open|>call tool="picker" index="1"<|sep|>'
    post = "<|close|>call<|sep|><|close|>tools<|sep|>"
    end = "<|close|>message<|sep|>"
    sample = (
        pre
        + '<|open|>argument key="pick" type="object"<|sep|>{"r": "v"}'
        + "<|close|>argument<|sep|>"
        + post
        + end
    )
    matcher = xg.GrammarMatcher(compiled)
    assert matcher.accept_string(sample)
    del vocab


def _tool_with_string_prop(prop: dict) -> FunctionToolParam:
    return FunctionToolParam(
        function={
            "name": "tagged",
            "parameters": {
                "type": "object",
                "properties": {"v": prop},
                "required": ["v"],
                "additionalProperties": False,
            },
        }
    )


def _string_tag_content(prop: dict):
    tag = get_kimi_k3_structural_tag([_tool_with_string_prop(prop)], [], "required", False)
    tags = [
        f
        for f in _walk_formats(tag.format)
        if type(f).__name__ == "TagFormat"
        and 'key="v"' in getattr(f, "begin", "")
    ]
    assert tags, "no string argument tag built"
    return tags[0].content


def test_pattern_simple_class_merges_with_length_bounds() -> None:
    """R17 P1: ^CLASS<quant>$ patterns merge EXACTLY with min/maxLength."""

    content = _string_tag_content(
        {"type": "string", "pattern": "^[0-9]+$", "maxLength": 5}
    )
    assert type(content).__name__ == "RegexFormat", type(content).__name__
    assert content.pattern == "[0-9]{1,5}", content.pattern

    content = _string_tag_content(
        {"type": "string", "pattern": "^[a-z]{2,8}$", "minLength": 3}
    )
    assert content.pattern == "[a-z]{3,8}", content.pattern

    # pattern only: unbounded quantifier rides the standing ceiling
    content = _string_tag_content({"type": "string", "pattern": "^[0-9]+$"})
    assert content.pattern == "[0-9]{1,4096}", content.pattern


def test_pattern_is_enforced_in_the_grammar() -> None:
    """The FSM must actually reject pattern-violating strict values."""

    xg = pytest.importorskip("xgrammar")
    compiler_cls = getattr(xg, "GrammarCompiler", None)
    if (
        compiler_cls is None
        or not hasattr(compiler_cls, "compile_structural_tag")
    ):
        pytest.skip("xgrammar lacks GrammarCompiler.compile_structural_tag")

    tag = get_kimi_k3_structural_tag(
        [
            _tool_with_string_prop(
                {"type": "string", "pattern": "^[A-Z][a-z]+$"}
            )
        ],
        [],
        "required",
        False,
    )
    vocab = [
        "a", "b", "c", "d", "e", "f", "g", "v", "<", "|", ">", '"', "=",
        "open", "close", "sep", "argument", "call", "tools", "response",
        "message", "string", "key", "tagged", "index", "H", "X", "e", "l",
        "o", "A", "B", "s", "t", "r", "n",
    ]
    compiled = compiler_cls(xg.TokenizerInfo(vocab)).compile_structural_tag(tag)
    pre = '<|close|>response<|sep|><|open|>tools<|sep|><|open|>call tool="tagged" index="1"<|sep|>'
    post = "<|close|>call<|sep|><|close|>tools<|sep|><|close|>message<|sep|>"

    def arg(val: str) -> str:
        return (
            f'<|open|>argument key="v" type="string"<|sep|>{val}'
            "<|close|>argument<|sep|>"
        )

    matcher = xg.GrammarMatcher(compiled)
    assert matcher.accept_string(pre + arg("Hello") + post)
    matcher = xg.GrammarMatcher(compiled)
    assert not matcher.accept_string(pre + arg("hello") + post)
    matcher = xg.GrammarMatcher(compiled)
    assert not matcher.accept_string(pre + arg("HELLO") + post)


def test_unanchored_pattern_falls_back_permissive() -> None:
    """R17 P1 caveat, PROVEN engine limit: xgrammar completes VALUES by
    prefix -- any value v plus a appended match is a containment witness
    ("urrgent no" -> "urrgent nourgent"), so JSON-Schema search semantics
    is unfalsifiable at the value end.  The grammar stays permissive
    (probed across regex/star/sequence composites; all accept a typo).
    Anchored regimes -- the realistic shapes -- are enforced exactly."""

    content = _string_tag_content({"type": "string", "pattern": "urgent"})
    assert type(content).__name__ == "AnyTextFormat", type(content).__name__


def test_parameters_false_rejects_invented_arguments() -> None:
    """R17 P3: JSON Schema ``parameters: false`` is a zero-parameter tool.

    NOTE on reachability: this wheel's ``FunctionToolParam`` validates
    ``parameters`` as dict-only, so a bool ``false`` dies with a 400
    upstream -- the shape arrives only through the builder's documented
    ``dict | bool`` contract (direct/internal callers).  There it must
    behave consistently with the empty-CLOSED-schema branch: ONLY the
    empty argument sequence is legal."""

    from vllm_hcu.tool_parsers.kimi_k3_structural_tag import (
        _k3_arguments_block,
    )

    block = _k3_arguments_block(False)
    assert type(block).__name__ == "ConstStringFormat", type(block).__name__
    assert block.value == "", block.value

    # ``True``/non-dict keeps the permissive fallback
    permissive = _k3_arguments_block(True)
    assert type(permissive).__name__ == "StarFormat", type(permissive).__name__

    # grammar-level proof via the REACHABLE dict twin (empty + closed):
    # only the zero-argument call matches -- same ConstStringFormat("")
    # the False branch returns.
    xg = pytest.importorskip("xgrammar")
    compiler_cls = getattr(
        xg, "GrammarCompiler", None
    )
    if (
        compiler_cls is None
        or not hasattr(compiler_cls, "compile_structural_tag")
    ):
        pytest.skip("xgrammar lacks GrammarCompiler.compile_structural_tag")
    tool = FunctionToolParam(
        function={
            "name": "noargs",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        }
    )
    tag = get_kimi_k3_structural_tag([tool], [], "required", False)
    vocab = [
        "a", "b", "c", "d", "e", "f", "g", "v", "<", "|", ">", '"', "=",
        "open", "close", "sep", "argument", "call", "tools", "response",
        "message", "string", "key", "noargs", "index", "x", "y",
    ]
    compiled = compiler_cls(xg.TokenizerInfo(vocab)).compile_structural_tag(tag)
    pre = '<|close|>response<|sep|><|open|>tools<|sep|><|open|>call tool="noargs" index="1"<|sep|>'
    post = "<|close|>call<|sep|><|close|>tools<|sep|><|close|>message<|sep|>"
    matcher = xg.GrammarMatcher(compiled)
    assert matcher.accept_string(pre + post)
    assert matcher.is_completed()
    matcher = xg.GrammarMatcher(compiled)
    invented = (
        pre
        + '<|open|>argument key="x" type="string"<|sep|>y'
        + "<|close|>argument<|sep|>"
        + post
    )
    assert not matcher.accept_string(invented)


def test_strict_opt_in_is_required_for_schema_constraint() -> None:
    """R13 P2: the wheel installs ONE structural tag for the whole list
    when any tool sets strict (registry :109); a sibling that merely
    OMITS the flag must stay unconstrained.  The omitted flag used to
    fall through ``strict is False`` into the schema channel, masking
    otherwise valid best-effort calls."""

    tag = get_kimi_k3_structural_tag(
        [_two_required_tool(name="weather"), _plain_sibling_tool()],
        [],
        "auto",
        False,
    )
    walked = _walk_formats(tag.format)
    names = [type(f).__name__ for f in walked]
    begins = [
        getattr(f, "begin", "")
        for f in walked
        if type(f).__name__ == "TagFormat"
    ]
    # the single permissive wildcard belongs to the omitting sibling ...
    assert names.count("PlusFormat") == 0, names  # weather coverage engaged
    assert (
        sum(1 for b in begins if b == "<|open|>argument ") == 1
    ), begins
    # ... while the strict tool keeps its schema-constrained tags
    assert any('tool="weather"' in b for b in begins), begins
    assert any('tool="plain_lookup"' in b for b in begins), begins
    assert any('key="alpha"' in b and 'type="string"' in b for b in begins), begins


def test_pattern_bounded_regex_regimes() -> None:
    """R17 R2-2/P3-1 unit contracts on _k3_bounded_string_regex."""

    from vllm_hcu.tool_parsers.kimi_k3_structural_tag import (
        _k3_bounded_string_regex,
    )

    # an unanchored pattern cannot be enforced, but a declared maxLength
    # must NOT be silently dropped (falls through to the length regime)
    regex = _k3_bounded_string_regex(
        {"type": "string", "pattern": "urgent", "maxLength": 10}
    )
    assert regex is not None and regex.endswith("{0,10}"), regex

    # no usable maxLength and an unenforceable pattern -> permissive None
    assert (
        _k3_bounded_string_regex({"type": "string", "pattern": "urgent"})
        is None
    )

    # ECMA-Annex-B literal quantifiers must be neither reinterpreted nor
    # crash the builder -- they fall to the verbatim anchored regime
    literal = _k3_bounded_string_regex(
        {"type": "string", "pattern": "^[a]{}$"}
    )
    assert literal == "(?:[a]{})", literal
    literal = _k3_bounded_string_regex(
        {"type": "string", "pattern": "^[a]{,5}$"}
    )
    assert literal == "(?:[a]{,5})", literal

    # non-simple anchored pattern: verbatim whole-value enforcement; the
    # declared length bounds are documented as NOT folded in
    verbatim = _k3_bounded_string_regex(
        {"type": "string", "pattern": "^[A-Z][a-z]+$", "maxLength": 10}
    )
    assert verbatim == "(?:[A-Z][a-z]+)", verbatim
