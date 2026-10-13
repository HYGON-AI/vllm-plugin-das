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
            },
        }
    )


def _union_tool(name: str = "fuzzy") -> FunctionToolParam:
    return FunctionToolParam(
        function={
            "name": name,
            "parameters": {
                "type": "object",
                "properties": {"mystery": {"type": ["string", "integer"]}},
                "required": ["mystery"],
            },
        }
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


def test_unexpressible_required_keeps_fork_shape() -> None:
    """A required prop mapped to the permissive tag must NOT be quantified:
    fall back to the portable single-Plus shape (fork parity)."""

    tag = get_kimi_k3_structural_tag([_union_tool()], [], "required", False)
    shapes = [type(f).__name__ for f in _walk_formats(tag.format)]
    # single Plus content=Or(tags) == the fork fallback shape; active
    # per-property coverage would show no Plus at all (R3 #2 bare tags)
    assert shapes.count("PlusFormat") == 1, shapes


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
