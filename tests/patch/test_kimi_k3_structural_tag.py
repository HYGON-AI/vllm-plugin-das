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
    from openai.types.shared import FunctionDefinition

    return FunctionToolParam(
        function=FunctionDefinition(
            name=name,
            description="run",
            parameters={
                "type": "object",
                "properties": {"code": {"type": "string"}},
                "required": ["code"],
            },
            strict=True,
        )
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
