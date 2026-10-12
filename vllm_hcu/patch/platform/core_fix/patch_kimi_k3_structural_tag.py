# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Register the K3 XTML strict-tool-calling builder with the wheel.

The wheel's (0.25.1+das) ``structural_tag_registry`` exposes the dispatch
but predates the kimi_k3 builder, so strict enforcement (``required``,
named tool choice, ``strict=true`` auto) raises
``ValueError: Unknown format type: kimi_k3`` inside
``get_model_structural_tag`` before generation can start (review
2026-10-12, P1 #2; verified against the wheel image, not only public
sources). This patch registers the verbatim fork builder
(``vllm_hcu/tool_parsers/kimi_k3_structural_tag.py``, ported from fork
877e6996) into the wheel's vLLM-owned registry
(``_VLLM_STRUCTURAL_TAG_REGISTRY`` via ``register_vllm_structural_tag``);
if the wheel predates that registry dict, it falls back to wrapping
``get_model_structural_tag`` with a pre-dispatch that serves kimi_k3
from the ported builder and delegates everything else unchanged.
"""

from __future__ import annotations

import functools
from types import ModuleType

from ._common import (
    PatchCompatibilityError,
    apply_once,
    load_exact_module,
    require_callable,
)

TARGET_MODULE = "vllm.tool_parsers.structural_tag_registry"
PATCH_ID = "platform.core_fix.kimi_k3_structural_tag"
TARGETS = (f"{TARGET_MODULE}.get_model_structural_tag",)
_MARKER = "_vllm_hcu_kimi_k3_structural_tag_registered"
_WRAPPER = "_vllm_hcu_kimi_k3_structural_tag_wrapper"
_MODEL = "kimi_k3"
_BUILDER = "get_kimi_k3_structural_tag"


def _load_builder() -> object:
    from vllm_hcu.tool_parsers.kimi_k3_structural_tag import (
        get_kimi_k3_structural_tag,
    )

    return get_kimi_k3_structural_tag


def _register_through_vllm_registry(module: ModuleType) -> bool:
    """Preferred path: the wheel owns a vLLM-structural-tag registry."""

    registry = getattr(module, "_VLLM_STRUCTURAL_TAG_REGISTRY", None)
    register = getattr(module, "register_vllm_structural_tag", None)
    if not isinstance(registry, dict) or not callable(register):
        return False
    builder = _load_builder()
    register(_MODEL)(builder)
    setattr(module, "_hcu_kimi_k3_builder", builder)
    return True


def _wrap_dispatch(module: ModuleType) -> None:
    """Fallback path: pre-dispatch kimi_k3 in front of the wheel lookup."""

    original = require_callable(module, "get_model_structural_tag", TARGETS[0])
    if getattr(original, _WRAPPER, False):
        return
    normalize = require_callable(
        module, "normalize_tool_choice", f"{TARGET_MODULE}.normalize_tool_choice"
    )
    dump_tools = require_callable(
        module, "_dump_tool_for_xgrammar", f"{TARGET_MODULE}._dump_tool_for_xgrammar"
    )
    dump_choice = require_callable(
        module,
        "_dump_tool_choice_for_xgrammar",
        f"{TARGET_MODULE}._dump_tool_choice_for_xgrammar",
    )
    # Per-fork dispatch: `if tool_choice == "auto" and not
    # _any_tool_strict(tools): return None` gates BEFORE builder lookup; a
    # wheel missing the probe would silently tighten auto semantics.
    strict_probe = require_callable(
        module, "_any_tool_strict", f"{TARGET_MODULE}._any_tool_strict"
    )
    builder = _load_builder()

    @functools.wraps(original)
    def hcu_get_model_structural_tag(model, tools, tool_choice, reasoning):
        # Mirror the wheel's own early gates: with no tools or an explicit
        # "none" the wheel returns None before the format lookup, and a
        # non-strict "auto" never enforces a grammar at all.
        if model == _MODEL and tools and tool_choice != "none":
            non_strict_auto = tool_choice == "auto" and not strict_probe(tools)
            if not non_strict_auto:
                function_tools, builtin_tools, simplified = normalize(
                    # _dump_tool_for_xgrammar takes ONE tool (fork loop).
                    [dump_tools(tool) for tool in tools],
                    dump_choice(tool_choice),
                )
                return builder(function_tools, builtin_tools, simplified, reasoning)
        return original(model, tools, tool_choice, reasoning)

    setattr(hcu_get_model_structural_tag, _WRAPPER, True)
    module.get_model_structural_tag = hcu_get_model_structural_tag  # type: ignore[assignment]


def apply_to_module(module: ModuleType) -> bool:
    registry_module = load_exact_module(TARGET_MODULE, module)
    if getattr(registry_module, _MARKER, False):
        return False
    if not _register_through_vllm_registry(registry_module):
        _wrap_dispatch(registry_module)
    setattr(registry_module, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    registry_module = load_exact_module(TARGET_MODULE, module)
    return apply_once(
        patch_id=PATCH_ID,
        targets=TARGETS,
        marker_owner=registry_module,
        marker=_MARKER,
        callback=lambda: apply_to_module(registry_module),
    )


__all__ = [
    "PATCH_ID",
    "TARGETS",
    "TARGET_MODULE",
    "PatchCompatibilityError",
    "apply",
    "apply_to_module",
]
