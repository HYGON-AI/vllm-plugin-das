# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Register the HCU-owned HYV4 tool parser with vLLM lazily."""

from __future__ import annotations

from types import ModuleType

from ._common import (
    PatchCompatibilityError,
    apply_once,
    load_exact_module,
    require_callable,
    require_positional_signature,
)

TARGET_MODULE = "vllm.tool_parsers"
PATCH_ID = "platform.core_fix.hy_v4_tool_parser_registry"
TARGETS = (f"{TARGET_MODULE}.ToolParserManager.lazy_parsers['hy_v4']",)
_MARKER = "_vllm_hcu_hy_v4_tool_parser_registered"
_PARSER_NAME = "hy_v4"
_PARSER_MODULE = "vllm_hcu.tool_parsers.hy_v4_tool_parser"
_PARSER_CLASS = "HYV4ToolParser"
_EXPECTED = (_PARSER_MODULE, _PARSER_CLASS)


def _get_manager(module: ModuleType) -> type:
    manager = getattr(module, "ToolParserManager", None)
    if not isinstance(manager, type):
        raise PatchCompatibilityError(
            "required HCU patch target vllm.tool_parsers.ToolParserManager is missing"
        )
    lazy_parsers = getattr(manager, "lazy_parsers", None)
    if not isinstance(lazy_parsers, dict):
        raise PatchCompatibilityError(
            "required HCU patch target "
            "vllm.tool_parsers.ToolParserManager.lazy_parsers is incompatible"
        )
    return manager


def apply_to_module(module: ModuleType) -> bool:
    tool_module = load_exact_module(TARGET_MODULE, module)
    manager = _get_manager(tool_module)
    if getattr(manager, _MARKER, False):
        return False

    existing = manager.lazy_parsers.get(_PARSER_NAME)
    if existing is not None and existing != _EXPECTED:
        raise PatchCompatibilityError(
            f"tool parser {_PARSER_NAME!r} is already registered as {existing!r}"
        )

    register = require_callable(
        manager,
        "register_lazy_module",
        "vllm.tool_parsers.ToolParserManager.register_lazy_module",
    )
    require_positional_signature(
        register,
        "vllm.tool_parsers.ToolParserManager.register_lazy_module",
        ("name", "module_path", "class_name"),
    )
    register(_PARSER_NAME, _PARSER_MODULE, _PARSER_CLASS)
    setattr(manager, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    tool_module = load_exact_module(TARGET_MODULE, module)
    manager = _get_manager(tool_module)
    return apply_once(
        patch_id=PATCH_ID,
        targets=TARGETS,
        marker_owner=manager,
        marker=_MARKER,
        callback=lambda: apply_to_module(tool_module),
    )


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
