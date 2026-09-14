# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Register the HCU-owned HYV4 reasoning parser with vLLM lazily."""

from __future__ import annotations

import sys
from types import ModuleType

from ._common import (
    PatchCompatibilityError,
    apply_once,
    load_exact_module,
    require_callable,
    require_positional_signature,
)

TARGET_MODULE = "vllm.reasoning"
PATCH_ID = "platform.core_fix.hy_v4_reasoning_parser_registry"
TARGETS = (f"{TARGET_MODULE}.ReasoningParserManager.lazy_parsers['hy_v4']",)
_MARKER = "_vllm_hcu_hy_v4_reasoning_parser_registered"
_PARSER_NAME = "hy_v4"
_PARSER_MODULE = "vllm_hcu.reasoning.hy_v4_reasoning_parser"
_PARSER_CLASS = "HYV4ReasoningParser"
_EXPECTED = (_PARSER_MODULE, _PARSER_CLASS)


def _get_manager(module: ModuleType) -> type:
    manager = getattr(module, "ReasoningParserManager", None)
    if not isinstance(manager, type):
        raise PatchCompatibilityError(
            "required HCU patch target "
            "vllm.reasoning.ReasoningParserManager is missing"
        )
    lazy_parsers = getattr(manager, "lazy_parsers", None)
    if not isinstance(lazy_parsers, dict):
        raise PatchCompatibilityError(
            "required HCU patch target "
            "vllm.reasoning.ReasoningParserManager.lazy_parsers is incompatible"
        )
    reasoning_parsers = getattr(manager, "reasoning_parsers", None)
    if not isinstance(reasoning_parsers, dict):
        raise PatchCompatibilityError(
            "required HCU patch target "
            "vllm.reasoning.ReasoningParserManager.reasoning_parsers is incompatible"
        )
    return manager


def _validate_owners(manager: type) -> None:
    existing_eager = manager.reasoning_parsers.get(_PARSER_NAME)
    if existing_eager is not None:
        parser_module = sys.modules.get(_PARSER_MODULE)
        expected_eager = getattr(parser_module, _PARSER_CLASS, None)
        if existing_eager is not expected_eager:
            raise PatchCompatibilityError(
                f"reasoning parser {_PARSER_NAME!r} is already registered as "
                f"{existing_eager!r}"
            )

    existing_lazy = manager.lazy_parsers.get(_PARSER_NAME)
    if existing_lazy is not None and existing_lazy != _EXPECTED:
        raise PatchCompatibilityError(
            f"reasoning parser {_PARSER_NAME!r} is already registered as "
            f"{existing_lazy!r}"
        )


def apply_to_module(module: ModuleType) -> bool:
    reasoning_module = load_exact_module(TARGET_MODULE, module)
    manager = _get_manager(reasoning_module)
    _validate_owners(manager)
    if getattr(manager, _MARKER, False):
        return False

    register = require_callable(
        manager,
        "register_lazy_module",
        "vllm.reasoning.ReasoningParserManager.register_lazy_module",
    )
    require_positional_signature(
        register,
        "vllm.reasoning.ReasoningParserManager.register_lazy_module",
        ("name", "module_path", "class_name"),
    )
    register(_PARSER_NAME, _PARSER_MODULE, _PARSER_CLASS)
    setattr(manager, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    reasoning_module = load_exact_module(TARGET_MODULE, module)
    manager = _get_manager(reasoning_module)
    _validate_owners(manager)
    return apply_once(
        patch_id=PATCH_ID,
        targets=TARGETS,
        marker_owner=manager,
        marker=_MARKER,
        callback=lambda: apply_to_module(reasoning_module),
    )


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
