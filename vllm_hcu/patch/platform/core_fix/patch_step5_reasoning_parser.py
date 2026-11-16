# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Runtime registration of the Step5 reasoning-parser name.

The historical vLLM source patch added ``"step5"`` to the reasoning-parser
registry as an alias of ``Step3p5ReasoningParser``. The current runtime keeps
the installed vLLM tree read-only and adds the same alias to the live registry
after ``vllm.reasoning`` has imported.
"""

from __future__ import annotations

from types import ModuleType

from ._common import (
    PatchCompatibilityError,
    apply_once,
    load_exact_module,
    require_callable,
)

TARGET_MODULE = "vllm.reasoning"
PATCH_ID = "platform.core_fix.step5_reasoning_parser"
TARGETS = (f"{TARGET_MODULE}.ReasoningParserManager",)
_MARKER = "_vllm_hcu_step5_reasoning_parser_registered"
_STEP5_ENTRY = ("vllm.reasoning.step3p5_reasoning_parser", "Step3p5ReasoningParser")


def _get_manager(module: ModuleType) -> type:
    manager = getattr(module, "ReasoningParserManager", None)
    if not isinstance(manager, type):
        raise PatchCompatibilityError(
            "required HCU patch target vllm.reasoning.ReasoningParserManager is "
            "missing"
        )
    lazy_parsers = getattr(manager, "lazy_parsers", None)
    eager_parsers = getattr(manager, "reasoning_parsers", None)
    if not isinstance(lazy_parsers, dict) or not isinstance(eager_parsers, dict):
        raise PatchCompatibilityError(
            "required HCU patch target vllm.reasoning.ReasoningParserManager "
            "registries are missing"
        )
    if lazy_parsers.get("step3p5") != _STEP5_ENTRY:
        raise PatchCompatibilityError(
            "vllm.reasoning step3p5 registration changed: "
            f"{lazy_parsers.get('step3p5')!r}"
        )
    return manager


def apply_to_module(module: ModuleType) -> bool:
    """Register ``step5`` as an alias of the Step3.5 reasoning parser."""

    parser_module = load_exact_module(TARGET_MODULE, module)
    manager = _get_manager(parser_module)
    if getattr(manager, _MARKER, False):
        return False

    register_lazy_module = require_callable(
        manager,
        "register_lazy_module",
        "vllm.reasoning.ReasoningParserManager.register_lazy_module",
    )

    lazy_parsers = manager.lazy_parsers
    eager_parsers = manager.reasoning_parsers
    existing_eager = eager_parsers.get("step5")
    existing_lazy = lazy_parsers.get("step5")
    if existing_eager is not None or existing_lazy is not None:
        if existing_eager is None and existing_lazy == _STEP5_ENTRY:
            setattr(manager, _MARKER, True)
            return False
        raise PatchCompatibilityError(
            "the target already registers a different step5 reasoning parser: "
            f"eager={existing_eager!r}, lazy={existing_lazy!r}"
        )

    register_lazy_module("step5", *_STEP5_ENTRY)
    if lazy_parsers.get("step5") != _STEP5_ENTRY:
        raise PatchCompatibilityError(
            "step5 reasoning-parser registration did not take effect"
        )
    setattr(manager, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    """Apply the Step5 reasoning-parser alias once."""

    parser_module = load_exact_module(TARGET_MODULE, module)
    manager = _get_manager(parser_module)

    return apply_once(
        patch_id=PATCH_ID,
        targets=TARGETS,
        marker_owner=manager,
        marker=_MARKER,
        callback=lambda: apply_to_module(parser_module),
    )


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
