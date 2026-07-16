# SPDX-License-Identifier: Apache-2.0
"""Small strict helpers for stage-3 runtime patch registration."""

from __future__ import annotations

from collections.abc import Callable
from types import ModuleType
from typing import Any


class Stage3CompatibilityError(RuntimeError):
    """A required vLLM v0.21 runtime target is missing or incompatible."""


def require_exact_module(module: ModuleType, expected_name: str) -> ModuleType:
    if not isinstance(module, ModuleType):
        raise Stage3CompatibilityError(
            f"required callback target {expected_name!r} is not a module"
        )
    actual_name = getattr(module, "__name__", None)
    if actual_name != expected_name:
        raise Stage3CompatibilityError(
            f"required callback expected module {expected_name!r}, got {actual_name!r}"
        )
    return module


def require_type(owner: object, name: str, target: str) -> type:
    value = getattr(owner, name, None)
    if not isinstance(value, type):
        raise Stage3CompatibilityError(f"required runtime target {target} is missing")
    return value


def require_callable(owner: object, name: str, target: str) -> Callable[..., Any]:
    value = getattr(owner, name, None)
    if not callable(value):
        raise Stage3CompatibilityError(f"required runtime target {target} is missing")
    return value


__all__ = [
    "Stage3CompatibilityError",
    "require_callable",
    "require_exact_module",
    "require_type",
]
