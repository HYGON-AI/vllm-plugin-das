# SPDX-License-Identifier: Apache-2.0
"""Strict vLLM v0.21 contract checks for platform framework adapters."""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable, Sequence
from types import ModuleType
from typing import Any


class PatchCompatibilityError(RuntimeError):
    """The imported object is not the audited vLLM v0.21 target."""


def load_exact_module(target: str, module: ModuleType | None) -> ModuleType:
    if module is None:
        try:
            module = importlib.import_module(target)
        except Exception as exc:
            raise PatchCompatibilityError(
                f"required HCU patch target {target!r} could not be imported"
            ) from exc
    if getattr(module, "__name__", None) != target:
        raise PatchCompatibilityError(
            f"required HCU patch expected module {target!r}, "
            f"got {getattr(module, '__name__', None)!r}"
        )
    return module


def require_class(owner: object, name: str, target: str) -> type:
    value = getattr(owner, name, None)
    if not isinstance(value, type):
        raise PatchCompatibilityError(f"required HCU patch target {target} is missing")
    return value


def require_callable(owner: object, name: str, target: str) -> Callable[..., Any]:
    value = getattr(owner, name, None)
    if not callable(value):
        raise PatchCompatibilityError(f"required HCU patch target {target} is missing")
    return value


def require_signature_prefix(
    function: Callable[..., Any], target: str, names: Sequence[str]
) -> inspect.Signature:
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError) as exc:
        raise PatchCompatibilityError(f"cannot inspect required target {target}") from exc
    parameters = tuple(signature.parameters.values())
    if len(parameters) < len(names) or tuple(p.name for p in parameters[: len(names)]) != tuple(names):
        raise PatchCompatibilityError(
            f"required HCU patch target {target} has incompatible signature {signature}"
        )
    return signature


def already_applied(
    owner: object,
    marker: str,
    wrapped: Sequence[tuple[object, str, str]],
) -> bool:
    if not getattr(owner, marker, False):
        return False
    for target_owner, name, wrapper_marker in wrapped:
        function = getattr(target_owner, name, None)
        if not callable(function) or not getattr(function, wrapper_marker, False):
            raise PatchCompatibilityError(
                f"required HCU patch marker for {name} is stale; restart the process"
            )
    return True


__all__ = [
    "PatchCompatibilityError",
    "already_applied",
    "load_exact_module",
    "require_callable",
    "require_class",
    "require_signature_prefix",
]
