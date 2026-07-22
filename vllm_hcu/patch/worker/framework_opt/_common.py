# SPDX-License-Identifier: Apache-2.0
"""Strict helpers shared by the audited target worker-framework adapters."""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable, Sequence
from types import ModuleType
from typing import Any


class PatchCompatibilityError(RuntimeError):
    """The imported module is not the audited target vLLM API."""


def load_exact_module(target: str, module: ModuleType | None) -> ModuleType:
    if module is None:
        try:
            module = importlib.import_module(target)
        except Exception as exc:
            raise PatchCompatibilityError(
                f"required HCU patch target {target!r} could not be imported"
            ) from exc
    actual = getattr(module, "__name__", None)
    if actual != target:
        raise PatchCompatibilityError(
            f"required HCU patch expected module {target!r}, got {actual!r}"
        )
    return module


def require_callable(owner: object, name: str, target: str) -> Callable[..., Any]:
    value = getattr(owner, name, None)
    if not callable(value):
        raise PatchCompatibilityError(f"required HCU patch target {target} is missing")
    return value


def require_class(owner: object, name: str, target: str) -> type:
    value = getattr(owner, name, None)
    if not isinstance(value, type):
        raise PatchCompatibilityError(f"required HCU patch target {target} is missing")
    return value


def require_exact_signature(
    function: Callable[..., Any],
    target: str,
    *,
    positional: Sequence[str] = (),
    keyword_only: Sequence[str] = (),
    defaults: dict[str, object] | None = None,
    var_positional: str | None = None,
    var_keyword: str | None = None,
) -> None:
    defaults = defaults or {}
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError) as exc:
        raise PatchCompatibilityError(
            f"cannot inspect required HCU patch target {target}"
        ) from exc

    expected: list[tuple[str, inspect._ParameterKind]] = [
        (name, inspect.Parameter.POSITIONAL_OR_KEYWORD) for name in positional
    ]
    expected.extend(
        (name, inspect.Parameter.KEYWORD_ONLY) for name in keyword_only
    )
    if var_positional is not None:
        expected.append((var_positional, inspect.Parameter.VAR_POSITIONAL))
    if var_keyword is not None:
        expected.append((var_keyword, inspect.Parameter.VAR_KEYWORD))
    actual = [(parameter.name, parameter.kind) for parameter in signature.parameters.values()]
    if actual != expected:
        raise PatchCompatibilityError(
            f"required HCU patch target {target} has incompatible signature {signature}"
        )
    for parameter in signature.parameters.values():
        expected_default = defaults.get(parameter.name, inspect.Parameter.empty)
        if parameter.default != expected_default:
            raise PatchCompatibilityError(
                f"required HCU patch target {target} has incompatible signature {signature}"
            )


def require_unpatched(
    owner: object, name: str, target: str, wrapper_marker: str
) -> Callable[..., Any]:
    function = require_callable(owner, name, target)
    if getattr(function, wrapper_marker, False):
        raise PatchCompatibilityError(
            f"required HCU patch target {target} is wrapped without its owner marker"
        )
    return function


def already_applied(
    owner: object,
    marker: str,
    wrapped: Sequence[tuple[object, str, str, str]],
) -> bool:
    if not getattr(owner, marker, False):
        return False
    for target_owner, name, target, wrapper_marker in wrapped:
        function = require_callable(target_owner, name, target)
        if not getattr(function, wrapper_marker, False):
            raise PatchCompatibilityError(
                f"required HCU patch marker for {target} is stale; restart the process"
            )
    return True


__all__ = [
    "PatchCompatibilityError",
    "already_applied",
    "load_exact_module",
    "require_callable",
    "require_class",
    "require_exact_signature",
    "require_unpatched",
]
