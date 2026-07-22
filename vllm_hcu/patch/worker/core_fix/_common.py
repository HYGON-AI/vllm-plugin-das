# SPDX-License-Identifier: Apache-2.0
"""Strict, registry-free helpers for worker core-fix import callbacks."""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable, Sequence
from types import ModuleType
from typing import Any


class PatchCompatibilityError(RuntimeError):
    """The imported vLLM module does not expose the audited target API."""


def load_exact_module(target: str, module: ModuleType | None) -> ModuleType:
    """Import *target* or validate a module delivered by the coordinator."""

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


def require_exact_signature(
    function: Callable[..., Any],
    target: str,
    *,
    positional: Sequence[str] = (),
    keyword_only: Sequence[str] = (),
    defaults: dict[str, object] | None = None,
) -> None:
    """Require the exact parameter names, kinds, and relied-upon defaults."""

    defaults = defaults or {}
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError) as exc:
        raise PatchCompatibilityError(
            f"cannot inspect required HCU patch target {target}"
        ) from exc

    parameters = tuple(signature.parameters.values())
    expected_names = tuple(positional) + tuple(keyword_only)
    if tuple(parameter.name for parameter in parameters) != expected_names:
        raise PatchCompatibilityError(
            f"required HCU patch target {target} has incompatible signature "
            f"{signature}"
        )
    positional_kind = inspect.Parameter.POSITIONAL_OR_KEYWORD
    if any(
        parameter.kind is not positional_kind
        for parameter in parameters[: len(positional)]
    ) or any(
        parameter.kind is not inspect.Parameter.KEYWORD_ONLY
        for parameter in parameters[len(positional) :]
    ):
        raise PatchCompatibilityError(
            f"required HCU patch target {target} has incompatible signature "
            f"{signature}"
        )

    for parameter in parameters:
        expected_default = defaults.get(parameter.name, inspect.Parameter.empty)
        if parameter.default != expected_default:
            raise PatchCompatibilityError(
                f"required HCU patch target {target} has incompatible signature "
                f"{signature}"
            )


def require_classmethod(
    owner: type,
    name: str,
    target: str,
    *,
    positional: Sequence[str],
) -> tuple[classmethod, Callable[..., Any]]:
    descriptor = vars(owner).get(name)
    if not isinstance(descriptor, classmethod):
        raise PatchCompatibilityError(
            f"required HCU patch target {target} must be a classmethod"
        )
    function = descriptor.__func__
    require_exact_signature(function, target, positional=positional)
    return descriptor, function


def require_model_init(owner: type, target: str) -> Callable[..., Any]:
    function = vars(owner).get("__init__")
    if not callable(function):
        raise PatchCompatibilityError(f"required HCU patch target {target} is missing")
    require_exact_signature(
        function,
        target,
        positional=("self",),
        keyword_only=("vllm_config", "prefix"),
        defaults={"prefix": ""},
    )
    return function


def get_hf_config(vllm_config: object, target: str) -> object:
    try:
        model_config = getattr(vllm_config, "model_config")
        return getattr(model_config, "hf_config")
    except AttributeError as exc:
        raise PatchCompatibilityError(
            f"required HCU patch runtime input {target} has no model_config.hf_config"
        ) from exc


__all__ = [
    "PatchCompatibilityError",
    "get_hf_config",
    "load_exact_module",
    "require_callable",
    "require_class",
    "require_classmethod",
    "require_exact_signature",
    "require_model_init",
]
