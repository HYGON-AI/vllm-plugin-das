# SPDX-License-Identifier: Apache-2.0
"""Strict helpers local to the atomic MoE migration group."""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable, Sequence
from types import ModuleType
from typing import Any

from .._common import (
    PatchCompatibilityError,
    load_exact_module,
    require_callable,
    require_class,
)


def require_parameter_names(
    function: Callable[..., Any],
    target: str,
    names: Sequence[str],
) -> inspect.Signature:
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError) as exc:
        raise PatchCompatibilityError(f"cannot inspect required HCU target {target}") from exc
    if tuple(signature.parameters) != tuple(names):
        raise PatchCompatibilityError(
            f"required HCU patch target {target} has incompatible signature {signature}"
        )
    return signature


def check_module_marker(
    module: ModuleType,
    marker: str,
    bindings: Sequence[tuple[object, str, str]],
) -> bool:
    if not getattr(module, marker, False):
        return False
    for owner, name, wrapper_marker in bindings:
        value = getattr(owner, name, None)
        if not callable(value) or not getattr(value, wrapper_marker, False):
            raise PatchCompatibilityError(
                f"stale HCU MoE marker for {module.__name__}.{name}; restart process"
            )
    return True


def marked_wrapper(original: Callable[..., Any], marker: str, implementation):
    wrapper = functools.wraps(original)(implementation)
    setattr(wrapper, marker, True)
    return wrapper


def require_replacement_module(
    module: ModuleType,
    replacement_name: str,
    targets: Sequence[str],
) -> None:
    if getattr(module, "__name__", None) != replacement_name:
        raise PatchCompatibilityError(
            f"{targets[0]} registers Torch custom ops and must be replaced before "
            f"import by {replacement_name}; post-import patching is forbidden"
        )


__all__ = [
    "PatchCompatibilityError",
    "check_module_marker",
    "load_exact_module",
    "marked_wrapper",
    "require_callable",
    "require_class",
    "require_parameter_names",
    "require_replacement_module",
]
