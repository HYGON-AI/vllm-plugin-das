# SPDX-License-Identifier: Apache-2.0
"""Strict helpers shared by the small platform compatibility patches."""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable, Sequence
from types import ModuleType
from typing import Any, TypeVar

from vllm_hcu.patch.runtime_state import PATCH_REGISTRY, PatchStatus, run_patch


class PatchCompatibilityError(RuntimeError):
    """The installed vLLM target does not match the supported v0.21 API."""


def load_exact_module(target: str, module: ModuleType | None) -> ModuleType:
    """Load *target* or validate a module supplied by an import callback."""

    if module is None:
        try:
            module = importlib.import_module(target)
        except Exception as exc:
            raise PatchCompatibilityError(
                f"required HCU patch target {target!r} could not be imported"
            ) from exc
    actual_name = getattr(module, "__name__", None)
    if actual_name != target:
        raise PatchCompatibilityError(
            f"required HCU patch expected module {target!r}, got {actual_name!r}"
        )
    return module


def require_callable(owner: object, name: str, target: str) -> Callable[..., Any]:
    value = getattr(owner, name, None)
    if not callable(value):
        raise PatchCompatibilityError(f"required HCU patch target {target} is missing")
    return value


def require_positional_signature(
    function: Callable[..., Any],
    target: str,
    names: Sequence[str],
    *,
    var_positional: str | None = None,
    var_keyword: str | None = None,
) -> None:
    """Validate the portion of a callable signature the adapter relies on."""

    try:
        parameters = tuple(inspect.signature(function).parameters.values())
    except (TypeError, ValueError) as exc:
        raise PatchCompatibilityError(
            f"cannot inspect required HCU patch target {target}"
        ) from exc

    expected_count = len(names)
    positional = parameters[:expected_count]
    positional_kinds = {
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    }
    if len(positional) != expected_count or any(
        parameter.kind not in positional_kinds for parameter in positional
    ):
        raise PatchCompatibilityError(
            f"required HCU patch target {target} has incompatible signature "
            f"{inspect.signature(function)}"
        )
    if tuple(parameter.name for parameter in positional) != tuple(names):
        raise PatchCompatibilityError(
            f"required HCU patch target {target} has incompatible signature "
            f"{inspect.signature(function)}"
        )

    remaining = parameters[expected_count:]
    expected_remaining: list[tuple[str, inspect._ParameterKind]] = []
    if var_positional is not None:
        expected_remaining.append((var_positional, inspect.Parameter.VAR_POSITIONAL))
    if var_keyword is not None:
        expected_remaining.append((var_keyword, inspect.Parameter.VAR_KEYWORD))
    actual_remaining = [(parameter.name, parameter.kind) for parameter in remaining]
    if actual_remaining != expected_remaining:
        raise PatchCompatibilityError(
            f"required HCU patch target {target} has incompatible signature "
            f"{inspect.signature(function)}"
        )


_T = TypeVar("_T")


def apply_once(
    *,
    patch_id: str,
    targets: str | Sequence[str],
    marker_owner: object,
    marker: str,
    callback: Callable[[], _T],
) -> bool:
    """Run one strict adapter and make module reloads fail visibly.

    A process-local applied record combined with a missing marker means the
    target module/class was replaced or reloaded.  Silently reporting that
    state as applied would leave vLLM running without the required behavior.
    """

    if getattr(marker_owner, marker, False):
        return False

    record = PATCH_REGISTRY.get(patch_id)
    if record is not None and record.status is PatchStatus.APPLIED:
        raise PatchCompatibilityError(
            f"required HCU patch {patch_id!r} was applied but its target was "
            "subsequently replaced or reloaded; restart the process"
        )

    applied_here = False

    def install_and_record() -> _T:
        nonlocal applied_here
        result = callback()
        applied_here = True
        return result

    run_patch(patch_id, targets, install_and_record)
    if not getattr(marker_owner, marker, False):
        raise PatchCompatibilityError(
            f"required HCU patch {patch_id!r} did not install its target marker"
        )
    return applied_here


__all__ = [
    "PatchCompatibilityError",
    "apply_once",
    "load_exact_module",
    "require_callable",
    "require_positional_signature",
]
