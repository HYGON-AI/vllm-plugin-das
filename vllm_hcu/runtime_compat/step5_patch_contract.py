# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Reviewed API contracts and idempotence for Step5 runtime adapters."""

from __future__ import annotations
import inspect
import hashlib
import textwrap


class Step5PatchCompatibilityError(RuntimeError):
    pass


def resolve(target, symbol):
    value = target
    for name in symbol.removeprefix(target.__name__ + ".").split("."):
        value = getattr(value, name, None)
        if value is None:
            raise Step5PatchCompatibilityError(
                f"Required Step5 target is missing: {symbol}"
            )
    return value


def already_applied(target, marker, symbols):
    previous = getattr(target, marker, None)
    if previous is None:
        return False
    if not isinstance(previous, dict) or any(
        resolve(target, name) is not previous.get(name) for name in symbols
    ):
        raise Step5PatchCompatibilityError(f"Stale Step5 patch marker: {marker}")
    return True


def remember_targets(target, marker, symbols):
    setattr(target, marker, {name: resolve(target, name) for name in symbols})


def require_signatures(target, signatures, reviewed_sources=()):
    for symbol, expected in signatures:
        value = resolve(target, symbol)
        if not callable(value):
            raise Step5PatchCompatibilityError(
                f"Required Step5 target is not callable: {symbol}"
            )
        actual = tuple(inspect.signature(value).parameters)
        if actual != expected:
            raise Step5PatchCompatibilityError(
                f"Incompatible Step5 target signature: {symbol}: expected {expected}, got {actual}"
            )
    for symbol, expected in reviewed_sources:
        value = inspect.unwrap(resolve(target, symbol))
        try:
            source = textwrap.dedent(inspect.getsource(value))
        except (TypeError, OSError) as exc:
            raise Step5PatchCompatibilityError(
                f"Unable to verify Step5 target source: {symbol}"
            ) from exc
        actual = hashlib.sha256(source.encode()).hexdigest()
        if actual not in expected:
            raise Step5PatchCompatibilityError(
                f"Unreviewed Step5 target source: {symbol}: {actual}"
            )
