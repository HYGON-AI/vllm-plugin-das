# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Static HCU test registration and deterministic duration partitioning.

The registration source is parsed with :mod:`ast`; importing the test modules is
deliberately avoided so the control-plane gate does not need vLLM, torch, or HCU.
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPOSITORY = Path(__file__).resolve().parents[3]
DEFAULT_REGISTRY = REPOSITORY / "tests" / "hcu_ci_registry.py"
REGISTER_FUNCTION = "register_hcu_ci"


class RegistrationError(ValueError):
    """Raised when static HCU registration is incomplete or ambiguous."""


@dataclass(frozen=True)
class HCURegistry:
    job: str
    target: str
    est_time: float
    disabled: str | None = None
    source_line: int = 0

    @property
    def test_file(self) -> str:
        return self.target.split("::", 1)[0]


def register_hcu_ci(
    *,
    job: str,
    target: str,
    est_time: float,
    disabled: str | None = None,
) -> None:
    """Document the literal-only registration API parsed by this module.

    Calls in ``tests/hcu_ci_registry.py`` are never executed by CI selection.
    Keeping this no-op implementation makes the source importable for editors.
    """


def _literal(node: ast.AST, *, path: Path, line: int, field: str) -> Any:
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError) as exc:
        raise RegistrationError(
            f"{path}:{line}: {field} must be a Python literal"
        ) from exc


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def parse_registry(path: Path = DEFAULT_REGISTRY) -> list[HCURegistry]:
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
    except (OSError, SyntaxError) as exc:
        raise RegistrationError(f"cannot parse HCU registry {path}: {exc}") from exc

    registrations: list[HCURegistry] = []
    allowed = {"job", "target", "est_time", "disabled"}
    required = {"job", "target", "est_time"}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or _call_name(node) != REGISTER_FUNCTION:
            continue
        if node.args:
            raise RegistrationError(
                f"{path}:{node.lineno}: {REGISTER_FUNCTION} accepts keyword arguments only"
            )
        values: dict[str, Any] = {}
        for keyword in node.keywords:
            if keyword.arg is None:
                raise RegistrationError(
                    f"{path}:{node.lineno}: **kwargs are not allowed in HCU registration"
                )
            if keyword.arg not in allowed:
                raise RegistrationError(
                    f"{path}:{node.lineno}: unsupported registration field {keyword.arg!r}"
                )
            if keyword.arg in values:
                raise RegistrationError(
                    f"{path}:{node.lineno}: duplicate registration field {keyword.arg!r}"
                )
            values[keyword.arg] = _literal(
                keyword.value,
                path=path,
                line=node.lineno,
                field=keyword.arg,
            )
        missing = sorted(required.difference(values))
        if missing:
            raise RegistrationError(
                f"{path}:{node.lineno}: registration is missing {missing}"
            )
        job = values["job"]
        target = values["target"]
        est_time = values["est_time"]
        disabled = values.get("disabled")
        if not isinstance(job, str) or not job:
            raise RegistrationError(f"{path}:{node.lineno}: job must be non-empty")
        if not isinstance(target, str) or not target.startswith("tests/"):
            raise RegistrationError(
                f"{path}:{node.lineno}: target must be a repository-relative tests/ path"
            )
        if (
            not isinstance(est_time, (int, float))
            or isinstance(est_time, bool)
            or est_time <= 0
        ):
            raise RegistrationError(
                f"{path}:{node.lineno}: est_time must be a positive number"
            )
        if disabled is not None and (not isinstance(disabled, str) or not disabled.strip()):
            raise RegistrationError(
                f"{path}:{node.lineno}: disabled must be a non-empty reason"
            )
        registrations.append(
            HCURegistry(
                job=job,
                target=target,
                est_time=float(est_time),
                disabled=disabled,
                source_line=node.lineno,
            )
        )
    if not registrations:
        raise RegistrationError(f"no {REGISTER_FUNCTION} calls found in {path}")
    return registrations


def _defined_nodeids(path: Path) -> tuple[set[str], set[str]]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError) as exc:
        raise RegistrationError(f"cannot inspect registered test {path}: {exc}") from exc
    functions: set[str] = set()
    classes: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.add(node.name)
        elif isinstance(node, ast.ClassDef):
            classes.add(node.name)
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    functions.add(f"{node.name}::{child.name}")
    return functions, classes


def validate_registrations(
    registrations: Sequence[HCURegistry],
    configured_jobs: Iterable[str],
) -> None:
    configured = set(configured_jobs)
    registered_jobs: set[str] = set()
    seen: set[tuple[str, str]] = set()
    inspected: dict[str, tuple[set[str], set[str]]] = {}
    for registration in registrations:
        key = (registration.job, registration.target)
        if key in seen:
            raise RegistrationError(
                f"duplicate HCU registration for job={registration.job!r}, "
                f"target={registration.target!r}"
            )
        seen.add(key)
        registered_jobs.add(registration.job)
        test_path = REPOSITORY / registration.test_file
        if not test_path.is_file() or not test_path.name.startswith("test_"):
            raise RegistrationError(
                f"registered target is not a test file: {registration.target}"
            )
        nodeid = registration.target.split("::", 1)[1] if "::" in registration.target else None
        if nodeid is not None:
            definitions = inspected.setdefault(
                registration.test_file,
                _defined_nodeids(test_path),
            )
            functions, classes = definitions
            if nodeid not in functions and nodeid not in classes:
                raise RegistrationError(
                    f"registered pytest node does not exist: {registration.target}"
                )
    unknown = sorted(registered_jobs.difference(configured))
    missing = sorted(configured.difference(registered_jobs))
    if unknown:
        raise RegistrationError(f"registry references unknown HCU jobs: {unknown}")
    if missing:
        raise RegistrationError(f"configured HCU jobs have no registration: {missing}")


def registrations_for_job(
    registrations: Sequence[HCURegistry],
    job: str,
    *,
    include_disabled: bool = False,
) -> list[HCURegistry]:
    return [
        item
        for item in registrations
        if item.job == job and (include_disabled or item.disabled is None)
    ]


def partition_registrations(
    registrations: Sequence[HCURegistry],
    partition_id: int,
    partition_size: int,
    *,
    live_estimates: Mapping[str, float] | None = None,
) -> list[HCURegistry]:
    """Assign registrations with deterministic longest-processing-time first."""

    if partition_size < 1:
        raise RegistrationError("partition_size must be at least 1")
    if partition_id < 0 or partition_id >= partition_size:
        raise RegistrationError(
            f"partition_id must be in [0, {partition_size}), got {partition_id}"
        )
    estimates = live_estimates or {}
    weighted = [
        replace(item, est_time=float(estimates.get(item.target, item.est_time)))
        for item in registrations
    ]
    if any(item.est_time <= 0 for item in weighted):
        raise RegistrationError("live estimates must be positive")
    partitions: list[list[HCURegistry]] = [[] for _ in range(partition_size)]
    totals = [0.0] * partition_size
    for item in sorted(weighted, key=lambda value: (-value.est_time, value.target)):
        destination = min(range(partition_size), key=lambda index: (totals[index], index))
        partitions[destination].append(item)
        totals[destination] += item.est_time
    return partitions[partition_id]


def load_live_estimates(path: Path | None) -> dict[str, float]:
    if path is None:
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RegistrationError(f"cannot load timing data {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise RegistrationError("timing data must be a target-to-seconds mapping")
    estimates: dict[str, float] = {}
    for target, seconds in raw.items():
        if (
            not isinstance(target, str)
            or not isinstance(seconds, (int, float))
            or isinstance(seconds, bool)
            or seconds <= 0
        ):
            raise RegistrationError(f"invalid timing entry: {target!r}={seconds!r}")
        estimates[target] = float(seconds)
    return estimates


def matrix_github_outputs(matrix: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """Return the shared workflow outputs for a planned HCU matrix."""

    planned = list(matrix)
    return {
        "matrix": json.dumps(planned, separators=(",", ":")),
        "has_jobs": str(bool(planned)).lower(),
    }
