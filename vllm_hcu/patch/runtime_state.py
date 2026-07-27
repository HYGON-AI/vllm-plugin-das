# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Process-local state for runtime HCU patches.

This module deliberately has no dependency on vLLM.  It is imported during
plugin discovery, including in spawned EngineCore and Worker processes, so it
must remain cheap and safe to import.
"""

from __future__ import annotations

import multiprocessing
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, replace
from enum import Enum
from typing import Callable, Iterator, Sequence, TypeVar


class ProcessRole(str, Enum):
    """The vLLM process in which a patch was observed."""

    MAIN = "Main"
    ENGINE_CORE = "EngineCore"
    WORKER = "Worker"


class PatchStatus(str, Enum):
    """Lifecycle of one runtime patch in the current process."""

    ARMED = "armed"
    APPLYING = "applying"
    APPLIED = "applied"
    SKIPPED = "skipped"
    FAILED = "failed"


class LatchedPatchError(RuntimeError):
    """Raised when an apply attempt reaches a previously failed patch."""


@dataclass(frozen=True, slots=True)
class PatchRecord:
    """Immutable snapshot of one patch state."""

    patch_id: str
    status: PatchStatus
    targets: tuple[str, ...]
    feature_enabled: bool = False
    failure_reason: str | None = None
    updated_at: float = 0.0


def _normalise_role(role: ProcessRole | str) -> ProcessRole:
    if isinstance(role, ProcessRole):
        return role
    normalised = role.strip().lower().replace("_", "").replace("-", "")
    aliases = {
        "main": ProcessRole.MAIN,
        "mainprocess": ProcessRole.MAIN,
        "enginecore": ProcessRole.ENGINE_CORE,
        "engine": ProcessRole.ENGINE_CORE,
        "worker": ProcessRole.WORKER,
    }
    try:
        return aliases[normalised]
    except KeyError as exc:
        allowed = ", ".join(item.value for item in ProcessRole)
        raise ValueError(f"unknown process role {role!r}; expected one of {allowed}") from exc


def detect_process_role() -> ProcessRole:
    """Best-effort role detection, overridable by ``VLLM_HCU_PROCESS_ROLE``.

    vLLM can choose multiprocessing names, therefore dispatchers should call
    :func:`set_process_role` when they know the exact role.  The heuristic is a
    safe default for diagnostics before that point.
    """

    explicit = os.getenv("VLLM_HCU_PROCESS_ROLE")
    if explicit:
        return _normalise_role(explicit)

    process_name = multiprocessing.current_process().name.lower()
    compact_name = process_name.replace("_", "").replace("-", "")
    if "enginecore" in compact_name:
        return ProcessRole.ENGINE_CORE
    if "worker" in compact_name:
        return ProcessRole.WORKER
    return ProcessRole.MAIN


class PatchRegistry:
    """Thread-safe, process-local registry for patch application state.

    Failed records are latched.  They can only be cleared by
    :meth:`reset_for_tests`, which prevents an import reload or a repeated
    plugin callback from accidentally registering a custom op twice.
    """

    def __init__(self) -> None:
        self._condition = threading.Condition(threading.RLock())
        self._records: dict[str, PatchRecord] = {}
        self._owners: dict[str, int] = {}
        self._role: ProcessRole | None = None

    @staticmethod
    def _validate_patch_id(patch_id: str) -> str:
        if not isinstance(patch_id, str) or not patch_id.strip():
            raise ValueError("patch_id must be a non-empty string")
        return patch_id.strip()

    @staticmethod
    def _normalise_targets(targets: str | Sequence[str]) -> tuple[str, ...]:
        if isinstance(targets, str):
            values = (targets,)
        else:
            values = tuple(targets)
        if not values or any(not isinstance(item, str) or not item.strip() for item in values):
            raise ValueError("targets must contain at least one non-empty symbol name")
        return tuple(item.strip() for item in values)

    def set_process_role(self, role: ProcessRole | str) -> None:
        with self._condition:
            self._role = _normalise_role(role)

    def process_role(self) -> ProcessRole:
        with self._condition:
            return self._role or detect_process_role()

    def declare(
        self,
        patch_id: str,
        targets: str | Sequence[str],
    ) -> PatchRecord:
        """Declare a patch as armed, or return its existing declaration."""

        patch_id = self._validate_patch_id(patch_id)
        target_tuple = self._normalise_targets(targets)
        with self._condition:
            existing = self._records.get(patch_id)
            if existing is not None:
                if existing.targets != target_tuple:
                    raise ValueError(
                        f"patch id {patch_id!r} is already bound to "
                        f"{existing.targets!r}, not {target_tuple!r}"
                    )
                return existing
            record = PatchRecord(
                patch_id=patch_id,
                status=PatchStatus.ARMED,
                targets=target_tuple,
                updated_at=time.time(),
            )
            self._records[patch_id] = record
            return record

    def begin(self, patch_id: str) -> bool:
        """Claim a patch for application.

        Returns ``False`` when the patch is already applied or skipped.  A
        concurrent caller waits for the owner.  A failed patch raises rather
        than retrying.
        """

        patch_id = self._validate_patch_id(patch_id)
        current_thread = threading.get_ident()
        with self._condition:
            if patch_id not in self._records:
                raise KeyError(f"patch {patch_id!r} has not been declared")
            while self._records[patch_id].status is PatchStatus.APPLYING:
                if self._owners.get(patch_id) == current_thread:
                    raise RuntimeError(f"re-entrant application of patch {patch_id!r}")
                self._condition.wait()

            record = self._records[patch_id]
            if record.status is PatchStatus.FAILED:
                raise LatchedPatchError(
                    f"patch {patch_id!r} previously failed: {record.failure_reason}"
                )
            if record.status in {PatchStatus.APPLIED, PatchStatus.SKIPPED}:
                return False

            self._records[patch_id] = replace(
                record,
                status=PatchStatus.APPLYING,
                failure_reason=None,
                updated_at=time.time(),
            )
            self._owners[patch_id] = current_thread
            return True

    def mark_applied(self, patch_id: str, *, feature_enabled: bool = True) -> PatchRecord:
        if not isinstance(feature_enabled, bool):
            raise TypeError("feature_enabled must be bool")
        with self._condition:
            record = self._require_record(patch_id)
            if record.status is PatchStatus.APPLIED:
                return record
            if record.status is not PatchStatus.APPLYING:
                raise RuntimeError(
                    f"cannot mark patch {patch_id!r} applied from {record.status.value!r}"
                )
            updated = replace(
                record,
                status=PatchStatus.APPLIED,
                feature_enabled=feature_enabled,
                failure_reason=None,
                updated_at=time.time(),
            )
            self._finish(patch_id, updated)
            return updated

    def mark_skipped(self, patch_id: str, reason: str | None = None) -> PatchRecord:
        with self._condition:
            record = self._require_record(patch_id)
            if record.status is PatchStatus.SKIPPED:
                return record
            if record.status not in {PatchStatus.ARMED, PatchStatus.APPLYING}:
                raise RuntimeError(
                    f"cannot skip patch {patch_id!r} from {record.status.value!r}"
                )
            updated = replace(
                record,
                status=PatchStatus.SKIPPED,
                # Preserve whether the user actually requested the feature.
                # This distinguishes "disabled and therefore skipped" from
                # "requested but unavailable" in the process report.
                feature_enabled=record.feature_enabled,
                failure_reason=reason,
                updated_at=time.time(),
            )
            self._finish(patch_id, updated)
            return updated

    def mark_failed(self, patch_id: str, error: BaseException | str) -> PatchRecord:
        reason = self._format_failure(error)
        with self._condition:
            record = self._require_record(patch_id)
            if record.status is PatchStatus.FAILED:
                return record
            if record.status in {PatchStatus.APPLIED, PatchStatus.SKIPPED}:
                raise RuntimeError(
                    f"cannot fail patch {patch_id!r} from {record.status.value!r}"
                )
            updated = replace(
                record,
                status=PatchStatus.FAILED,
                # Failure describes the apply result, not whether the user
                # requested the feature.  Preserve the observed request state
                # so diagnostics can distinguish a disabled path from an
                # enabled feature whose required patch failed.
                feature_enabled=record.feature_enabled,
                failure_reason=reason,
                updated_at=time.time(),
            )
            self._finish(patch_id, updated)
            return updated

    def set_feature_enabled(self, patch_id: str, enabled: bool) -> PatchRecord:
        """Update the observed feature state of an armed/applied patch.

        Dispatchers first arm exact import callbacks and may only obtain the
        deserialised ``VllmConfig`` later in Worker construction.  Recording
        the state while a callback is still armed (or was capability-skipped)
        makes ``patch_report()`` truthful at every phase.
        """

        if not isinstance(enabled, bool):
            raise TypeError("enabled must be bool")
        with self._condition:
            record = self._require_record(patch_id)
            if record.status not in {
                PatchStatus.ARMED,
                PatchStatus.APPLIED,
                PatchStatus.SKIPPED,
            }:
                raise RuntimeError(
                    "feature state requires an armed, applied, or skipped patch, "
                    f"got {record.status.value!r}"
                )
            updated = replace(record, feature_enabled=enabled, updated_at=time.time())
            self._records[patch_id] = updated
            return updated

    def get(self, patch_id: str) -> PatchRecord | None:
        with self._condition:
            return self._records.get(patch_id)

    def snapshot(self) -> tuple[PatchRecord, ...]:
        with self._condition:
            return tuple(self._records[key] for key in sorted(self._records))

    def report(self) -> dict[str, object]:
        """Return a JSON-serialisable report for the current process."""

        pid = os.getpid()
        role = self.process_role().value
        patches: dict[str, object] = {}
        for record in self.snapshot():
            patches[record.patch_id] = {
                "pid": pid,
                "process_role": role,
                "status": record.status.value,
                "targets": list(record.targets),
                "failure_reason": record.failure_reason,
                "feature_enabled": record.feature_enabled,
                "updated_at": record.updated_at,
            }
        return {
            "pid": pid,
            "process_role": role,
            "patches": patches,
        }

    def reset_for_tests(self) -> None:
        """Clear all state.  Production code must never call this method."""

        with self._condition:
            self._records.clear()
            self._owners.clear()
            self._role = None
            self._condition.notify_all()

    def _require_record(self, patch_id: str) -> PatchRecord:
        patch_id = self._validate_patch_id(patch_id)
        try:
            return self._records[patch_id]
        except KeyError as exc:
            raise KeyError(f"patch {patch_id!r} has not been declared") from exc

    def _finish(self, patch_id: str, record: PatchRecord) -> None:
        self._records[patch_id] = record
        self._owners.pop(patch_id, None)
        self._condition.notify_all()

    @staticmethod
    def _format_failure(error: BaseException | str) -> str:
        if isinstance(error, BaseException):
            message = str(error)
            return f"{type(error).__name__}: {message}" if message else type(error).__name__
        if not isinstance(error, str) or not error.strip():
            raise ValueError("failure reason must be a non-empty string or exception")
        return error.strip()


_T = TypeVar("_T")


PATCH_REGISTRY = PatchRegistry()


def set_process_role(role: ProcessRole | str) -> None:
    PATCH_REGISTRY.set_process_role(role)


def patch_report() -> dict[str, object]:
    return PATCH_REGISTRY.report()


def run_patch(
    patch_id: str,
    targets: str | Sequence[str],
    callback: Callable[[], _T],
    *,
    feature_enabled: bool = True,
    registry: PatchRegistry = PATCH_REGISTRY,
) -> _T | None:
    """Apply one patch once and latch any failure.

    ``None`` is returned for an already-applied/skipped patch.  A previously
    failed patch raises :class:`LatchedPatchError` without rerunning callback.
    """

    registry.declare(patch_id, targets)
    if not registry.begin(patch_id):
        return None
    try:
        result = callback()
    except BaseException as exc:
        registry.mark_failed(patch_id, exc)
        raise
    registry.mark_applied(patch_id, feature_enabled=feature_enabled)
    return result


@contextmanager
def applying_patch(
    patch_id: str,
    targets: str | Sequence[str],
    *,
    feature_enabled: bool = True,
    registry: PatchRegistry = PATCH_REGISTRY,
) -> Iterator[bool]:
    """Context-manager form of :func:`run_patch`."""

    registry.declare(patch_id, targets)
    should_apply = registry.begin(patch_id)
    if not should_apply:
        yield False
        return
    try:
        yield True
    except BaseException as exc:
        registry.mark_failed(patch_id, exc)
        raise
    else:
        registry.mark_applied(patch_id, feature_enabled=feature_enabled)


__all__ = [
    "PATCH_REGISTRY",
    "LatchedPatchError",
    "PatchRecord",
    "PatchRegistry",
    "PatchStatus",
    "ProcessRole",
    "applying_patch",
    "detect_process_role",
    "patch_report",
    "run_patch",
    "set_process_role",
]
