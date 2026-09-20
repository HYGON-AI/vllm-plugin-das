# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Lazy, process-local registration of downstream HCU scheduler adapters.

Adapters inherit HcuScheduler or HcuAsyncScheduler and retain their constructor
validation. Register paths from a general plugin before config construction;
registering an adapter never imports its scheduler/model stack.
"""

from __future__ import annotations

_SCHEDULER_ADAPTERS: dict[str, bool] = {}
_BUILTIN_PATHS = frozenset(
    {
        "vllm.v1.core.sched.scheduler.Scheduler",
        "vllm_hcu.v1.core.sched.scheduler.HcuScheduler",
        "vllm_hcu.v1.core.sched.scheduler.HcuAsyncScheduler",
    }
)


def register_scheduler_adapter(class_path: str, *, async_scheduling: bool) -> None:
    """Register an explicit adapter path; identical registrations are idempotent."""
    if (
        not isinstance(class_path, str)
        or "." not in class_path
        or not all(part.isidentifier() for part in class_path.split("."))
    ):
        raise ValueError("A scheduler adapter requires a qualified class path")
    if type(async_scheduling) is not bool:
        raise TypeError("async_scheduling must be bool")
    if class_path in _BUILTIN_PATHS:
        raise ValueError("Built-in scheduler selection cannot be overridden")
    previous = _SCHEDULER_ADAPTERS.get(class_path)
    if previous is not None and previous != async_scheduling:
        raise ValueError(f"Conflicting scheduler registration: {class_path}")
    _SCHEDULER_ADAPTERS[class_path] = async_scheduling


def get_scheduler_adapter_mode(class_path: object) -> bool | None:
    """Return the registered async requirement, or None for an unknown path."""
    if not isinstance(class_path, str):
        return None
    return _SCHEDULER_ADAPTERS.get(class_path)
