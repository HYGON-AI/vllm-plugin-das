# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Select HcuMultiprocExecutor through vLLM's executor backend entry."""

from __future__ import annotations

import functools
import inspect
from types import ModuleType

from ._common import (
    PatchCompatibilityError,
    already_applied,
    load_exact_module,
    require_callable,
    require_class,
    require_signature_prefix,
)

TARGET_MODULE = "vllm.v1.executor.multiproc_executor"
PATCH_ID = "platform.framework_opt.hcu_multiproc_executor"
TARGETS = (
    f"{TARGET_MODULE}.MultiprocExecutor",
    f"{TARGET_MODULE}.MultiprocExecutor.collective_rpc",
    f"{TARGET_MODULE}.MessageQueue.recv",
    f"{TARGET_MODULE}.FutureWrapper",
    f"{TARGET_MODULE}.WorkerProc",
    f"{TARGET_MODULE}.WorkerProc.ResponseStatus",
    "vllm_hcu.v1.executor.multiproc_executor.HcuMultiprocExecutor",
)
_MARKER = "_vllm_hcu_multiproc_contract_validated"
_RECV_MARKER = "_vllm_hcu_non_negative_timeout"
HCU_MULTIPROC_EXECUTOR_PATH = (
    "vllm_hcu.v1.executor.multiproc_executor.HcuMultiprocExecutor"
)
UPSTREAM_MULTIPROC_EXECUTOR_PATH = TARGET_MODULE + ".MultiprocExecutor"


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    executor = require_class(target, "MultiprocExecutor", TARGETS[0])
    message_queue = require_class(
        target,
        "MessageQueue",
        f"{TARGET_MODULE}.MessageQueue",
    )
    if already_applied(
        target,
        _MARKER,
        ((message_queue, "recv", _RECV_MARKER),),
    ):
        return False
    require_signature_prefix(
        require_callable(executor, "_init_executor", f"{TARGETS[0]}._init_executor"),
        f"{TARGETS[0]}._init_executor",
        ("self",),
    )
    require_signature_prefix(
        require_callable(executor, "collective_rpc", TARGETS[1]),
        TARGETS[1],
        (
            "self",
            "method",
            "timeout",
            "args",
            "kwargs",
            "non_block",
            "unique_reply_rank",
            "kv_output_aggregator",
            "ec_output_aggregator",
        ),
    )
    future_wrapper = require_class(target, "FutureWrapper", TARGETS[3])
    require_signature_prefix(
        future_wrapper,
        TARGETS[3],
        ("futures_queue", "get_response", "aggregate"),
    )
    worker_proc = require_class(target, "WorkerProc", TARGETS[4])
    response_status = require_class(worker_proc, "ResponseStatus", TARGETS[5])
    if not hasattr(response_status, "SUCCESS"):
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGETS[5]}.SUCCESS is missing"
        )
    try:
        signature = inspect.signature(message_queue)
    except (TypeError, ValueError) as exc:
        raise PatchCompatibilityError("cannot inspect vLLM MessageQueue") from exc
    if "max_chunks" not in signature.parameters:
        raise PatchCompatibilityError(
            f"vLLM MessageQueue lacks required max_chunks parameter: {signature}"
        )
    original_recv = require_callable(message_queue, "recv", TARGETS[2])
    if getattr(original_recv, _RECV_MARKER, False):
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGETS[2]} is wrapped without "
            "its owner marker"
        )
    require_signature_prefix(
        original_recv,
        TARGETS[2],
        ("socket", "timeout"),
    )

    @functools.wraps(original_recv)
    def hcu_recv(socket, timeout):
        timeout = None if timeout is None else max(0.0, timeout)
        return original_recv(socket, timeout)

    setattr(hcu_recv, _RECV_MARKER, True)
    setattr(message_queue, "recv", staticmethod(hcu_recv))
    setattr(target, _MARKER, True)
    return True


def select_hcu_multiproc_executor(vllm_config: object) -> bool:
    parallel_config = getattr(vllm_config, "parallel_config", None)
    if parallel_config is None or not hasattr(
        parallel_config, "distributed_executor_backend"
    ):
        raise PatchCompatibilityError(
            "vllm_config.parallel_config.distributed_executor_backend is missing"
        )
    selected = parallel_config.distributed_executor_backend
    if selected in ("mp", UPSTREAM_MULTIPROC_EXECUTOR_PATH):
        # Executor.get_class resolves qualified strings lazily at engine start.
        parallel_config.distributed_executor_backend = HCU_MULTIPROC_EXECUTOR_PATH
        return True
    if selected == HCU_MULTIPROC_EXECUTOR_PATH:
        return False
    return False


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = [
    "HCU_MULTIPROC_EXECUTOR_PATH",
    "PATCH_ID",
    "TARGET_MODULE",
    "TARGETS",
    "UPSTREAM_MULTIPROC_EXECUTOR_PATH",
    "apply",
    "apply_to_module",
    "select_hcu_multiproc_executor",
]
