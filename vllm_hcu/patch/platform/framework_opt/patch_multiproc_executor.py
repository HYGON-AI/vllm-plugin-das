# SPDX-License-Identifier: Apache-2.0
"""Select HcuMultiprocExecutor through vLLM's executor backend entry."""

from __future__ import annotations

import inspect
from types import ModuleType

from ._common import PatchCompatibilityError, load_exact_module, require_callable, require_class, require_signature_prefix

TARGET_MODULE = "vllm.v1.executor.multiproc_executor"
PATCH_ID = "platform.framework_opt.hcu_multiproc_executor"
TARGETS = (
    f"{TARGET_MODULE}.MultiprocExecutor",
    "vllm_hcu.v1.executor.multiproc_executor.HcuMultiprocExecutor",
)
_MARKER = "_vllm_hcu_multiproc_contract_validated"
HCU_MULTIPROC_EXECUTOR_PATH = (
    "vllm_hcu.v1.executor.multiproc_executor.HcuMultiprocExecutor"
)
UPSTREAM_MULTIPROC_EXECUTOR_PATH = TARGET_MODULE + ".MultiprocExecutor"


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    if getattr(target, _MARKER, False):
        return False
    executor = require_class(target, "MultiprocExecutor", TARGETS[0])
    require_signature_prefix(
        require_callable(executor, "_init_executor", f"{TARGETS[0]}._init_executor"),
        f"{TARGETS[0]}._init_executor",
        ("self",),
    )
    message_queue = require_class(target, "MessageQueue", f"{TARGET_MODULE}.MessageQueue")
    try:
        signature = inspect.signature(message_queue)
    except (TypeError, ValueError) as exc:
        raise PatchCompatibilityError("cannot inspect vLLM MessageQueue") from exc
    if "max_chunks" not in signature.parameters:
        raise PatchCompatibilityError(
            f"vLLM MessageQueue lacks required max_chunks parameter: {signature}"
        )
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
