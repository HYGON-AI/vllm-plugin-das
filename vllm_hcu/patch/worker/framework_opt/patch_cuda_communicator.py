# SPDX-License-Identifier: Apache-2.0
"""Early module exchange for HCU CustomAllreduce used by CudaCommunicator."""

from __future__ import annotations

from types import ModuleType

from vllm_hcu.patch.import_coordinator import (
    IMPORT_COORDINATOR,
    ExactImportCoordinator,
    ImportRegistration,
)

from ._common import (
    PatchCompatibilityError,
    load_exact_module,
    require_callable,
    require_class,
    require_exact_signature,
)

TARGET_MODULE = "vllm.distributed.device_communicators.cuda_communicator"
CUSTOM_ALLREDUCE_MODULE = "vllm.distributed.device_communicators.custom_all_reduce"
HCU_CUSTOM_ALLREDUCE_MODULE = "vllm_hcu.distributed.device_communicators.custom_all_reduce"
PATCH_ID = "worker.framework_opt.communicator.hcu_custom_allreduce_exchange"
TARGETS = (
    CUSTOM_ALLREDUCE_MODULE,
    f"{TARGET_MODULE}.CudaCommunicator.__init__",
)
_MARKER = "_vllm_hcu_cuda_communicator_validated"


def register(
    coordinator: ExactImportCoordinator = IMPORT_COORDINATOR,
) -> ImportRegistration:
    return coordinator.register_replacement(
        PATCH_ID,
        CUSTOM_ALLREDUCE_MODULE,
        HCU_CUSTOM_ALLREDUCE_MODULE,
        targets=(CUSTOM_ALLREDUCE_MODULE, HCU_CUSTOM_ALLREDUCE_MODULE),
        late_policy="fail",
    )


def apply_to_module(module: ModuleType) -> bool:
    cuda = load_exact_module(TARGET_MODULE, module)
    if getattr(cuda, _MARKER, False):
        return False
    communicator = require_class(
        cuda, "CudaCommunicator", f"{TARGET_MODULE}.CudaCommunicator"
    )
    init = require_callable(communicator, "__init__", TARGETS[1])
    require_exact_signature(
        init,
        TARGETS[1],
        positional=(
            "self",
            "cpu_group",
            "device",
            "device_group",
            "unique_name",
            "global_ranks",
            "global_world_size",
            "tcp_store_group",
        ),
        defaults={
            "device": None,
            "device_group": None,
            "unique_name": "",
            "global_ranks": None,
            "global_world_size": None,
            "tcp_store_group": None,
        },
    )
    if "all_to_all_single" in vars(communicator):
        raise PatchCompatibilityError(
            "clean vLLM v0.21 unexpectedly contains the stale CUDA all-to-all "
            "method that the retired legacy source patch attempted to remove"
        )
    setattr(cuda, _MARKER, True)
    return True


def apply(
    module: ModuleType | None = None,
    *,
    coordinator: ExactImportCoordinator = IMPORT_COORDINATOR,
) -> bool:
    register(coordinator)
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = [
    "CUSTOM_ALLREDUCE_MODULE",
    "HCU_CUSTOM_ALLREDUCE_MODULE",
    "PATCH_ID",
    "TARGET_MODULE",
    "TARGETS",
    "apply",
    "apply_to_module",
    "register",
]
