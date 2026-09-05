# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Early module exchange for HCU CustomAllreduce used by CudaCommunicator."""

from __future__ import annotations

import functools
from types import ModuleType

from vllm_hcu.patch.config import get_hcu_config

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
_WRAPPER = "_vllm_hcu_cuda_communicator_wrapper"


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
        init = getattr(getattr(cuda, "CudaCommunicator", None), "__init__", None)
        if not callable(init) or not getattr(init, _WRAPPER, False):
            raise PatchCompatibilityError(
                "HCU CUDA communicator marker is stale; restart the process"
            )
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
            "use_all2all",
        ),
        defaults={
            "device": None,
            "device_group": None,
            "unique_name": "",
            "global_ranks": None,
            "global_world_size": None,
            "tcp_store_group": None,
            "use_all2all": False,
        },
    )
    if "all_to_all_single" in vars(communicator):
        raise PatchCompatibilityError(
            "clean audited target vLLM unexpectedly contains the stale CUDA all-to-all "
            "method that the retired legacy source patch attempted to remove"
        )

    @functools.wraps(init)
    def hcu_init(
        self,
        cpu_group,
        device=None,
        device_group=None,
        unique_name="",
        global_ranks=None,
        global_world_size=None,
        tcp_store_group=None,
        use_all2all=False,
    ):
        init(
            self,
            cpu_group,
            device,
            device_group,
            unique_name,
            global_ranks,
            global_world_size,
            tcp_store_group,
            use_all2all,
        )
        from vllm.config import get_current_vllm_config_or_none

        vllm_config = get_current_vllm_config_or_none()
        if vllm_config is None or not get_hcu_config(vllm_config).deepep_auto:
            return
        # CudaCommunicator is constructed for DP/TP/PP as well as EP groups.
        # ``deepep_auto`` owns only the official EP all-to-all communicator;
        # the other groups must retain their normal collective managers.
        if not getattr(self, "use_all2all", False):
            return
        from vllm.distributed.device_communicators.all2all import (
            DeepEPAutoAll2AllManager,
        )

        self.all2all_manager = DeepEPAutoAll2AllManager(
            self.cpu_group, tcp_store_group
        )

    setattr(hcu_init, _WRAPPER, True)
    setattr(communicator, "_vllm_hcu_original_init", init)
    setattr(communicator, "__init__", hcu_init)
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
