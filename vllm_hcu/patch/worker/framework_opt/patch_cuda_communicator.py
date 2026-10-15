# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Early module exchange for HCU CustomAllreduce used by CudaCommunicator."""

from __future__ import annotations

import functools
import os
from contextlib import suppress
from types import ModuleType

import torch

from vllm_hcu.patch.config import get_hcu_config
from vllm_hcu.platforms import envs as henvs

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
    f"{TARGET_MODULE}.CudaCommunicator.all_gatherv",
    f"{TARGET_MODULE}.CudaCommunicator.destroy",
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


def _maybe_init_ep_custom_allreduce(communicator, unique_name: str) -> None:
    """Give the EP group a custom all-reduce for VLLM_HCU_EP_FUSED_COMBINE.

    Upstream restricts custom all-reduce to the TP group; the fused MoE
    combine (moe_runner._ep_fused_all_reduce) all-reduces over EP instead.
    """
    if not henvs.optional_custom_op_enabled(
        os.environ.get("VLLM_HCU_EP_FUSED_COMBINE", "0") == "1"
    ):
        return
    if unique_name.split(":")[0] != "ep" or communicator.world_size <= 1:
        return
    if communicator.ca_comm is not None:
        return
    from vllm.distributed.parallel_state import _ENABLE_CUSTOM_ALL_REDUCE

    if not _ENABLE_CUSTOM_ALL_REDUCE:
        return
    from vllm.distributed.device_communicators.custom_all_reduce import (
        CustomAllreduce,
    )

    communicator.ca_comm = CustomAllreduce(
        group=communicator.cpu_group, device=communicator.device
    )
    from vllm.logger import init_logger

    init_logger(__name__).info_once(
        "EP group '%s' custom all-reduce for fused MoE combine: %s",
        unique_name,
        "disabled" if communicator.ca_comm.disabled else "enabled",
    )


def _maybe_init_dp_ipc_all_gather(communicator, unique_name: str) -> None:
    """IPC all-gather for the naive DP/EP MoE dispatch (VLLM_HCU_DP_CUSTOM_AG).

    NCCL pays ~55 us per dispatch all-gather at decode sizes; the IPC kernel
    takes ~15 us.  Used by ``hcu_all_gatherv`` when the gather fits.
    """
    communicator._hcu_ipc_ag = None
    if not henvs.optional_custom_op_enabled(
        os.environ.get("VLLM_HCU_DP_CUSTOM_AG", "0") == "1"
    ):
        return
    if unique_name.split(":")[0] != "dp" or not 2 <= communicator.world_size <= 8:
        return
    from vllm.distributed.parallel_state import in_the_same_node_as

    if not all(in_the_same_node_as(communicator.cpu_group, source_rank=0)):
        return
    from vllm_hcu.distributed.device_communicators.ipc_all_gather import (
        IpcAllGather,
    )

    communicator._hcu_ipc_ag = IpcAllGather(
        communicator.cpu_group, communicator.device
    )
    from vllm.logger import init_logger

    init_logger(__name__).info_once(
        "DP group '%s' uses the IPC all-gather for MoE dispatch", unique_name
    )


def _close_ipc_all_gather(communicator) -> None:
    ag = getattr(communicator, "_hcu_ipc_ag", None)
    communicator._hcu_ipc_ag = None
    if ag is not None:
        ag.close()


def apply_to_module(module: ModuleType) -> bool:
    cuda = load_exact_module(TARGET_MODULE, module)
    if getattr(cuda, _MARKER, False):
        communicator = getattr(cuda, "CudaCommunicator", None)
        for method_name in ("__init__", "all_gatherv", "destroy"):
            method = getattr(communicator, method_name, None)
            if not callable(method) or not getattr(method, _WRAPPER, False):
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
    all_gatherv = require_callable(communicator, "all_gatherv", TARGETS[2])
    destroy = require_callable(communicator, "destroy", TARGETS[3])
    require_exact_signature(
        all_gatherv, TARGETS[2], positional=("self", "input_", "dim", "sizes"),
        defaults={"dim": 0, "sizes": None},
    )
    require_exact_signature(destroy, TARGETS[3], positional=("self",), defaults={})
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
        try:
            _maybe_init_ep_custom_allreduce(self, unique_name)
            _maybe_init_dp_ipc_all_gather(self, unique_name)
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
        except BaseException:
            with suppress(Exception):
                _close_ipc_all_gather(self)
            raise

    @functools.wraps(destroy)
    def hcu_destroy(self):
        try:
            _close_ipc_all_gather(self)
        finally:
            result = destroy(self)
        return result

    @functools.wraps(all_gatherv)
    def hcu_all_gatherv(self, input_, dim=0, sizes=None):
        ag = getattr(self, "_hcu_ipc_ag", None)
        if ag is not None and dim == 0:
            single = isinstance(input_, torch.Tensor)
            tensors = [input_] if single else list(input_)
            if tensors:
                if sizes is None:
                    if tensors[0].dim() == 0:
                        raise ValueError("all-gather inputs must have a row dimension")
                    sizes = [tensors[0].shape[0]] * self.world_size
                use_ipc = ag.can_gather(tensors, sizes)
                # Normalize both paths. Local contiguity must not decide which
                # collective peers enter, and NCCL also expects packed storage.
                tensors = [tensor.contiguous() for tensor in tensors]
                if use_ipc:
                    outs = ag.gather(tensors, sizes)
                    return outs[0] if single else outs
                input_ = tensors[0] if single else tensors
        return all_gatherv(self, input_, dim, sizes)

    setattr(hcu_all_gatherv, _WRAPPER, True)
    setattr(hcu_destroy, _WRAPPER, True)
    setattr(communicator, "all_gatherv", hcu_all_gatherv)
    setattr(communicator, "destroy", hcu_destroy)
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
