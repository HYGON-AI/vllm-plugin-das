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
from vllm_hcu.runtime_compat.step5 import is_step5_config

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

    # UP-02 opt-in: the fused subclass adds can_quant_gather/gather_quant_prod on top of the
    # production communicator; the env gate matches the deferred-quant patch.
    if os.environ.get("VLLM_HCU_UP02_FUSED_DISPATCH", "0") == "1":
        from vllm_hcu.distributed.device_communicators.ipc_fused_dispatch import (
            IpcQuantAllGather,
        )

        communicator._hcu_ipc_ag = IpcQuantAllGather(
            communicator.cpu_group, communicator.device
        )
    else:
        communicator._hcu_ipc_ag = IpcAllGather(
            communicator.cpu_group, communicator.device
        )
    from vllm.logger import init_logger

    init_logger(__name__).info_once(
        "DP group '%s' uses the IPC all-gather for MoE dispatch", unique_name
    )


_UP02_MARKER = "_hcu_up02_quant_pending"


def _up02_marked(tensor) -> bool:
    """True when `tensor` carries a deferred per-token FP8 quantization."""
    return torch.is_tensor(tensor) and bool(getattr(tensor, _UP02_MARKER, False))


def _up02_clear(tensor) -> None:
    """Retire the marker once the deferred quantization has been honoured."""
    if torch.is_tensor(tensor):
        tensor._hcu_up02_quant_pending = False


def _up02_guard_unconsumable(input_, ag, dim) -> None:
    """Fail fast when a deferred-quant activation reaches a gather that cannot honour it.

    The deferral contract is narrow: the marked activation must arrive at the DP IPC
    all-gather as the first of the four production dispatch tensors.  Any other shape of
    call (single tensor, a different tensor count, dim != 0, no IPC all-gather) would gather
    an unquantized activation alongside a scale placeholder that nothing has written, and the
    caller would consume uninitialized memory as a per-token scale.  That must not happen
    quietly, so raise instead of proceeding.
    """
    first = input_ if torch.is_tensor(input_) else (
        input_[0] if isinstance(input_, (list, tuple)) and input_ else None
    )
    if not _up02_marked(first):
        return
    if (
        ag is not None
        and dim == 0
        and isinstance(input_, (list, tuple))
        and len(input_) == 4
    ):
        return
    raise RuntimeError(
        "UP-02: a deferred per-token FP8 quantization reached an all-gather that cannot "
        "consume it (ag=%s, dim=%s, payload=%s). The activation is still unquantized and "
        "its scale is an unwritten placeholder; refusing to gather uninitialized scales."
        % (
            "none" if ag is None else type(ag).__name__,
            dim,
            "tensor" if torch.is_tensor(input_) else
            "%d tensors" % len(input_) if isinstance(input_, (list, tuple)) else
            type(input_).__name__,
        )
    )


def _up02_prepare_dispatch(ag, tensors, sizes):
    """UP-02 opt-in: marked deferred-quant dispatch -> (gathered outs | None, tensors).

    A marked first tensor means prepare() skipped the local quant and passed the bf16
    activation (plus a scale placeholder) so the gather can quantize in the IPC prologue.
    Returns the gathered [q, w, ids, s] when the fused path runs, else None with the tensor
    list rewritten to the quantized [q, w, ids, s] so the standard gather path stays correct.

    Once the marker is seen, every exit either runs the fused gather or rewrites the pair with
    a real quantization, and the marker is retired.  Returning a marked tensor unchanged would
    leave the standard gather with an unquantized activation and an uninitialized scale.
    """
    if len(tensors) != 4 or not sizes:
        return None, tensors
    t0, t1, t2, t3 = tensors
    if not _up02_marked(t0):
        return None, tensors
    fused_ready = (
        torch.is_tensor(t1) and torch.is_tensor(t2) and torch.is_tensor(t3)
        and t0.dtype is torch.bfloat16 and t1.dtype is torch.float32
        and t2.dtype is torch.int32 and t3.dtype is torch.float32
        and t1.shape == t2.shape and t1.shape[0] == t0.shape[0]
        and t3.shape == (t0.shape[0], 1)
        and t0.is_contiguous() and t1.is_contiguous() and t2.is_contiguous()
    )
    if fused_ready:
        can = getattr(ag, "can_quant_gather", None)
        if can is not None:
            try:
                if can(t0, t1, t2, sizes):
                    outs = ag.gather_quant_prod(t0, t1, t2, t3, sizes)
                    _up02_clear(t0)
                    return outs, tensors
            except Exception:
                pass
    if not (torch.is_tensor(t1) and torch.is_tensor(t2)):
        raise RuntimeError(
            "UP-02: deferred-quant activation paired with a non-tensor dispatch payload; "
            "refusing to gather an unwritten scale placeholder"
        )
    from vllm_hcu.ops import token_fp8_quant as tq

    q, s = tq.token_fp8_quant(t0)
    raw = [q, t1, t2, s]
    _up02_clear(t0)
    if ag.can_gather(raw, sizes):
        return ag.gather(raw, sizes), raw
    return None, raw


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
        from vllm.config import get_current_vllm_config_or_none

        self._hcu_dp_packed_ag = False
        try:
            # Preserve the upstream opt-in collective and cleanup lifecycle.
            # Only the new packed/Fabric providers are Step5-specific.
            _maybe_init_ep_custom_allreduce(self, unique_name)
            _maybe_init_dp_ipc_all_gather(self, unique_name)
            vllm_config = get_current_vllm_config_or_none()
            step5_owner = vllm_config is not None and is_step5_config(vllm_config)
            self._hcu_dp_packed_ag = (
                step5_owner
                and henvs.optional_custom_op_enabled(
                    os.environ.get("VLLM_HCU_DP_PACKED_AG", "0") == "1"
                )
                and unique_name.split(":")[0] == "dp"
                and 2 <= self.world_size <= 16
            )
            from vllm_hcu.distributed.device_communicators.step5_dp_fabric import maybe_install
            maybe_install(self, cpu_group, device, unique_name,
                          vllm_config=vllm_config)
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
        _up02_guard_unconsumable(input_, ag, dim)
        if getattr(self, "_hcu_dp_packed_ag", False) and ag is None:
            from vllm_hcu.distributed.device_communicators.packed_all_gather import (
                packed_step5_all_gatherv,
            )

            packed = packed_step5_all_gatherv(
                self, input_, all_gatherv, dim=dim, sizes=sizes
            )
            if packed is not None:
                return packed
        if ag is not None and dim == 0:
            single = isinstance(input_, torch.Tensor)
            tensors = [input_] if single else list(input_)
            if tensors:
                if sizes is None:
                    if tensors[0].dim() == 0:
                        raise ValueError("all-gather inputs must have a row dimension")
                    sizes = [tensors[0].shape[0]] * self.world_size
                if not single and len(tensors) == 4:
                    outs, tensors = _up02_prepare_dispatch(ag, tensors, sizes)
                    if outs is not None:
                        return outs
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
