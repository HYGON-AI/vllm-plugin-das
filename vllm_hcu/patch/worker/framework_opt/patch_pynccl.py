# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Optional PyNccl communicator all-to-all method for capable RCCL builds."""

from __future__ import annotations

from types import ModuleType

from ._common import (
    PatchCompatibilityError,
    load_exact_module,
    require_callable,
    require_class,
    require_exact_signature,
)

TARGET_MODULE = "vllm.distributed.device_communicators.pynccl"
PATCH_ID = "worker.framework_opt.communicator.pynccl_all_to_all"
TARGETS = (f"{TARGET_MODULE}.PyNcclCommunicator.all_to_all_single",)
_MARKER = "_vllm_hcu_pynccl_all_to_all_applied"
_PROBE_MARKER = "_vllm_hcu_pynccl_all_to_all_probe"
_WRAPPER = "_vllm_hcu_pynccl_all_to_all_wrapper"


def _unavailable(module: ModuleType, reason: str, required: bool) -> bool:
    setattr(module, _PROBE_MARKER, reason)
    if required:
        raise RuntimeError(
            "PyNccl all-to-all was explicitly requested but is unavailable: " + reason
        )
    return False


def apply_to_module(module: ModuleType, *, required: bool = False) -> bool:
    pynccl = load_exact_module(TARGET_MODULE, module)
    communicator = require_class(
        pynccl, "PyNcclCommunicator", f"{TARGET_MODULE}.PyNcclCommunicator"
    )
    if getattr(pynccl, _MARKER, False):
        method = require_callable(communicator, "all_to_all_single", TARGETS[0])
        if not getattr(method, _WRAPPER, False):
            raise PatchCompatibilityError(
                f"required HCU patch marker for {TARGETS[0]} is stale"
            )
        return False
    previous_probe = getattr(pynccl, _PROBE_MARKER, None)
    if previous_probe is not None:
        return _unavailable(pynccl, str(previous_probe), required)
    if "all_to_all_single" in vars(communicator):
        raise PatchCompatibilityError(
            f"audited target vLLM API {TARGETS[0]} unexpectedly exists"
        )
    reduce_scatter = require_callable(
        communicator, "reduce_scatter", f"{TARGET_MODULE}.PyNcclCommunicator.reduce_scatter"
    )
    require_exact_signature(
        reduce_scatter,
        f"{TARGET_MODULE}.PyNcclCommunicator.reduce_scatter",
        positional=("self", "output_tensor", "input_tensor", "op", "stream"),
        defaults={"op": pynccl.ReduceOp.SUM, "stream": None},
    )

    wrapper = __import__(
        "vllm.distributed.device_communicators.pynccl_wrapper",
        fromlist=["NCCLLibrary"],
    )
    library_class = require_class(
        wrapper,
        "NCCLLibrary",
        "vllm.distributed.device_communicators.pynccl_wrapper.NCCLLibrary",
    )
    library_method = getattr(library_class, "ncclAllToAll", None)
    if not callable(library_method) or not getattr(
        wrapper, "_vllm_hcu_pynccl_all_to_all_applied", False
    ):
        return _unavailable(
            pynccl,
            "RCCL ncclAllToAll binding was not registered before PyNccl import",
            required,
        )

    def all_to_all_single(self, output_tensor, input_tensor, stream=None):
        if self.disabled:
            return None
        if input_tensor.device != self.device:
            raise AssertionError(
                f"this nccl communicator is created to work on {self.device}, "
                f"but the input tensor is on {input_tensor.device}"
            )
        if output_tensor.device != self.device:
            raise AssertionError(
                f"this nccl communicator is created to work on {self.device}, "
                f"but the output tensor is on {output_tensor.device}"
            )
        if input_tensor.numel() % self.world_size:
            raise ValueError(
                "PyNccl all-to-all input elements must be divisible by world_size"
            )
        if output_tensor.numel() != input_tensor.numel():
            raise ValueError("PyNccl all-to-all input and output sizes must match")
        if output_tensor.dtype != input_tensor.dtype:
            raise TypeError("PyNccl all-to-all input and output dtypes must match")
        if stream is None:
            stream = pynccl.current_stream()
        self.nccl.ncclAllToAll(
            pynccl.buffer_type(input_tensor.data_ptr()),
            pynccl.buffer_type(output_tensor.data_ptr()),
            input_tensor.numel() // self.world_size,
            pynccl.ncclDataTypeEnum.from_torch(input_tensor.dtype),
            self.comm,
            pynccl.cudaStream_t(stream.cuda_stream),
        )
        return output_tensor

    setattr(all_to_all_single, _WRAPPER, True)
    setattr(communicator, "all_to_all_single", all_to_all_single)
    setattr(pynccl, _PROBE_MARKER, "available")
    setattr(pynccl, _MARKER, True)
    return True


def apply(module: ModuleType | None = None, *, required: bool = False) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module), required=required)


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
