# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Opt-in exact-size registered host allocations for HCU expert offloading."""

from __future__ import annotations

import functools
import os
from types import ModuleType

from ._common import (
    load_exact_module,
    require_callable,
    require_class,
    require_exact_signature,
)
from .patch_hcu_uva_buffer import _allocate_registered_buffer

TARGET_MODULE = "vllm.model_executor.offloader.uva"
PATCH_ID = "worker.core_fix.hcu_uva_offloader"
TARGETS = (f"{TARGET_MODULE}.UVAOffloader._maybe_offload_to_cpu",)
_MARKER = "_vllm_hcu_registered_uva_offloader"


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    offloader = require_class(target, "UVAOffloader", f"{TARGET_MODULE}.UVAOffloader")
    original = require_callable(offloader, "_maybe_offload_to_cpu", TARGETS[0])
    require_exact_signature(
        original, TARGETS[0], positional=("self", "module", "prefix"),
        defaults={"prefix": ""},
    )
    if getattr(original, _MARKER, False):
        return False

    @functools.wraps(original)
    def hcu_offload(self, module, prefix=""):
        if (
            os.getenv("VLLM_HCU_REGISTERED_CPU_OFFLOAD", "0") != "1"
            or not self.uva_offloading
            or not self.pin_memory
        ):
            return original(self, module, prefix)
        first = next(module.parameters(), None)
        if (
            first is None
            or first.device.type == "cpu"
            or self.cpu_offload_bytes >= self.cpu_offload_max_bytes
        ):
            return module

        for name, parameter in module.named_parameters():
            if self.cpu_offload_bytes >= self.cpu_offload_max_bytes:
                break
            if parameter.device.type == "cpu" or getattr(
                parameter, "_vllm_is_uva_offloaded", False
            ):
                continue
            if self.cpu_offload_params and not any(
                f".{part}." in f".{prefix}{name}."
                for part in self.cpu_offload_params
            ):
                continue
            if parameter.numel() == 0:
                continue

            cpu, mapping = _allocate_registered_buffer(
                tuple(parameter.shape), parameter.dtype, module
            )
            owners = getattr(module, "_hcu_registered_offload_storage", None)
            if owners is None:
                owners = []
                module._hcu_registered_offload_storage = owners
            owners.append((cpu, mapping))
            cpu.copy_(parameter.detach())
            parameter.data = target.get_accelerator_view_from_cpu_tensor(cpu)
            parameter._vllm_is_uva_offloaded = True
            nbytes = cpu.numel() * cpu.element_size()
            self.cpu_offload_bytes += nbytes
            target.logger.info_once(
                "HCU expert CPU offload uses exact-size registered anonymous "
                "host memory (VLLM_HCU_REGISTERED_CPU_OFFLOAD=1)."
            )
        return module

    setattr(hcu_offload, _MARKER, True)
    setattr(offloader, "_vllm_hcu_original_maybe_offload_to_cpu", original)
    setattr(offloader, "_maybe_offload_to_cpu", hcu_offload)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
