# SPDX-License-Identifier: Apache-2.0
"""Custom-SP extensions for DeviceCommunicatorBase."""

from __future__ import annotations

import functools
from types import ModuleType

from vllm_hcu.patch.config import get_hcu_config

from ._common import (
    PatchCompatibilityError,
    already_applied,
    load_exact_module,
    require_callable,
    require_class,
    require_exact_signature,
)

TARGET_MODULE = "vllm.distributed.device_communicators.base_device_communicator"
PATCH_ID = "worker.framework_opt.communicator.base_custom_sp"
TARGETS = (
    f"{TARGET_MODULE}.DeviceCommunicatorBase.all_to_all_single",
    f"{TARGET_MODULE}.DeviceCommunicatorBase.__init__",
)
_MARKER = "_vllm_hcu_base_communicator_applied"
_WRAPPER = "_vllm_hcu_base_communicator_wrapper"


def apply_to_module(module: ModuleType) -> bool:
    base_module = load_exact_module(TARGET_MODULE, module)
    communicator = require_class(
        base_module, "DeviceCommunicatorBase", f"{TARGET_MODULE}.DeviceCommunicatorBase"
    )
    wrapped = ((communicator, "__init__", TARGETS[1], _WRAPPER),)
    if getattr(base_module, _MARKER, False):
        if not already_applied(base_module, _MARKER, wrapped):
            return False
        all_to_all = require_callable(communicator, "all_to_all_single", TARGETS[0])
        if not getattr(all_to_all, _WRAPPER, False):
            raise PatchCompatibilityError(
                f"required HCU patch marker for {TARGETS[0]} is stale"
            )
        return False

    if "all_to_all_single" in vars(communicator):
        raise PatchCompatibilityError(
            f"audited v0.21 target {TARGETS[0]} unexpectedly already exists"
        )
    original_init = require_callable(communicator, "__init__", TARGETS[1])
    require_exact_signature(
        original_init,
        TARGETS[1],
        positional=(
            "self",
            "cpu_group",
            "device",
            "device_group",
            "unique_name",
            "global_ranks",
            "global_world_size",
        ),
        defaults={
            "device": None,
            "device_group": None,
            "unique_name": "",
            "global_ranks": None,
            "global_world_size": None,
        },
    )
    # This method is the stable insertion anchor for the runtime adapter.
    reduce_scatter = require_callable(communicator, "reduce_scatter", TARGETS[0])
    require_exact_signature(
        reduce_scatter,
        f"{TARGET_MODULE}.DeviceCommunicatorBase.reduce_scatter",
        positional=("self", "input_", "dim"),
        defaults={"dim": -1},
    )

    from vllm_hcu.distributed.device_communicators import framework_runtime

    @functools.wraps(original_init)
    def hcu_init(
        self,
        cpu_group,
        device=None,
        device_group=None,
        unique_name="",
        global_ranks=None,
        global_world_size=None,
    ):
        original_init(
            self,
            cpu_group,
            device,
            device_group,
            unique_name,
            global_ranks,
            global_world_size,
        )
        from vllm.config import get_current_vllm_config_or_none

        vllm_config = get_current_vllm_config_or_none()
        custom_sp = get_hcu_config(vllm_config).enable_custom_sp
        if custom_sp and self.is_ep_communicator:
            self.use_all2all = True

    def hcu_all_to_all_single(self, output, input):
        return framework_runtime.all_to_all_single(self, output, input)

    setattr(hcu_init, _WRAPPER, True)
    setattr(hcu_all_to_all_single, _WRAPPER, True)
    setattr(communicator, "_vllm_hcu_original_init", original_init)
    setattr(communicator, "__init__", hcu_init)
    setattr(communicator, "all_to_all_single", hcu_all_to_all_single)
    setattr(base_module, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
