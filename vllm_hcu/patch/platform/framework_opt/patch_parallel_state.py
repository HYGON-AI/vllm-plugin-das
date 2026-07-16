# SPDX-License-Identifier: Apache-2.0
"""Add HCU's communicator-backed all-to-all method to GroupCoordinator."""

from __future__ import annotations

from types import ModuleType

from ._common import (
    PatchCompatibilityError,
    load_exact_module,
    require_callable,
    require_class,
    require_signature_prefix,
)

TARGET_MODULE = "vllm.distributed.parallel_state"
PATCH_ID = "platform.framework_opt.group_coordinator_all_to_all"
TARGETS = (f"{TARGET_MODULE}.GroupCoordinator.all_to_all_single",)
_MARKER = "_vllm_hcu_group_all_to_all_applied"
_METHOD_MARKER = "_vllm_hcu_group_all_to_all_method"


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    group = require_class(target, "GroupCoordinator", f"{TARGET_MODULE}.GroupCoordinator")
    require_signature_prefix(
        require_callable(
            group,
            "reduce_scatter",
            f"{TARGET_MODULE}.GroupCoordinator.reduce_scatter",
        ),
        f"{TARGET_MODULE}.GroupCoordinator.reduce_scatter",
        ("self", "input_", "dim"),
    )
    if getattr(group, _MARKER, False):
        method = getattr(group, "all_to_all_single", None)
        if not callable(method) or not getattr(method, _METHOD_MARKER, False):
            raise PatchCompatibilityError(
                "HCU GroupCoordinator all-to-all marker is stale; restart the process"
            )
        return False
    if "all_to_all_single" in vars(group):
        raise PatchCompatibilityError(f"required HCU-owned target {TARGETS[0]} already exists")

    def all_to_all_single(self, output, input):
        if self.device_communicator is None:
            raise ValueError("No device communicator found")
        method = getattr(self.device_communicator, "all_to_all_single", None)
        if not callable(method):
            raise RuntimeError(
                "selected HCU device communicator does not implement all_to_all_single"
            )
        return method(output, input)

    setattr(all_to_all_single, _METHOD_MARKER, True)
    setattr(group, "all_to_all_single", all_to_all_single)
    setattr(group, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
