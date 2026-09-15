# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Avoid NCCL profile-buffer reservation for CPU-staged EPLB on HCU.

The audited vLLM v0.25.1 implementation makes every EPLB communicator inherit
``needs_profile_buffer_reservation=True``.  That is appropriate for the NCCL
communicators, but the Gloo communicator moves expert weights through CPU
staging and never performs a device collective.  Reserving a full-layer device
``all_gather`` buffer for that backend can therefore OOM large HCU models before
the first request without reserving anything the real transfer path uses.
"""

from __future__ import annotations

from types import ModuleType

from ._common import PatchCompatibilityError, load_exact_module, require_class

TARGET_MODULE = "vllm.distributed.eplb.eplb_communicator"
PATCH_ID = "worker.framework_opt.eplb.gloo_profile_reservation"
TARGETS = (
    f"{TARGET_MODULE}.EplbCommunicator.needs_profile_buffer_reservation",
    (
        f"{TARGET_MODULE}.TorchDistGlooStagedEplbCommunicator."
        "needs_profile_buffer_reservation"
    ),
)
_MARKER = "_vllm_hcu_gloo_eplb_profile_reservation_applied"
_PROPERTY_MARKER = "_vllm_hcu_gloo_eplb_profile_reservation_property"


def _patched_property_is_valid(gloo_communicator: type) -> bool:
    profile_policy = gloo_communicator.__dict__.get(
        "needs_profile_buffer_reservation"
    )
    return (
        isinstance(profile_policy, property)
        and callable(profile_policy.fget)
        and getattr(profile_policy.fget, _PROPERTY_MARKER, False)
    )


def apply_to_module(module: ModuleType) -> bool:
    communicator = load_exact_module(TARGET_MODULE, module)
    base = require_class(
        communicator,
        "EplbCommunicator",
        f"{TARGET_MODULE}.EplbCommunicator",
    )
    gloo = require_class(
        communicator,
        "TorchDistGlooStagedEplbCommunicator",
        f"{TARGET_MODULE}.TorchDistGlooStagedEplbCommunicator",
    )
    require_class(
        communicator,
        "TorchDistNcclEplbCommunicator",
        f"{TARGET_MODULE}.TorchDistNcclEplbCommunicator",
    )
    require_class(
        communicator,
        "PyNcclEplbCommunicator",
        f"{TARGET_MODULE}.PyNcclEplbCommunicator",
    )

    if getattr(communicator, _MARKER, False):
        if not _patched_property_is_valid(gloo):
            raise PatchCompatibilityError(
                "required HCU Gloo EPLB profile-policy marker is stale; "
                "restart the process"
            )
        return False

    base_policy = base.__dict__.get("needs_profile_buffer_reservation")
    if not isinstance(base_policy, property) or not callable(base_policy.fget):
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGETS[0]} is incompatible"
        )
    if not issubclass(gloo, base):
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGETS[1]} is not an EPLB communicator"
        )
    if "needs_profile_buffer_reservation" in gloo.__dict__:
        raise PatchCompatibilityError(
            f"audited target {gloo.__name__} already defines "
            "needs_profile_buffer_reservation"
        )

    def hcu_needs_profile_buffer_reservation(self) -> bool:
        del self
        return False

    setattr(hcu_needs_profile_buffer_reservation, _PROPERTY_MARKER, True)
    setattr(
        gloo,
        "needs_profile_buffer_reservation",
        property(hcu_needs_profile_buffer_reservation),
    )
    setattr(communicator, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = [
    "PATCH_ID",
    "TARGET_MODULE",
    "TARGETS",
    "PatchCompatibilityError",
    "apply",
    "apply_to_module",
]
