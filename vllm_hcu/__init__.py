# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# Modified by Hygon Information Technology Co., Ltd., 2026.
"""vLLM plugin entry points for the HCU runtime integration.

Plugin discovery must remain dependency-light: the platform phase only arms
exact import callbacks, while the general plugins pre-arm Worker callbacks
before registering models or importing custom operators.  Process-local
registry records make every entry point idempotent and latch failures so a
second plugin callback cannot retry a partially registered custom op.
"""

from __future__ import annotations

import importlib
import os
import threading


def _is_uuid_device_mask(value: str | None) -> bool:
    return bool(value) and all(
        device_id.startswith("GPU-") for device_id in value.split(",")
    )


def _sync_hip_cuda_visible_devices() -> ValueError | None:
    """Synchronize masks that need no device discovery.

    Conflicts and UUID masks are deferred until the HCU platform callback is
    selected. Raising while the entry point is loaded would let vLLM silently
    omit the HCU plugin during platform discovery.
    """

    hip_is_set = "HIP_VISIBLE_DEVICES" in os.environ
    cuda_is_set = "CUDA_VISIBLE_DEVICES" in os.environ
    hip_visible_devices = os.environ.get("HIP_VISIBLE_DEVICES")
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")

    if (
        hip_is_set
        and cuda_is_set
        and hip_visible_devices != cuda_visible_devices
    ):
        return ValueError(
            "Inconsistent GPU visibility env vars: "
            f"HIP_VISIBLE_DEVICES='{hip_visible_devices}' vs "
            f"CUDA_VISIBLE_DEVICES='{cuda_visible_devices}'. "
            "Please set only one, or ensure they match."
        )
    visible_devices = hip_visible_devices if hip_is_set else cuda_visible_devices
    if _is_uuid_device_mask(visible_devices):
        return None
    if hip_is_set:
        assert hip_visible_devices is not None
        os.environ["CUDA_VISIBLE_DEVICES"] = hip_visible_devices
    elif cuda_is_set:
        assert cuda_visible_devices is not None
        os.environ["HIP_VISIBLE_DEVICES"] = cuda_visible_devices
    return None


def _resolve_uuid_visible_devices() -> None:
    """Expose an integer mask to CUDA-only HCU control-plane consumers."""

    hip_visible_devices = os.environ.get("HIP_VISIBLE_DEVICES")
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    uuid_mask = (
        hip_visible_devices
        if _is_uuid_device_mask(hip_visible_devices)
        else cuda_visible_devices
        if _is_uuid_device_mask(cuda_visible_devices)
        else None
    )
    if uuid_mask is None:
        return

    management_api = importlib.import_module("amdsmi")
    initialized = False
    try:
        management_api.amdsmi_init()
        initialized = True
        handles = management_api.amdsmi_get_processor_handles()
        physical_uuids: list[str] = []
        for handle in handles:
            serial = str(
                management_api.amdsmi_get_gpu_asic_info(handle)["asic_serial"]
            ).lower()
            physical_uuids.append(serial.removeprefix("0x"))

        physical_device_ids: list[int] = []
        for entry in uuid_mask.split(","):
            candidate = entry.removeprefix("GPU-").lower()
            matches = [
                index
                for index, physical_uuid in enumerate(physical_uuids)
                if physical_uuid.startswith(candidate)
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"HIP device UUID {entry!r} matched {len(matches)} "
                    "physical devices; expected exactly one."
                )
            physical_device_id = matches[0]
            if physical_device_id in physical_device_ids:
                raise ValueError(
                    f"HIP device UUID mask {uuid_mask!r} selects physical "
                    f"device {physical_device_id} more than once."
                )
            physical_device_ids.append(physical_device_id)
    finally:
        if initialized:
            management_api.amdsmi_shut_down()

    resolved_mask = ",".join(map(str, physical_device_ids))
    if hip_visible_devices is None:
        os.environ["HIP_VISIBLE_DEVICES"] = uuid_mask
    if cuda_visible_devices not in (None, uuid_mask, resolved_mask):
        raise ValueError(
            "Inconsistent GPU visibility env vars after resolving HIP UUIDs: "
            f"HIP_VISIBLE_DEVICES='{os.environ.get('HIP_VISIBLE_DEVICES')}' vs "
            f"CUDA_VISIBLE_DEVICES='{cuda_visible_devices}'."
        )
    os.environ["CUDA_VISIBLE_DEVICES"] = resolved_mask


_DEVICE_VISIBILITY_CONFLICT = _sync_hip_cuda_visible_devices()

from vllm_hcu.lightop_env import configure_lightop_environment

configure_lightop_environment()

from vllm_hcu.compatibility import (
    VllmCompatibilityError,
    ensure_vllm_compatible,
)


_MODEL_REGISTRY_PATCH_ID = "plugin.general.model_registry"
_MODEL_REGISTRY_TARGETS = (
    "vllm.model_executor.models.registry.ModelRegistry",
    "vllm_hcu.models.register_model",
)
_OPS_REGISTRY_PATCH_ID = "plugin.general.ops_registry"
_OPS_REGISTRY_TARGETS = ("vllm_hcu.ops",)
_PLATFORM_CLASS_PATH = "vllm_hcu.platforms.hcu.HCUPlatform"
_PLATFORM_PLUGIN_LOCK = threading.RLock()
_PLATFORM_INIT_FAILURE: Exception | None = None


def _apply_platform_preserving_role() -> None:
    """Apply the platform phase without losing an already-known process role.

    The platform dispatcher performs best-effort role detection.  An
    ``EngineCoreProc``/Worker boundary is more authoritative, so preserve that
    process-local value across defensive plugin re-entry.
    """

    from vllm_hcu.patch import PATCH_REGISTRY, apply_platform_patches
    from vllm_hcu.patch.runtime_state import set_process_role

    role = PATCH_REGISTRY.process_role()
    apply_platform_patches()
    set_process_role(role)


def _prepare_general_plugin() -> None:
    _apply_platform_preserving_role()
    from vllm_hcu.patch.worker import prepare_worker_patches

    prepare_worker_patches()


def _ensure_platform_plugin_ready() -> None:
    """Fail closed after a platform-initialization error was latched."""

    if _PLATFORM_INIT_FAILURE is None:
        return
    error = _PLATFORM_INIT_FAILURE
    if isinstance(error, VllmCompatibilityError):
        raise VllmCompatibilityError(
            "HCU platform compatibility initialization previously failed: "
            f"{error}"
        ) from error
    raise RuntimeError(
        "HCU platform patch initialization previously failed: "
        f"{type(error).__name__}: {error}"
    ) from error


def hcu_platform_plugin() -> str:
    """Arm platform patches and return the public vLLM platform class path."""

    global _PLATFORM_INIT_FAILURE
    with _PLATFORM_PLUGIN_LOCK:
        _ensure_platform_plugin_ready()
        try:
            if _DEVICE_VISIBILITY_CONFLICT is not None:
                raise _DEVICE_VISIBILITY_CONFLICT
            _resolve_uuid_visible_devices()
            ensure_vllm_compatible()
            _apply_platform_preserving_role()
        except Exception as exc:
            # The audited target vLLM invokes every platform plugin once inside a broad
            # exception-swallowing activation probe and invokes the selected
            # plugin a second time outside that probe.  Return the class path
            # once so HCU remains selected, then make the second invocation
            # (or direct class import) surface the latched compatibility error.
            _PLATFORM_INIT_FAILURE = exc
            return _PLATFORM_CLASS_PATH
        return _PLATFORM_CLASS_PATH


def hcu_platform_register_model() -> None:
    """Register HCU model classes once in the current process."""

    ensure_vllm_compatible()
    _prepare_general_plugin()
    from vllm_hcu.patch.runtime_state import run_patch

    def register_models() -> None:
        from vllm_hcu.models import register_model

        register_model()

    run_patch(
        _MODEL_REGISTRY_PATCH_ID,
        _MODEL_REGISTRY_TARGETS,
        register_models,
    )


def hcu_platform_register_ops() -> None:
    """Import and register HCU custom operators once in the current process."""

    ensure_vllm_compatible()
    _prepare_general_plugin()
    from vllm_hcu.patch.runtime_state import run_patch

    run_patch(
        _OPS_REGISTRY_PATCH_ID,
        _OPS_REGISTRY_TARGETS,
        lambda: importlib.import_module("vllm_hcu.ops"),
    )


__all__ = [
    "hcu_platform_plugin",
    "hcu_platform_register_model",
    "hcu_platform_register_ops",
]
