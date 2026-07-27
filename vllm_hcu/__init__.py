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
import threading

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
