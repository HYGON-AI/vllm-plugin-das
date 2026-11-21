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
import logging
import threading


def _attach_vllm_log_handler() -> None:
    """Make ``vllm_hcu`` log records reach vLLM's handler.

    vLLM's logging config declares only the ``vllm`` logger and installs no
    root handler, so records from ``vllm_hcu.*`` propagate to a handler-less
    root and Python's ``lastResort`` drops everything below WARNING: every
    ``logger.info`` in this package was being discarded in every process.
    Mirror vLLM's handler and level onto the package logger. A
    ``VLLM_LOGGING_CONFIG_PATH`` that already configures ``vllm_hcu``, or any
    other pre-installed handler, is left untouched.

    Deliberately does not import vLLM. Importing it from this module's body
    runs vLLM's plugin discovery while ``vllm_hcu`` is still half-initialized,
    so discovery cannot resolve ``vllm_hcu:hcu_platform_plugin`` and vLLM
    drops the HCU platform plugin with nothing but an error line -- the engine
    then runs on the wrong platform. ``logging.getLogger`` touches no module,
    so when vLLM has not configured its logger yet there is simply nothing to
    mirror; this is idempotent and the plugin entry points call it again once
    vLLM is up.
    """

    package_logger = logging.getLogger("vllm_hcu")
    if package_logger.handlers:
        return
    vllm_logger = logging.getLogger("vllm")
    if not vllm_logger.handlers:
        return
    for handler in vllm_logger.handlers:
        package_logger.addHandler(handler)
    # getEffectiveLevel, not .level: a custom VLLM_LOGGING_CONFIG_PATH may
    # attach a handler to "vllm" without setting a level, and copying that
    # NOTSET while propagate is off would resolve through the root logger to
    # WARNING and drop every info record again -- the exact bug this fixes.
    package_logger.setLevel(vllm_logger.getEffectiveLevel())
    package_logger.propagate = False


_attach_vllm_log_handler()

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

    _attach_vllm_log_handler()
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

    _attach_vllm_log_handler()
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
