# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Platform-phase patch dispatcher for the HCU plugin."""

from __future__ import annotations

import threading

from vllm_hcu.compatibility import ensure_vllm_compatible
from vllm_hcu.patch.import_coordinator import IMPORT_COORDINATOR
from vllm_hcu.patch.module_exchange import register_all_module_exchanges
from vllm_hcu.patch.runtime_callbacks import register_runtime_method_callbacks
from vllm_hcu.patch.runtime_state import detect_process_role, set_process_role
from vllm_hcu.patch.tokenizer_callbacks import register_tokenizer_callbacks

from .core_fix import register_platform_core_callbacks
from .framework_opt import (
    patch_engine_core,
    patch_distributed_utils,
    patch_kv_cache_coordinator,
    patch_kv_connector_factory,
    patch_mooncake_connector,
    patch_multiproc_executor,
    patch_output_processor,
    patch_outputs,
    patch_parallel_state,
    patch_scheduler,
)


_DISPATCH_LOCK = threading.RLock()


# This inventory is deliberately explicit.  Registration order matters when
# platform targets were imported before plugin discovery: the exact import
# coordinator applies those callbacks immediately in the order below.  Keep
# the HCU-owned Mooncake contract ahead of its factory registration, then arm
# the vLLM integration points from low-level KV/distributed contracts through
# scheduling and engine output.
_ORDERED_FRAMEWORK_ADAPTERS = (
    patch_distributed_utils,
    patch_mooncake_connector,
    patch_kv_connector_factory,
    patch_parallel_state,
    patch_kv_cache_coordinator,
    patch_scheduler,
    patch_engine_core,
    patch_output_processor,
    patch_multiproc_executor,
    patch_outputs,
)


def _register_platform_framework_callbacks() -> None:
    """Arm platform/framework adapters without importing their targets."""

    for adapter in _ORDERED_FRAMEWORK_ADAPTERS:
        IMPORT_COORDINATOR.register_callback(
            adapter.PATCH_ID,
            adapter.TARGET_MODULE,
            adapter.apply_to_module,
            targets=adapter.TARGETS,
        )


def platform_framework_callback_names() -> tuple[tuple[str, str], ...]:
    """Return the frozen platform/framework callback inventory for tests."""

    return tuple(
        (adapter.PATCH_ID, adapter.TARGET_MODULE)
        for adapter in _ORDERED_FRAMEWORK_ADAPTERS
    )


def apply_platform_patches() -> None:
    """Install and arm the complete platform-phase runtime patch set.

    The operation is process-local, thread-safe, and idempotent.  Whole-module
    replacements are armed before any post-import callbacks, with
    ``modular_kernel`` first inside the explicit replacement inventory.
    Required compatibility failures propagate and are latched by the
    coordinator; startup must not silently fall back to source mutation or an
    unpatched official implementation.
    """

    ensure_vllm_compatible()
    with _DISPATCH_LOCK:
        set_process_role(detect_process_role())
        # Install the finder and publish all custom-op/module replacements
        # under one coordinator fence.  A third-party import can only run
        # before this batch begins or after the complete inventory is visible;
        # it cannot observe the sequential-registration midpoint.
        with IMPORT_COORDINATOR.registration_batch():
            IMPORT_COORDINATOR.install()
            from vllm_hcu.patch.worker import (
                _register_worker_cold_replacements,
            )

            _register_worker_cold_replacements(IMPORT_COORDINATOR)
            register_all_module_exchanges(IMPORT_COORDINATOR)
        register_platform_core_callbacks(IMPORT_COORDINATOR)
        register_tokenizer_callbacks(IMPORT_COORDINATOR)
        register_runtime_method_callbacks(IMPORT_COORDINATOR)
        _register_platform_framework_callbacks()
        # Some targets can recursively trigger platform discovery while their
        # own loader is still running.  The coordinator leaves those callbacks
        # armed; every defensive plugin re-entry drains modules whose imports
        # have completed and re-raises any latched compatibility failure.
        IMPORT_COORDINATOR.drain_ready_callbacks()


__all__ = ["apply_platform_patches", "platform_framework_callback_names"]
