# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Runtime patch foundations for the vLLM-HCU plugin.

Feature dispatchers live in the ``platform`` and ``worker`` subpackages.  The
objects exported here are dependency-light and safe during plugin discovery.
"""

from .config import (
    HcuFeatureConfig,
    get_hcu_config,
    pop_hcu_feature_kwargs,
    set_hcu_config,
    write_hcu_config,
)
from .import_coordinator import (
    IMPORT_COORDINATOR,
    ExactImportCoordinator,
    ImportAction,
    ImportRegistration,
    LateModuleReplacementError,
    ModuleReloadBlockedError,
    install_import_coordinator,
    register_module_replacement,
    register_post_import,
    uninstall_import_coordinator,
)
from .runtime_state import (
    PATCH_REGISTRY,
    LatchedPatchError,
    PatchRecord,
    PatchRegistry,
    PatchStatus,
    ProcessRole,
    applying_patch,
    detect_process_role,
    patch_report,
    run_patch,
    set_process_role,
)
from vllm_hcu.compatibility import ensure_vllm_compatible


def apply_platform_patches() -> None:
    """Apply the process-local platform phase without importing vLLM eagerly."""

    ensure_vllm_compatible()
    from .platform import apply_platform_patches as _apply

    _apply()


def apply_worker_patches(vllm_config=None) -> None:
    """Apply the process-local Worker phase against an optional vLLM config."""

    ensure_vllm_compatible()
    from .worker import apply_worker_patches as _apply

    _apply(vllm_config)


__all__ = [
    "IMPORT_COORDINATOR",
    "PATCH_REGISTRY",
    "ExactImportCoordinator",
    "HcuFeatureConfig",
    "ImportAction",
    "ImportRegistration",
    "LatchedPatchError",
    "LateModuleReplacementError",
    "ModuleReloadBlockedError",
    "PatchRecord",
    "PatchRegistry",
    "PatchStatus",
    "ProcessRole",
    "applying_patch",
    "apply_platform_patches",
    "apply_worker_patches",
    "detect_process_role",
    "get_hcu_config",
    "install_import_coordinator",
    "patch_report",
    "pop_hcu_feature_kwargs",
    "register_module_replacement",
    "register_post_import",
    "run_patch",
    "set_hcu_config",
    "set_process_role",
    "uninstall_import_coordinator",
    "write_hcu_config",
]
