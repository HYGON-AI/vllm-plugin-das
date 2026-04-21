# SPDX-License-Identifier: Apache-2.0

from .patch_utils import (
    import_hook,
    _register_patches,
    patch_module_class_function,
)

_HCU_SOURCE_PATCHES_READY = False
_HCU_SOURCE_PATCHES_IN_PROGRESS = False
_HCU_OP_PATCHES_READY = False
_HCU_OP_PATCHES_IN_PROGRESS = False


def _run_once(
    ready_flag: str,
    in_progress_flag: str,
    init_fn,
) -> None:
    """Run an initializer once and guard against re-entrant calls."""
    global _HCU_SOURCE_PATCHES_READY, _HCU_SOURCE_PATCHES_IN_PROGRESS
    global _HCU_OP_PATCHES_READY, _HCU_OP_PATCHES_IN_PROGRESS

    if globals()[ready_flag] or globals()[in_progress_flag]:
        return

    globals()[in_progress_flag] = True
    try:
        init_fn()
        globals()[ready_flag] = True
    finally:
        globals()[in_progress_flag] = False


def _init_hcu_source_patches() -> None:
    """Apply minimal source/import patches during plugin initialization."""
    def _init() -> None:
        _register_patches()
        import_hook()

    _run_once(
        "_HCU_SOURCE_PATCHES_READY",
        "_HCU_SOURCE_PATCHES_IN_PROGRESS",
        _init,
    )


def _init_hcu_op_patches() -> None:
    """Apply heavier runtime/ops patches after plugin initialization."""
    def _init() -> None:
        _init_hcu_source_patches()
        patch_module_class_function()
    _run_once(
        "_HCU_OP_PATCHES_READY",
        "_HCU_OP_PATCHES_IN_PROGRESS",
        _init,
    )

def hcu_platform_plugin():
    """Register the HCU platform."""
    _init_hcu_source_patches()
    return "vllm_hcu.platforms.hcu.HCUPlatform"

# def hcu_platform_register_model():
#     """Register models for training and inference"""
#     from .model_executor.models import register_model as _reg
#     _reg()
    
def hcu_platform_register_ops():
    _init_hcu_op_patches()
    import vllm_hcu.ops
