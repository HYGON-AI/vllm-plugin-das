# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Gate static EPLB at vLLM's common model-loader entry point."""

from __future__ import annotations

from types import ModuleType

from .patch_model_loader_static_eplb import (
    ENTRYPOINT_TARGET_MODULE as TARGET_MODULE,
    ENTRYPOINT_TARGETS as TARGETS,
    apply_entrypoint_to_module,
)


PATCH_ID = "worker.framework_opt.model_loader.static_eplb_capability"


def apply_to_module(module: ModuleType) -> bool:
    return apply_entrypoint_to_module(module)


def apply(module: ModuleType | None = None) -> bool:
    from ._common import load_exact_module

    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
