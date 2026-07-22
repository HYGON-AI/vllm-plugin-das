# SPDX-License-Identifier: Apache-2.0
"""Explicit platform/core callback ordering for the audited target vLLM API.

Importing this package is side-effect free.  The dispatcher calls
``register_platform_core_callbacks`` after the exact import coordinator is
installed; callbacks then run immediately for already-loaded exact targets or
after the first successful load of an as-yet-unloaded target.
"""

from __future__ import annotations

from vllm_hcu.patch.import_coordinator import (
    IMPORT_COORDINATOR,
    ExactImportCoordinator,
    ImportRegistration,
)

from . import (
    patch_compilation_config,
    patch_engine_args,
    patch_envs,
    patch_hy_v3_reasoning_parser,
    patch_hy_v3_tool_parser,
    patch_import_utils,
    patch_slimquant_registry,
    patch_vllm_config,
)


# This tuple is deliberately hand-ordered.  Do not replace it with package or
# directory discovery: ordering is part of the plugin contract and must remain
# reviewable when vLLM is upgraded.
_ORDERED_ADAPTERS = (
    patch_envs,
    patch_import_utils,
    patch_engine_args,
    patch_compilation_config,
    patch_vllm_config,
    patch_slimquant_registry,
    patch_hy_v3_reasoning_parser,
    patch_hy_v3_tool_parser,
)


def register_platform_core_callbacks(
    coordinator: ExactImportCoordinator = IMPORT_COORDINATOR,
) -> tuple[ImportRegistration, ...]:
    """Arm every platform/core adapter in deterministic order."""

    registrations: list[ImportRegistration] = []
    for adapter in _ORDERED_ADAPTERS:
        if adapter is patch_engine_args:
            # ``arg_utils`` itself can trigger the first platform discovery
            # before EngineArgs/AsyncEngineArgs are defined.  Arm its exact
            # decorator boundary before publishing the normal post-import
            # callback; the public patch registry keeps both paths one-shot.
            patch_engine_args.arm_partial_import_bridge()
        registrations.append(
            coordinator.register_callback(
                adapter.PATCH_ID,
                adapter.TARGET_MODULE,
                adapter.apply_to_module,
                targets=adapter.TARGETS,
            )
        )
    return tuple(registrations)


def platform_core_callback_names() -> tuple[tuple[str, str], ...]:
    """Return the frozen callback inventory for doctor/tests."""

    return tuple(
        (adapter.PATCH_ID, adapter.TARGET_MODULE) for adapter in _ORDERED_ADAPTERS
    )


__all__ = [
    "platform_core_callback_names",
    "register_platform_core_callbacks",
]
