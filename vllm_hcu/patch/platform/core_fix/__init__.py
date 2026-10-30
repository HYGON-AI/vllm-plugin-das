# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
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
    patch_cache_config,
    patch_compilation_config,
    patch_config_utils,
    patch_decoder_replay_compilation,
    patch_engram_config,
    patch_engine_args,
    patch_envs,
    patch_hy_v3_reasoning_parser,
    patch_hy_v3_tool_parser,
    patch_import_utils,
    patch_layer_name,
    patch_nixl_utils,
    patch_qwen4_exp_pp,
    patch_slimquant_registry,
    patch_vllm_config,
)


# This tuple is deliberately hand-ordered.  Do not replace it with package or
# directory discovery: ordering is part of the plugin contract and must remain
# reviewable when vLLM is upgraded.
_ORDERED_ADAPTERS = (
    patch_envs,
    patch_config_utils,
    patch_engram_config,
    patch_import_utils,
    patch_layer_name,
    patch_nixl_utils,
    patch_engine_args,
    # After EngineArgs wrappers so CLI / create_engine_config compose correctly.
    patch_cache_config,
    patch_compilation_config,
    # After compilation_config so CompilationConfig class exists / is patched.
    patch_decoder_replay_compilation,
    patch_vllm_config,
    patch_qwen4_exp_pp,
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
        if adapter is patch_layer_name:
            patch_layer_name.arm_partial_import_bridge()
        if adapter is patch_engine_args:
            # ``arg_utils`` itself can trigger the first platform discovery
            # before EngineArgs/AsyncEngineArgs are defined.  Arm its exact
            # decorator boundary before publishing the normal post-import
            # callback; the public patch registry keeps both paths one-shot.
            patch_engine_args.arm_partial_import_bridge()
        if adapter is patch_cache_config:
            # CacheConfig exact-import callback.
            registrations.append(
                coordinator.register_callback(
                    adapter.PATCH_ID,
                    adapter.TARGET_MODULE,
                    adapter.apply_to_module,
                    targets=adapter.TARGETS,
                )
            )
            # EngineArgs wiring must run after patch_engine_args wrappers.
            registrations.append(
                coordinator.register_callback(
                    adapter.ENGINE_PATCH_ID,
                    adapter.ENGINE_TARGET_MODULE,
                    adapter.apply_engine_args_to_module,
                    targets=adapter.ENGINE_TARGETS,
                )
            )
            continue
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

    names: list[tuple[str, str]] = []
    for adapter in _ORDERED_ADAPTERS:
        names.append((adapter.PATCH_ID, adapter.TARGET_MODULE))
        if adapter is patch_cache_config:
            names.append(
                (adapter.ENGINE_PATCH_ID, adapter.ENGINE_TARGET_MODULE)
            )
    return tuple(names)


__all__ = [
    "platform_core_callback_names",
    "register_platform_core_callbacks",
]
