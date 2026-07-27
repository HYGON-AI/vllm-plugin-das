# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Exact, lazy whole-module exchanges for the HCU platform.

The legacy hook matched ancestor imports and loaded every HCU implementation
as one batch.  This module only registers complete canonical names with the
exact import coordinator.  Registration validates HCU source locations on
disk, but importing an implementation is deferred until its canonical module
is requested.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from .import_coordinator import (
    IMPORT_COORDINATOR,
    ExactImportCoordinator,
    ImportRegistration,
)

_Entry = tuple[str, str, str]

# (patch id, canonical vLLM module, HCU implementation module)
_MODULAR_KERNEL: tuple[_Entry, ...] = (
    (
        "module_exchange.modular_kernel",
        "vllm.model_executor.layers.fused_moe.modular_kernel",
        "vllm_hcu.model_executor.layers.fused_moe.modular_kernel",
    ),
)

_BASE_LINEAR: tuple[_Entry, ...] = (
    (
        "module_exchange.base_linear.linear",
        "vllm.model_executor.layers.linear",
        "vllm_hcu.model_executor.layers.linear",
    ),
)

_DEEPSEEK_V4: tuple[_Entry, ...] = (
    (
        "module_exchange.deepseek_v4.sparse_swa",
        "vllm.v1.attention.backends.mla.sparse_swa",
        "vllm_hcu.v1.attention.backends.mla.sparse_swa",
    ),
)

_DEEP_GEMM: tuple[_Entry, ...] = (
    (
        "module_exchange.deep_gemm.utils",
        "vllm.model_executor.layers.fused_moe.deep_gemm_utils",
        "vllm_hcu.model_executor.layers.fused_moe.deep_gemm_utils",
    ),
    (
        "module_exchange.deep_gemm.experts",
        "vllm.model_executor.layers.fused_moe.experts.deep_gemm_moe",
        "vllm_hcu.model_executor.layers.fused_moe.experts.deep_gemm_moe",
    ),
    (
        "module_exchange.deep_gemm.batched_experts",
        "vllm.model_executor.layers.fused_moe.experts.batched_deep_gemm_moe",
        "vllm_hcu.model_executor.layers.fused_moe.experts.batched_deep_gemm_moe",
    ),
)

_ATTENTION: tuple[_Entry, ...] = (
    (
        "module_exchange.attention.sparse_indexer",
        "vllm.model_executor.layers.sparse_attn_indexer",
        "vllm_hcu.model_executor.layers.sparse_attn_indexer",
    ),
)

_ALL_GROUPS: tuple[tuple[_Entry, ...], ...] = (
    _MODULAR_KERNEL,
    _BASE_LINEAR,
    # DeepSeek attention imports the sparse indexer through its canonical
    # alias, so the indexer's AITER-bearing replacement must be armed first.
    _ATTENTION,
    _DEEPSEEK_V4,
    _DEEP_GEMM,
)

_HCU_PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def _validate_hcu_replacement_path(module_name: str) -> None:
    """Validate an HCU module path without importing any package/module."""

    parts = module_name.split(".")
    if len(parts) < 2 or parts[0] != "vllm_hcu" or any(
        not part.isidentifier() for part in parts
    ):
        raise ValueError(
            f"replacement must be an absolute vllm_hcu module name: {module_name!r}"
        )
    relative = _HCU_PACKAGE_ROOT.joinpath(*parts[1:])
    module_file = relative.with_suffix(".py")
    package_file = relative / "__init__.py"
    if not module_file.is_file() and not package_file.is_file():
        raise ModuleNotFoundError(
            f"HCU replacement module {module_name!r} has no source at "
            f"{module_file} or {package_file}"
        )


def _validate_entries(entries: Iterable[_Entry]) -> tuple[_Entry, ...]:
    materialized = tuple(entries)
    patch_ids: set[str] = set()
    canonical_names: set[str] = set()
    for patch_id, canonical, replacement in materialized:
        if not patch_id or patch_id in patch_ids:
            raise ValueError(
                f"duplicate or empty module-exchange patch id: {patch_id!r}"
            )
        if not canonical.startswith("vllm.") or canonical == replacement:
            raise ValueError(f"invalid canonical module exchange target: {canonical!r}")
        if canonical in canonical_names:
            raise ValueError(
                f"duplicate canonical module exchange target: {canonical!r}"
            )
        _validate_hcu_replacement_path(replacement)
        patch_ids.add(patch_id)
        canonical_names.add(canonical)
    return materialized


def _register_entries(
    entries: Iterable[_Entry],
    coordinator: ExactImportCoordinator,
) -> tuple[ImportRegistration, ...]:
    with coordinator.registration_batch():
        checked = _validate_entries(entries)
        return tuple(
            coordinator.register_replacement(
                patch_id,
                canonical,
                replacement,
                targets=(canonical, replacement),
                late_policy="fail",
            )
            for patch_id, canonical, replacement in checked
        )


def register_modular_kernel_exchange(
    coordinator: ExactImportCoordinator = IMPORT_COORDINATOR,
) -> tuple[ImportRegistration, ...]:
    """Arm the HCU-owned modular-kernel replacement first."""

    return _register_entries(_MODULAR_KERNEL, coordinator)


def register_base_linear_exchanges(
    coordinator: ExactImportCoordinator = IMPORT_COORDINATOR,
) -> tuple[ImportRegistration, ...]:
    return _register_entries(_BASE_LINEAR, coordinator)


def register_deepseek_v4_exchanges(
    coordinator: ExactImportCoordinator = IMPORT_COORDINATOR,
) -> tuple[ImportRegistration, ...]:
    return _register_entries(_DEEPSEEK_V4, coordinator)


def register_deep_gemm_exchanges(
    coordinator: ExactImportCoordinator = IMPORT_COORDINATOR,
) -> tuple[ImportRegistration, ...]:
    return _register_entries(_DEEP_GEMM, coordinator)


def register_attention_exchanges(
    coordinator: ExactImportCoordinator = IMPORT_COORDINATOR,
) -> tuple[ImportRegistration, ...]:
    return _register_entries(_ATTENTION, coordinator)


def register_all_module_exchanges(
    coordinator: ExactImportCoordinator = IMPORT_COORDINATOR,
) -> tuple[ImportRegistration, ...]:
    """Register all exchanges in explicit safety order without importing them."""

    with coordinator.registration_batch():
        # Validate the complete set before mutating coordinator state.  The
        # modular kernel is then armed first so consumers cannot import its
        # official implementation before the HCU-owned alias is visible.
        # Nested group fences use the coordinator's RLock and keep this whole
        # inventory one atomic visibility unit.
        _validate_entries(entry for group in _ALL_GROUPS for entry in group)
        registrations: list[ImportRegistration] = []
        for group in _ALL_GROUPS:
            registrations.extend(_register_entries(group, coordinator))
        return tuple(registrations)


def module_exchange_names() -> tuple[tuple[str, str], ...]:
    """Return the frozen canonical/replacement pairs for diagnostics."""

    return tuple(
        (canonical, replacement)
        for group in _ALL_GROUPS
        for _, canonical, replacement in group
    )


__all__ = [
    "module_exchange_names",
    "register_all_module_exchanges",
    "register_attention_exchanges",
    "register_base_linear_exchanges",
    "register_deep_gemm_exchanges",
    "register_deepseek_v4_exchanges",
    "register_modular_kernel_exchange",
]
