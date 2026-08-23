# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Keep unitary coordinator KV ownership independent of PCP."""

from __future__ import annotations

import functools
import inspect
from types import ModuleType

from ._common import (
    PatchCompatibilityError,
    already_applied,
    load_exact_module,
    require_callable,
    require_class,
)


TARGET_MODULE = "vllm.v1.core.kv_cache_coordinator"
PATCH_ID = "platform.framework_opt.pcp_kv_cache_coordinator"
TARGETS = (f"{TARGET_MODULE}.UnitaryKVCacheCoordinator.__init__",)
_MARKER = "_vllm_hcu_pcp_kv_cache_coordinator_applied"
_WRAPPER = "_vllm_hcu_pcp_unitary_coordinator_init_wrapper"
_PARAMETERS = (
    "self",
    "kv_cache_config",
    "max_model_len",
    "max_num_batched_tokens",
    "use_eagle",
    "enable_caching",
    "enable_kv_cache_events",
    "dcp_world_size",
    "pcp_world_size",
    "scheduler_block_size",
    "hash_block_size",
    "metrics_collector",
)


def apply_to_module(module: ModuleType) -> bool:
    coordinator = load_exact_module(TARGET_MODULE, module)
    unitary = require_class(
        coordinator,
        "UnitaryKVCacheCoordinator",
        f"{TARGET_MODULE}.UnitaryKVCacheCoordinator",
    )
    if already_applied(
        coordinator,
        _MARKER,
        ((unitary, "__init__", _WRAPPER),),
    ):
        return False

    original = require_callable(unitary, "__init__", TARGETS[0])
    signature = inspect.signature(original)
    parameters = tuple(signature.parameters.values())
    if (
        tuple(signature.parameters) != _PARAMETERS
        or any(
            parameter.kind is not inspect.Parameter.POSITIONAL_OR_KEYWORD
            for parameter in parameters
        )
        or any(
            parameter.default
            is not (
                None
                if parameter.name == "metrics_collector"
                else inspect.Parameter.empty
            )
            for parameter in parameters
        )
    ):
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGETS[0]} has incompatible "
            f"signature {signature}"
        )
    base = require_class(
        coordinator,
        "KVCacheCoordinator",
        f"{TARGET_MODULE}.KVCacheCoordinator",
    )
    if unitary.__bases__ != (base,):
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGETS[0]} has incompatible base class"
        )

    @functools.wraps(original)
    def hcu_init(
        self,
        kv_cache_config,
        max_model_len,
        max_num_batched_tokens,
        use_eagle,
        enable_caching,
        enable_kv_cache_events,
        dcp_world_size,
        pcp_world_size,
        scheduler_block_size,
        hash_block_size,
        metrics_collector=None,
    ):
        if pcp_world_size == 1:
            return original(
                self,
                kv_cache_config,
                max_model_len,
                max_num_batched_tokens,
                use_eagle,
                enable_caching,
                enable_kv_cache_events,
                dcp_world_size,
                pcp_world_size,
                scheduler_block_size,
                hash_block_size,
                metrics_collector,
            )

        super(unitary, self).__init__(
            kv_cache_config,
            max_model_len,
            max_num_batched_tokens,
            use_eagle,
            enable_caching,
            enable_kv_cache_events,
            dcp_world_size,
            pcp_world_size,
            scheduler_block_size,
            hash_block_size,
            metrics_collector,
        )
        self.kv_cache_spec = self.kv_cache_config.kv_cache_groups[0].kv_cache_spec
        self.block_size = self.kv_cache_spec.block_size
        self.dcp_world_size = dcp_world_size
        self.pcp_world_size = pcp_world_size
        if dcp_world_size > 1:
            self.block_size *= dcp_world_size
        assert not enable_caching or hash_block_size == self.block_size, (
            "UnitaryKVCacheCoordinator assumes hash_block_size == block_size"
        )
        assert len(self.kv_cache_config.kv_cache_groups) == 1, (
            "UnitaryKVCacheCoordinator assumes only one kv cache group"
        )
        self.single_type_managers[0].use_eagle = 0 in self.eagle_group_ids
        return None

    setattr(hcu_init, _WRAPPER, True)
    setattr(unitary, "_vllm_hcu_original_init", original)
    setattr(unitary, "__init__", hcu_init)
    setattr(coordinator, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
