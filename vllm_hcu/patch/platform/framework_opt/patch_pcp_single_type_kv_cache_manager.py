# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Keep per-request KV allocation and hash lookup independent of PCP."""

from __future__ import annotations

import functools
import inspect
from types import ModuleType

from ._common import (
    PatchCompatibilityError,
    load_exact_module,
    require_callable,
    require_class,
)


TARGET_MODULE = "vllm.v1.core.single_type_kv_cache_manager"
PATCH_ID = "platform.framework_opt.pcp_single_type_kv_cache_manager"
TARGETS = (
    f"{TARGET_MODULE}.SingleTypeKVCacheManager.__init__",
    f"{TARGET_MODULE}.FullAttentionManager.find_longest_cache_hit",
)
_MARKER = "_vllm_hcu_pcp_single_type_kv_cache_manager_applied"
_INIT_WRAPPER = "_vllm_hcu_pcp_single_type_manager_init_wrapper"
_HASH_WRAPPER = "_vllm_hcu_pcp_full_attention_hash_wrapper"

_INIT_PARAMETERS = (
    "self",
    "kv_cache_spec",
    "block_pool",
    "enable_caching",
    "kv_cache_group_id",
    "scheduler_block_size",
    "dcp_world_size",
    "pcp_world_size",
    "max_admission_blocks_per_request",
)
_HASH_PARAMETERS = (
    "cls",
    "block_hashes",
    "max_length",
    "kv_cache_group_ids",
    "block_pool",
    "kv_cache_spec",
    "drop_eagle_block",
    "alignment_tokens",
    "dcp_world_size",
    "pcp_world_size",
)


def _require_exact_signature(
    function, target: str, names: tuple[str, ...], defaults: dict[str, object]
) -> None:
    signature = inspect.signature(function)
    parameters = tuple(signature.parameters.values())
    if (
        tuple(signature.parameters) != names
        or any(
            parameter.kind is not inspect.Parameter.POSITIONAL_OR_KEYWORD
            for parameter in parameters
        )
        or any(
            parameter.default
            is not defaults.get(parameter.name, inspect.Parameter.empty)
            for parameter in parameters
        )
    ):
        raise PatchCompatibilityError(
            f"required HCU patch target {target} has incompatible "
            f"signature {signature}"
        )


def apply_to_module(module: ModuleType) -> bool:
    managers = load_exact_module(TARGET_MODULE, module)
    single_type_manager = require_class(
        managers,
        "SingleTypeKVCacheManager",
        f"{TARGET_MODULE}.SingleTypeKVCacheManager",
    )
    full_attention_manager = require_class(
        managers,
        "FullAttentionManager",
        f"{TARGET_MODULE}.FullAttentionManager",
    )

    hash_descriptor = vars(full_attention_manager).get("find_longest_cache_hit")
    if not isinstance(hash_descriptor, classmethod):
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGETS[1]} must be a classmethod"
        )

    if getattr(managers, _MARKER, False):
        init = getattr(single_type_manager, "__init__", None)
        current_hash_descriptor = vars(full_attention_manager).get(
            "find_longest_cache_hit"
        )
        if (
            not callable(init)
            or not getattr(init, _INIT_WRAPPER, False)
            or not isinstance(current_hash_descriptor, classmethod)
            or not getattr(
                current_hash_descriptor.__func__, _HASH_WRAPPER, False
            )
        ):
            raise PatchCompatibilityError(
                "required HCU PCP single-type manager marker is stale; "
                "restart the process"
            )
        return False

    original_init = require_callable(
        single_type_manager, "__init__", TARGETS[0]
    )
    original_hash = hash_descriptor.__func__
    _require_exact_signature(
        original_init,
        TARGETS[0],
        _INIT_PARAMETERS,
        {
            "dcp_world_size": 1,
            "pcp_world_size": 1,
            "max_admission_blocks_per_request": None,
        },
    )
    _require_exact_signature(
        original_hash,
        TARGETS[1],
        _HASH_PARAMETERS,
        {"dcp_world_size": 1, "pcp_world_size": 1},
    )

    @functools.wraps(original_init)
    def hcu_init(
        self,
        kv_cache_spec,
        block_pool,
        enable_caching,
        kv_cache_group_id,
        scheduler_block_size,
        dcp_world_size=1,
        pcp_world_size=1,
        max_admission_blocks_per_request=None,
    ):
        if pcp_world_size == 1:
            return original_init(
                self,
                kv_cache_spec,
                block_pool,
                enable_caching,
                kv_cache_group_id,
                scheduler_block_size,
                dcp_world_size,
                pcp_world_size,
                max_admission_blocks_per_request,
            )

        result = original_init(
            self,
            kv_cache_spec,
            block_pool,
            enable_caching,
            kv_cache_group_id,
            scheduler_block_size,
            dcp_world_size,
            1,
            max_admission_blocks_per_request,
        )
        self.pcp_world_size = pcp_world_size
        return result

    @functools.wraps(original_hash)
    def hcu_find_longest_cache_hit(
        cls,
        block_hashes,
        max_length,
        kv_cache_group_ids,
        block_pool,
        kv_cache_spec,
        drop_eagle_block,
        alignment_tokens,
        dcp_world_size=1,
        pcp_world_size=1,
    ):
        if pcp_world_size == 1:
            return original_hash(
                cls,
                block_hashes,
                max_length,
                kv_cache_group_ids,
                block_pool,
                kv_cache_spec,
                drop_eagle_block,
                alignment_tokens,
                dcp_world_size,
                pcp_world_size,
            )
        return original_hash(
            cls,
            block_hashes,
            max_length,
            kv_cache_group_ids,
            block_pool,
            kv_cache_spec,
            drop_eagle_block,
            alignment_tokens,
            dcp_world_size,
            1,
        )

    setattr(hcu_init, _INIT_WRAPPER, True)
    setattr(hcu_find_longest_cache_hit, _HASH_WRAPPER, True)
    setattr(
        single_type_manager,
        "_vllm_hcu_original_init",
        original_init,
    )
    setattr(
        full_attention_manager,
        "_vllm_hcu_original_find_longest_cache_hit",
        hash_descriptor,
    )
    setattr(single_type_manager, "__init__", hcu_init)
    setattr(
        full_attention_manager,
        "find_longest_cache_hit",
        classmethod(hcu_find_longest_cache_hit),
    )
    setattr(managers, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
