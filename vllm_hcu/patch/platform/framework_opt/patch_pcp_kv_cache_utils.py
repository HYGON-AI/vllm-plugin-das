# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Keep scheduler and hash block sizing independent of PCP ownership."""

from __future__ import annotations

import functools
import inspect
import math
from types import ModuleType

from ._common import (
    PatchCompatibilityError,
    already_applied,
    load_exact_module,
    require_callable,
    require_class,
)


TARGET_MODULE = "vllm.v1.core.kv_cache_utils"
PATCH_ID = "platform.framework_opt.pcp_kv_cache_utils"
TARGETS = (f"{TARGET_MODULE}.resolve_kv_cache_block_sizes",)
_MARKER = "_vllm_hcu_pcp_kv_cache_utils_applied"
_WRAPPER = "_vllm_hcu_pcp_kv_cache_utils_wrapper"


def apply_to_module(module: ModuleType) -> bool:
    kv_cache_utils = load_exact_module(TARGET_MODULE, module)
    if already_applied(
        kv_cache_utils,
        _MARKER,
        ((kv_cache_utils, "resolve_kv_cache_block_sizes", _WRAPPER),),
    ):
        return False

    original = require_callable(
        kv_cache_utils, "resolve_kv_cache_block_sizes", TARGETS[0]
    )
    signature = inspect.signature(original)
    parameters = tuple(signature.parameters.values())
    if (
        tuple(signature.parameters) != ("kv_cache_config", "vllm_config")
        or any(
            parameter.kind is not inspect.Parameter.POSITIONAL_OR_KEYWORD
            or parameter.default is not inspect.Parameter.empty
            for parameter in parameters
        )
    ):
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGETS[0]} has incompatible "
            f"signature {signature}"
        )

    attention_spec = require_class(
        kv_cache_utils,
        "AttentionSpec",
        f"{TARGET_MODULE}.AttentionSpec",
    )
    mamba_spec = require_class(
        kv_cache_utils,
        "MambaSpec",
        f"{TARGET_MODULE}.MambaSpec",
    )

    @functools.wraps(original)
    def hcu_resolve_kv_cache_block_sizes(kv_cache_config, vllm_config):
        parallel_config = vllm_config.parallel_config
        pcp = parallel_config.prefill_context_parallel_size
        if pcp == 1:
            return original(kv_cache_config, vllm_config)

        cache_config = vllm_config.cache_config
        dcp = parallel_config.decode_context_parallel_size
        groups = kv_cache_config.kv_cache_groups

        if len(groups) <= 1:
            block_size = cache_config.block_size * dcp
            return block_size, block_size

        group_block_sizes = [
            group.kv_cache_spec.block_size * dcp
            if isinstance(group.kv_cache_spec, attention_spec)
            else group.kv_cache_spec.block_size
            for group in groups
        ]
        scheduler_block_size = math.lcm(*group_block_sizes)

        connector_enabled = vllm_config.kv_transfer_config is not None
        if not (cache_config.enable_prefix_caching or connector_enabled):
            return scheduler_block_size, scheduler_block_size

        if any(
            isinstance(group.kv_cache_spec, mamba_spec)
            and group.kv_cache_spec.block_size != cache_config.block_size
            for group in groups
        ):
            return scheduler_block_size, scheduler_block_size

        requested = cache_config.hash_block_size
        hash_block_size = (
            requested
            if requested is not None
            else math.gcd(*group_block_sizes)
        )
        if any(
            block_size % hash_block_size != 0
            for block_size in group_block_sizes
        ):
            raise ValueError(
                f"Invalid hash_block_size={hash_block_size}; all KV cache "
                "group block sizes must be divisible by hash_block_size. "
                f"Got group block sizes={group_block_sizes}."
            )
        return scheduler_block_size, hash_block_size

    setattr(hcu_resolve_kv_cache_block_sizes, _WRAPPER, True)
    setattr(
        kv_cache_utils,
        "_vllm_hcu_original_resolve_kv_cache_block_sizes",
        original,
    )
    setattr(
        kv_cache_utils,
        "resolve_kv_cache_block_sizes",
        hcu_resolve_kv_cache_block_sizes,
    )
    setattr(kv_cache_utils, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
