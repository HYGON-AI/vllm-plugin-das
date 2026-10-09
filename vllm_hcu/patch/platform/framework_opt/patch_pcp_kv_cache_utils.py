# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Keep scheduler and hash block sizing independent of PCP ownership."""

from __future__ import annotations

import functools
import inspect
from types import ModuleType

from ._common import (
    PatchCompatibilityError,
    already_applied,
    load_exact_module,
    require_callable,
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

    @functools.wraps(original)
    def hcu_resolve_kv_cache_block_sizes(kv_cache_config, vllm_config):
        # Current vLLM already resolves cache ownership from DCP only and
        # handles prefix_match_unit, prefix-cacheable groups, and Mamba align
        # mode.  Reimplementing that policy here made this adapter depend on
        # the removed CacheConfig.hash_block_size field.  Keep the wrapper as
        # a stable plugin integration point, but delegate sizing to vLLM so
        # PCP never carries a stale copy of the cache-layout algorithm.
        return original(kv_cache_config, vllm_config)

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
