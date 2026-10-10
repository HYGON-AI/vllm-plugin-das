# SPDX-License-Identifier: Apache-2.0
"""Use replicated cache ownership for the GLM5Next PCP hybrid layout."""

from __future__ import annotations

import functools
import inspect
from types import ModuleType

from ._common import load_exact_module, require_callable, require_class

TARGET_MODULE = "vllm.v1.core.kv_cache_coordinator"
PATCH_ID = "platform.framework_opt.glm53_pcp_hybrid_kv_cache"
TARGETS = (f"{TARGET_MODULE}.HybridKVCacheCoordinator.__init__",)
_MARKER = "_hcu_glm53_pcp_hybrid_cache"


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    cls = require_class(target, "HybridKVCacheCoordinator", TARGETS[0])
    original = require_callable(cls, "__init__", TARGETS[0])
    if getattr(original, _MARKER, False):
        return False
    signature = inspect.signature(original)

    @functools.wraps(original)
    def initialize(*args, **kwargs):
        from vllm.v1.kv_cache_interface import KpoolTailSpec, MambaSpec

        bound = signature.bind(*args, **kwargs)
        width = bound.arguments["pcp_world_size"]
        specs = [
            group.kv_cache_spec
            for group in bound.arguments["kv_cache_config"].kv_cache_groups
        ]
        if (
            width == 8
            and any(isinstance(spec, MambaSpec) for spec in specs)
            and any(isinstance(spec, KpoolTailSpec) for spec in specs)
        ):
            # HCU PCP gathers MLA KV writes and replicates KDA/K-pool state.
            # Each rank owns a complete logical cache, not one eighth of it.
            bound.arguments["pcp_world_size"] = 1
        return original(*bound.args, **bound.kwargs)

    setattr(initialize, _MARKER, True)
    cls.__init__ = initialize
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))
