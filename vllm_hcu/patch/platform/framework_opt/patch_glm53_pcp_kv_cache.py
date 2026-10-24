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
_OWNERSHIP = "_hcu_glm53_replicated_cache_pcp_width"


def _has_glm53_state_groups(kv_cache_config) -> bool:
    from vllm.v1.kv_cache_interface import KpoolTailSpec, MambaSpec

    specs = [group.kv_cache_spec for group in kv_cache_config.kv_cache_groups]
    return any(isinstance(spec, MambaSpec) for spec in specs) and any(
        isinstance(spec, KpoolTailSpec) for spec in specs
    )


def mark_glm53_cache_ownership(kv_cache_config, vllm_config) -> None:
    """Stamp the model-specific contract where config and cache meet.

    EngineCore resolves block sizes on the scheduler cache before constructing
    its scheduler/coordinator. No distributed group or ambient global config
    is needed, and unrelated hybrid models remain untouched.
    """
    from vllm_hcu.v1.glm53_kda_cp_config import validate_glm53_pcp_topology

    # A cache object reused with another config must not retain our old policy.
    if hasattr(kv_cache_config, _OWNERSHIP):
        delattr(kv_cache_config, _OWNERSHIP)
    model = getattr(vllm_config, "model_config", None)
    if getattr(model, "architectures", None) not in (
        ["Glm5NextForConditionalGeneration"], ["Glm5NextForCausalLM"],
    ):
        return
    size = vllm_config.parallel_config.prefill_context_parallel_size
    if size <= 1 or not _has_glm53_state_groups(kv_cache_config):
        return
    validate_glm53_pcp_topology(vllm_config)
    setattr(kv_cache_config, _OWNERSHIP, size)


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    cls = require_class(target, "HybridKVCacheCoordinator", TARGETS[0])
    original = require_callable(cls, "__init__", TARGETS[0])
    if getattr(original, _MARKER, False):
        return False
    signature = inspect.signature(original)

    @functools.wraps(original)
    def initialize(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        width = bound.arguments["pcp_world_size"]
        cache_config = bound.arguments["kv_cache_config"]
        owner_width = getattr(cache_config, _OWNERSHIP, None)
        if width > 1 and owner_width is not None:
            if width != owner_width:
                raise ValueError("GLM5Next cache ownership/PCP width mismatch")
            if not _has_glm53_state_groups(cache_config):
                raise ValueError("GLM5Next replicated cache state groups changed")
            # Logical token/block ownership is replicated across PCP. KDA
            # per-head state dimensions are separately sized by head_size.
            bound.arguments["pcp_world_size"] = 1
        return original(*bound.args, **bound.kwargs)

    setattr(initialize, _MARKER, True)
    cls.__init__ = initialize
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))
