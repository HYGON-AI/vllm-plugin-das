# SPDX-License-Identifier: Apache-2.0
"""Bind validated maps after construction and before checkpoint loading."""
from __future__ import annotations

import functools
import sys
from contextvars import ContextVar

from ._common import load_exact_module, require_exact_signature, PatchCompatibilityError

TARGET_MODULE = "vllm.model_executor.model_loader.utils"
PATCH_ID = "worker.framework_opt.model_loader.static_eplb_preload"
TARGETS = (f"{TARGET_MODULE}.initialize_model",)
INITIALIZER_ALIASES = ("vllm.model_executor.model_loader.base_loader",
    "vllm.model_executor.model_loader.tensorizer_loader", "vllm.model_executor.models.mllama4")
_MARKER = "_vllm_hcu_static_preload"
_DEPTH = ContextVar("hcu_static_initialization_depth", default=0)


def validate_static_loader(vllm_config, load_config):
    from vllm_hcu.patch.config import bind_hcu_eplb_config
    bind_hcu_eplb_config(vllm_config)
    path = getattr(getattr(vllm_config, "parallel_config", None), "_vllm_hcu_expert_map_path", None)
    if path and getattr(load_config, "load_format", None) not in {"auto", "pt", "safetensors", "npcache", "mistral"}:
        raise ValueError("Static EPLB requires an audited default model loader format")


def apply_to_module(module):
    target = load_exact_module(TARGET_MODULE, module)
    original = target.initialize_model
    if getattr(target, _MARKER, False):
        if not getattr(original, _MARKER, False):
            raise PatchCompatibilityError("Static EPLB initializer marker is stale")
        return False
    require_exact_signature(original, TARGETS[0], positional=("vllm_config",),
        keyword_only=("prefix", "model_class", "model_config"),
        defaults=dict(prefix="", model_class=None, model_config=None))
    aliases = []
    for name in INITIALIZER_ALIASES:
        alias = sys.modules.get(name)
        if alias is None:
            continue
        value = getattr(alias, "initialize_model", None)
        if value is None and getattr(getattr(alias, "__spec__", None), "_initializing", False):
            continue
        if value is not original:
            raise PatchCompatibilityError(f"Static EPLB initializer alias mismatch: {name}")
        aliases.append(alias)

    @functools.wraps(original)
    def initialize(vllm_config, *, prefix="", model_class=None, model_config=None):
        from vllm_hcu.model_executor.layers.fused_moe.static_eplb import bind_static_eplb_plan
        from vllm_hcu.patch.config import bind_hcu_eplb_config
        bind_hcu_eplb_config(vllm_config)
        depth = _DEPTH.get()
        token = _DEPTH.set(depth + 1)
        try:
            model = original(vllm_config, prefix=prefix, model_class=model_class,
                             model_config=model_config)
            if depth == 0:
                bind_static_eplb_plan(vllm_config, model)
            return model
        finally:
            _DEPTH.reset(token)
    setattr(initialize, _MARKER, True)
    target.initialize_model = initialize
    for alias in aliases:
        alias.initialize_model = initialize
    setattr(target, _MARKER, True)
    return True


def apply(module=None):
    return apply_to_module(load_exact_module(TARGET_MODULE, module))
