# SPDX-License-Identifier: Apache-2.0
"""Lightly-CP adapter for the official MLA wrapper class."""

from __future__ import annotations

import functools
from types import ModuleType

from vllm_hcu.patch.config import get_hcu_config

from ._common import PatchCompatibilityError, already_applied, load_exact_module, require_callable, require_class, require_exact_signature

TARGET_MODULE = "vllm.model_executor.layers.mla"
PATCH_ID = "worker.op_opt.mla.lightly_cp_wrapper"
TARGETS = (
    f"{TARGET_MODULE}.MultiHeadLatentAttentionWrapper.__init__",
    f"{TARGET_MODULE}.MultiHeadLatentAttentionWrapper.forward",
)
_MARKER = "_vllm_hcu_mla_lightly_cp_applied"
_WRAPPER = "_vllm_hcu_mla_lightly_cp_wrapper"


def apply_to_module(module: ModuleType) -> bool:
    mla = load_exact_module(TARGET_MODULE, module)
    cls = require_class(mla, "MultiHeadLatentAttentionWrapper", f"{TARGET_MODULE}.MultiHeadLatentAttentionWrapper")
    wrapped = ((cls, "__init__", TARGETS[0], _WRAPPER), (cls, "forward", TARGETS[1], _WRAPPER))
    if already_applied(mla, _MARKER, wrapped):
        return False
    original_init = require_callable(cls, "__init__", TARGETS[0])
    require_exact_signature(
        original_init, TARGETS[0],
        positional=("self", "hidden_size", "num_heads", "scale", "qk_nope_head_dim",
                    "qk_rope_head_dim", "v_head_dim", "q_lora_rank", "kv_lora_rank",
                    "mla_modules", "cache_config", "quant_config", "prefix", "skip_topk"),
        defaults={"cache_config": None, "quant_config": None, "prefix": "", "skip_topk": False},
    )
    original_forward = require_callable(cls, "forward", TARGETS[1])
    require_exact_signature(
        original_forward, TARGETS[1],
        positional=("self", "positions", "hidden_states", "llama_4_scaling"),
        defaults={"llama_4_scaling": None},
    )
    if "skip_topk" not in original_init.__code__.co_names:
        raise PatchCompatibilityError(
            "clean v0.25.1 target MLA constructor no longer stores skip_topk"
        )
    if "skip_topk" not in original_forward.__code__.co_names:
        raise PatchCompatibilityError(
            "clean v0.25.1 target MLA forward no longer guards skip_topk"
        )

    @functools.wraps(original_init)
    def hcu_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        from vllm.config import get_current_vllm_config_or_none

        self._hcu_feature_config = get_hcu_config(get_current_vllm_config_or_none())

    @functools.wraps(original_forward)
    def hcu_forward(self, positions, hidden_states, llama_4_scaling=None):
        config = getattr(self, "_hcu_feature_config", None)
        if config is None:
            raise RuntimeError("HCU MLA feature config was not initialized")
        if not config.enable_lightly_cp:
            return original_forward(self, positions, hidden_states, llama_4_scaling)
        from vllm_hcu.model_executor.layers.mla_runtime import lightly_cp_mla_wrapper_forward

        return lightly_cp_mla_wrapper_forward(
            self, positions, hidden_states, llama_4_scaling, config
        )

    setattr(hcu_init, _WRAPPER, True)
    setattr(hcu_forward, _WRAPPER, True)
    setattr(cls, "_vllm_hcu_original_init", original_init)
    setattr(cls, "_vllm_hcu_original_forward", original_forward)
    setattr(cls, "__init__", hcu_init)
    setattr(cls, "forward", hcu_forward)
    setattr(mla, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = [ "PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
