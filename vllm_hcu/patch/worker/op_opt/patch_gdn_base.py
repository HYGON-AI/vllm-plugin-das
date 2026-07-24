# SPDX-License-Identifier: Apache-2.0
"""Qwen-local HCU state-dtype delta for vLLM v0.25.1 GDN attention."""

from __future__ import annotations

import functools
from types import ModuleType

from ._common import (
    already_applied,
    load_exact_module,
    require_callable,
    require_class,
    require_exact_signature,
)

TARGET_MODULE = "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn"
PATCH_ID = "worker.op_opt.mamba.gdn.base_state_dtype"
TARGETS = (f"{TARGET_MODULE}.QwenGatedDeltaNetAttention.get_state_dtype",)
_MARKER = "_vllm_hcu_gdn_base_applied"
_WRAPPER = "_vllm_hcu_gdn_base_wrapper"


def apply_to_module(module: ModuleType) -> bool:
    qwen = load_exact_module(TARGET_MODULE, module)
    cls = require_class(
        qwen,
        "QwenGatedDeltaNetAttention",
        f"{TARGET_MODULE}.QwenGatedDeltaNetAttention",
    )
    wrapped = ((cls, "get_state_dtype", TARGETS[0], _WRAPPER),)
    if already_applied(qwen, _MARKER, wrapped):
        return False
    state_dtype = require_callable(cls, "get_state_dtype", TARGETS[0])
    require_exact_signature(state_dtype, TARGETS[0], positional=("self",))

    @functools.wraps(state_dtype)
    def hcu_state_dtype(self):
        from vllm_hcu.platforms import envs as henvs

        if not (
            henvs.VLLM_HCU_MAMBA_SSM_CACHE_DTYPE
            and henvs.VLLM_HCU_USE_CUSTOM_OPS
        ):
            return state_dtype(self)
        from vllm.model_executor.layers.mamba.mamba_utils import (
            MambaStateDtypeCalculator,
        )

        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            self.model_config.dtype,
            self.cache_config.mamba_cache_dtype,
            "auto",
        )

    setattr(hcu_state_dtype, _WRAPPER, True)
    setattr(cls, "_vllm_hcu_original_get_state_dtype", state_dtype)
    setattr(cls, "get_state_dtype", hcu_state_dtype)
    setattr(qwen, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
