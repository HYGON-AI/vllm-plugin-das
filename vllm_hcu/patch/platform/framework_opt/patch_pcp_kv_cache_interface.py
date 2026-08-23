# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Keep full-attention memory capacity independent of PCP."""

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


TARGET_MODULE = "vllm.v1.kv_cache_interface"
PATCH_ID = "platform.framework_opt.pcp_kv_cache_interface"
TARGETS = (f"{TARGET_MODULE}.FullAttentionSpec.max_memory_usage_bytes",)
_MARKER = "_vllm_hcu_pcp_kv_cache_interface_applied"
_WRAPPER = "_vllm_hcu_pcp_full_attention_memory_wrapper"


def apply_to_module(module: ModuleType) -> bool:
    interface = load_exact_module(TARGET_MODULE, module)
    full_attention_spec = require_class(
        interface,
        "FullAttentionSpec",
        f"{TARGET_MODULE}.FullAttentionSpec",
    )
    if already_applied(
        interface,
        _MARKER,
        ((full_attention_spec, "max_memory_usage_bytes", _WRAPPER),),
    ):
        return False

    original = require_callable(
        full_attention_spec,
        "max_memory_usage_bytes",
        TARGETS[0],
    )
    signature = inspect.signature(original)
    parameters = tuple(signature.parameters.values())
    if (
        tuple(signature.parameters) != ("self", "vllm_config")
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

    cdiv = require_callable(interface, "cdiv", f"{TARGET_MODULE}.cdiv")

    @functools.wraps(original)
    def hcu_max_memory_usage_bytes(self, vllm_config):
        parallel_config = vllm_config.parallel_config
        if parallel_config.prefill_context_parallel_size == 1:
            return original(self, vllm_config)

        max_model_len = vllm_config.model_config.max_model_len
        dcp_world_size = parallel_config.decode_context_parallel_size
        if dcp_world_size > 1:
            max_model_len = cdiv(max_model_len, dcp_world_size)
        return cdiv(max_model_len, self.block_size) * self.page_size_bytes

    setattr(hcu_max_memory_usage_bytes, _WRAPPER, True)
    setattr(
        full_attention_spec,
        "_vllm_hcu_original_max_memory_usage_bytes",
        original,
    )
    setattr(
        full_attention_spec,
        "max_memory_usage_bytes",
        hcu_max_memory_usage_bytes,
    )
    setattr(interface, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
