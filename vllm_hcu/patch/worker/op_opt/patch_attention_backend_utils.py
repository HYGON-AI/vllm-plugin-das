# SPDX-License-Identifier: Apache-2.0
"""Attach CPU sequence lengths to causal-conv metadata after exact import."""

from __future__ import annotations

import functools
from types import ModuleType

from ._common import already_applied, load_exact_module, require_callable, require_exact_signature

TARGET_MODULE = "vllm.v1.attention.backends.utils"
PATCH_ID = "worker.op_opt.attention.causal_conv_metadata"
TARGETS = (f"{TARGET_MODULE}.compute_causal_conv1d_metadata",)
_MARKER = "_vllm_hcu_causal_conv_metadata_applied"
_WRAPPER = "_vllm_hcu_causal_conv_metadata_wrapper"


def apply_to_module(module: ModuleType) -> bool:
    utils = load_exact_module(TARGET_MODULE, module)
    if already_applied(utils, _MARKER, ((utils, "compute_causal_conv1d_metadata", TARGETS[0], _WRAPPER),)):
        return False
    original = require_callable(utils, "compute_causal_conv1d_metadata", TARGETS[0])
    require_exact_signature(
        original, TARGETS[0], positional=("query_start_loc_p_cpu",),
        keyword_only=("device",),
    )

    @functools.wraps(original)
    def hcu_metadata(query_start_loc_p_cpu, *, device):
        result = original(query_start_loc_p_cpu, device=device)
        if not isinstance(result, tuple) or len(result) != 3 or not isinstance(result[0], dict):
            raise RuntimeError("vLLM causal-conv metadata returned an incompatible value")
        result[0]["seqlens"] = query_start_loc_p_cpu.diff().tolist()
        return result

    setattr(hcu_metadata, _WRAPPER, True)
    setattr(utils, "_vllm_hcu_original_compute_causal_conv1d_metadata", original)
    setattr(utils, "compute_causal_conv1d_metadata", hcu_metadata)
    setattr(utils, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
