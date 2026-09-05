# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Attach HCU PCP plans to vLLM 0.28 attention metadata."""

from __future__ import annotations

import functools
import hashlib
import inspect
import textwrap
from types import ModuleType

from ._common import (
    PatchCompatibilityError,
    already_applied,
    load_exact_module,
    require_callable,
    require_class,
    require_exact_signature,
)

TARGET_MODULE = "vllm.v1.worker.gpu.model_states.default"
PATCH_ID = "worker.framework_opt.pcp.default_model_state_metadata"
TARGETS = (
    f"{TARGET_MODULE}.DefaultModelState.prepare_attn",
)
_MARKER = "_vllm_hcu_pcp_model_state_applied"
_WRAPPER = "_vllm_hcu_pcp_model_state_wrapper"
_V028_PREPARE_ATTN_SOURCE_SHA256 = (
    "eff5cf447236a900df70ef1f91a5868f6a676124d6220cbc9a041e582df0e728"
)


def _attach_pcp_plan(attn_metadata: dict[str, object], pcp_plan: object) -> None:
    """Attach a plan only to metadata types that explicitly support it."""
    if pcp_plan is None:
        return
    visited: set[int] = set()
    for metadata in attn_metadata.values():
        metadata_id = id(metadata)
        if metadata_id in visited:
            continue
        visited.add(metadata_id)
        if hasattr(metadata, "pcp_plan"):
            setattr(metadata, "pcp_plan", pcp_plan)


def _require_source_fingerprint(function, target: str, expected: str) -> None:
    try:
        source = textwrap.dedent(inspect.getsource(function))
    except (OSError, TypeError) as exc:
        raise PatchCompatibilityError(
            f"required HCU patch target {target} source fingerprint could "
            f"not be computed: expected sha256={expected}, "
            f"actual=<unavailable>; {type(exc).__name__}: {exc}"
        ) from exc
    actual = hashlib.sha256(source.encode("utf-8")).hexdigest()
    if actual != expected:
        raise PatchCompatibilityError(
            f"required HCU patch target {target} source fingerprint "
            f"mismatch: expected sha256={expected}, actual sha256={actual}"
        )


def apply_to_module(module: ModuleType) -> bool:
    default = load_exact_module(TARGET_MODULE, module)
    model_state = require_class(
        default,
        "DefaultModelState",
        f"{TARGET_MODULE}.DefaultModelState",
    )
    wrapped = ((model_state, "prepare_attn", TARGETS[0], _WRAPPER),)
    if already_applied(default, _MARKER, wrapped):
        return False

    original_prepare_attn = require_callable(
        model_state, "prepare_attn", TARGETS[0]
    )
    require_exact_signature(
        original_prepare_attn,
        TARGETS[0],
        positional=(
            "self",
            "input_batch",
            "cudagraph_mode",
            "block_tables",
            "slot_mappings",
            "attn_groups",
            "kv_cache_config",
            "for_capture",
        ),
        defaults={"for_capture": False},
    )
    _require_source_fingerprint(
        original_prepare_attn,
        TARGETS[0],
        _V028_PREPARE_ATTN_SOURCE_SHA256,
    )
    original_code = getattr(original_prepare_attn, "__code__", None)
    original_required_names = {
        "CUDAGraphMode",
        "build_attn_metadata",
        "compute_mm_prefix_ranges",
        "query_start_loc_np",
        "query_start_loc",
        "is_prefilling_np",
        "max_query_len",
    }
    if original_code is None or not original_required_names.issubset(
        original_code.co_names
    ):
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGETS[0]} no longer matches the "
            "audited v0.28 metadata path"
        )

    @functools.wraps(original_prepare_attn)
    def hcu_prepare_attn(
        self,
        input_batch,
        cudagraph_mode,
        block_tables,
        slot_mappings,
        attn_groups,
        kv_cache_config,
        for_capture=False,
    ):
        attn_metadata = original_prepare_attn(
            self,
            input_batch,
            cudagraph_mode,
            block_tables,
            slot_mappings,
            attn_groups,
            kv_cache_config,
            for_capture,
        )
        pcp_size = int(
            self.vllm_config.parallel_config.prefill_context_parallel_size
        )
        if pcp_size > 1:
            _attach_pcp_plan(
                attn_metadata,
                getattr(input_batch, "_vllm_hcu_pcp_plan", None),
            )
        return attn_metadata

    setattr(hcu_prepare_attn, _WRAPPER, True)
    setattr(
        model_state,
        "_vllm_hcu_original_prepare_attn",
        original_prepare_attn,
    )
    setattr(model_state, "prepare_attn", hcu_prepare_attn)
    setattr(default, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
