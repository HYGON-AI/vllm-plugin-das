# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Attach PCP request-phase metadata to v0.25.1 DefaultModelState.

This selectively backports the DefaultModelState portion of vLLM upstream
commit b6ff8a2f50.  The pinned v0.25.1 ``build_attn_metadata`` predates its
``is_prefilling`` parameter, so the adapter uses the existing model-specific
metadata hook to populate the same ``CommonAttentionMetadata`` field without
changing the upstream helper's public signature.
"""

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
    f"{TARGET_MODULE}.build_attn_metadata",
)
_MARKER = "_vllm_hcu_pcp_model_state_applied"
_WRAPPER = "_vllm_hcu_pcp_model_state_wrapper"
_V0251_PREPARE_ATTN_SOURCE_SHA256 = (
    "f71a0efd2110430144bd006e9e478dcecb7b07d6eeb841a91c789a6301b9d0b9"
)
_V0251_BUILD_ATTN_METADATA_SOURCE_SHA256 = (
    "62890a577c861312c50b7400cf00908e223999fe2e9e400a0738a469655d2cd1"
)


class _PCPRequestPhaseMetadata:
    """Supply request phase and GQA PCP plan through the metadata hook."""

    def __init__(self, is_prefilling, pcp_plan=None):
        self.is_prefilling = is_prefilling
        self.pcp_plan = pcp_plan

    def get_extra_common_attn_kwargs(self, kv_cache_group_id, num_reqs):
        del kv_cache_group_id
        return {"is_prefilling": self.is_prefilling[:num_reqs]}

    def get_extra_attn_kwargs(self, attn_metadata_builder, num_reqs):
        del num_reqs
        if (
            self.pcp_plan is not None
            and getattr(attn_metadata_builder, "supports_pcp_plan", False)
        ):
            return {"pcp_plan": self.pcp_plan}
        return {}


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


def _validate_build_attn_metadata(build_attn_metadata) -> None:
    positional = (
        "attn_groups",
        "num_reqs",
        "num_tokens",
        "query_start_loc_gpu",
        "query_start_loc_cpu",
        "max_query_len",
        "seq_lens",
        "max_seq_len",
        "block_tables",
        "slot_mappings",
        "kv_cache_config",
        "seq_lens_cpu_upper_bound",
        "dcp_local_seq_lens",
        "positions",
        "mm_req_doc_ranges",
        "model_specific_attn_metadata",
        "for_cudagraph_capture",
        "causal",
        "rswa_prefix_lens",
    )
    require_exact_signature(
        build_attn_metadata,
        TARGETS[1],
        positional=positional,
        defaults={
            "seq_lens_cpu_upper_bound": None,
            "dcp_local_seq_lens": None,
            "positions": None,
            "mm_req_doc_ranges": None,
            "model_specific_attn_metadata": None,
            "for_cudagraph_capture": False,
            "causal": True,
            "rswa_prefix_lens": None,
        },
    )
    _require_source_fingerprint(
        build_attn_metadata,
        TARGETS[1],
        _V0251_BUILD_ATTN_METADATA_SOURCE_SHA256,
    )
    function_globals = getattr(build_attn_metadata, "__globals__", {})
    metadata_class = function_globals.get("CommonAttentionMetadata")
    fields = getattr(metadata_class, "__dataclass_fields__", {})
    if "is_prefilling" not in fields:
        raise PatchCompatibilityError(
            "required vLLM 0.25.1 CommonAttentionMetadata.is_prefilling "
            "field is missing"
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
        _V0251_PREPARE_ATTN_SOURCE_SHA256,
    )
    original_code = getattr(original_prepare_attn, "__code__", None)
    original_required_names = {
        "CUDAGraphMode",
        "build_attn_metadata",
        "compute_mm_prefix_ranges",
        "query_start_loc_np",
        "query_start_loc",
    }
    # is_prefilling_np is the one expected addition, so the pinned original
    # must contain every other audited name and must not already own that path.
    if original_code is None or not original_required_names.issubset(
        original_code.co_names
    ):
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGETS[0]} no longer matches the "
            "audited v0.25.1 metadata path"
        )
    if "is_prefilling_np" in original_code.co_names:
        raise PatchCompatibilityError(
            f"audited target vLLM API {TARGETS[0]} unexpectedly already "
            "propagates request-phase metadata"
        )

    build_attn_metadata = require_callable(
        default, "build_attn_metadata", TARGETS[1]
    )
    _validate_build_attn_metadata(build_attn_metadata)
    compute_mm_prefix_ranges = require_callable(
        default,
        "compute_mm_prefix_ranges",
        f"{TARGET_MODULE}.compute_mm_prefix_ranges",
    )
    torch = getattr(default, "torch", None)
    cudagraph_mode_class = getattr(default, "CUDAGraphMode", None)
    if torch is None or cudagraph_mode_class is None:
        raise PatchCompatibilityError(
            f"required HCU patch dependencies for {TARGETS[0]} are missing"
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
        pcp_size = int(
            self.vllm_config.parallel_config.prefill_context_parallel_size
        )
        if pcp_size == 1:
            return original_prepare_attn(
                self,
                input_batch,
                cudagraph_mode,
                block_tables,
                slot_mappings,
                attn_groups,
                kv_cache_config,
                for_capture,
            )

        if cudagraph_mode == cudagraph_mode_class.FULL:
            num_reqs = input_batch.num_reqs_after_padding
            num_tokens = input_batch.num_tokens_after_padding
        else:
            num_reqs = input_batch.num_reqs
            num_tokens = input_batch.num_tokens
        query_start_loc_cpu = torch.from_numpy(
            input_batch.query_start_loc_np[: num_reqs + 1]
        )
        query_start_loc_gpu = input_batch.query_start_loc[: num_reqs + 1]
        max_query_len = input_batch.num_scheduled_tokens.max().item()
        seq_lens_cpu_upper_bound = input_batch.seq_lens_cpu_upper_bound
        if for_capture:
            max_seq_len = self.max_model_len
        else:
            max_seq_len = seq_lens_cpu_upper_bound[:num_reqs].max().item()
        req_doc_ranges = None
        if (
            self.supports_mm_inputs
            and self.encoder_cache is not None
            and self.model_config.is_mm_prefix_lm
        ):
            req_doc_ranges = compute_mm_prefix_ranges(
                req_ids=input_batch.req_ids,
                mm_features=self.encoder_cache.mm_features,
                sliding_window=self.model_config.get_sliding_window(),
            )
        request_phase = _PCPRequestPhaseMetadata(
            torch.from_numpy(input_batch.is_prefilling_np),
            pcp_plan=getattr(input_batch, "_vllm_hcu_pcp_plan", None),
        )
        return build_attn_metadata(
            attn_groups=attn_groups,
            num_reqs=num_reqs,
            num_tokens=num_tokens,
            query_start_loc_gpu=query_start_loc_gpu,
            query_start_loc_cpu=query_start_loc_cpu,
            max_query_len=max_query_len,
            seq_lens=input_batch.seq_lens,
            max_seq_len=max_seq_len,
            block_tables=block_tables,
            slot_mappings=slot_mappings,
            kv_cache_config=kv_cache_config,
            seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
            dcp_local_seq_lens=input_batch.dcp_local_seq_lens,
            positions=input_batch.positions,
            mm_req_doc_ranges=req_doc_ranges,
            model_specific_attn_metadata=request_phase,
            for_cudagraph_capture=for_capture,
            rswa_prefix_lens=input_batch.prompt_lens,
        )

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
