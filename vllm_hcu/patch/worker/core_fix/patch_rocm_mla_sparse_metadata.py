# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Keep supported Triton sparse MLA routes independent of AITER helpers."""

from __future__ import annotations

import functools
import importlib
from types import ModuleType

import torch

from ._common import (
    PatchCompatibilityError,
    load_exact_module,
    require_callable,
    require_class,
    require_exact_signature,
)

TARGET_MODULE = "vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse"
PATCH_ID = "worker.core_fix.rocm_mla_sparse.triton_metadata_fallback"
_MARKER = "_vllm_hcu_rocm_mla_sparse_metadata_fallback_applied"
_WRAPPER_MARKER = "_vllm_hcu_rocm_mla_sparse_metadata_init_wrapper"
_ROUTE_WRAPPER_MARKER = "_vllm_hcu_rocm_mla_sparse_mtp_route_wrapper"


def _empty_metadata_info(*args, **kwargs):
    del args, kwargs
    return ((0, torch.int32),) * 6


def apply_to_module(module: ModuleType) -> bool:
    sparse = load_exact_module(TARGET_MODULE, module)
    builder = require_class(
        sparse,
        "ROCMAiterMLASparseMetadataBuilder",
        f"{TARGET_MODULE}.ROCMAiterMLASparseMetadataBuilder",
    )
    original = require_callable(
        builder,
        "__init__",
        f"{TARGET_MODULE}.ROCMAiterMLASparseMetadataBuilder.__init__",
    )
    original_route = require_callable(
        sparse,
        "_use_rocm_sparse_triton",
        f"{TARGET_MODULE}._use_rocm_sparse_triton",
    )
    if getattr(sparse, _MARKER, False):
        if (
            not getattr(vars(builder).get("__init__"), _WRAPPER_MARKER, False)
            or not getattr(original_route, _ROUTE_WRAPPER_MARKER, False)
        ):
            raise PatchCompatibilityError(
                f"required HCU patch marker for {TARGET_MODULE} is stale"
            )
        return False

    require_exact_signature(
        original_route,
        f"{TARGET_MODULE}._use_rocm_sparse_triton",
        keyword_only=(
            "kv_cache_dtype",
            "head_size",
            "kv_lora_rank",
            "num_prefills",
            "num_decodes",
            "num_decode_tokens",
            "max_query_len",
        ),
    )

    @functools.wraps(original_route)
    def hcu_use_rocm_sparse_triton(
        *,
        kv_cache_dtype,
        head_size,
        kv_lora_rank,
        num_prefills,
        num_decodes,
        num_decode_tokens,
        max_query_len,
    ):
        # Backport vLLM #55239. The ragged Triton kernel indexes metadata per
        # query token, so MTP verification rows have the same capability
        # requirements as ordinary decode rows. The older selector excluded
        # them and sent BF16 NoPE verification through an incompatible AITER
        # sparse-MLA kernel.
        del num_decode_tokens, max_query_len
        return (
            not kv_cache_dtype.startswith("fp8")
            and head_size == kv_lora_rank
            and (num_prefills > 0 or num_decodes > 0)
        )

    @functools.wraps(original)
    def hcu_builder_init(self, *args, **kwargs):
        aiter = importlib.import_module("aiter")
        if hasattr(aiter, "get_mla_metadata_info_v1"):
            return original(self, *args, **kwargs)

        # This HCU AITER build intentionally carries MoE kernels but not the
        # upstream sparse-MLA metadata API.  The rope-free BF16 GLM path uses
        # vLLM's Triton sparse attention, so persistent AITER metadata is dead
        # state.  Let upstream initialize its remaining buffers unchanged.
        aiter.get_mla_metadata_info_v1 = _empty_metadata_info
        try:
            original(self, *args, **kwargs)
        finally:
            if getattr(aiter, "get_mla_metadata_info_v1", None) is _empty_metadata_info:
                delattr(aiter, "get_mla_metadata_info_v1")
        self._use_persistent_metadata = False

    setattr(hcu_builder_init, _WRAPPER_MARKER, True)
    setattr(hcu_use_rocm_sparse_triton, _ROUTE_WRAPPER_MARKER, True)
    setattr(builder, "_vllm_hcu_original_init", original)
    setattr(builder, "__init__", hcu_builder_init)
    setattr(sparse, "_vllm_hcu_original_use_rocm_sparse_triton", original_route)
    setattr(sparse, "_use_rocm_sparse_triton", hcu_use_rocm_sparse_triton)
    setattr(sparse, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "apply", "apply_to_module"]
