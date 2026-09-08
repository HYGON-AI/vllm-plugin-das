# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""HCU sparse FlashMLA adapter selected through the official backend registry."""

from __future__ import annotations

import functools
from types import ModuleType

from ._common import (
    PatchCompatibilityError,
    already_applied,
    load_exact_module,
    require_callable,
    require_class,
    require_exact_signature,
)

TARGET_MODULE = "vllm.v1.attention.backends.mla.flashmla_sparse"
PATCH_ID = "worker.op_opt.mla.flashmla_sparse_hcu"
TARGETS = (
    f"{TARGET_MODULE}.FlashMLASparseMetadataBuilder.build",
    f"{TARGET_MODULE}.FlashMLASparseImpl._fp8_flash_mla_kernel",
    f"{TARGET_MODULE}.FlashMLASparseImpl._bf16_flash_mla_kernel",
    f"{TARGET_MODULE}.FlashMLASparseMetadataBuilder._build_fp8_separate_prefill_decode",
)
_MARKER = "_vllm_hcu_flashmla_sparse_applied"
_WRAPPER = "_vllm_hcu_flashmla_sparse_wrapper"


def apply_to_module(module: ModuleType) -> bool:
    flash = load_exact_module(TARGET_MODULE, module)
    builder_cls = require_class(flash, "FlashMLASparseMetadataBuilder", f"{TARGET_MODULE}.FlashMLASparseMetadataBuilder")
    impl_cls = require_class(flash, "FlashMLASparseImpl", f"{TARGET_MODULE}.FlashMLASparseImpl")
    wrapped = (
        (builder_cls, "build", TARGETS[0], _WRAPPER),
        (impl_cls, "_fp8_flash_mla_kernel", TARGETS[1], _WRAPPER),
        (impl_cls, "_bf16_flash_mla_kernel", TARGETS[2], _WRAPPER),
    )
    if already_applied(flash, _MARKER, wrapped):
        return False
    build = require_callable(builder_cls, "build", TARGETS[0])
    require_exact_signature(
        build, TARGETS[0],
        positional=("self", "common_prefix_len", "common_attn_metadata", "fast_build"),
        defaults={"fast_build": False},
    )
    fp8 = require_callable(impl_cls, "_fp8_flash_mla_kernel", TARGETS[1])
    require_exact_signature(
        fp8, TARGETS[1],
        positional=("self", "q", "kv_c_and_k_pe_cache", "topk_indices", "kernel_metadata"),
    )
    bf16 = require_callable(impl_cls, "_bf16_flash_mla_kernel", TARGETS[2])
    require_exact_signature(
        bf16, TARGETS[2],
        positional=(
            "self",
            "q",
            "kv_c_and_k_pe_cache",
            "topk_indices",
            "topk_length",
            "actual_num_heads",
        ),
        defaults={"topk_length": None, "actual_num_heads": None},
    )
    build_fp8_separate_prefill_decode = require_callable(
        builder_cls, "_build_fp8_separate_prefill_decode", TARGETS[3]
    )
    require_exact_signature(
        build_fp8_separate_prefill_decode,
        TARGETS[3],
        positional=("self", "common_attn_metadata", "metadata"),
    )

    # Validate and collect every replacement before mutating the target module.
    from vllm_hcu.v1.attention.ops import flashmla as hcu_flashmla

    hcu_bindings = {}
    for name in (
        "FlashMLASchedMeta",
        "flash_mla_sparse_fwd",
        "flash_mla_with_kvcache",
        "get_mla_metadata",
    ):
        value = getattr(hcu_flashmla, name, None)
        if value is None:
            raise RuntimeError(
                f"required HCU FlashMLA symbol {name} is unavailable"
            )
        hcu_bindings[name] = value

    @functools.wraps(build)
    def hcu_build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        result = build(self, common_prefix_len, common_attn_metadata, fast_build)
        from vllm_hcu.platforms import envs as henvs
        from vllm_hcu.model_executor.layers.attention.pcp import (
            effective_pcp_world_size,
        )

        vllm_config = getattr(self, "vllm_config", None)
        if vllm_config is None:
            pcp_world_size = int(
                getattr(common_attn_metadata, "pcp_world_size", 1)
            )
            cp_kv_cache_interleave_size = int(
                getattr(
                    common_attn_metadata,
                    "cp_kv_cache_interleave_size",
                    1,
                )
            )
        else:
            parallel_config = getattr(vllm_config, "parallel_config", None)
            pcp_world_size = getattr(
                parallel_config,
                "prefill_context_parallel_size",
                None,
            )
            if pcp_world_size is None:
                raise PatchCompatibilityError(
                    "required vLLM 0.25.1 prefill_context_parallel_size "
                    "is missing from sparse MLA metadata builder"
                )
            pcp_world_size = int(pcp_world_size)
            cp_kv_cache_interleave_size = getattr(
                parallel_config,
                "cp_kv_cache_interleave_size",
                None,
            )
            if cp_kv_cache_interleave_size is None:
                raise PatchCompatibilityError(
                    "required vLLM 0.25.1 cp_kv_cache_interleave_size "
                    "is missing from sparse MLA metadata builder"
                )
            cp_kv_cache_interleave_size = int(cp_kv_cache_interleave_size)
        pcp_world_size = effective_pcp_world_size(pcp_world_size)
        if not henvs.VLLM_HCU_USE_FP8_MIXED_BATCH:
            if getattr(result, "fp8_use_mixed_batch", False):
                result.fp8_extra_metadata = build_fp8_separate_prefill_decode(
                    self,
                    common_attn_metadata,
                    result,
                )
                result.fp8_use_mixed_batch = False
        result.num_kv_actual_tokens = getattr(
            common_attn_metadata,
            "num_kv_actual_tokens",
            common_attn_metadata.num_actual_tokens,
        )
        result.pcp_world_size = pcp_world_size
        result.cp_kv_cache_interleave_size = cp_kv_cache_interleave_size
        return result

    @functools.wraps(fp8)
    def hcu_fp8(self, q, kv_c_and_k_pe_cache, topk_indices, kernel_metadata):
        if not flash.current_platform.is_rocm():
            return fp8(self, q, kv_c_and_k_pe_cache, topk_indices, kernel_metadata)
        out, lse = flash.flash_mla_with_kvcache(
            q=q,
            k_cache=kv_c_and_k_pe_cache.view(flash.torch.uint8).unsqueeze(-2),
            block_table=kernel_metadata.dummy_block_table,
            head_dim_v=512,
            cache_seqlens=kernel_metadata.cache_lens,
            tile_scheduler_metadata=kernel_metadata.scheduler_metadata,
            is_fp8_kvcache=True,
            indices=topk_indices,
            softmax_scale=self.softmax_scale,
        )
        return out, lse

    @functools.wraps(bf16)
    def hcu_bf16(
        self,
        q,
        kv_c_and_k_pe_cache,
        topk_indices,
        topk_length=None,
        actual_num_heads=None,
    ):
        if not flash.current_platform.is_rocm():
            return bf16(
                self,
                q,
                kv_c_and_k_pe_cache,
                topk_indices,
                topk_length,
                actual_num_heads,
            )
        if actual_num_heads is None:
            actual_num_heads = q.shape[1]
        num_tokens = q.shape[0]
        cache = kv_c_and_k_pe_cache.view(-1, 1, kv_c_and_k_pe_cache.shape[-1])
        indices = topk_indices.view(num_tokens, 1, -1)
        output, _, lse = flash.flash_mla_sparse_fwd(
            q,
            cache,
            indices,
            self.softmax_scale,
            topk_length=topk_length,
        )
        return output[:, :actual_num_heads, :], lse[:, :actual_num_heads]

    for function in (hcu_build, hcu_fp8, hcu_bf16):
        setattr(function, _WRAPPER, True)
    # Apply all target mutations only after validation and wrapper construction.
    # The official backend class remains registered and no module alias is used.
    for name, value in hcu_bindings.items():
        setattr(flash, name, value)
    setattr(builder_cls, "_vllm_hcu_original_build", build)
    setattr(impl_cls, "_vllm_hcu_original_fp8_kernel", fp8)
    setattr(impl_cls, "_vllm_hcu_original_bf16_kernel", bf16)
    setattr(builder_cls, "build", hcu_build)
    setattr(impl_cls, "_fp8_flash_mla_kernel", hcu_fp8)
    setattr(impl_cls, "_bf16_flash_mla_kernel", hcu_bf16)
    setattr(flash, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
