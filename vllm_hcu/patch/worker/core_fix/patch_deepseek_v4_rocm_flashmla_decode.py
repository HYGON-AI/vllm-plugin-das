# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Opt-in FlashMLA sparse prefill/decode for native ROCm DeepSeek-V4."""

from __future__ import annotations

import functools
from types import ModuleType

import torch

from ._common import (
    PatchCompatibilityError,
    load_exact_module,
    require_callable,
    require_class,
    require_exact_signature,
)

TARGET_MODULE = "vllm.models.deepseek_v4.amd.rocm"
PATCH_ID = "worker.core_fix.deepseek_v4_rocm.flashmla_sparse_decode"
_MARKER = "_vllm_hcu_flashmla_sparse_decode_applied"


@functools.cache
def _require_flashmla_ready() -> None:
    """Fail loudly, once, if the FlashMLA sparse decode path cannot run here.

    ``is_flashmla_sparse_supported`` only proves the Python package imports.
    The decode call additionally needs the compiled ``sparse_decode_fwd``
    entry point, and the FP8 cache bytes must be OCP E4M3: FlashMLA decodes
    the 584-byte fp8_ds_mla rows as OCP, while the SWA cache writer follows
    ``current_platform.is_fp8_fnuz()`` (FNUZ on gfx942-class parts), which
    would silently misscale by ~1.87x.
    """
    from vllm_hcu.v1.attention.ops.flashmla import is_flashmla_sparse_supported

    supported, reason = is_flashmla_sparse_supported()
    if not supported:
        raise RuntimeError(f"DeepSeek-V4 FlashMLA decode unavailable: {reason}")
    import flash_mla.cuda as flash_mla_cuda

    if not callable(getattr(flash_mla_cuda, "sparse_decode_fwd", None)):
        raise RuntimeError(
            "DeepSeek-V4 FlashMLA decode unavailable: the installed flash_mla "
            "extension does not export sparse_decode_fwd"
        )
    from vllm.platforms import current_platform

    if current_platform.is_fp8_fnuz():
        raise RuntimeError(
            "DeepSeek-V4 FlashMLA decode unavailable: this platform stores the "
            "SWA cache as FNUZ FP8, but FlashMLA reads fp8_ds_mla rows as OCP"
        )


@functools.cache
def _require_flashmla_prefill_ready() -> None:
    """Validate the compiled sparse-prefill entry point before model execution."""
    from vllm_hcu.v1.attention.ops.flashmla import is_flashmla_sparse_supported

    supported, reason = is_flashmla_sparse_supported()
    if not supported:
        raise RuntimeError(f"DeepSeek-V4 FlashMLA prefill unavailable: {reason}")
    import flash_mla.cuda as flash_mla_cuda

    if not callable(getattr(flash_mla_cuda, "sparse_prefill_fwd", None)):
        raise RuntimeError(
            "DeepSeek-V4 FlashMLA prefill unavailable: the installed flash_mla "
            "extension does not export sparse_prefill_fwd"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("DeepSeek-V4 FlashMLA prefill requires an available ROCm device")
    arch = str(torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName)
    arch = arch.split(":", 1)[0]
    if arch not in ("gfx936", "gfx938"):
        raise RuntimeError(
            "DeepSeek-V4 FlashMLA prefill unavailable: no validated sparse "
            f"prefill kernel for {arch or 'unknown ROCm architecture'}"
        )


def apply_to_module(module: ModuleType) -> bool:
    rocm = load_exact_module(TARGET_MODULE, module)
    attention_cls = require_class(rocm, "DeepseekV4ROCMAiterMLAAttention", TARGET_MODULE)
    builder_cls = require_class(
        rocm, "DeepseekV4ROCMAiterSparseSWAMetadataBuilder", TARGET_MODULE
    )
    mla_builder_cls = require_class(
        rocm, "DeepseekV4ROCMAiterMLASparseMetadataBuilder", TARGET_MODULE
    )
    original_decode = require_callable(attention_cls, "_forward_decode", TARGET_MODULE)
    original_prefill = require_callable(attention_cls, "_forward_prefill", TARGET_MODULE)
    original_scheduler = require_callable(builder_cls, "build_tile_scheduler", TARGET_MODULE)
    original_swa_build = require_callable(builder_cls, "build", TARGET_MODULE)
    original_mla_build = require_callable(mla_builder_cls, "build", TARGET_MODULE)
    original_swa_init = require_callable(builder_cls, "__init__", TARGET_MODULE)
    original_mla_init = require_callable(mla_builder_cls, "__init__", TARGET_MODULE)
    if getattr(rocm, _MARKER, False):
        if not all(
            getattr(fn, _MARKER, False)
            for fn in (
                attention_cls._forward_decode,
                attention_cls._forward_prefill,
                builder_cls.build,
                builder_cls.build_tile_scheduler,
                builder_cls.__init__,
                mla_builder_cls.build,
                mla_builder_cls.__init__,
            )
        ):
            raise PatchCompatibilityError("stale DeepSeek-V4 FlashMLA sparse attention patch")
        return False
    require_exact_signature(
        original_decode,
        f"{TARGET_MODULE}.DeepseekV4ROCMAiterMLAAttention._forward_decode",
        positional=("self", "q", "kv_cache", "swa_metadata", "attn_metadata", "swa_only", "output"),
    )
    require_exact_signature(
        original_prefill,
        f"{TARGET_MODULE}.DeepseekV4ROCMAiterMLAAttention._forward_prefill",
        positional=(
            "self", "q", "positions", "compressed_k_cache", "swa_k_cache",
            "output", "attn_metadata", "swa_metadata",
        ),
    )
    require_exact_signature(
        original_scheduler,
        f"{TARGET_MODULE}.DeepseekV4ROCMAiterSparseSWAMetadataBuilder.build_tile_scheduler",
        positional=("self", "num_decode_tokens"),
    )
    for fn, name in (
        (original_swa_build, "DeepseekV4ROCMAiterSparseSWAMetadataBuilder.build"),
        (original_mla_build, "DeepseekV4ROCMAiterMLASparseMetadataBuilder.build"),
    ):
        require_exact_signature(
            fn,
            f"{TARGET_MODULE}.{name}",
            positional=("self", "common_prefix_len", "common_attn_metadata", "fast_build"),
            defaults={"fast_build": False},
        )

    @functools.wraps(original_swa_init)
    def swa_builder_init(self, *args, **kwargs):
        original_swa_init(self, *args, **kwargs)
        from vllm_hcu.platforms import envs as henvs

        if henvs.VLLM_HCU_DEEPSEEK_V4_ROCM_FLASHMLA_DECODE:
            # FlashMLA consumes the dense SWA indices; release the AITER-only
            # ragged buffers (max_tokens * window_size int32) right away.
            self.decode_swa_ragged_indices_buffer = None
            self.decode_swa_ragged_indptr_buffer = None

    @functools.wraps(original_mla_init)
    def mla_builder_init(self, *args, **kwargs):
        original_mla_init(self, *args, **kwargs)
        from vllm_hcu.platforms import envs as henvs

        if henvs.VLLM_HCU_DEEPSEEK_V4_ROCM_FLASHMLA_DECODE:
            self.c128a_decode_topk_ragged_indices_buffer = None
            self.c128a_decode_topk_ragged_indptr_buffer = None

    @functools.wraps(original_swa_build)
    def build_swa_metadata(self, common_prefix_len, common_attn_metadata, fast_build=False):
        from vllm_hcu.platforms import envs as henvs

        if not henvs.VLLM_HCU_DEEPSEEK_V4_ROCM_FLASHMLA_DECODE:
            return original_swa_build(self, common_prefix_len, common_attn_metadata, fast_build)
        # The base builder already supplies dense SWA indices and the tile
        # scheduler. The native subclass adds only AITER's ragged copy.
        base = rocm.DeepseekSparseSWAMetadataBuilder.build(
            self, common_prefix_len, common_attn_metadata, fast_build
        )
        return rocm.DeepseekV4ROCMAiterSparseSWAMetadata(**vars(base))

    @functools.wraps(original_mla_build)
    def build_mla_metadata(self, common_prefix_len, common_attn_metadata, fast_build=False):
        from vllm_hcu.platforms import envs as henvs

        if not henvs.VLLM_HCU_DEEPSEEK_V4_ROCM_FLASHMLA_DECODE:
            return original_mla_build(self, common_prefix_len, common_attn_metadata, fast_build)
        # The base builder already computes C128A dense global top-k and
        # lengths. The native subclass adds only AITER's ragged conversion.
        base = rocm.DeepseekV4FlashMLAMetadataBuilder.build(
            self, common_prefix_len, common_attn_metadata, fast_build
        )
        return rocm.DeepseekV4ROCMAiterMLASparseMetadata(**vars(base))

    @functools.wraps(original_scheduler)
    def build_tile_scheduler(self, num_decode_tokens):
        result = original_scheduler(self, num_decode_tokens)
        from vllm_hcu.platforms import envs as henvs

        if not henvs.VLLM_HCU_DEEPSEEK_V4_ROCM_FLASHMLA_DECODE or num_decode_tokens == 0:
            return result
        _require_flashmla_ready()
        from vllm_hcu.v1.attention.ops.flashmla import get_mla_metadata

        for layer_type in self._layer_types:
            result[layer_type] = get_mla_metadata()[0]
        return result

    @functools.wraps(original_decode)
    def forward_decode(self, q, kv_cache, swa_metadata, attn_metadata, swa_only, output):
        from vllm_hcu.platforms import envs as henvs

        if not henvs.VLLM_HCU_DEEPSEEK_V4_ROCM_FLASHMLA_DECODE:
            return original_decode(
                self, q, kv_cache, swa_metadata, attn_metadata, swa_only, output
            )

        _require_flashmla_ready()
        from vllm.models.deepseek_v4.common.ops import (
            compute_global_topk_indices_and_lens,
        )
        from vllm_hcu.v1.attention.ops.flashmla import flash_mla_with_kvcache

        num_tokens = swa_metadata.num_decode_tokens
        if q.ndim != 3 or q.shape[0] != num_tokens or q.shape[-1] != 512 or output.shape != q.shape:
            raise ValueError("FlashMLA decode requires one 512-wide output per query")
        if not (q.shape[1] <= 16 or q.shape[1] in (64, 128)):
            raise ValueError(
                "The installed FlashMLA FP8 sparse decode kernels cover local "
                f"query head counts <=16, 64, and 128; got {q.shape[1]}. "
                "TP2 (32 local heads) is the unsupported configuration."
            )
        if self.swa_cache_layer.kv_cache.dtype != torch.uint8 or self.swa_cache_layer.kv_cache.shape[-1] != 584:
            raise ValueError("FlashMLA decode requires 584-byte fp8_ds_mla SWA cache rows")
        swa_indices = swa_metadata.decode_swa_indices
        swa_lens = swa_metadata.decode_swa_lens
        if swa_indices is None or swa_lens is None:
            raise ValueError("FlashMLA decode requires dense SWA indices and lengths")
        if swa_indices.dtype != torch.int32 or swa_lens.dtype != torch.int32:
            raise TypeError("FlashMLA sparse indices and lengths must be int32")
        if swa_indices.shape[:2] != (num_tokens, 1):
            raise ValueError("FlashMLA SWA indices must have shape (tokens, 1, topk)")
        topk_indices = topk_lens = None
        if not swa_only:
            if kv_cache is None or attn_metadata is None:
                raise ValueError("FlashMLA compressed decode requires KV cache and metadata")
            if kv_cache.dtype != torch.uint8 or kv_cache.shape[-1] != 584:
                raise ValueError("FlashMLA decode requires 584-byte fp8_ds_mla compressed cache rows")
            if self.compress_ratio == 4:
                if self.topk_indices_buffer is None or swa_metadata.is_valid_token is None:
                    raise ValueError("C4A decode requires topk buffer and valid-token mask")
                topk_indices, topk_lens = compute_global_topk_indices_and_lens(
                    self.topk_indices_buffer[:num_tokens],
                    swa_metadata.token_to_req_indices,
                    attn_metadata.block_table[:swa_metadata.num_decodes],
                    attn_metadata.block_size // self.compress_ratio,
                    swa_metadata.is_valid_token[:num_tokens],
                )
                topk_indices = topk_indices.view(num_tokens, 1, -1)
            elif self.compress_ratio == 128:
                topk_indices = attn_metadata.c128a_global_decode_topk_indices
                topk_lens = attn_metadata.c128a_decode_topk_lens
            else:
                raise ValueError(f"Unsupported compress_ratio={self.compress_ratio}")
            if topk_indices is None or topk_lens is None:
                raise ValueError("FlashMLA decode requires dense global topk indices and lengths")
            if topk_indices.dtype != torch.int32 or topk_lens.dtype != torch.int32:
                raise TypeError("FlashMLA topk indices and lengths must be int32")
            if topk_indices.shape[:2] != (num_tokens, 1):
                raise ValueError("FlashMLA topk indices must have shape (tokens, 1, topk)")

        if swa_only:
            tile_metadata = swa_metadata.tile_sched_swaonly
        elif self.compress_ratio == 4:
            tile_metadata = swa_metadata.tile_sched_c4a
        else:
            tile_metadata = swa_metadata.tile_sched_c128a
        if tile_metadata is None:
            raise ValueError("FlashMLA tile scheduler metadata was not built")

        # Both caches store FP8 bytes. A singleton head dimension is a view,
        # so this preserves the native cache layout and graph replay addresses.
        swa_cache = self.swa_cache_layer.kv_cache.unsqueeze(-2)
        extra_cache = kv_cache.unsqueeze(-2) if kv_cache is not None else None
        out, _ = flash_mla_with_kvcache(
            q=q.unsqueeze(1),
            k_cache=swa_cache,
            block_table=None,
            cache_seqlens=None,
            head_dim_v=512,
            tile_scheduler_metadata=tile_metadata,
            is_fp8_kvcache=True,
            indices=swa_indices,
            topk_length=swa_lens,
            softmax_scale=self.scale,
            attn_sink=self.attn_sink,
            extra_k_cache=extra_cache,
            extra_indices_in_kvcache=topk_indices,
            extra_topk_length=topk_lens,
        )
        if out.squeeze(1).shape != output.shape:
            raise RuntimeError("FlashMLA returned an unexpected output shape")
        output.copy_(out.squeeze(1).to(output.dtype))

    @functools.wraps(original_prefill)
    def forward_prefill(
        self,
        q,
        positions,
        compressed_k_cache,
        swa_k_cache,
        output,
        attn_metadata,
        swa_metadata,
    ):
        from vllm_hcu.platforms import envs as henvs

        if not henvs.VLLM_HCU_DEEPSEEK_V4_ROCM_FLASHMLA_PREFILL:
            return original_prefill(
                self, q, positions, compressed_k_cache, swa_k_cache, output,
                attn_metadata, swa_metadata,
            )

        _require_flashmla_prefill_ready()
        from vllm_hcu.v1.attention.ops.flashmla import flash_mla_sparse_fwd

        swa_only = attn_metadata is None
        num_prefills = swa_metadata.num_prefills
        num_decode_tokens = swa_metadata.num_decode_tokens
        seq_lens = swa_metadata.prefill_seq_lens
        gather_lens = swa_metadata.prefill_gather_lens
        query_start_loc_cpu = swa_metadata.query_start_loc_cpu
        query_start_loc = swa_metadata.query_start_loc
        if seq_lens is None or gather_lens is None:
            raise ValueError("FlashMLA prefill requires sequence and gather lengths")
        if query_start_loc_cpu is None or query_start_loc is None:
            raise ValueError("FlashMLA prefill requires query start locations")
        if q.ndim != 3 or q.shape[-1] != 512 or output.shape != q.shape:
            raise ValueError("FlashMLA prefill requires [tokens, heads, 512] q/output")
        if q.dtype != torch.bfloat16:
            raise TypeError(f"FlashMLA sparse prefill requires bfloat16 q, got {q.dtype}")

        prefill_token_base = query_start_loc_cpu[swa_metadata.num_decodes]
        if swa_only:
            if self.topk_indices_buffer is None:
                raise ValueError("SWA-only prefill requires the shared topk buffer")
            topk_indices = self.topk_indices_buffer[num_decode_tokens:]
            top_k = 0
            compressed_pool_size = 0
        else:
            if compressed_k_cache is None:
                raise ValueError("compressed prefill requires its KV cache")
            if self.compress_ratio == 4:
                if self.topk_indices_buffer is None:
                    raise ValueError("C4A prefill requires the topk buffer")
                topk_indices = self.topk_indices_buffer[num_decode_tokens:]
                topk_indices = topk_indices[:swa_metadata.num_prefill_tokens]
            elif self.compress_ratio == 128:
                topk_indices = attn_metadata.c128a_prefill_topk_indices
            else:
                raise ValueError(f"Unsupported compress_ratio={self.compress_ratio}")
            if topk_indices is None:
                raise ValueError("FlashMLA prefill requires dense topk indices")
            top_k = topk_indices.shape[-1]
            compressed_pool_size = (
                self.max_model_len + self.compress_ratio - 1
            ) // self.compress_ratio

        workspace_width = (
            compressed_pool_size + self.window_size + self.max_num_batched_tokens
        )
        workspace = rocm.current_workspace_manager().get_simultaneous(
            ((self.PREFILL_CHUNK_SIZE, workspace_width, q.shape[-1]), torch.bfloat16),
        )[0]
        num_chunks = (
            num_prefills + self.PREFILL_CHUNK_SIZE - 1
        ) // self.PREFILL_CHUNK_SIZE
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * self.PREFILL_CHUNK_SIZE
            chunk_end = min(chunk_start + self.PREFILL_CHUNK_SIZE, num_prefills)
            chunk_size = chunk_end - chunk_start
            if not swa_only:
                block_table = attn_metadata.block_table[swa_metadata.num_decodes:]
                rocm.dequantize_and_gather_k_cache(
                    workspace[:chunk_size],
                    compressed_k_cache,
                    seq_lens=seq_lens[chunk_start:chunk_end] // self.compress_ratio,
                    gather_lens=None,
                    block_table=block_table[chunk_start:chunk_end],
                    block_size=attn_metadata.block_size // self.compress_ratio,
                    offset=0,
                    use_fnuz=False,
                )

            swa_block_table = swa_metadata.block_table[swa_metadata.num_decodes:]
            rocm.dequantize_and_gather_k_cache(
                workspace[:chunk_size],
                swa_k_cache,
                seq_lens=seq_lens[chunk_start:chunk_end],
                gather_lens=gather_lens[chunk_start:chunk_end],
                block_table=swa_block_table[chunk_start:chunk_end],
                block_size=swa_metadata.block_size,
                offset=compressed_pool_size,
                use_fnuz=rocm.current_platform.is_fp8_fnuz(),
            )

            query_start = (
                query_start_loc_cpu[swa_metadata.num_decodes + chunk_start]
                - prefill_token_base
            )
            query_end = (
                query_start_loc_cpu[swa_metadata.num_decodes + chunk_end]
                - prefill_token_base
            )
            combined_indices, combined_lens = rocm.combine_topk_swa_indices(
                topk_indices[query_start:query_end],
                query_start_loc[
                    swa_metadata.num_decodes + chunk_start:
                    swa_metadata.num_decodes + chunk_end + 1
                ],
                seq_lens[chunk_start:chunk_end],
                gather_lens[chunk_start:chunk_end],
                self.window_size,
                self.compress_ratio,
                top_k,
                workspace_width,
                compressed_pool_size,
            )
            chunk_output, _, _ = flash_mla_sparse_fwd(
                q=q[query_start:query_end],
                kv=workspace.view(-1, 1, q.shape[-1]),
                indices=combined_indices.unsqueeze(1),
                sm_scale=self.scale,
                d_v=512,
                attn_sink=self.attn_sink,
                topk_length=combined_lens,
            )
            output[query_start:query_end].copy_(chunk_output.to(output.dtype))

    for fn in (
        forward_decode,
        forward_prefill,
        build_swa_metadata,
        build_mla_metadata,
        build_tile_scheduler,
        swa_builder_init,
        mla_builder_init,
    ):
        setattr(fn, _MARKER, True)
    setattr(attention_cls, "_forward_decode", forward_decode)
    setattr(attention_cls, "_forward_prefill", forward_prefill)
    setattr(builder_cls, "build_tile_scheduler", build_tile_scheduler)
    setattr(builder_cls, "build", build_swa_metadata)
    setattr(builder_cls, "__init__", swa_builder_init)
    setattr(mla_builder_cls, "build", build_mla_metadata)
    setattr(mla_builder_cls, "__init__", mla_builder_init)
    setattr(rocm, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "apply", "apply_to_module"]
