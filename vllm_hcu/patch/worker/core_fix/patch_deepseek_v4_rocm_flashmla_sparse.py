# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Sparse prefill/decode for native ROCm DeepSeek-V4.

Both paths require VLLM_HCU_USE_CUSTOM_OPS and their individual enable flags.
VLLM_HCU_DEEPSEEK_V4_ROCM_SPARSE_PREFILL_BACKEND selects "flashmla" (default)
or "boltops" for prefill only. Decode always requires native FlashMLA.
Set these variables before starting the worker; providers are cached.
"""

from __future__ import annotations

import functools
import inspect
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
PATCH_ID = "worker.core_fix.deepseek_v4_rocm.flashmla_sparse"
_DECODE_MARKER = "_vllm_hcu_flashmla_sparse_decode_applied"
_PREFILL_MARKER = "_vllm_hcu_flashmla_sparse_prefill_applied"


def _flashmla_prefill_padded_heads(num_heads: int) -> int | None:
    """Return the sparse-prefill kernel width for a TP-local Q layout."""
    if 0 < num_heads <= 64:
        return 64
    if 64 < num_heads <= 128:
        return 128
    return None


def _flashmla_decode_enabled() -> bool:
    from vllm_hcu.platforms import envs as henvs

    return henvs.optional_custom_op_enabled(
        henvs.VLLM_HCU_DEEPSEEK_V4_ROCM_FLASHMLA_DECODE
    )


def _flashmla_prefill_enabled() -> bool:
    from vllm_hcu.platforms import envs as henvs

    return henvs.optional_custom_op_enabled(
        henvs.VLLM_HCU_DEEPSEEK_V4_ROCM_FLASHMLA_PREFILL
    )


def _prefill_backend() -> str:
    from vllm_hcu.platforms import envs as henvs

    backend = henvs.VLLM_HCU_DEEPSEEK_V4_ROCM_SPARSE_PREFILL_BACKEND
    if backend not in ("flashmla", "boltops"):
        raise ValueError(
            "VLLM_HCU_DEEPSEEK_V4_ROCM_SPARSE_PREFILL_BACKEND must be "
            f"flashmla or boltops, got {backend!r}"
        )
    return backend


def _flashmla_decode_supports_heads(num_heads: int) -> bool:
    return 0 < num_heads <= 16 or num_heads in (64, 128)


def _builder_uses_flashmla_decode(builder) -> bool:
    config = builder.vllm_config
    # BF16/auto caches must retain the native metadata and decode path.
    if config.cache_config.cache_dtype != "fp8_ds_mla":
        return False
    num_heads = config.model_config.hf_config.num_attention_heads
    tp_size = config.parallel_config.tensor_parallel_size
    return _flashmla_decode_supports_heads(num_heads // tp_size)


def _require_flashmla_sparse_kernel(path: str) -> None:
    from vllm_hcu.v1.attention.ops.flashmla import is_flashmla_sparse_supported

    supported, reason = is_flashmla_sparse_supported()
    if not supported:
        raise RuntimeError(f"DeepSeek-V4 FlashMLA {path} unavailable: {reason}")
    import flash_mla.cuda as flash_mla_cuda

    entry_point = f"sparse_{path}_fwd"
    if not callable(getattr(flash_mla_cuda, entry_point, None)):
        raise RuntimeError(
            f"DeepSeek-V4 FlashMLA {path} unavailable: the installed flash_mla "
            f"extension does not export {entry_point}"
        )


@functools.cache
def _require_flashmla_ready() -> None:
    # Packed cache bytes must be OCP E4M3; FNUZ silently changes their scale.
    _require_flashmla_sparse_kernel("decode")
    from vllm.platforms import current_platform

    if current_platform.is_fp8_fnuz():
        raise RuntimeError(
            "DeepSeek-V4 FlashMLA decode unavailable: this platform stores the "
            "SWA cache as FNUZ FP8, but FlashMLA reads fp8_ds_mla rows as OCP"
        )


@functools.cache
def _require_flashmla_prefill_ready() -> None:
    """Validate the selected sparse-prefill provider before model execution."""
    if _prefill_backend() == "boltops":
        from vllm_hcu.v1.attention.ops.flashmla import _resolve_sparse_mla_fwd

        # Reuse the dispatcher's BoltOPs availability and signature checks.
        _resolve_sparse_mla_fwd("boltops")
        return
    _require_flashmla_sparse_kernel("prefill")
    if not torch.cuda.is_available():
        raise RuntimeError("DeepSeek-V4 FlashMLA prefill requires an available ROCm device")
    arch = str(torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName)
    arch = arch.split(":", 1)[0]
    if arch not in ("gfx936", "gfx938"):
        raise RuntimeError(
            "DeepSeek-V4 FlashMLA prefill unavailable: no validated sparse "
            f"prefill kernel for {arch or 'unknown ROCm architecture'}"
        )


def _apply_decode_to_module(module: ModuleType) -> bool:
    if not _flashmla_decode_enabled():
        return False
    rocm = load_exact_module(TARGET_MODULE, module)
    attention_cls = require_class(rocm, "DeepseekV4ROCMAiterMLAAttention", TARGET_MODULE)
    builder_cls = require_class(
        rocm, "DeepseekV4ROCMAiterSparseSWAMetadataBuilder", TARGET_MODULE
    )
    mla_builder_cls = require_class(
        rocm, "DeepseekV4ROCMAiterMLASparseMetadataBuilder", TARGET_MODULE
    )
    original_decode = require_callable(attention_cls, "_forward_decode", TARGET_MODULE)
    original_scheduler = require_callable(builder_cls, "build_tile_scheduler", TARGET_MODULE)
    original_swa_build = require_callable(builder_cls, "build", TARGET_MODULE)
    original_mla_build = require_callable(mla_builder_cls, "build", TARGET_MODULE)
    original_swa_init = require_callable(builder_cls, "__init__", TARGET_MODULE)
    original_mla_init = require_callable(mla_builder_cls, "__init__", TARGET_MODULE)
    if getattr(rocm, _DECODE_MARKER, False):
        if not all(
            getattr(fn, _DECODE_MARKER, False)
            for fn in (
                attention_cls._forward_decode,
                builder_cls.build,
                builder_cls.build_tile_scheduler,
                builder_cls.__init__,
                mla_builder_cls.build,
                mla_builder_cls.__init__,
            )
        ):
            raise PatchCompatibilityError("stale DeepSeek-V4 FlashMLA sparse attention patch")
        return False
    decode_has_adaptive_splits = (
        "adaptive_splits" in inspect.signature(original_decode).parameters
    )
    require_exact_signature(
        original_decode,
        f"{TARGET_MODULE}.DeepseekV4ROCMAiterMLAAttention._forward_decode",
        positional=("self", "q", "kv_cache", "swa_metadata", "attn_metadata", "swa_only", "output")
        + (("adaptive_splits",) if decode_has_adaptive_splits else ()),
    )
    require_exact_signature(
        original_scheduler,
        f"{TARGET_MODULE}.DeepseekV4ROCMAiterSparseSWAMetadataBuilder.build_tile_scheduler",
        positional=("self", "num_decode_tokens"),
    )
    swa_has_replay_start = (
        "replay_start" in inspect.signature(original_swa_build).parameters
    )
    require_exact_signature(
        original_swa_build,
        f"{TARGET_MODULE}.DeepseekV4ROCMAiterSparseSWAMetadataBuilder.build",
        positional=("self", "common_prefix_len", "common_attn_metadata", "fast_build")
        + (("replay_start",) if swa_has_replay_start else ()),
        defaults={
            "fast_build": False,
            **({"replay_start": None} if swa_has_replay_start else {}),
        },
    )
    require_exact_signature(
        original_mla_build,
        f"{TARGET_MODULE}.DeepseekV4ROCMAiterMLASparseMetadataBuilder.build",
        positional=("self", "common_prefix_len", "common_attn_metadata", "fast_build"),
        defaults={"fast_build": False},
    )

    @functools.wraps(original_swa_init)
    def swa_builder_init(self, *args, **kwargs):
        original_swa_init(self, *args, **kwargs)
        if (
            _flashmla_decode_enabled()
            and _builder_uses_flashmla_decode(self)
        ):
            # FlashMLA consumes the dense SWA indices; release the AITER-only
            # ragged buffers (max_tokens * window_size int32) right away.
            self.decode_swa_ragged_indices_buffer = None
            self.decode_swa_ragged_indptr_buffer = None

    @functools.wraps(original_mla_init)
    def mla_builder_init(self, *args, **kwargs):
        original_mla_init(self, *args, **kwargs)
        if (
            _flashmla_decode_enabled()
            and _builder_uses_flashmla_decode(self)
        ):
            self.c128a_decode_topk_ragged_indices_buffer = None
            self.c128a_decode_topk_ragged_indptr_buffer = None

    @functools.wraps(original_swa_build)
    def build_swa_metadata(
        self, common_prefix_len, common_attn_metadata, fast_build=False, replay_start=None
    ):
        replay_kwargs = {"replay_start": replay_start} if swa_has_replay_start else {}
        if replay_start is not None and not swa_has_replay_start:
            raise PatchCompatibilityError("this vLLM SWA builder does not support replay_start")
        if (
            not _flashmla_decode_enabled()
            or not _builder_uses_flashmla_decode(self)
        ):
            return original_swa_build(
                self, common_prefix_len, common_attn_metadata, fast_build, **replay_kwargs
            )
        # The base builder already supplies dense SWA indices and the tile
        # scheduler. The native subclass adds only AITER's ragged copy.
        base = rocm.DeepseekSparseSWAMetadataBuilder.build(
            self, common_prefix_len, common_attn_metadata, fast_build, **replay_kwargs
        )
        return rocm.DeepseekV4ROCMAiterSparseSWAMetadata(**vars(base))

    @functools.wraps(original_mla_build)
    def build_mla_metadata(self, common_prefix_len, common_attn_metadata, fast_build=False):
        if (
            not _flashmla_decode_enabled()
            or not _builder_uses_flashmla_decode(self)
        ):
            return original_mla_build(self, common_prefix_len, common_attn_metadata, fast_build)
        # The base builder already computes C128A dense global top-k and
        # lengths. The native subclass adds only AITER's ragged conversion.
        base = rocm.DeepseekV4SparseMLAMetadataBuilder.build(
            self, common_prefix_len, common_attn_metadata, fast_build
        )
        return rocm.DeepseekV4ROCMAiterMLASparseMetadata(**vars(base))

    @functools.wraps(original_scheduler)
    def build_tile_scheduler(self, num_decode_tokens):
        result = original_scheduler(self, num_decode_tokens)
        if (
            not _flashmla_decode_enabled()
            or not _builder_uses_flashmla_decode(self)
            or num_decode_tokens == 0
        ):
            return result
        _require_flashmla_ready()
        from vllm_hcu.v1.attention.ops.flashmla import get_mla_metadata

        for layer_type in self._layer_types:
            result[layer_type] = get_mla_metadata()[0]
        return result

    @functools.wraps(original_decode)
    def forward_decode(
        self, q, kv_cache, swa_metadata, attn_metadata, swa_only, output,
        adaptive_splits=False,
    ):
        decode_kwargs = (
            {"adaptive_splits": adaptive_splits} if decode_has_adaptive_splits else {}
        )

        if (
            not _flashmla_decode_enabled()
            or self.swa_cache_layer.kv_cache.dtype != torch.uint8
            or self.swa_cache_layer.kv_cache.shape[-1] != 584
            or (q.ndim == 3 and not _flashmla_decode_supports_heads(q.shape[1]))
        ):
            return original_decode(
                self, q, kv_cache, swa_metadata, attn_metadata, swa_only, output, **decode_kwargs
            )
        _require_flashmla_ready()
        from vllm.models.deepseek_v4.common.ops import (
            compute_global_topk_indices_and_lens,
        )
        from vllm_hcu.v1.attention.ops.flashmla import flash_mla_with_kvcache

        num_tokens = swa_metadata.num_decode_tokens
        if q.ndim != 3 or q.shape[0] != num_tokens or q.shape[-1] != 512 or output.shape != q.shape:
            raise ValueError("FlashMLA decode requires one 512-wide output per query")
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
                raise ValueError(
                    "FlashMLA decode requires 584-byte fp8_ds_mla compressed cache rows"
                )
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
        out = out.squeeze(1)
        if out.shape != output.shape:
            raise RuntimeError("FlashMLA returned an unexpected output shape")
        output.copy_(out.to(output.dtype))

    for fn in (
        forward_decode,
        build_swa_metadata,
        build_mla_metadata,
        build_tile_scheduler,
        swa_builder_init,
        mla_builder_init,
    ):
        setattr(fn, _DECODE_MARKER, True)
    setattr(attention_cls, "_forward_decode", forward_decode)
    setattr(builder_cls, "build_tile_scheduler", build_tile_scheduler)
    setattr(builder_cls, "build", build_swa_metadata)
    setattr(builder_cls, "__init__", swa_builder_init)
    setattr(mla_builder_cls, "build", build_mla_metadata)
    setattr(mla_builder_cls, "__init__", mla_builder_init)
    setattr(rocm, _DECODE_MARKER, True)
    return True


def _apply_prefill_to_module(module: ModuleType) -> bool:
    if not _flashmla_prefill_enabled():
        return False
    rocm = load_exact_module(TARGET_MODULE, module)
    original = require_callable(rocm, "rocm_sparse_attn_prefill", TARGET_MODULE)
    if getattr(rocm, _PREFILL_MARKER, False):
        if not getattr(original, _PREFILL_MARKER, False):
            raise PatchCompatibilityError("stale DeepSeek-V4 FlashMLA sparse prefill patch")
        return False
    require_exact_signature(
        original,
        f"{TARGET_MODULE}.rocm_sparse_attn_prefill",
        positional=(
            "q", "kv", "indices", "topk_length", "scale", "head_dim",
            "nope_head_dim", "rope_head_dim", "attn_sink", "output",
            "ragged_indices", "ragged_indptr",
        ),
        defaults={"ragged_indices": None, "ragged_indptr": None},
    )

    @functools.wraps(original)
    def sparse_prefill(
        q, kv, indices, topk_length, scale, head_dim, nope_head_dim,
        rope_head_dim, attn_sink, output, ragged_indices=None, ragged_indptr=None,
    ):
        # Sparse prefill kernels require 64/128 heads; pad TP-local queries.
        padded_heads = (
            _flashmla_prefill_padded_heads(q.shape[1]) if q.ndim == 3 else None
        )
        if (
            not _flashmla_prefill_enabled()
            or indices is None or topk_length is None or attn_sink is None
            or (
                q.ndim == 3
                and (
                    padded_heads is None
                    or q.shape[-1] != 512
                    or q.dtype != torch.bfloat16
                )
            )
        ):
            return original(
                q, kv, indices, topk_length, scale, head_dim, nope_head_dim,
                rope_head_dim, attn_sink, output, ragged_indices, ragged_indptr,
            )
        _require_flashmla_prefill_ready()
        from vllm_hcu.v1.attention.ops.flashmla import flash_mla_sparse_fwd

        if q.ndim != 3 or q.shape[-1] != 512 or output.shape != q.shape:
            raise ValueError("FlashMLA prefill requires [tokens, heads, 512] q/output")
        if kv.ndim != 3 or kv.shape[-2:] != (1, 512):
            raise ValueError("FlashMLA prefill requires [tokens, 1, 512] KV")
        if indices.ndim != 2 or indices.shape[0] != q.shape[0]:
            raise ValueError("FlashMLA prefill requires one dense index row per query")
        if topk_length.shape != (q.shape[0],):
            raise ValueError("FlashMLA prefill requires one top-k length per query")
        if attn_sink.ndim != 1 or attn_sink.shape[0] != q.shape[1]:
            raise ValueError("FlashMLA prefill requires one attention sink per query head")

        actual_heads = q.shape[1]
        assert padded_heads is not None
        if actual_heads != padded_heads:
            q_padded = q.new_zeros((q.shape[0], padded_heads, q.shape[2]))
            q_padded[:, :actual_heads].copy_(q)
            q = q_padded

            # Padded heads are discarded, but FlashMLA still requires the sink
            # tensor to match h_q. Zero-fill mirrors SGLang's DSV4 TP padding.
            sink_padded = attn_sink.new_zeros((padded_heads,))
            sink_padded[:actual_heads].copy_(attn_sink)
            attn_sink = sink_padded
        chunk_output, _, _ = flash_mla_sparse_fwd(
            q=q,
            kv=kv,
            indices=indices.unsqueeze(1),
            sm_scale=scale,
            d_v=head_dim,
            attn_sink=attn_sink,
            topk_length=topk_length,
            backend=_prefill_backend(),
        )
        output.copy_(chunk_output[:, :actual_heads].to(output.dtype))

    setattr(sparse_prefill, _PREFILL_MARKER, True)
    setattr(rocm, "rocm_sparse_attn_prefill", sparse_prefill)
    setattr(rocm, _PREFILL_MARKER, True)
    return True


def apply_to_module(module: ModuleType) -> bool:
    decode_changed = _apply_decode_to_module(module)
    prefill_changed = _apply_prefill_to_module(module)
    return decode_changed or prefill_changed


def apply(module: ModuleType | None = None) -> bool:
    if not (
        _flashmla_decode_enabled()
        or _flashmla_prefill_enabled()
    ):
        return False
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "apply", "apply_to_module"]
