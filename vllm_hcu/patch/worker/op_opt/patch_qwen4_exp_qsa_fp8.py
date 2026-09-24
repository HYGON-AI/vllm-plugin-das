# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Enable an FP8 main KV cache for the Qwen4Exp QSA attention path.

The QSA owner in ``vllm.models.qwen4_exp.amd.qsa`` hard-requires a BF16 main
cache in five places: the backend's ``supported_kv_cache_dtypes``, two
constructor guards, the implementation constructor, and the ``forward_qsa``
dtype assertion.  Its read kernel is also BF16-only.  None of that file is
editable here, so this module patches the class from the outside:

* ``Qwen4ExpQSAAttention.__init__`` runs against a temporarily neutralized
  ``cache_dtype`` so every guard sees the BF16 configuration it expects; the
  real FP8 dtype is restored on both the layer and its ``impl`` afterwards.
  This also covers ``Qwen4ExpQSAFlashAttentionImpl.__init__``, which is built
  inside that call.
* ``forward_qsa`` keeps the official BF16 path untouched and, for an FP8 main
  cache, views the cache as FP8 and dispatches to the plugin's dequantizing
  kernel (``vllm_hcu.v1.attention.backends.qsa_fp8``) with the layer's
  per-tensor scales.
* ``do_kv_cache_update`` writes FP8 through the HCU writer
  (``torch.ops.hcu_ops``), which handles ``fp8_e4m3``/``fp8_e5m2`` with
  scales; the inherited vLLM writer is left in charge for every other dtype.

The QSA *indexer* side cache is a separate BF16 ``QSAKeyStateCache`` and is
intentionally left alone, so token selection keeps full precision.
"""

from __future__ import annotations

import functools
from types import ModuleType

import torch

from vllm.utils.torch_utils import (
    canonicalize_singleton_dim_strides,
    kv_cache_dtype_str_to_dtype,
)
from vllm.v1.attention.backend import AttentionType

import vllm_hcu.hcu_ops  # noqa: F401 - registers torch.ops.hcu_ops

from ._common import (
    PatchCompatibilityError,
    already_applied,
    load_exact_module,
    require_callable,
    require_class,
)
from vllm_hcu.v1.attention.backends.flash_attn import _split_kv_cache
from vllm_hcu.v1.attention.backends.qsa_fp8 import (
    qsa_sparse_paged_attention_fp8,
)

TARGET_MODULE = "vllm.models.qwen4_exp.amd.qsa"
PATCH_ID = "worker.op_opt.qwen4_exp.qsa.fp8_kv_cache"
TARGETS = (
    f"{TARGET_MODULE}.Qwen4ExpQSAFlashAttentionBackend.supported_kv_cache_dtypes",
    f"{TARGET_MODULE}.Qwen4ExpQSAAttention.__init__",
    f"{TARGET_MODULE}.Qwen4ExpQSAFlashAttentionImpl.forward_qsa",
    f"{TARGET_MODULE}.Qwen4ExpQSAFlashAttentionImpl.do_kv_cache_update",
)
_MARKER = "_vllm_hcu_qwen4_exp_qsa_fp8_applied"
_WRAPPER = "_vllm_hcu_qsa_fp8_wrapper"

# Cache dtypes this patch enables.  ``fp8`` is vLLM's alias for e4m3.
_FP8_CACHE_DTYPES = ("fp8", "fp8_e4m3", "fp8_e5m2")

_FP8_TORCH_DTYPES = {
    "fp8": torch.float8_e4m3fn,
    "fp8_e4m3": torch.float8_e4m3fn,
    "fp8_e5m2": torch.float8_e5m2,
}


def _is_fp8_cache(kv_cache_dtype: object) -> bool:
    return isinstance(kv_cache_dtype, str) and kv_cache_dtype in _FP8_CACHE_DTYPES


def _fp8_view(cache: torch.Tensor, kv_cache_dtype: str) -> torch.Tensor:
    """Reinterpret a byte-stored cache as the FP8 dtype it holds."""

    dtype = _FP8_TORCH_DTYPES[kv_cache_dtype]
    if cache.dtype == dtype:
        return cache
    if cache.element_size() != 1:
        raise PatchCompatibilityError(
            "QSA FP8 cache must be stored one byte per element, got "
            f"{cache.element_size()} bytes"
        )
    return cache.view(dtype)


def _require_bf16_cache_supported(qsa: ModuleType) -> type:
    """Fail loudly if a future wheel already implements FP8 QSA itself."""

    backend = require_class(
        qsa,
        "Qwen4ExpQSAFlashAttentionBackend",
        f"{TARGET_MODULE}.Qwen4ExpQSAFlashAttentionBackend",
    )
    declared = tuple(getattr(backend, "supported_kv_cache_dtypes", ()) or ())
    if any(dtype in declared for dtype in _FP8_CACHE_DTYPES):
        raise PatchCompatibilityError(
            "Qwen4Exp QSA already declares FP8 cache dtypes; the HCU FP8 "
            "patch would shadow an official implementation"
        )
    return backend


def apply_to_module(module: ModuleType) -> bool:
    qsa = load_exact_module(TARGET_MODULE, module)
    backend_cls = require_class(
        qsa,
        "Qwen4ExpQSAFlashAttentionBackend",
        f"{TARGET_MODULE}.Qwen4ExpQSAFlashAttentionBackend",
    )
    attention_cls = require_class(
        qsa, "Qwen4ExpQSAAttention", f"{TARGET_MODULE}.Qwen4ExpQSAAttention"
    )
    impl_cls = require_class(
        qsa,
        "Qwen4ExpQSAFlashAttentionImpl",
        f"{TARGET_MODULE}.Qwen4ExpQSAFlashAttentionImpl",
    )
    wrapped = (
        (attention_cls, "__init__", TARGETS[1], _WRAPPER),
        (impl_cls, "forward_qsa", TARGETS[2], _WRAPPER),
        (impl_cls, "do_kv_cache_update", TARGETS[3], _WRAPPER),
    )
    if already_applied(qsa, _MARKER, wrapped):
        return False

    _require_bf16_cache_supported(qsa)

    original_init = require_callable(attention_cls, "__init__", TARGETS[1])
    original_forward_qsa = require_callable(impl_cls, "forward_qsa", TARGETS[2])
    original_cache_update = require_callable(
        impl_cls, "do_kv_cache_update", TARGETS[3]
    )

    @functools.wraps(original_init)
    def hcu_qsa_init(
        self,
        *,
        vllm_config,
        config,
        layer_id,
        quant_config=None,
        reduce_results=True,
        prefix="",
    ):
        cache_config = getattr(vllm_config, "cache_config", None)
        if cache_config is None or not _is_fp8_cache(
            getattr(cache_config, "cache_dtype", None)
        ):
            return original_init(
                self,
                vllm_config=vllm_config,
                config=config,
                layer_id=layer_id,
                quant_config=quant_config,
                reduce_results=reduce_results,
                prefix=prefix,
            )

        # The official constructor rejects FP8 in three places and derives the
        # cache storage dtype from the config.  Present the BF16 configuration
        # it expects, then restore the real dtype on the constructed objects so
        # the KV-cache spec still allocates one byte per element.
        requested_dtype = cache_config.cache_dtype
        cache_config.cache_dtype = "bfloat16"
        try:
            result = original_init(
                self,
                vllm_config=vllm_config,
                config=config,
                layer_id=layer_id,
                quant_config=quant_config,
                reduce_results=reduce_results,
                prefix=prefix,
            )
        finally:
            cache_config.cache_dtype = requested_dtype

        storage_dtype = kv_cache_dtype_str_to_dtype(requested_dtype, None)
        if getattr(storage_dtype, "itemsize", None) != 1:
            raise PatchCompatibilityError(
                f"unexpected FP8 cache storage dtype for {requested_dtype!r}"
            )
        self.kv_cache_dtype = requested_dtype
        self.kv_cache_torch_dtype = storage_dtype
        impl = getattr(self, "impl", None)
        if impl is None:
            raise PatchCompatibilityError("QSA owner built without an impl")
        # Keep the impl's view of the cache consistent with the layer's; the
        # official constructor never sees the FP8 dtype.
        impl.kv_cache_dtype = requested_dtype
        return result

    @functools.wraps(original_forward_qsa)
    def hcu_forward_qsa(
        self,
        layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output,
        token_to_req,
        output_scale=None,
        output_block_scale=None,
    ):
        if not _is_fp8_cache(self.kv_cache_dtype):
            return original_forward_qsa(
                self,
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                token_to_req,
                output_scale,
                output_block_scale,
            )

        # Same argument checks and cache view construction as the official
        # implementation, with the FP8 dequantizing kernel substituted.
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("QSA does not support fused output quantization")
        if self.alibi_slopes is not None or self.sinks is not None:
            raise NotImplementedError("QSA does not support ALiBi or attention sinks")
        if self.sliding_window != (-1, -1):
            raise NotImplementedError("QSA does not support sliding-window attention")

        num_tokens = attn_metadata.num_actual_tokens
        output.zero_()
        if num_tokens == 0:
            return output

        topk_buffer = getattr(layer, "topk_indices_buffer", None)
        if topk_buffer is None:
            raise RuntimeError("QSA owner did not provide its top-k buffer")
        logical_indices = topk_buffer[:num_tokens]
        token_to_req = token_to_req[:num_tokens]
        key_cache, value_cache = kv_cache.transpose(1, 2).split(
            self.head_size, dim=-1
        )
        key_cache = canonicalize_singleton_dim_strides(key_cache)
        value_cache = canonicalize_singleton_dim_strides(value_cache)
        if query.dtype != torch.bfloat16:
            raise NotImplementedError("Qwen4Exp QSA requires BF16 Q")
        key_cache = _fp8_view(key_cache, self.kv_cache_dtype)
        value_cache = _fp8_view(value_cache, self.kv_cache_dtype)

        qsa_sparse_paged_attention_fp8(
            query[:num_tokens],
            key_cache,
            value_cache,
            logical_indices,
            attn_metadata.block_table,
            token_to_req,
            output[:num_tokens],
            k_scale=getattr(layer, "_k_scale", None),
            v_scale=getattr(layer, "_v_scale", None),
        )
        return output

    @functools.wraps(original_cache_update)
    def hcu_do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):
        if not _is_fp8_cache(self.kv_cache_dtype):
            return original_cache_update(self, layer, key, value, kv_cache, slot_mapping)

        # The inherited vLLM writer does not accept fp8_e5m2 on this build; the
        # HCU writer handles both FP8 formats and takes the layer's scales.
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            return
        key_cache, value_cache = _split_kv_cache(kv_cache, self.head_size)
        torch.ops.hcu_ops.reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )

    setattr(hcu_qsa_init, _WRAPPER, True)
    setattr(hcu_forward_qsa, _WRAPPER, True)
    setattr(hcu_do_kv_cache_update, _WRAPPER, True)
    setattr(attention_cls, "_vllm_hcu_original_init", original_init)
    setattr(impl_cls, "_vllm_hcu_original_forward_qsa", original_forward_qsa)
    setattr(impl_cls, "_vllm_hcu_original_do_kv_cache_update", original_cache_update)
    setattr(attention_cls, "__init__", hcu_qsa_init)
    setattr(impl_cls, "forward_qsa", hcu_forward_qsa)
    setattr(impl_cls, "do_kv_cache_update", hcu_do_kv_cache_update)
    setattr(
        backend_cls,
        "supported_kv_cache_dtypes",
        list(backend_cls.supported_kv_cache_dtypes) + list(_FP8_CACHE_DTYPES),
    )
    setattr(qsa, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
