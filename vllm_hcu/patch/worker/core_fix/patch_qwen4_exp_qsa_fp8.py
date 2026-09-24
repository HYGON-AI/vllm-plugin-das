# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Enable per-tensor FP8 main KV cache for the Qwen4Exp QSA owner.

The QSA indexer keeps its BF16 raw/compressed side caches. Only the main
sparse-GQA reader consumes an FP8 view of the vLLM-managed uint8 KV storage.
"""

from __future__ import annotations

import functools
import importlib
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
from vllm_hcu.v1.attention.backends.qsa import get_qsa_fp8_reader

TARGET_MODULE = "vllm.models.qwen4_exp.amd.qsa"
PATCH_ID = "worker.core_fix.qwen4_exp.qsa.fp8_kv_cache"
_ATTENTION_TARGET = f"{TARGET_MODULE}.Qwen4ExpQSAAttention.__init__"
_IMPL_INIT_TARGET = f"{TARGET_MODULE}.Qwen4ExpQSAFlashAttentionImpl.__init__"
_FORWARD_TARGET = f"{TARGET_MODULE}.Qwen4ExpQSAFlashAttentionImpl.forward_qsa"
TARGETS = (_ATTENTION_TARGET, _IMPL_INIT_TARGET, _FORWARD_TARGET)
_MARKER = "_vllm_hcu_qwen4_exp_qsa_fp8_applied"
_ATTENTION_WRAPPER = "_vllm_hcu_qsa_fp8_attention_init"
_IMPL_INIT_WRAPPER = "_vllm_hcu_qsa_fp8_impl_init"
_FORWARD_WRAPPER = "_vllm_hcu_qsa_fp8_forward"
_READER_ATTR = "_vllm_hcu_qsa_fp8_reader"
_FP8_CACHE_DTYPES = ("fp8", "fp8_e4m3", "fp8_e5m2")


def _already_applied(owner: object, wrapped: tuple[tuple, ...]) -> bool:
    if not getattr(owner, _MARKER, False):
        return False
    for target_owner, name, target, wrapper_marker in wrapped:
        function = require_callable(target_owner, name, target)
        if not getattr(function, wrapper_marker, False):
            raise PatchCompatibilityError(
                f"required HCU patch marker for {target} is stale; restart the process"
            )
    return True


def _require_variadic_init(function) -> None:
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError) as exc:
        raise PatchCompatibilityError(
            f"cannot inspect required HCU patch target {_IMPL_INIT_TARGET}"
        ) from exc
    parameters = tuple(signature.parameters.values())
    expected = (
        ("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
        ("args", inspect.Parameter.VAR_POSITIONAL),
        ("kwargs", inspect.Parameter.VAR_KEYWORD),
    )
    if tuple((item.name, item.kind) for item in parameters) != expected:
        raise PatchCompatibilityError(
            f"required HCU patch target {_IMPL_INIT_TARGET} has incompatible "
            f"signature {signature}"
        )


def _fp8_view_dtype(cache_dtype: str) -> torch.dtype:
    if cache_dtype in ("fp8", "fp8_e4m3"):
        return torch.float8_e4m3fn
    if cache_dtype == "fp8_e5m2":
        return torch.float8_e5m2
    raise ValueError(f"unsupported QSA FP8 cache dtype: {cache_dtype}")


def _upstream_triton_fp8_reader():
    ops = importlib.import_module("vllm.models.qwen4_exp.amd.ops.qsa")
    reader = getattr(ops, "qsa_sparse_paged_attention_fp8", None)
    return reader if callable(reader) else None


def _configure_fp8_impl(impl, cache_dtype: str) -> None:
    impl.kv_cache_dtype = cache_dtype
    impl.supports_quant_query_input = False
    setattr(
        impl,
        _READER_ATTR,
        get_qsa_fp8_reader(triton_fp8=_upstream_triton_fp8_reader()),
    )


def apply_to_module(module: ModuleType) -> bool:
    owner = load_exact_module(TARGET_MODULE, module)
    attention_cls = require_class(
        owner, "Qwen4ExpQSAAttention", _ATTENTION_TARGET
    )
    impl_cls = require_class(
        owner, "Qwen4ExpQSAFlashAttentionImpl", _IMPL_INIT_TARGET
    )
    backend_cls = require_class(
        owner,
        "Qwen4ExpQSAFlashAttentionBackend",
        f"{TARGET_MODULE}.Qwen4ExpQSAFlashAttentionBackend",
    )
    wrapped = (
        (attention_cls, "__init__", _ATTENTION_TARGET, _ATTENTION_WRAPPER),
        (impl_cls, "__init__", _IMPL_INIT_TARGET, _IMPL_INIT_WRAPPER),
        (impl_cls, "forward_qsa", _FORWARD_TARGET, _FORWARD_WRAPPER),
    )
    if _already_applied(owner, wrapped):
        return False

    original_attention_init = require_callable(
        attention_cls, "__init__", _ATTENTION_TARGET
    )
    require_exact_signature(
        original_attention_init,
        _ATTENTION_TARGET,
        positional=("self",),
        keyword_only=(
            "vllm_config",
            "config",
            "layer_id",
            "quant_config",
            "reduce_results",
            "prefix",
        ),
        defaults={
            "quant_config": None,
            "reduce_results": True,
            "prefix": "",
        },
    )
    original_impl_init = require_callable(impl_cls, "__init__", _IMPL_INIT_TARGET)
    _require_variadic_init(original_impl_init)
    original_forward = require_callable(impl_cls, "forward_qsa", _FORWARD_TARGET)
    require_exact_signature(
        original_forward,
        _FORWARD_TARGET,
        positional=(
            "self",
            "layer",
            "query",
            "key",
            "value",
            "kv_cache",
            "attn_metadata",
            "output",
            "token_to_req",
            "output_scale",
            "output_block_scale",
        ),
        defaults={"output_scale": None, "output_block_scale": None},
    )
    canonicalize = require_callable(
        owner,
        "canonicalize_singleton_dim_strides",
        f"{TARGET_MODULE}.canonicalize_singleton_dim_strides",
    )

    @functools.wraps(original_impl_init)
    def hcu_impl_init(self, *args, **kwargs):
        try:
            original_impl_init(self, *args, **kwargs)
        except NotImplementedError as exc:
            cache_dtype = getattr(self, "kv_cache_dtype", None)
            if (
                cache_dtype not in _FP8_CACHE_DTYPES
                or str(exc) != "Qwen4Exp QSA requires a BF16 main KV cache"
            ):
                raise
            _configure_fp8_impl(self, cache_dtype)

    @functools.wraps(original_attention_init)
    def hcu_attention_init(
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
        cache_dtype = getattr(cache_config, "cache_dtype", None)
        if cache_dtype not in _FP8_CACHE_DTYPES:
            return original_attention_init(
                self,
                vllm_config=vllm_config,
                config=config,
                layer_id=layer_id,
                quant_config=quant_config,
                reduce_results=reduce_results,
                prefix=prefix,
            )

        # Upstream owns all QSA construction but rejects FP8 before it can
        # create the BF16 indexer side caches. Present its supported default
        # only during construction, restore the shared config unconditionally,
        # then bind FP8 exclusively to the main cache owner.
        cache_config.cache_dtype = "auto"
        try:
            original_attention_init(
                self,
                vllm_config=vllm_config,
                config=config,
                layer_id=layer_id,
                quant_config=quant_config,
                reduce_results=reduce_results,
                prefix=prefix,
            )
        finally:
            cache_config.cache_dtype = cache_dtype
        self.kv_cache_dtype = cache_dtype
        self.kv_cache_torch_dtype = torch.uint8
        _configure_fp8_impl(self.impl, cache_dtype)

    @functools.wraps(original_forward)
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
        if self.kv_cache_dtype not in _FP8_CACHE_DTYPES:
            return original_forward(
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
        del key, value
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "QSA does not support fused output quantization"
            )
        if self.alibi_slopes is not None or self.sinks is not None:
            raise NotImplementedError(
                "QSA does not support ALiBi or attention sinks"
            )
        if self.sliding_window != (-1, -1):
            raise NotImplementedError(
                "QSA does not support sliding-window attention"
            )

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
        key_cache = canonicalize(key_cache).view(
            _fp8_view_dtype(self.kv_cache_dtype)
        )
        value_cache = canonicalize(value_cache).view(
            _fp8_view_dtype(self.kv_cache_dtype)
        )
        if query.dtype != torch.bfloat16:
            raise NotImplementedError("Qwen4Exp QSA requires BF16 queries")

        reader = getattr(self, _READER_ATTR, None)
        if not callable(reader):
            raise RuntimeError("QSA FP8 reader was not initialized before capture")
        reader(
            query[:num_tokens],
            key_cache,
            value_cache,
            logical_indices,
            attn_metadata.block_table,
            token_to_req,
            out=output[:num_tokens],
            k_scale=layer._k_scale,
            v_scale=layer._v_scale,
        )
        return output

    setattr(hcu_attention_init, _ATTENTION_WRAPPER, True)
    setattr(hcu_impl_init, _IMPL_INIT_WRAPPER, True)
    setattr(hcu_forward_qsa, _FORWARD_WRAPPER, True)
    original_supported = list(backend_cls.supported_kv_cache_dtypes)
    supported = list(dict.fromkeys((*original_supported, *_FP8_CACHE_DTYPES)))

    try:
        backend_cls.supported_kv_cache_dtypes = supported
        setattr(attention_cls, "__init__", hcu_attention_init)
        setattr(impl_cls, "__init__", hcu_impl_init)
        setattr(impl_cls, "forward_qsa", hcu_forward_qsa)
        setattr(owner, _MARKER, True)
    except BaseException:
        backend_cls.supported_kv_cache_dtypes = original_supported
        setattr(attention_cls, "__init__", original_attention_init)
        setattr(impl_cls, "__init__", original_impl_init)
        setattr(impl_cls, "forward_qsa", original_forward)
        if hasattr(owner, _MARKER):
            delattr(owner, _MARKER)
        raise
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
