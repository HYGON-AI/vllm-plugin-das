# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""HCU attention policy for the Kimi-K3/K3.1 DSpark draft (``dspark_mla``).

The upstream draft (``vllm.models.kimi_k3.nvidia.dspark_mla``) builds every
decoder layer's attention as a DENSE ``MultiHeadLatentAttention``
(``use_sparse=False``, head 576 = 512 NoPE latent + 64 RoPE) and hands it the
run-level ``cache_config``.  Kimi-K3.1 normalizes the run KV dtype to the
sparse-only ``fp8_ds_mla`` record (656 B/token) for its DSA target layers, so
the draft asks the platform for ``576 + fp8_ds_mla + dense``, which none of
the HCU backends accept (FLASHMLA/TRITON_MLA reject the dtype, the sparse
backends reject dense) -> ``No valid attention backend found for hip``.

Policy (applied per draft decoder layer by
``patch_kimi_k3_dspark_draft_attention``):

* KV dtype -- when, and only when, the run dtype is ``fp8_ds_mla``, build the
  draft layers with ``VLLM_HCU_K3_DSPARK_DRAFT_KV_DTYPE`` (default ``auto``:
  model-dtype BF16 latent, 1152 B/token).  The upstream K3 MLA layer supports
  BF16 / plain FP8 / fp8_ds_mla natively per layer (fused insert + prefill +
  decode, grouped context-KV insert) and publishes the dtype through its own
  ``MLAAttentionSpec``, so the KV pool allocates the draft group per spec.
* Backend -- ``VLLM_HCU_K3_DSPARK_DRAFT_MLA_BACKEND``: ``auto`` keeps the
  platform dense-MLA priority (FLASHMLA -> TRITON_MLA, Triton first with
  ``VLLM_HCU_USE_CUSTOM_OPS=0``); ``FLASHMLA`` / ``TRITON_MLA`` pin the draft
  layers only, resolved straight through the platform so the target model's
  selection and vLLM's cached selector are never consulted for the pin.

Nothing here touches a device at import or call time.
"""

from __future__ import annotations

import contextlib
import importlib
from collections.abc import Callable, Iterator
from typing import Any

# The only run dtype a dense HCU MLA backend cannot serve (sparse-only).
SPARSE_ONLY_KV_CACHE_DTYPE = "fp8_ds_mla"
# Requested draft dtype -> CacheDType string handed to the draft layers.
_DRAFT_KV_CACHE_DTYPES = {
    "auto": "auto",
    "bfloat16": "bfloat16",
    "fp8_e4m3": "fp8_e4m3",
}
_LATENT_BYTES_PER_TOKEN = {
    # 576 latent elements per token (kv_lora_rank 512 + rope 64).
    "auto": 2,
    "bfloat16": 2,
    "float16": 2,
    "fp8": 1,
    "fp8_e4m3": 1,
    "fp8_e5m2": 1,
}
# fp8_ds_mla: 512 fp8 NoPE + 4x fp32 tile scales + 64 bf16 RoPE.
_FP8_DS_MLA_BYTES_PER_TOKEN = 656
DENSE_MLA_BACKENDS = ("FLASHMLA", "TRITON_MLA")


def resolve_draft_kv_cache_dtype(
    run_cache_dtype: str | None, requested: str
) -> str | None:
    """KV dtype for the draft layers, or ``None`` to inherit the run dtype.

    Only the sparse-only ``fp8_ds_mla`` run dtype is rewritten; every other
    run dtype is already accepted by the dense HCU MLA backends and stays.
    """

    if requested == "inherit":
        return None
    if requested not in _DRAFT_KV_CACHE_DTYPES:
        raise ValueError(
            "DSpark draft KV dtype must be one of "
            f"{(*_DRAFT_KV_CACHE_DTYPES, 'inherit')}, got {requested!r}"
        )
    if run_cache_dtype != SPARSE_ONLY_KV_CACHE_DTYPE:
        return None
    return _DRAFT_KV_CACHE_DTYPES[requested]


def mla_latent_bytes_per_token(cache_dtype: str, head_size: int = 576) -> int:
    """Bytes one token occupies in a latent MLA page of ``cache_dtype``."""

    if cache_dtype == SPARSE_ONLY_KV_CACHE_DTYPE:
        if head_size != 576:
            raise ValueError("fp8_ds_mla is defined for the 576-wide latent only")
        return _FP8_DS_MLA_BYTES_PER_TOKEN
    try:
        return head_size * _LATENT_BYTES_PER_TOKEN[cache_dtype]
    except KeyError:
        raise ValueError(f"unsupported MLA cache dtype {cache_dtype!r}") from None


def selector_config_kwargs(
    fields: tuple[str, ...],
    *,
    head_size: int,
    dtype: Any,
    kv_cache_dtype: str | None,
    use_sparse: bool,
    decoder_attn_type: Any,
) -> dict[str, Any]:
    """Draft MLA selector fields, filtered to the running vLLM's schema."""

    values: dict[str, Any] = {
        "head_size": head_size,
        "dtype": dtype,
        "kv_cache_dtype": kv_cache_dtype,
        "block_size": None,
        "use_mla": True,
        "has_sink": False,
        "use_sparse": use_sparse,
        "use_mm_prefix": False,
        "attn_type": decoder_attn_type,
    }
    return {name: value for name, value in values.items() if name in fields}


def _resolve_qualname(path: str) -> Any:
    module_name, _, attr = path.rpartition(".")
    return getattr(importlib.import_module(module_name), attr)


def make_pinned_get_attn_backend(
    backend_name: str,
    original: Callable[..., Any],
) -> Callable[..., Any]:
    """``get_attn_backend`` stand-in that pins dense draft MLA to a backend.

    Non-MLA or sparse requests are delegated to ``original`` unchanged.  The
    pin goes through ``Platform.get_attn_backend_cls`` so platform validation
    and the custom-op master fallback still apply; the global
    ``attention_config.backend`` and vLLM's selector cache are bypassed only
    for the pinned draft request.
    """

    if backend_name not in DENSE_MLA_BACKENDS:
        raise ValueError(
            f"DSpark draft MLA backend must be one of {DENSE_MLA_BACKENDS}, "
            f"got {backend_name!r}"
        )

    def get_attn_backend(
        head_size: int,
        dtype: Any,
        kv_cache_dtype: str | None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        # The K3 MLA layer passes the selector flags by keyword; anything
        # else is not the audited dense draft request.
        if args or not kwargs.get("use_mla", False) or kwargs.get(
            "use_sparse", False
        ):
            return original(head_size, dtype, kv_cache_dtype, *args, **kwargs)

        from vllm.platforms import current_platform
        from vllm.v1.attention.backend import AttentionType
        from vllm.v1.attention.backends.registry import AttentionBackendEnum
        from vllm.v1.attention.selector import AttentionSelectorConfig

        selector_config = AttentionSelectorConfig(
            **selector_config_kwargs(
                AttentionSelectorConfig._fields,
                head_size=head_size,
                dtype=dtype,
                kv_cache_dtype=kv_cache_dtype,
                use_sparse=False,
                decoder_attn_type=AttentionType.DECODER,
            )
        )
        path = current_platform.get_attn_backend_cls(
            AttentionBackendEnum[backend_name],
            selector_config,
            num_heads=kwargs.get("num_heads"),
        )
        return _resolve_qualname(path)

    get_attn_backend._vllm_hcu_dspark_pin = backend_name  # type: ignore[attr-defined]
    return get_attn_backend


@contextlib.contextmanager
def draft_attention_scope(
    cache_config: Any,
    draft_kv_cache_dtype: str | None,
    mla_module: Any,
    backend_name: str,
) -> Iterator[None]:
    """Construction-only scope for one draft decoder layer.

    The draft layer captures ``cache_config.cache_dtype`` into its own
    ``kv_cache_dtype`` (and its spec) during ``__init__``; the shared run
    config is restored unconditionally afterwards, so the target model and
    every later reader keep the run dtype.
    """

    previous_dtype = getattr(cache_config, "cache_dtype", None)
    previous_get_attn_backend = getattr(mla_module, "get_attn_backend", None)
    swap_dtype = cache_config is not None and draft_kv_cache_dtype is not None
    swap_backend = backend_name != "auto"
    if swap_backend and previous_get_attn_backend is None:
        raise RuntimeError(
            f"{getattr(mla_module, '__name__', mla_module)!r} has no "
            "get_attn_backend to pin the DSpark draft backend through"
        )
    try:
        if swap_dtype:
            cache_config.cache_dtype = draft_kv_cache_dtype
        if swap_backend:
            mla_module.get_attn_backend = make_pinned_get_attn_backend(
                backend_name, previous_get_attn_backend
            )
        yield
    finally:
        if swap_backend:
            mla_module.get_attn_backend = previous_get_attn_backend
        if swap_dtype:
            cache_config.cache_dtype = previous_dtype


__all__ = [
    "DENSE_MLA_BACKENDS",
    "SPARSE_ONLY_KV_CACHE_DTYPE",
    "draft_attention_scope",
    "make_pinned_get_attn_backend",
    "mla_latent_bytes_per_token",
    "resolve_draft_kv_cache_dtype",
    "selector_config_kwargs",
]
