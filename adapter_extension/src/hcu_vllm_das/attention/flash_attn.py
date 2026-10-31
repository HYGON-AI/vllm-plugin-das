"""HCU vendor varlen FlashAttention backend for vLLM 0.32."""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import Any, ClassVar

import torch
from vllm.config import get_current_vllm_config_or_none
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
)
from vllm.v1.kv_cache_layout import KVCacheLayout

_FA4_ONLY_ARGUMENTS = ("dynamic_causal", "mask_mod", "aux_tensors")
_E5M2_KV_CACHE_DTYPE = "fp8_e5m2"
_KERNEL_BLOCK_SIZE = 64
_REQUIRED_VLLM_SEAMS = (
    "_get_kv_cache_views",
    "_get_kv_cache_write_views",
    "_run_flash_attention",
)
_MISSING_VLLM_SEAMS = tuple(
    name
    for name in _REQUIRED_VLLM_SEAMS
    if not callable(getattr(FlashAttentionImpl, name, None))
)
if _MISSING_VLLM_SEAMS:
    missing = ", ".join(_MISSING_VLLM_SEAMS)
    raise RuntimeError(
        "HCU FLASH_ATTN requires the vLLM 0.32 HCU attention seams; "
        f"missing: {missing}. Install the matching HCU vLLM wheel."
    )


def _vendor_varlen_func() -> Callable[..., Any]:
    try:
        function = import_module("flash_attn").flash_attn_varlen_func
    except (AttributeError, ImportError) as error:
        raise RuntimeError(
            "HCU FLASH_ATTN requires flash_attn.flash_attn_varlen_func"
        ) from error
    if not callable(function):
        raise TypeError(
            "HCU FLASH_ATTN requires callable flash_attn.flash_attn_varlen_func"
        )
    return function


def _layout_name(layout: KVCacheLayout) -> str:
    return getattr(layout, "name", str(layout))


def _upstream_constructor_args(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[tuple[Any, ...], dict[str, Any], bool]:
    """Hide only HCU E5M2 from upstream's NVIDIA capability guard."""
    upstream_args = list(args)
    upstream_kwargs = dict(kwargs)
    if "kv_cache_dtype" in upstream_kwargs:
        uses_e5m2 = upstream_kwargs["kv_cache_dtype"] == _E5M2_KV_CACHE_DTYPE
        if uses_e5m2:
            upstream_kwargs["kv_cache_dtype"] = "auto"
    elif len(upstream_args) > 6:
        uses_e5m2 = upstream_args[6] == _E5M2_KV_CACHE_DTYPE
        if uses_e5m2:
            upstream_args[6] = "auto"
    else:
        uses_e5m2 = False
    return tuple(upstream_args), upstream_kwargs, uses_e5m2


def _e5m2_byte_view(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.element_size() != 1:
        raise ValueError("HCU E5M2 FlashAttention requires one-byte K/V storage")
    if tensor.dtype == torch.float8_e5m2:
        return tensor
    return tensor.view(torch.float8_e5m2)


def _dense_kv_cache_views(
    kv_cache: torch.Tensor,
    head_size: int,
    layout: KVCacheLayout,
) -> tuple[torch.Tensor, torch.Tensor]:
    if kv_cache.ndim != 4:
        raise ValueError("HCU FLASH_ATTN requires a 4D KV cache")
    num_blocks, num_heads, block_size, packed_head_size = kv_cache.shape
    if packed_head_size != 2 * head_size:
        raise ValueError(
            "HCU FLASH_ATTN requires KV cache content size 2 * head_size"
        )
    if layout == KVCacheLayout.LBHNC:
        expected_inner_strides = (
            block_size * packed_head_size,
            packed_head_size,
            1,
        )
        shape = (num_blocks, num_heads, block_size, head_size)
        inner_strides = (block_size * head_size, head_size, 1)
    elif layout == KVCacheLayout.LBNHC:
        expected_inner_strides = (
            packed_head_size,
            num_heads * packed_head_size,
            1,
        )
        shape = (num_blocks, block_size, num_heads, head_size)
        inner_strides = (num_heads * head_size, head_size, 1)
    else:
        raise ValueError(
            "HCU FLASH_ATTN received unsupported resolved KV cache layout: "
            f"{_layout_name(layout)}"
        )
    if kv_cache.stride()[1:] != expected_inner_strides:
        raise ValueError(
            "HCU FLASH_ATTN KV cache strides do not match resolved layout "
            f"{_layout_name(layout)}"
        )

    half_page = num_heads * block_size * head_size
    block_stride = kv_cache.stride(0)
    if block_stride < 2 * half_page:
        raise ValueError("HCU FLASH_ATTN KV cache blocks overlap")
    strides = (block_stride, *inner_strides)
    storage_offset = kv_cache.storage_offset()
    key_cache = torch.as_strided(
        kv_cache,
        size=shape,
        stride=strides,
        storage_offset=storage_offset,
    )
    value_cache = torch.as_strided(
        kv_cache,
        size=shape,
        stride=strides,
        storage_offset=storage_offset + half_page,
    )
    return key_cache, value_cache


class HcuFlashAttentionImpl(FlashAttentionImpl):
    """Expose dense HND or NHD pages to the HCU vendor varlen ABI."""

    supports_dcp = False

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        upstream_args, upstream_kwargs, uses_e5m2 = _upstream_constructor_args(
            args,
            kwargs,
        )
        super().__init__(*upstream_args, **upstream_kwargs)
        if uses_e5m2:
            self.kv_cache_dtype = _E5M2_KV_CACHE_DTYPE
            # The vendor E5M2 path consumes BF16/FP16 Q with E5M2 K/V.
            self.supports_quant_query_input = False
        vllm_config = get_current_vllm_config_or_none()
        self.cache_config = (
            vllm_config.cache_config if vllm_config is not None else None
        )
        if self.vllm_flash_attn_version is None:
            self.vllm_flash_attn_version = 2

    @property
    def kv_cache_layout(self) -> KVCacheLayout:
        if self.cache_config is None:
            raise RuntimeError(
                "HCU FLASH_ATTN requires an active vLLM cache configuration"
            )
        layout = self.cache_config.get_resolved_kv_cache_layout()
        if layout not in (KVCacheLayout.LBHNC, KVCacheLayout.LBNHC):
            raise ValueError(
                "HCU FLASH_ATTN received unsupported resolved KV cache layout: "
                f"{_layout_name(layout)}"
            )
        return layout

    def _get_kv_cache_views(
        self, kv_cache: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _dense_kv_cache_views(
            kv_cache,
            self.head_size,
            self.kv_cache_layout,
        )

    def _get_kv_cache_write_views(
        self, kv_cache: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key_cache, value_cache = self._get_kv_cache_views(kv_cache)
        if self.kv_cache_layout == KVCacheLayout.LBHNC:
            return key_cache.transpose(1, 2), value_cache.transpose(1, 2)
        return key_cache, value_cache

    def _run_flash_attention(self, **kwargs: Any) -> Any:
        for name in _FA4_ONLY_ARGUMENTS:
            value = kwargs.pop(name, None)
            if value is not None:
                raise NotImplementedError(
                    f"HCU FlashAttention FA2 does not support {name}"
                )
        max_query_len = kwargs.get("max_seqlen_q")
        query = kwargs.get("q")
        cu_query_lens = kwargs.get("cu_seqlens_q")
        if (
            kwargs.get("block_table") is not None
            and isinstance(max_query_len, int)
            and 1 < max_query_len < 16
            and isinstance(query, torch.Tensor)
            and isinstance(cu_query_lens, torch.Tensor)
            and max_query_len * (cu_query_lens.shape[0] - 1) == query.shape[0]
        ):
            # The vendor wrapper otherwise classifies rectangular short
            # prefill as MTP and can select an incompatible direct-paged tile.
            # max_seqlen_q is an upper bound, so incrementing it preserves the
            # ragged sequence contract while selecting the regular prefill path.
            kwargs["max_seqlen_q"] = max_query_len + 1
        kwargs["layout"] = (
            "bhsd"
            if self.kv_cache_layout == KVCacheLayout.LBHNC
            else "bshd"
        )
        if self.kv_cache_dtype == _E5M2_KV_CACHE_DTYPE:
            kwargs["k"] = _e5m2_byte_view(kwargs["k"])
            kwargs["v"] = _e5m2_byte_view(kwargs["v"])
        kwargs["fa_version"] = 2
        return _vendor_varlen_func()(**kwargs)


class HcuFlashAttentionBackend(FlashAttentionBackend):
    """Select dense HND/NHD HCU views through vLLM's backend seam."""

    supported_kv_cache_dtypes: ClassVar[list[str]] = [
        "auto",
        "float16",
        "bfloat16",
        _E5M2_KV_CACHE_DTYPE,
    ]

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: str | None) -> bool:
        if kv_cache_dtype == _E5M2_KV_CACHE_DTYPE:
            return True
        return super().supports_kv_cache_dtype(kv_cache_dtype)

    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: str | None,
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        use_mm_prefix: bool,
        device_capability: Any,
    ) -> str | None:
        if kv_cache_dtype == _E5M2_KV_CACHE_DTYPE:
            return None
        return super().supports_combination(
            head_size,
            dtype,
            kv_cache_dtype,
            block_size,
            use_mla,
            has_sink,
            use_sparse,
            use_mm_prefix,
            device_capability,
        )

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return head_size % 64 == 0 and head_size <= 256

    @classmethod
    def supports_block_size(cls, block_size: int | None) -> bool:
        return block_size is None or block_size == _KERNEL_BLOCK_SIZE

    @classmethod
    def supports_dcp(cls) -> bool:
        return False

    @classmethod
    def supports_mm_prefix(cls) -> bool:
        return False

    @classmethod
    def supports_sink(cls) -> bool:
        return False

    @classmethod
    def supports_rswa(cls) -> bool:
        return False

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == "decoder"

    @classmethod
    def supports_sliding_window(cls) -> bool:
        return False

    @classmethod
    def supports_non_causal(cls) -> bool:
        return False

    @classmethod
    def supports_batch_invariance(cls) -> bool:
        return False

    @classmethod
    def supports_per_head_quant_scales(cls) -> bool:
        return False

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[KVCacheLayout, ...]:
        return (KVCacheLayout.LBHNC, KVCacheLayout.LBNHC)

    @classmethod
    def get_supported_kernel_block_sizes(
        cls, kv_cache_spec: Any = None
    ) -> list[int]:
        del kv_cache_spec
        return [_KERNEL_BLOCK_SIZE]

    @classmethod
    def get_preferred_block_size(cls, default_block_size: int) -> int:
        del default_block_size
        return _KERNEL_BLOCK_SIZE

    @staticmethod
    def get_impl_cls() -> type[HcuFlashAttentionImpl]:
        return HcuFlashAttentionImpl


__all__ = [
    "HcuFlashAttentionBackend",
    "HcuFlashAttentionImpl",
]
