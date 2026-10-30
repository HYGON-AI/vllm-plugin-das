"""HCU vendor varlen FlashAttention backend for vLLM 0.32."""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import Any

import torch
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
)
from vllm.v1.kv_cache_layout import KVCacheLayout

_FA4_ONLY_ARGUMENTS = ("dynamic_causal", "mask_mod", "aux_tensors")
_KERNEL_BLOCK_SIZE = 64


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


def _dense_hnd_kv_cache_views(
    kv_cache: torch.Tensor,
    head_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if kv_cache.ndim != 4:
        raise ValueError("HCU FLASH_ATTN requires a 4D KV cache")
    num_blocks, num_heads, block_size, packed_head_size = kv_cache.shape
    if packed_head_size != 2 * head_size:
        raise ValueError(
            "HCU FLASH_ATTN requires KV cache content size 2 * head_size"
        )
    expected_inner_strides = (
        block_size * packed_head_size,
        packed_head_size,
        1,
    )
    if kv_cache.stride()[1:] != expected_inner_strides:
        raise ValueError("HCU FLASH_ATTN requires the LBHNC KV cache layout")

    half_page = num_heads * block_size * head_size
    block_stride = kv_cache.stride(0)
    if block_stride < 2 * half_page:
        raise ValueError("HCU FLASH_ATTN KV cache blocks overlap")
    shape = (num_blocks, num_heads, block_size, head_size)
    strides = (block_stride, block_size * head_size, head_size, 1)
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
    """Expose dense HND pages to the HCU vendor varlen ABI."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if self.vllm_flash_attn_version is None:
            self.vllm_flash_attn_version = 2

    def _get_kv_cache_views(
        self, kv_cache: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _dense_hnd_kv_cache_views(kv_cache, self.head_size)

    def _get_kv_cache_write_views(
        self, kv_cache: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key_cache, value_cache = self._get_kv_cache_views(kv_cache)
        return key_cache.transpose(1, 2), value_cache.transpose(1, 2)

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
        kwargs["layout"] = "bhsd"
        kwargs["fa_version"] = 2
        return _vendor_varlen_func()(**kwargs)


class HcuFlashAttentionBackend(FlashAttentionBackend):
    """Select the dense HND HCU implementation through vLLM's backend seam."""

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[KVCacheLayout, ...]:
        return (KVCacheLayout.LBHNC,)

    @classmethod
    def get_supported_kernel_block_sizes(
        cls, kv_cache_spec: Any = None
    ) -> list[int]:
        del kv_cache_spec
        return [_KERNEL_BLOCK_SIZE]

    @classmethod
    def get_preferred_block_size(cls, default_block_size: int) -> int:
        return max(default_block_size, _KERNEL_BLOCK_SIZE)

    @staticmethod
    def get_impl_cls() -> type[HcuFlashAttentionImpl]:
        return HcuFlashAttentionImpl


__all__ = [
    "HcuFlashAttentionBackend",
    "HcuFlashAttentionImpl",
]
