"""HCU vendor varlen FlashAttention through vLLM's public backend API."""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import Any, ClassVar

import torch
from vllm.config import get_current_vllm_config_or_none
from vllm.v1.attention.backend import AttentionImpl
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionMetadata,
)
from vllm.v1.kv_cache_layout import KVCacheLayout

_FA4_ONLY_ARGUMENTS = ("dynamic_causal", "mask_mod", "aux_tensors")
_E5M2_KV_CACHE_DTYPE = "fp8_e5m2"
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


def _layout_name(layout: KVCacheLayout) -> str:
    return getattr(layout, "name", str(layout))


def _e5m2_byte_view(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.element_size() != 1:
        raise ValueError("HCU E5M2 FlashAttention requires one-byte K/V storage")
    if tensor.dtype == torch.float8_e5m2:
        return tensor
    return tensor.view(torch.float8_e5m2)


def _reshape_and_cache_e5m2(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
) -> None:
    from hcu_vllm_das.attention.e5m2_cache import reshape_and_cache_e5m2

    reshape_and_cache_e5m2(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        k_scale,
        v_scale,
    )


def _dense_kv_cache_views(
    kv_cache: torch.Tensor,
    head_size: int,
    layout: KVCacheLayout,
) -> tuple[torch.Tensor, torch.Tensor]:
    if kv_cache.ndim != 4:
        raise ValueError("HCU FLASH_ATTN requires a 4D KV cache")
    num_blocks, num_heads, block_size, packed_head_size = kv_cache.shape
    if packed_head_size != 2 * head_size:
        raise ValueError("HCU FLASH_ATTN requires KV cache content size 2 * head_size")
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


class HcuFlashAttentionImpl(AttentionImpl[FlashAttentionMetadata]):
    """Own the vendor call and cache ABI without replacing Core internals."""

    supports_dcp = False

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: str = "decoder",
        kv_sharing_target_layer_name: str | None = None,
        sinks: torch.Tensor | None = None,
    ) -> None:
        if attn_type != "decoder" or sliding_window is not None or sinks is not None:
            raise NotImplementedError("HCU Flash supports full decoder attention only")
        if self.dcp_world_size != 1 or self.pcp_world_size != 1:
            raise NotImplementedError("HCU Flash does not support context parallelism")
        if not HcuFlashAttentionBackend.supports_head_size(head_size):
            raise ValueError(f"unsupported HCU Flash head size: {head_size}")
        if not HcuFlashAttentionBackend.supports_kv_cache_dtype(kv_cache_dtype):
            raise ValueError(f"unsupported HCU Flash KV cache dtype: {kv_cache_dtype}")
        self.num_heads = num_heads
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        if self.num_kv_heads <= 0 or num_heads <= 0 or num_heads % self.num_kv_heads:
            raise ValueError("HCU Flash requires positive grouped query heads")
        self.head_size = head_size
        self.scale = scale
        self.kv_cache_dtype = kv_cache_dtype
        self.attn_type = attn_type
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name
        self.alibi_slopes = (
            torch.tensor(alibi_slopes, dtype=torch.float32, device="cuda")
            if alibi_slopes is not None
            else None
        )
        self.logits_soft_cap = 0.0 if logits_soft_cap is None else logits_soft_cap
        # Both vendor paths consume unquantized BF16/FP16 Q. Keep the public
        # base's fusion support disabled: native fused writers use another ABI.
        self.supports_quant_query_input = False
        self.vllm_flash_attn_version = 2
        vllm_config = get_current_vllm_config_or_none()
        self.cache_config = (
            vllm_config.cache_config if vllm_config is not None else None
        )

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata | None,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "HCU Flash does not support fused output quantization"
            )
        if attn_metadata is None:
            return output.fill_(0)
        if (
            attn_metadata.causal is not True
            or attn_metadata.mm_prefix_query_range_tensor is not None
            or attn_metadata.rswa_prefix_lens is not None
            or attn_metadata.dcp_context_kv_lens is not None
        ):
            raise NotImplementedError("unsupported HCU Flash attention metadata")
        tokens = attn_metadata.num_actual_tokens
        if tokens == 0:
            return output
        key_cache, value_cache = self._get_kv_cache_views(kv_cache)
        descale_shape = (attn_metadata.query_start_loc.shape[0] - 1, self.num_kv_heads)
        uses_e5m2 = self.kv_cache_dtype == _E5M2_KV_CACHE_DTYPE
        self._run_flash_attention(
            q=query[:tokens],
            k=key_cache,
            v=value_cache,
            out=output[:tokens],
            cu_seqlens_q=attn_metadata.query_start_loc,
            max_seqlen_q=attn_metadata.max_query_len,
            seqused_k=attn_metadata.seq_lens,
            max_seqlen_k=attn_metadata.max_seq_len,
            block_table=attn_metadata.block_table,
            softmax_scale=self.scale,
            causal=True,
            alibi_slopes=self.alibi_slopes,
            window_size=(-1, -1),
            softcap=self.logits_soft_cap,
            q_descale=None,
            k_descale=layer._k_scale.expand(descale_shape) if uses_e5m2 else None,
            v_descale=layer._v_scale.expand(descale_shape) if uses_e5m2 else None,
        )
        return output

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

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        key_cache, value_cache = self._get_kv_cache_write_views(kv_cache)
        if self.kv_cache_dtype != _E5M2_KV_CACHE_DTYPE:
            from vllm._custom_ops import reshape_and_cache_flash

            reshape_and_cache_flash(
                key,
                value,
                key_cache,
                value_cache,
                slot_mapping,
                self.kv_cache_dtype,
                layer._k_scale,
                layer._v_scale,
            )
            return
        _reshape_and_cache_e5m2(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            layer._k_scale,
            layer._v_scale,
        )

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
            "bhsd" if self.kv_cache_layout == KVCacheLayout.LBHNC else "bshd"
        )
        if self.kv_cache_dtype == _E5M2_KV_CACHE_DTYPE:
            kwargs["k"] = _e5m2_byte_view(kwargs["k"])
            kwargs["v"] = _e5m2_byte_view(kwargs["v"])
        kwargs["fa_version"] = 2
        if (
            self.kv_cache_dtype == _E5M2_KV_CACHE_DTYPE
            and kwargs.get("block_table") is not None
            and isinstance(max_query_len, int)
            and max_query_len >= 1
        ):
            from hcu_vllm_das.attention.e5m2_scales import run_e5m2_attention

            return run_e5m2_attention(_vendor_varlen_func(), kwargs)
        return _vendor_varlen_func()(**kwargs)


class HcuFlashAttentionBackend(FlashAttentionBackend):
    """Select dense HND/NHD HCU views through vLLM's backend seam."""

    @classmethod
    def validate_configuration(cls, *args, **kwargs):
        from hcu_vllm_das.flash_activation import require_published

        require_published(cls)
        return super().validate_configuration(*args, **kwargs)

    supported_kv_cache_dtypes: ClassVar[list[str]] = [
        "auto",
        "float16",
        "bfloat16",
        _E5M2_KV_CACHE_DTYPE,
    ]

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: str | None) -> bool:
        return kv_cache_dtype is None or kv_cache_dtype in cls.supported_kv_cache_dtypes

    @classmethod
    def supports_compute_capability(cls, capability: Any) -> bool:
        # Adapter admission verifies real HCU. NVIDIA SM numbers do not express
        # this provider's capability; do not fabricate an NVIDIA architecture.
        return True

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
        if use_mla or has_sink or use_sparse or use_mm_prefix:
            return "HCU Flash supports ordinary full decoder attention only"
        return None

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return head_size in (64, 128, 192, 256)

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
    def get_supported_kernel_block_sizes(cls, kv_cache_spec: Any = None) -> list[int]:
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
