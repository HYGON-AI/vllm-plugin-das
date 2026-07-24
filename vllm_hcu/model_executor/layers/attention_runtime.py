# SPDX-License-Identifier: Apache-2.0
"""HCU-owned attention implementations used by the v0.25.1 runtime adapters.

The module is imported only after vLLM's attention layer has finished loading.
Keeping the implementation here avoids embedding a large copied method in a
patch callback and, importantly, registers the fused custom op exactly once.
"""

from __future__ import annotations

from types import ModuleType

import torch

from vllm.config import CacheConfig, get_current_vllm_config
from vllm.model_executor.layers.attention.attention import Attention
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.backend import AttentionBackend, AttentionType

from vllm_hcu.platforms import envs as henvs


def split_kv_cache(kv_cache: object) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(kv_cache, (tuple, list)):
        if len(kv_cache) != 2:
            raise ValueError(f"expected two split KV cache tensors, got {len(kv_cache)}")
        return kv_cache[0], kv_cache[1]
    if not isinstance(kv_cache, torch.Tensor):
        raise TypeError(f"unsupported KV cache type: {type(kv_cache).__name__}")
    if kv_cache.ndim >= 1 and kv_cache.shape[0] == 2:
        return kv_cache.unbind(0)
    raise ValueError(
        "expected stacked KV cache dimension of size 2 at axis 0, "
        f"got shape {tuple(kv_cache.shape)}"
    )


def init_kv_cache_quant_e5m2(
    upstream: ModuleType,
    layer: torch.nn.Module,
    quant_config: QuantizationConfig | None,
    prefix: str,
) -> None:
    """v0.25.1 target ``_init_kv_cache_quant`` plus the HCU E5M2 delta."""

    upstream.set_default_quant_scales(layer, register_buffer=True)
    layer._o_scale_float = None
    quant_method = (
        quant_config.get_quant_method(layer, prefix=prefix) if quant_config else None
    )
    if upstream.should_load_quant_weights(quant_method):
        if not isinstance(quant_method, upstream.BaseKVCacheMethod):
            raise TypeError(
                "HCU FP8 KV-cache initialization requires BaseKVCacheMethod, "
                f"got {type(quant_method).__name__}"
            )
        layer.quant_method = quant_method
        layer.quant_method.create_weights(layer)


def attention_forward(
    upstream: ModuleType,
    self: Attention,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output_shape: torch.Size | None = None,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Attention forward preserving v0.25.1 semantics for HCU custom KV layout.

    HCU's split key/value cache is not a single tensor.  Calling the Python
    implementation keeps the data dependency dummy on the query device while
    retaining the exact official implementation for every feature-off call.
    """

    if self.calculate_kv_scales:
        torch.ops.vllm.maybe_calc_kv_scales(
            query,
            key,
            value,
            upstream._encode_layer_name(self.layer_name),
        )
    if output_dtype is None:
        output_dtype = query.dtype
    if self.query_quant is not None:
        if self.kv_cache_dtype not in {"fp8", "fp8_e4m3", "fp8_e5m2", "nvfp4"}:
            raise ValueError(
                "unsupported HCU quantized attention KV-cache dtype "
                f"{self.kv_cache_dtype!r}"
            )
        if self.impl.supports_quant_query_input:
            query, _ = self.query_quant(query, self._q_scale)

    if output_shape is None:
        num_tokens = query.shape[0]
        output_shape = torch.Size((num_tokens, self.num_heads * self.head_size_v))
    output = torch.empty(output_shape, dtype=output_dtype, device=query.device)
    hidden_size = output_shape[-1]
    query = query.view(-1, self.num_heads, self.head_size)
    output = output.view(-1, self.num_heads, self.head_size_v)
    if key is not None:
        key = key.view(-1, self.num_kv_heads, self.head_size)
    if value is not None:
        value = value.view(-1, self.num_kv_heads, self.head_size_v)

    kv_cache_dummy_dep = None
    if (
        not self.attn_backend.forward_includes_kv_cache_update
        and self.kv_sharing_target_layer_name is None
        and key is not None
        and value is not None
    ):
        layer_name = upstream._resolve_layer_name(self.layer_name)
        _, attn_layer, kv_cache, layer_slot_mapping = upstream.get_attention_context(
            layer_name
        )
        if layer_slot_mapping is not None:
            update = getattr(attn_layer.impl, "do_kv_cache_update", None)
            if not callable(update):
                raise RuntimeError(
                    f"{attn_layer.impl.__class__.__name__} does not support KV cache update"
                )
            update(attn_layer, key, value, kv_cache, layer_slot_mapping)
        # HCU's custom cache is a (key, value) pair and has no ``.device``.
        kv_cache_dummy_dep = torch.empty(0, device=key.device, dtype=key.dtype)
    upstream.unified_attention_with_output(
        query,
        key,
        value,
        output,
        self.layer_name,
        kv_cache_dummy_dep=kv_cache_dummy_dep,
    )
    return output.view(-1, hidden_size)


class FusedQkvSplitRmsNormRopeAttention(Attention):
    """Attention layer using lightop's fused QKV/RMSNorm/RoPE/KV-store op."""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        use_alibi_sqrt: bool | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        logits_soft_cap: float | None = None,
        per_layer_sliding_window: int | None = None,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        attn_backend: type[AttentionBackend] | None = None,
        head_size_v: int | None = None,
        **extra_impl_args,
    ) -> None:
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            use_alibi_sqrt,
            cache_config,
            quant_config,
            logits_soft_cap,
            per_layer_sliding_window,
            prefix,
            attn_type,
            kv_sharing_target_layer_name,
            attn_backend,
            head_size_v,
            **extra_impl_args,
        )
        self.block_size = get_current_vllm_config().cache_config.block_size

    def forward(
        self,
        qkv: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        weight_q_norm: torch.Tensor,
        weight_k_norm: torch.Tensor,
        epsilon: float,
        output_shape: torch.Size | None = None,
        is_neox: bool = False,
    ) -> torch.Tensor:
        output_dtype = qkv.dtype
        num_tokens = qkv.shape[0]
        if output_shape is None:
            output_shape = torch.Size(
                (num_tokens, self.num_heads * self.head_size_v)
            )
        output = torch.empty(output_shape, dtype=output_dtype, device=qkv.device)
        output = output.view(-1, self.num_heads, self.head_size_v)
        hidden_size = output_shape[-1]
        q_size = self.num_heads * self.head_size
        kv_size = self.num_kv_heads * self.head_size
        query, key, value = torch.ops.vllm.fused_qkv_split_rmsnorm_rope_kv_store(
            qkv=qkv,
            positions=positions,
            layer_name=self.layer_name,
            kv_cache_dtype=self.kv_cache_dtype,
            cos_sin_cache=cos_sin_cache,
            weight_q_norm=weight_q_norm,
            weight_k_norm=weight_k_norm,
            epsilon=epsilon,
            head_size=self.head_size,
            head_size_v=self.head_size_v,
            q_size=q_size,
            kv_size=kv_size,
            block_size=self.block_size,
            is_neox=is_neox,
        )
        torch.ops.vllm.unified_attention_with_output(
            query,
            key,
            value,
            output,
            self.layer_name,
            kv_cache_dummy_dep=None,
        )
        return output.view(-1, hidden_size)


def fused_qkv_split_rmsnorm_rope_kv_store_impl(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    layer_name: str,
    kv_cache_dtype: str,
    cos_sin_cache: torch.Tensor,
    weight_q_norm: torch.Tensor,
    weight_k_norm: torch.Tensor,
    epsilon: float,
    head_size: int,
    head_size_v: int,
    q_size: int,
    kv_size: int,
    block_size: int,
    is_neox: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from vllm.forward_context import get_forward_context

    num_tokens = qkv.shape[0]
    forward_context = get_forward_context()
    slot_mapping = forward_context.slot_mapping
    if not isinstance(slot_mapping, dict):
        raise TypeError(f"Expected slot_mapping dict, got {type(slot_mapping).__name__}")
    layer_slot_mapping = slot_mapping.get(layer_name)
    attn_layer = forward_context.no_compile_layers[layer_name]
    kv_cache = attn_layer.kv_cache

    if layer_slot_mapping is not None:
        key_cache, value_cache = split_kv_cache(kv_cache)
        if kv_cache_dtype.startswith("fp8"):
            from vllm_hcu.v1.attention.backends.flash_attn import (
                HcuFlashAttentionBackend,
            )

            fp8_dtype = HcuFlashAttentionBackend.get_fp8_dtype_for_flashattn(
                kv_cache_dtype
            )
            key_cache = key_cache.view(fp8_dtype)
            value_cache = value_cache.view(fp8_dtype)
    else:
        key_cache = torch.empty(0, device=qkv.device, dtype=qkv.dtype)
        value_cache = torch.empty(0, device=qkv.device, dtype=qkv.dtype)

    try:
        from lightop import split_qkv_rms_rotary_embedding_fuse_with_kv_store_quant
    except ImportError as exc:
        raise RuntimeError(
            "VLLM_HCU_USE_FUSED_QKV_SPLIT_RMS_ROPE_KVSTORE requires lightop"
        ) from exc

    q, k, v = split_qkv_rms_rotary_embedding_fuse_with_kv_store_quant(
        positions,
        qkv.contiguous(),
        q_size,
        kv_size,
        cos_sin_cache,
        head_dim=head_size,
        page_size=block_size,
        k_buffer=key_cache,
        v_buffer=value_cache,
        kv_cache_loc=layer_slot_mapping,
        is_neox=is_neox,
        weight_q=weight_q_norm,
        weight_k=weight_k_norm,
        output_dtype=qkv.dtype,
        kv_cache_dtype=kv_cache_dtype,
        epsilon=epsilon,
        residual_q=None,
        residual_k=None,
        k_scale=None,
        v_scale=None,
    )
    q = q.contiguous().view(num_tokens, q_size // head_size, head_size)
    k = k.contiguous().view(num_tokens, kv_size // head_size_v, head_size_v)
    v = v.contiguous().view(num_tokens, kv_size // head_size_v, head_size_v)
    return q, k, v


def fused_qkv_split_rmsnorm_rope_kv_store_fake(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    layer_name: str,
    kv_cache_dtype: str,
    cos_sin_cache: torch.Tensor,
    weight_q_norm: torch.Tensor,
    weight_k_norm: torch.Tensor,
    epsilon: float,
    head_size: int,
    head_size_v: int,
    q_size: int,
    kv_size: int,
    block_size: int,
    is_neox: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del (
        positions,
        layer_name,
        kv_cache_dtype,
        cos_sin_cache,
        weight_q_norm,
        weight_k_norm,
        epsilon,
        block_size,
        is_neox,
    )
    num_tokens = qkv.shape[0]
    q = torch.empty(
        (num_tokens, q_size // head_size, head_size),
        device=qkv.device,
        dtype=qkv.dtype,
    )
    k = torch.empty(
        (num_tokens, kv_size // head_size_v, head_size_v),
        device=qkv.device,
        dtype=qkv.dtype,
    )
    v = torch.empty_like(k)
    return q, k, v


direct_register_custom_op(
    op_name="fused_qkv_split_rmsnorm_rope_kv_store",
    op_func=fused_qkv_split_rmsnorm_rope_kv_store_impl,
    mutates_args=[],
    fake_impl=fused_qkv_split_rmsnorm_rope_kv_store_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)


__all__ = [
    "FusedQkvSplitRmsNormRopeAttention",
    "attention_forward",
    "init_kv_cache_quant_e5m2",
]
