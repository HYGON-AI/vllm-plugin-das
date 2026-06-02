# SPDX-License-Identifier: Apache-2.0

"""
Patch for vllm.model_executor.layers.attention.attention unified_kv_cache_update
"""

PATCHES = [
(
    "import vllm.envs as envs",
    "import vllm.envs as envs\nimport vllm_hcu.platforms.envs as henvs",
),

(
"""
    return torch.empty(0, device=kv_cache.device, dtype=kv_cache.dtype)
""",
"""
    if henvs.VLLM_HCU_USE_CUSTOM_FLASH_ATTN:
        return torch.empty(0, device=key.device, dtype=key.dtype)
    else:
        return torch.empty(0, device=kv_cache.device, dtype=kv_cache.dtype)
""",
),

####################################### support fp8_e5m2 ########################################
(
"""
        if layer.kv_cache_dtype == "fp8_e5m2":
            raise ValueError("fp8_e5m2 kv-cache is not supported with fp8 checkpoints.")
""",
"""
        #if layer.kv_cache_dtype == "fp8_e5m2":
        #    raise ValueError("fp8_e5m2 kv-cache is not supported with fp8 checkpoints.")
""",
),

(
"""
            assert self.kv_cache_dtype in {"fp8", "fp8_e4m3", "nvfp4"}
""",
"""
            assert self.kv_cache_dtype in {"fp8", "fp8_e4m3", "fp8_e5m2", "nvfp4"}
""",
),
####################################### support fp8_e5m2 ########################################

####################################### FusedQkvSplitRmsNormRopeAttention ########################################
(
"""
def maybe_calc_kv_scales(
""",
"""
class FusedQkvSplitRmsNormRopeAttention(Attention):
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
        super().__init__(num_heads, head_size, scale,
                       num_kv_heads, alibi_slopes,
                       use_alibi_sqrt, cache_config,
                       quant_config, logits_soft_cap,
                       per_layer_sliding_window,
                       prefix, attn_type,
                       kv_sharing_target_layer_name,
                       attn_backend,
                       head_size_v,
                       **extra_impl_args)
        
        vllm_config = get_current_vllm_config()
        self.block_size = vllm_config.cache_config.block_size
        
    def forward(
        self,
        qkv: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        weight_q_norm: torch.Tensor,
        weight_k_norm: torch.Tensor,
        epsilon: float,
        # For some alternate attention backends like MLA the attention output
        # shape does not match the query shape, so we optionally let the model
        # definition specify the output tensor shape.
        output_shape: torch.Size | None = None,
        is_neox: bool = False,
    ) -> torch.Tensor:

        output_dtype = qkv.dtype
        num_tokens = qkv.shape[0]

        if output_shape is None:
            # Handle both 2D [num_tokens, hidden] and
            # 3D [num_tokens, heads, head_dim] query
            output_shape = torch.Size(
                (num_tokens, self.num_heads * self.head_size_v)
            )
        output = torch.empty(output_shape, dtype=output_dtype, device=qkv.device)
        output = output.view(-1, self.num_heads, self.head_size_v)
        hidden_size = output_shape[-1]

        q_size = self.num_heads * self.head_size
        kv_size = self.num_kv_heads * self.head_size
        query, key, value = torch.ops.vllm.fused_qkv_split_rmsnorm_rope_kv_store(qkv=qkv,
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
                                                 is_neox=is_neox)

        kv_cache_dummy_dep = None
        torch.ops.vllm.unified_attention_with_output(
            query,
            key,
            value,
            output,
            self.layer_name,
            kv_cache_dummy_dep=kv_cache_dummy_dep,
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
    is_neox: bool = False)-> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    num_tokens = qkv.shape[0]
    forward_context = get_forward_context()
    
    slot_mapping = forward_context.slot_mapping
    layer_slot_mapping = slot_mapping.get(layer_name)
    assert isinstance(slot_mapping, dict), (
        f"Expected slot_mapping to be a dict, got {type(slot_mapping)}. "
    )
    
    attn_layer = forward_context.no_compile_layers[layer_name]
    kv_cache = attn_layer.kv_cache

    if layer_slot_mapping is not None:
        if henvs.VLLM_HCU_USE_CUSTOM_FLASH_ATTN:
            key_cache, value_cache = kv_cache
        else:
            key_cache, value_cache = kv_cache.unbind(0)

        if kv_cache_dtype.startswith("fp8"):
            # queries are quantized in the attention layer
            from vllm_hcu.v1.attention.backends.flash_attn import HcuFlashAttentionBackend
            kv_cache_dtype = HcuFlashAttentionBackend.get_fp8_dtype_for_flashattn(
                kv_cache_dtype
            )
            key_cache = key_cache.view(kv_cache_dtype)
            value_cache = value_cache.view(kv_cache_dtype)
    else:
        key_cache = torch.empty([0], device=qkv.device, dtype=qkv.dtype)
        value_cache = torch.empty([0], device=qkv.device, dtype=qkv.dtype)

    from lightop import split_qkv_rms_rotary_embedding_fuse_with_kv_store_quant
    q, k, v = split_qkv_rms_rotary_embedding_fuse_with_kv_store_quant(positions,
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
    q = q.contiguous().view(num_tokens, q_size//head_size, head_size)
    k = k.contiguous().view(num_tokens, kv_size//head_size_v, head_size_v)
    v = v.contiguous().view(num_tokens, kv_size//head_size_v, head_size_v)
    return q, k ,v


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
    num_token = qkv.shape[0]
    q = torch.empty((num_token, q_size//head_size, head_size), device=qkv.device, dtype=qkv.dtype)
    k = torch.empty((num_token, kv_size//head_size_v, head_size_v), device=qkv.device, dtype=qkv.dtype)
    v = torch.empty((num_token, kv_size//head_size_v, head_size_v), device=qkv.device, dtype=qkv.dtype)
    return q, k, v

direct_register_custom_op(
    op_name="fused_qkv_split_rmsnorm_rope_kv_store",
    op_func=fused_qkv_split_rmsnorm_rope_kv_store_impl,
    mutates_args=["qkv", "positions"],
    fake_impl=fused_qkv_split_rmsnorm_rope_kv_store_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)

def maybe_calc_kv_scales(
""",
),
####################################### FusedQkvSplitRmsNormRopeAttention ########################################

]