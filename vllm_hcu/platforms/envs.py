# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

import os
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    VLLM_USE_NN : bool = False
    VLLM_HCU_USE_FLASH_ATTN: bool = False
    VLLM_HCU_USE_FLASH_ATTN_UNIFIED: bool = True
    VLLM_HCU_USE_FLASH_ATTN_VARLEN: bool = False
    VLLM_HCU_USE_CUSTOM_FLASH_ATTN: bool = False
    VLLM_HCU_USE_FLASHMLA: bool = False
    VLLM_USE_OPT_CAT: bool = False
    VLLM_HCU_USE_CAT_MLA: bool = False
    VLLM_HCU_DISABLE_DSA: bool = False
    VLLM_HCU_USE_FP8_MIXED_BATCH: bool = False
    VLLM_HCU_USE_CUSTOM_QUANTIZATION_GEMM : bool = False
    VLLM_HCU_USE_CUSTOM_OPS : bool = False
    VLLM_HCU_USE_CUSTOM_SILU_AND_MUL : bool = False
    VLLM_HCU_USE_CUSTOM_GEMMA_RMS_NORM : bool = False
    VLLM_HCU_USE_SKIP_WEIGHT_DEBUG : bool = False
    VLLM_HCU_USE_CUSTOM_TOPK_TOPP_SAMPLER : bool = False
    VLLM_HCU_USE_CUSTOM_RMS_NORM : bool = False
    VLLM_HCU_USE_CUSTOM_AITER_FLA : bool = False
    VLLM_HCU_PP_LAYER_PARTITION_D : Optional[str] = None
    VLLM_HCU_USE_FUSE_MOE_GATE : bool = False
    VLLM_HCU_USE_CUSTOM_CAUSAL_CONV1D : bool = False
    VLLM_HCU_USE_DP_CONNECTOR : bool = False
    VLLM_HCU_LIGHTLY_CP_THRESHOLD: int = 2048
    VLLM_HCU_USE_LIGHTOP_TOPK: bool = False
    VLLM_HCU_USE_LIGHTOP_SPARSE_MLA_TOPK: bool = True
    VLLM_HCU_USE_AITER_MHC: bool = True
    VLLM_HCU_USE_TILELANG_MHC_PRENORM: bool = True
    VLLM_HCU_DEEPSEEK_V4_ROCM_DECODE_FALLBACK: bool = False
    VLLM_HCU_DEEPSEEK_V4_ROCM_FAST_WOA: bool = True
    VLLM_HCU_ENABLE_DEEPSEEK_V4_MULTI_STREAM: bool = True
    VLLM_HCU_DEEPSEEK_V4_MULTI_STREAM_GEMM_TOKEN_THRESHOLD: int = 16384
    VLLM_HCU_DEEPSEEK_V4_MULTI_STREAM_COMPRESSOR_TOKEN_THRESHOLD: int = 16384
    VLLM_HCU_ENABLE_DEEPSEEK_V4_C128_COMPRESSOR: bool = True
    VLLM_HCU_ENABLE_DEEPSEEK_V4_CACHE_WINDOW: bool = False
    VLLM_HCU_USE_AITER_W8A8_FP8_MOE: bool = False
    VLLM_HCU_USE_LIGHTOP_MOE_ALIGN: bool = False
    VLLM_HCU_USE_LIGHTOP_EP_SCATTER: bool = True
    VLLM_HCU_USE_LIGHTOP_PER_TOKEN_QUANT_FP8: bool = False
    VLLM_HCU_USE_FUSED_RMS_QUANT: bool = False
    VLLM_HCU_USE_FUSED_SILU_MUL_QUANT: bool = False
    VLLM_HCU_USE_FUSED_QKV_SPLIT_RMS_ROPE_KVSTORE: bool = False
    VLLM_HCU_FLASH_ATTN_BLOCK_ALIGNMENT_SIZE: Optional[int] = None
    VLLM_HCU_MAMBA_SSM_CACHE_DTYPE: bool = False
    VLLM_HCU_USE_PD_SPLIT: bool = False
    VLLM_HCU_USE_AITER_W4A16_MOE: bool = False
    VLLM_HCU_USE_TORCH_EPLB_MAP_RECORD: bool = False
    VLLM_HCU_USE_AITER_W16A16_MOE_SHUFFLE: bool = True
    VLLM_HCU_USE_AITER_MOE_CONFIG: bool = True
    VLLM_HCU_MOONCAKE_TTFT_TRACE: bool = False
    VLLM_HCU_DEEPEP_NUM_SMS: Optional[int] = None
    VLLM_HCU_DPSK_V4_DEEPEP_LL_USE_HCU_DISPATCH_API: bool = False
    VLLM_HCU_SHARED_EXPERTS_STREAM_FORCE: bool = False
    VLLM_HCU_SHARED_EXPERTS_EARLY_LAUNCH: bool = False
    VLLM_HCU_ENABLE_REQUEST_CUDAGRAPH_BUCKETS: bool = False

def maybe_convert_int(value: Optional[str]) -> Optional[int]:
    """
    如果值是None，则返回None；否则将字符串转换为整数并返回。
    
    Args:
        value (Optional[str], optional): 要转换的可选字符串. Defaults to None.
    
    Returns:
        Optional[int]: 如果值是None，则返回None；否则将字符串转换为整数并返回.
    """
    if value is None:
        return None
    return int(value)


def resolve_hcu_flash_attn_mode(explicit_mode: Optional[str]) -> str:
    """Resolve an HCU flash-attention sub-mode without touching vLLM schema.

    An explicit sidecar value wins.  The legacy environment switches retain
    their historical priority, and the flagless HCU default uses the CUTLASS
    unified implementation.
    """

    if explicit_mode is not None:
        normalized = explicit_mode.lower()
        aliases = {
            "unified": "cutlass",
            "classic": "classic",
            "cutlass": "cutlass",
            "varlen": "varlen",
            "custom": "custom",
        }
        if normalized not in aliases:
            raise ValueError(
                f"Unsupported HCU flash attention mode: {explicit_mode!r}."
            )
        return aliases[normalized]

    if os.environ.get(
        "VLLM_HCU_USE_CUSTOM_FLASH_ATTN", "False"
    ).lower() in ("true", "1"):
        return "custom"
    if os.environ.get(
        "VLLM_HCU_USE_FLASH_ATTN_VARLEN", "False"
    ).lower() in ("true", "1"):
        return "varlen"
    # Only an explicitly enabled legacy switch participates in priority
    # resolution. The flagless CUTLASS default is applied below, so an
    # explicitly requested classic mode can still take effect.
    if os.environ.get(
        "VLLM_HCU_USE_FLASH_ATTN_UNIFIED", "False"
    ).lower() in ("true", "1"):
        return "cutlass"
    if os.environ.get("VLLM_HCU_USE_FLASH_ATTN", "False").lower() in (
        "true",
        "1",
    ):
        return "classic"
    return "cutlass"

hcu_vllm_environment_variables: dict[str, Callable[[], Any]] = {
    # path to the logs of redirect-output, abstrac of related are ok

    # If set, vLLM will transpose weight to use nn layout
    "VLLM_USE_NN":
    lambda: (os.environ.get("VLLM_USE_NN", "True").lower() in 
             ("true", "1")),
    # vLLM will use FlashAttention Backend on hcu, office attention layerout blocksize 128
    "VLLM_HCU_USE_FLASH_ATTN":
    lambda: (os.environ.get("VLLM_HCU_USE_FLASH_ATTN", "False").lower() in
             ("true", "1")),
    # vLLM will use FlashAttention Backend (varlen_fwd_unified) on hcu, cutlass attention layerout blocksize 64 for qwen3.5
    "VLLM_HCU_USE_FLASH_ATTN_UNIFIED":
    lambda: (os.environ.get("VLLM_HCU_USE_FLASH_ATTN_UNIFIED", "True").lower() in
             ("true", "1")),
    # Select flash_attn.flash_attn_varlen_func without changing legacy paths.
    "VLLM_HCU_USE_FLASH_ATTN_VARLEN":
    lambda: (os.environ.get("VLLM_HCU_USE_FLASH_ATTN_VARLEN", "False").lower() in
             ("true", "1")),
    # vLLM will use custom FlashAttention (convert kv cache) Backend on hcu,  not office attention layerout blocksize 64 
    "VLLM_HCU_USE_CUSTOM_FLASH_ATTN":
    lambda: (os.environ.get("VLLM_HCU_USE_CUSTOM_FLASH_ATTN", "False").lower() in
             ("true", "1")),
    # vLLM will use FlashMLA Backend on hcu
    "VLLM_HCU_USE_FLASHMLA":
    lambda: (os.environ.get("VLLM_HCU_USE_FLASHMLA", "False").lower() in
             ("true", "1")),
    # vLLM will use use opt cat 
    "VLLM_USE_OPT_CAT":
    lambda: (os.environ.get("VLLM_USE_OPT_CAT", "True").lower() in
             ("true", "1")),
    # vLLM will use use fused cat and mla
    "VLLM_HCU_USE_CAT_MLA":
    lambda: (os.environ.get("VLLM_HCU_USE_CAT_MLA", "True").lower() in
             ("true", "1")),
    # If set, vllm will disable DSA
    "VLLM_HCU_DISABLE_DSA":
        lambda: (os.environ.get("VLLM_HCU_DISABLE_DSA", "False").lower() in
                    ("true", "1")),  
    # If set, vllm will use mixed P/D batch for fp8 (num_attention_heads / tp < 32)
    "VLLM_HCU_USE_FP8_MIXED_BATCH":
        lambda: (os.getenv('VLLM_HCU_USE_FP8_MIXED_BATCH', 'True').lower() in
                 ("true", "1")),  
    # If set, control hcu custom gemm including w8a8 int8/fp8 etc
    "VLLM_HCU_USE_CUSTOM_QUANTIZATION_GEMM":
    lambda: (os.environ.get("VLLM_HCU_USE_CUSTOM_QUANTIZATION_GEMM", "True").lower() in
             ("true", "1")),
    # If set, control hcu custom unfused or fused kernel ops
    "VLLM_HCU_USE_CUSTOM_OPS":
    lambda: (os.environ.get("VLLM_HCU_USE_CUSTOM_OPS", "True").lower() in
             ("true", "1")),
    # If set, control hcu custom silu and mul op
    "VLLM_HCU_USE_CUSTOM_SILU_AND_MUL":
    lambda: (os.environ.get("VLLM_HCU_USE_CUSTOM_SILU_AND_MUL", "True").lower() in
             ("true", "1")),
    "VLLM_HCU_USE_CUSTOM_GEMMA_RMS_NORM":
    lambda: (os.environ.get("VLLM_HCU_USE_CUSTOM_GEMMA_RMS_NORM", "True").lower() in
             ("true", "1")),
    "VLLM_HCU_USE_SKIP_WEIGHT_DEBUG":
    lambda: (os.environ.get("VLLM_HCU_USE_SKIP_WEIGHT_DEBUG", "False").lower() in
             ("true", "1")),
    "VLLM_HCU_USE_CUSTOM_TOPK_TOPP_SAMPLER":
    lambda: (os.environ.get("VLLM_HCU_USE_CUSTOM_TOPK_TOPP_SAMPLER", "False").lower() in
            ("true", "1")),
    "VLLM_HCU_USE_CUSTOM_RMS_NORM":
    lambda: (os.environ.get("VLLM_HCU_USE_CUSTOM_RMS_NORM", "True").lower() in
             ("true", "1")),
    "VLLM_HCU_USE_CUSTOM_AITER_FLA":
    lambda: (os.environ.get("VLLM_HCU_USE_CUSTOM_AITER_FLA", "True").lower() in
             ("true", "1")),
            
    # Pipeline stage partition strategy
    "VLLM_HCU_PP_LAYER_PARTITION_D":
    lambda: os.getenv("VLLM_HCU_PP_LAYER_PARTITION_D", None),
    "VLLM_HCU_USE_FUSE_MOE_GATE":
    lambda: (os.environ.get("VLLM_HCU_USE_FUSE_MOE_GATE", "True").lower() in
             ("true", "1")),
    "VLLM_HCU_USE_CUSTOM_CAUSAL_CONV1D":
    lambda: (os.environ.get("VLLM_HCU_USE_CUSTOM_CAUSAL_CONV1D", "True").lower() in
             ("true", "1")),
    # vllm use dp connector
    "VLLM_HCU_USE_DP_CONNECTOR":
    lambda: (os.environ.get("VLLM_HCU_USE_DP_CONNECTOR", "False").lower() in
             ("true", "1")),

    # MLA_CP enable threshold
    "VLLM_HCU_LIGHTLY_CP_THRESHOLD":
        lambda: int(os.getenv("VLLM_HCU_LIGHTLY_CP_THRESHOLD", "2048")),

    # If use lightop top_k_per_row_prefill impl, please set True
    "VLLM_HCU_USE_LIGHTOP_TOPK":
        lambda: (os.environ.get("VLLM_HCU_USE_LIGHTOP_TOPK", "True").lower() in
                    ("true", "1")),

    # If set, use LightOp top_k_per_row kernels for sparse MLA prefill/decode indices.
    "VLLM_HCU_USE_LIGHTOP_SPARSE_MLA_TOPK":
        lambda: (os.environ.get("VLLM_HCU_USE_LIGHTOP_SPARSE_MLA_TOPK", "True").lower() in
                    ("true", "1")),

    # If use AITER MHC impl, please set True
    "VLLM_HCU_USE_AITER_MHC":
        lambda: (os.environ.get("VLLM_HCU_USE_AITER_MHC", "True").lower() in
                    ("true", "1")),

    # If use TileLang MHC prenorm impl, please set True
    "VLLM_HCU_USE_TILELANG_MHC_PRENORM":
        lambda: (os.environ.get("VLLM_HCU_USE_TILELANG_MHC_PRENORM", "True").lower() in
                    ("true", "1")),

    # Whether to route DeepSeek V4 ROCm decode through the legacy fallback.
    "VLLM_HCU_DEEPSEEK_V4_ROCM_DECODE_FALLBACK":
        lambda: (os.environ.get("VLLM_HCU_DEEPSEEK_V4_ROCM_DECODE_FALLBACK", "False").lower() in
                    ("true", "1")),

    # Whether to use the local inverse-RoPE + BF16 einsum path for DeepSeek V4 ROCm WO_A.
    "VLLM_HCU_DEEPSEEK_V4_ROCM_FAST_WOA":
        lambda: (os.environ.get("VLLM_HCU_DEEPSEEK_V4_ROCM_FAST_WOA", "True").lower() in
                    ("true", "1")),

    # Whether to create aux streams for DeepSeek V4 attention input GEMM overlap.
    "VLLM_HCU_ENABLE_DEEPSEEK_V4_MULTI_STREAM":
        lambda: (os.environ.get("VLLM_HCU_ENABLE_DEEPSEEK_V4_MULTI_STREAM", "True").lower() in
                    ("true", "1")),

    # Token-count cutoff for DeepSeek V4 multi-stream attention input GEMM overlap.
    "VLLM_HCU_DEEPSEEK_V4_MULTI_STREAM_GEMM_TOKEN_THRESHOLD":
        lambda: int(os.getenv("VLLM_HCU_DEEPSEEK_V4_MULTI_STREAM_GEMM_TOKEN_THRESHOLD", "16384")),

    # Token threshold for overlapping DeepSeek V4 wq_b/cache insert with sparse compressor.
    "VLLM_HCU_DEEPSEEK_V4_MULTI_STREAM_COMPRESSOR_TOKEN_THRESHOLD":
        lambda: int(os.getenv("VLLM_HCU_DEEPSEEK_V4_MULTI_STREAM_COMPRESSOR_TOKEN_THRESHOLD", "16384")),

    # Use the split C128 compressor path that writes state and then stores only
    # C128 boundary tokens to KV cache.
    "VLLM_HCU_ENABLE_DEEPSEEK_V4_C128_COMPRESSOR":
        lambda: (os.environ.get("VLLM_HCU_ENABLE_DEEPSEEK_V4_C128_COMPRESSOR", "True").lower() in
                    ("true", "1")),

    # If set, split DeepSeek V4 cache kernels into block windows once cache
    # block count is large enough to risk large-offset addressing.
    "VLLM_HCU_ENABLE_DEEPSEEK_V4_CACHE_WINDOW":
        lambda: (os.environ.get("VLLM_HCU_ENABLE_DEEPSEEK_V4_CACHE_WINDOW", "False").lower() in
                    ("true", "1")),

    # W8A8 fp8 moe use aiter 
    "VLLM_HCU_USE_AITER_W8A8_FP8_MOE":
        lambda: (os.environ.get("VLLM_HCU_USE_AITER_W8A8_FP8_MOE", "False").lower() in
                    ("true", "1")),
    "VLLM_HCU_USE_LIGHTOP_MOE_ALIGN":
        lambda: (os.environ.get("VLLM_HCU_USE_LIGHTOP_MOE_ALIGN", "True").lower() in
                    ("true", "1")),

    # DeepEP HT permute: use lightop op.ep_scatter when True (also requires
    # VLLM_HCU_USE_CUSTOM_OPS), triton kernels when False.
    "VLLM_HCU_USE_LIGHTOP_EP_SCATTER":
        lambda: (os.environ.get("VLLM_HCU_USE_LIGHTOP_EP_SCATTER", "True").lower() in
                 ("true", "1")),

    # If set, use LightOp per-token fp8 quant for dynamic PER_TOKEN QuantFP8.
    "VLLM_HCU_USE_LIGHTOP_PER_TOKEN_QUANT_FP8":
        lambda: (os.environ.get("VLLM_HCU_USE_LIGHTOP_PER_TOKEN_QUANT_FP8", "False").lower() in
                 ("true", "1")),

    # If use fused rmsnorm and quant, please set True
    "VLLM_HCU_USE_FUSED_RMS_QUANT":
        lambda: (os.environ.get("VLLM_HCU_USE_FUSED_RMS_QUANT", "False").lower() in
                 ("true", "1")),

    # If use fused silu and mul and quant, please set True
    "VLLM_HCU_USE_FUSED_SILU_MUL_QUANT":
        lambda: (os.environ.get("VLLM_HCU_USE_FUSED_SILU_MUL_QUANT", "True").lower() in
                 ("true", "1")),

    #If set to 1/True, enable fuse split qkv+rmsnorm+rope+kv_update just like glm4.7 moe attention.
    "VLLM_HCU_USE_FUSED_QKV_SPLIT_RMS_ROPE_KVSTORE":
        lambda: (os.environ.get("VLLM_HCU_USE_FUSED_QKV_SPLIT_RMS_ROPE_KVSTORE", "True").lower() in
                 ("true", "1")),

    # Optional override for flash-attn kernel block alignment size.
    "VLLM_HCU_FLASH_ATTN_BLOCK_ALIGNMENT_SIZE":
    lambda: maybe_convert_int(
        os.getenv("VLLM_HCU_FLASH_ATTN_BLOCK_ALIGNMENT_SIZE", None)
    ),

    # Force mamba SSM cache dtype to "auto" when enabled.
    "VLLM_HCU_MAMBA_SSM_CACHE_DTYPE":
    lambda: (os.environ.get("VLLM_HCU_MAMBA_SSM_CACHE_DTYPE", "False").lower() in
             ("true", "1")),

    # vLLM will split prefill and decode, not mix up
    "VLLM_HCU_USE_PD_SPLIT":
        lambda: (os.environ.get("VLLM_HCU_USE_PD_SPLIT", "False").lower() in
                 ("true", "1")),

    # If use custom AITER_W4A16_MOE impl, please set True
    "VLLM_HCU_USE_AITER_W4A16_MOE":
        lambda: (os.environ.get("VLLM_HCU_USE_AITER_W4A16_MOE", "True").lower() in
                    ("true", "1")),

    # If set, use torch ops for EPLB logical-to-physical mapping and load
    # recording instead of the fused Triton kernel.
    "VLLM_HCU_USE_TORCH_EPLB_MAP_RECORD":
        lambda: (os.environ.get("VLLM_HCU_USE_TORCH_EPLB_MAP_RECORD", "False").lower() in
                    ("true", "1")),

    # If use shuffle for AITER W16A16 MoE, please set True (default True)
    "VLLM_HCU_USE_AITER_W16A16_MOE_SHUFFLE":
        lambda: (os.environ.get("VLLM_HCU_USE_AITER_W16A16_MOE_SHUFFLE", "True").lower() in
                    ("true", "1")),

    # If use AITER MoE config (solution_id lookup), please set True (default True)
    "VLLM_HCU_USE_AITER_MOE_CONFIG":
        lambda: (os.environ.get("VLLM_HCU_USE_AITER_MOE_CONFIG", "True").lower() in
                    ("true", "1")),

    # Emit Mooncake TTFT_EVENT DEBUG lines with wall-clock ts for PD TTFT analysis.
    "VLLM_HCU_MOONCAKE_TTFT_TRACE":
        lambda: (os.environ.get("VLLM_HCU_MOONCAKE_TTFT_TRACE", "False").lower() in
                    ("true", "1")),

    # Optional override for DeepEP Buffer.set_num_sms().
    "VLLM_HCU_DEEPEP_NUM_SMS":
    lambda: maybe_convert_int(os.getenv("VLLM_HCU_DEEPEP_NUM_SMS", None)),

    # If set, use the HCU DeepEP LL dispatch API with topk weights and
    # quant_group_size. Default keeps the legacy DeepEP API path.
    "VLLM_HCU_DPSK_V4_DEEPEP_LL_USE_HCU_DISPATCH_API":
        lambda: (os.environ.get("VLLM_HCU_DPSK_V4_DEEPEP_LL_USE_HCU_DISPATCH_API", "False").lower() in
                    ("true", "1")),

    # Prefer running shared_experts on the auxiliary stream even when the
    # modular MoE kernel can own shared_experts internally.
    "VLLM_HCU_SHARED_EXPERTS_STREAM_FORCE":
        lambda: (os.environ.get("VLLM_HCU_SHARED_EXPERTS_STREAM_FORCE", "False").lower() in
                    ("true", "1")),

    # Launch shared_experts on the auxiliary stream before routed MoE/DeepEP,
    # and wait only when the shared output is consumed.
    "VLLM_HCU_SHARED_EXPERTS_EARLY_LAUNCH":
        lambda: (os.environ.get("VLLM_HCU_SHARED_EXPERTS_EARLY_LAUNCH", "False").lower() in
                    ("true", "1")),

    # Enable request-count-oriented cudagraph capture buckets up to request size 256.
    "VLLM_HCU_ENABLE_REQUEST_CUDAGRAPH_BUCKETS":
        lambda: (os.environ.get("VLLM_HCU_ENABLE_REQUEST_CUDAGRAPH_BUCKETS", "False").lower() in
                    ("true", "1")),
}

# end-env-vars-definition

def __getattr__(name: str):
    """
    当调用不存在的属性时，该函数被调用。如果属性是hcu_vllm_environment_variables中的一个，则返回相应的值。否则引发AttributeError异常。
    
    Args:
        name (str): 要获取的属性名称。
    
    Raises:
        AttributeError (Exception): 如果属性不是hcu_vllm_environment_variables中的一个，则会引发此异常。
    
    Returns:
        Any, optional: 如果属性是hcu_vllm_environment_variables中的一个，则返回相应的值；否则返回None。
    """
    # lazy evaluation of environment variables
    if name in hcu_vllm_environment_variables:
        return hcu_vllm_environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """
    返回一个包含所有可见的变量名称的列表。
    
    返回值（list）：一个包含所有可见的变量名称的列表，这些变量是通过`xhcu_vllm_environment_variables`字典定义的。
    
    Returns:
        List[str]: 一个包含所有可见的变量名称的列表。
                   这些变量是通过`hcu_vllm_environment_variables`字典定义的。
    """
    return list(hcu_vllm_environment_variables.keys())


def is_set(name: str):
    """Check if an environment variable is explicitly set."""
    if name in hcu_vllm_environment_variables:
        return name in os.environ
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
