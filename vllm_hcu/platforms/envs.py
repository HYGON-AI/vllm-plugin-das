# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

import functools
import logging
import os
from typing import TYPE_CHECKING, Any, Callable, Optional

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    VLLM_USE_NN : bool = False
    VLLM_HCU_USE_FLASH_ATTN: bool = False
    VLLM_HCU_USE_FLASH_ATTN_UNIFIED: bool = False
    VLLM_HCU_USE_FLASH_ATTN_VARLEN: bool = True
    VLLM_HCU_USE_FLASHMLA: bool = False
    VLLM_USE_OPT_CAT: bool = False
    VLLM_HCU_USE_CAT_MLA: bool = True
    VLLM_HCU_DISABLE_DSA: bool = False
    VLLM_HCU_USE_FP8_MIXED_BATCH: bool = False
    VLLM_HCU_USE_CUSTOM_QUANTIZATION_GEMM : bool = False
    VLLM_HCU_USE_CUSTOM_OPS : bool = False
    VLLM_HCU_GLM53_ROUTER_GEMV: bool = True
    VLLM_HCU_QSA_BACKEND: str = "cutlass"
    VLLM_HCU_USE_CUSTOM_SILU_AND_MUL : bool = False
    VLLM_HCU_USE_CUSTOM_GEMMA_RMS_NORM : bool = False
    VLLM_HCU_USE_SKIP_WEIGHT_DEBUG : bool = False
    VLLM_HCU_USE_CUSTOM_TOPK_TOPP_SAMPLER : bool = False
    VLLM_HCU_USE_CUSTOM_RMS_NORM : bool = False
    VLLM_HCU_USE_CUSTOM_AITER_FLA : bool = False
    VLLM_HCU_GDN_DECODE_BACKEND: str = "hcu-aiter"
    VLLM_HCU_PP_LAYER_PARTITION_D : Optional[str] = None
    VLLM_HCU_USE_FUSE_MOE_GATE : bool = True
    VLLM_HCU_USE_LIGHTOP_SQRTSOFTPLUS_GATE: bool = True
    VLLM_HCU_USE_LIGHTOP_QWEN_RMSNORM_GATED: bool = True
    VLLM_HCU_USE_LIGHTOP_W16A16_MOE: bool = False
    VLLM_HCU_USE_LIGHTOP_AWQ: bool = False
    VLLM_HCU_INDEXER_KCACHE_LAYOUT: str = "preshuffle"
    VLLM_HCU_USE_LIGHTOP_MLA_DECODE_CAT: bool = True
    VLLM_HCU_USE_CUSTOM_CAUSAL_CONV1D : bool = False
    VLLM_HCU_USE_DP_CONNECTOR : bool = False
    VLLM_HCU_LIGHTLY_CP_THRESHOLD: int = 2048
    VLLM_HCU_USE_LIGHTOP_TOPK: bool = False
    VLLM_HCU_USE_LIGHTOP_SPARSE_MLA_TOPK: bool = True
    VLLM_HCU_USE_LIGHTOP_FAST_TOPK_TRANSFORM: bool = True
    VLLM_HCU_USE_AITER_MHC: bool = True
    VLLM_HCU_USE_TILELANG_MHC_PRENORM: bool = True
    VLLM_HCU_DEEPSEEK_V4_ROCM_DECODE_FALLBACK: bool = False
    VLLM_HCU_DEEPSEEK_V4_ROCM_FLASHMLA_DECODE: bool = True
    VLLM_HCU_DEEPSEEK_V4_ROCM_FLASHMLA_PREFILL: bool = True
    VLLM_HCU_DEEPSEEK_V4_ROCM_FLASHMLA_PREFILL_HEAD_PADDING: bool = True
    VLLM_HCU_DEEPSEEK_V4_ROCM_SPARSE_PREFILL_BACKEND: str = "flashmla"
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
    VLLM_HCU_FUSED_MOE_CHUNK_SIZE: Optional[int] = None
    VLLM_HCU_USE_GLOBAL_MOE_CACHE: bool = False
    VLLM_HCU_USE_FUSED_RMS_QUANT: bool = False
    VLLM_HCU_USE_FUSE_SILU_AND_MUL: bool = False
    VLLM_HCU_USE_FUSED_SILU_MUL_QUANT: bool = False
    VLLM_HCU_USE_FUSED_QKV_SPLIT_RMS_ROPE_KVSTORE: bool = False
    VLLM_HCU_USE_FUSED_RMS_ROPE: bool = True
    VLLM_HCU_FLASH_ATTN_BLOCK_ALIGNMENT_SIZE: Optional[int] = None
    VLLM_HCU_MAMBA_SSM_CACHE_DTYPE: bool = False
    VLLM_HCU_USE_AITER_W4A16_MOE: bool = False
    VLLM_HCU_USE_TORCH_EPLB_MAP_RECORD: bool = False
    VLLM_HCU_USE_AITER_MOE_SHUFFLE: bool = True
    VLLM_HCU_USE_AITER_MOE_CONFIG: bool = True
    VLLM_HCU_MOONCAKE_TTFT_TRACE: bool = False
    VLLM_HCU_DEEPEP_NUM_SMS: Optional[int] = None
    VLLM_HCU_DPSK_V4_DEEPEP_LL_USE_HCU_DISPATCH_API: bool = False
    VLLM_HCU_SHARED_EXPERTS_STREAM_FORCE: bool = False
    VLLM_HCU_DEEPSEEK_V4_PCP_LOCAL_QUERIES: bool = True
    VLLM_HCU_KV_ZERO_BY_GROUP: bool = True
    VLLM_HCU_DEEPSEEK_V4_PCP_LOCAL_SLICES: bool = True
    VLLM_HCU_COMPRESSOR_GEMM_BACKEND: str = "blaslt"
    VLLM_HCU_C128_HEAD_SPLITS: int = 4
    VLLM_HCU_INVERSE_ROPE_SUFFIX: bool = True
    VLLM_HCU_SHARED_EXPERTS_EARLY_LAUNCH: bool = False
    VLLM_HCU_DEEPEP_AUTO_SHARED_WEIGHTS: bool = True
    VLLM_HCU_MOE_WORKSPACE_GROWTH_MB: int = 512
    VLLM_HCU_DEEPEP_HT_EXACT_LAYOUT: bool = True
    VLLM_HCU_SHARED_EXPERTS_DEEPEP_EPLB_OVERLAP: bool = True
    VLLM_HCU_ENABLE_REQUEST_CUDAGRAPH_BUCKETS: bool = False
    VLLM_HCU_PLE_CPU_OFFLOAD: bool = False
    VLLM_HCU_PLE_PREFETCH_STREAM: bool = False
    VLLM_HCU_QWEN4_EXP_HC_SP: bool = True
    VLLM_HCU_DP_INT8_ALLGATHER: bool = False
    VLLM_HCU_MHC_FUSED_RMSNORM: bool = True
    VLLM_HCU_GLM5NEXT_KDA_STRIDED_INPUTS: bool = True

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


def _environment_flag(raw: str) -> bool:
    return raw.lower() in ("true", "1")


def custom_ops_enabled() -> bool:
    """Return whether the HCU custom-op master switch is enabled."""
    return _environment_flag(os.environ.get("VLLM_HCU_USE_CUSTOM_OPS", "True"))


def optional_custom_op_enabled(feature_enabled: bool = True) -> bool:
    """Resolve an optional optimized operator under the process master.

    Read the master directly from ``os.environ`` so a materialized lazy module
    attribute cannot freeze the effective policy for later consumers.
    """

    return custom_ops_enabled() and bool(feature_enabled)


def linear_gate_pcp_shard_enabled(
    *, tp_size: int, pcp_size: int, pp_size: int = 1
) -> bool:
    """Resolve HY4 linear-gate PCP sharding under the custom-op master."""

    raw = os.environ.get("VLLM_HCU_ENABLE_LINEAR_GATE_PCP_SHARD")
    if raw is None:
        requested = (tp_size, pcp_size, pp_size) == (1, 8, 1)
    else:
        requested = raw.strip().lower() in ("1", "true", "yes", "on")
    return optional_custom_op_enabled(requested)


def qwen4_exp_hc_sp_enabled() -> bool:
    """Resolve Qwen4Exp HC sequence parallelism under the master switch."""

    return optional_custom_op_enabled(
        _environment_flag(os.environ.get("VLLM_HCU_QWEN4_EXP_HC_SP", "True"))
    )


def dp_int8_allgather_enabled() -> bool:
    """Resolve INT8 MoE AllGatherv transport under the custom-op master."""

    return optional_custom_op_enabled(
        _environment_flag(
            os.environ.get("VLLM_HCU_DP_INT8_ALLGATHER", "False")
        )
    )


def k3_sp_moe_requested() -> bool:
    """Raw Kimi-K3 SP-MoE opt-in parse (child switch only).

    Keep this the ONLY parser of ``VLLM_HCU_K3_SP_MOE`` spelling: it feeds
    :func:`k3_sp_moe_enabled` (master AND child), which the platform
    persists into the compilation-cache partition and the SP patch's
    ``_switch_enabled`` delegates to, so neither can disagree on spelling.
    """
    return os.environ.get("VLLM_HCU_K3_SP_MOE", "0").strip().lower() in (
        "1",
        "true",
    )


def k3_sp_moe_enabled() -> bool:
    """Effective Kimi-K3 SP-MoE opt-in under the custom-op master.

    The row-sharded forward rewires KimiDecoderLayer at patch-apply time
    (``o_proj.reduce_results`` plus the ``hcu_k3_sp_*`` collectives), so the
    resolved master-and-child result must be persisted before VllmConfig
    computes its compilation-cache hash: opposite-mode AOT graphs cannot
    share an entry.
    """
    return optional_custom_op_enabled(k3_sp_moe_requested())


def glm53_router_gemv_enabled() -> bool:
    """Resolve the GLM-5.3 router GEMV under the custom-op master switch."""

    return optional_custom_op_enabled(
        _environment_flag(
            os.environ.get("VLLM_HCU_GLM53_ROUTER_GEMV", "True")
        )
    )


def custom_quantization_gemm_enabled() -> bool:
    """Resolve the quantized-GEMM child switch under the process master."""

    return optional_custom_op_enabled(
        _environment_flag(
            os.environ.get("VLLM_HCU_USE_CUSTOM_QUANTIZATION_GEMM", "True")
        )
    )


def lightop_awq_enabled() -> bool:
    """Resolve the opt-in LightOp AutoAWQ backend under the master switch."""

    return optional_custom_op_enabled(
        _environment_flag(os.environ.get("VLLM_HCU_USE_LIGHTOP_AWQ", "False"))
    )


INDEXER_KCACHE_LAYOUT_ENV = "VLLM_HCU_INDEXER_KCACHE_LAYOUT"
INDEXER_KCACHE_LAYOUTS = ("preshuffle", "normal")


def indexer_kcache_layout() -> str:
    """Resolve the gfx938 GLM5Next sparse-indexer K-cache page layout.

    ``preshuffle`` keeps the official 16x16 tiled KPool pages, decoded by the
    HCU Triton direct reader. ``normal`` writes token-major pages and decodes
    them with LightOp ``paged_mqa_logits``. The layout is part of the KV-cache
    contents, so every rank and every disaggregated prefill/decode peer must
    use the same value. Disabling the HCU custom-op master switch always keeps
    the official ``preshuffle`` layout.
    """

    raw = os.environ.get(INDEXER_KCACHE_LAYOUT_ENV, "preshuffle").strip().lower()
    if raw not in INDEXER_KCACHE_LAYOUTS:
        raise ValueError(
            f"{INDEXER_KCACHE_LAYOUT_ENV} must be one of "
            f"{INDEXER_KCACHE_LAYOUTS}, got {raw!r}"
        )
    if not custom_ops_enabled():
        return "preshuffle"
    return raw


def ple_prefetch_enabled() -> bool:
    """Resolve PLE stream prefetch under the custom-op master switch."""
    return custom_ops_enabled() and _environment_flag(
        os.environ.get("VLLM_HCU_PLE_PREFETCH_STREAM", "False")
    )


def fused_qwen4_exp_ple_enabled() -> bool:
    """Select fused Qwen4Exp PLE under the custom-op master switch."""

    return custom_ops_enabled()


def fused_qwen3_rms_rope_enabled() -> bool:
    """Resolve Qwen3 fused Q/K RMSNorm plus RoPE under the master switch."""

    return optional_custom_op_enabled(
        _environment_flag(
            os.environ.get("VLLM_HCU_USE_FUSED_RMS_ROPE", "True")
        )
    )


@functools.lru_cache(maxsize=1)
def resolve_aiter_moe_shuffle() -> bool:
    """Resolve the unified AITER MoE weight-shuffle switch."""

    raw = os.environ.get("VLLM_HCU_USE_AITER_MOE_SHUFFLE")
    return True if raw is None else _environment_flag(raw)


@functools.lru_cache(maxsize=1)
def resolve_aiter_moe_config_compat() -> bool:
    """Keep the obsolete selector switch as an enabled compatibility alias."""

    raw = os.environ.get("VLLM_HCU_USE_AITER_MOE_CONFIG")
    if raw is not None and not _environment_flag(raw):
        logger.warning(
            "VLLM_HCU_USE_AITER_MOE_CONFIG=0 is deprecated and ignored; "
            "unified AITER MoE routing always uses AiterMoeConfig"
        )
    return True


def resolve_hcu_flash_attn_mode(explicit_mode: Optional[str]) -> str:
    """Resolve an HCU flash-attention sub-mode without touching vLLM schema.

    An explicit sidecar value wins.  The legacy environment switches retain
    their historical priority, and the flagless HCU default uses the native
    varlen implementation.
    """

    if explicit_mode is not None:
        normalized = explicit_mode.lower()
        aliases = {
            "unified": "cutlass",
            "classic": "classic",
            "cutlass": "cutlass",
            "varlen": "varlen",
        }
        if normalized not in aliases:
            raise ValueError(
                f"Unsupported HCU flash attention mode: {explicit_mode!r}."
            )
        return aliases[normalized]

    if os.environ.get(
        "VLLM_HCU_USE_FLASH_ATTN_VARLEN", "False"
    ).lower() in ("true", "1"):
        return "varlen"
    # Only an explicitly enabled legacy switch participates in priority
    # resolution. The flagless varlen default is applied below, so an
    # explicitly requested classic or CUTLASS mode can still take effect.
    if os.environ.get(
        "VLLM_HCU_USE_FLASH_ATTN_UNIFIED", "False"
    ).lower() in ("true", "1"):
        return "cutlass"
    if os.environ.get("VLLM_HCU_USE_FLASH_ATTN", "False").lower() in (
        "true",
        "1",
    ):
        return "classic"
    return "varlen"

hcu_vllm_environment_variables: dict[str, Callable[[], Any]] = {
    # path to the logs of redirect-output, abstrac of related are ok

    # If set, vLLM will transpose weight to use nn layout
    "VLLM_USE_NN":
    lambda: (os.environ.get("VLLM_USE_NN", "True").lower() in 
             ("true", "1")),
    "VLLM_HCU_USE_FUSED_RMS_ROPE":
    lambda: _environment_flag(
        os.environ.get("VLLM_HCU_USE_FUSED_RMS_ROPE", "True")
    ),
    # vLLM will use FlashAttention Backend on hcu, office attention layerout blocksize 128
    "VLLM_HCU_USE_FLASH_ATTN":
    lambda: (os.environ.get("VLLM_HCU_USE_FLASH_ATTN", "False").lower() in
             ("true", "1")),
    # vLLM will use FlashAttention Backend (varlen_fwd_unified) on hcu, cutlass attention layerout blocksize 64 for qwen3.5
    "VLLM_HCU_USE_FLASH_ATTN_UNIFIED":
    lambda: (os.environ.get("VLLM_HCU_USE_FLASH_ATTN_UNIFIED", "False").lower() in
             ("true", "1")),
    # Select flash_attn.flash_attn_varlen_func by default without changing
    # legacy paths.
    "VLLM_HCU_USE_FLASH_ATTN_VARLEN":
    lambda: (os.environ.get("VLLM_HCU_USE_FLASH_ATTN_VARLEN", "True").lower() in
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
    # Control HCU custom GEMM for W8A8 INT8/FP8. This feature switch is also
    # gated by VLLM_HCU_USE_CUSTOM_OPS; disabling the master switch routes
    # quantized linear GEMM through the target vLLM Triton implementations.
    "VLLM_HCU_USE_CUSTOM_QUANTIZATION_GEMM":
    lambda: (os.environ.get("VLLM_HCU_USE_CUSTOM_QUANTIZATION_GEMM", "True").lower() in
             ("true", "1")),
    # If set, control hcu custom unfused or fused kernel ops
    "VLLM_HCU_USE_CUSTOM_OPS":
    custom_ops_enabled,
    "VLLM_HCU_GLM53_ROUTER_GEMV":
    glm53_router_gemv_enabled,
    # Select the BF16 QSA implementation independently of the generic
    # FLASH_ATTN backend mode. The BF16 dispatcher applies
    # VLLM_HCU_USE_CUSTOM_OPS as its master gate; FP8 QSA is selected by the
    # cache dtype and uses its dedicated reader/writer independently.
    "VLLM_HCU_QSA_BACKEND":
    lambda: os.environ.get("VLLM_HCU_QSA_BACKEND", "cutlass").strip().lower(),
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
    # Deprecated and ignored: GDN prefill is selected by
    # --gdn-prefill-backend and GDN decode by VLLM_HCU_GDN_DECODE_BACKEND.
    "VLLM_HCU_USE_CUSTOM_AITER_FLA":
    lambda: (os.environ.get("VLLM_HCU_USE_CUSTOM_AITER_FLA", "True").lower() in
             ("true", "1")),
    # Select the GDN decode kernel provider: triton (vLLM FLA), hcu-boltops
    # or hcu-aiter (default). VLLM_HCU_USE_CUSTOM_OPS=0 forces hcu-boltops. On HCU the GDN
    # prefill provider is selected by --gdn-prefill-backend instead.
    "VLLM_HCU_GDN_DECODE_BACKEND":
    lambda: os.environ.get(
        "VLLM_HCU_GDN_DECODE_BACKEND", "hcu-aiter"
    ).strip().lower(),
            
    # Pipeline stage partition strategy
    "VLLM_HCU_PP_LAYER_PARTITION_D":
    lambda: os.getenv("VLLM_HCU_PP_LAYER_PARTITION_D", None),
    "VLLM_HCU_USE_FUSE_MOE_GATE":
        lambda: (os.environ.get("VLLM_HCU_USE_FUSE_MOE_GATE", "True").lower() in
                    ("true", "1")),

    # Use LightOp for non-hash DeepSeek V4 sqrt-softplus routing. Hash layers
    # always retain vLLM's official operator.
    "VLLM_HCU_USE_LIGHTOP_SQRTSOFTPLUS_GATE":
    lambda: (os.environ.get(
            "VLLM_HCU_USE_LIGHTOP_SQRTSOFTPLUS_GATE", "True"
        ).lower() in ("true", "1")),
    # Strict BF16 width-128/256 Qwen gated RMSNorm route. Unsupported layouts,
    # activations and row geometries retain vLLM's Triton implementation.
    "VLLM_HCU_USE_LIGHTOP_QWEN_RMSNORM_GATED":
    lambda: (os.environ.get(
            "VLLM_HCU_USE_LIGHTOP_QWEN_RMSNORM_GATED", "True"
        ).lower() in ("true", "1")),
    # Opt-in because this backend installs a LightOp-specific packed weight
    # layout during process_weights_after_loading().
    "VLLM_HCU_USE_LIGHTOP_W16A16_MOE":
    lambda: (os.environ.get(
            "VLLM_HCU_USE_LIGHTOP_W16A16_MOE", "False"
        ).lower() in ("true", "1")),
    # Use categorized LightOp ds_cat for the validated FlashMLA decode layout.
    # VLLM_USE_OPT_CAT remains a compatibility opt-out; custom ops master it.
    "VLLM_HCU_USE_LIGHTOP_MLA_DECODE_CAT":
    lambda: (os.environ.get(
            "VLLM_HCU_USE_LIGHTOP_MLA_DECODE_CAT", "True"
        ).lower() in ("true", "1")),

    "VLLM_HCU_USE_CUSTOM_CAUSAL_CONV1D":
    lambda: (os.environ.get("VLLM_HCU_USE_CUSTOM_CAUSAL_CONV1D", "True").lower() in
             ("true", "1")),
    "VLLM_HCU_USE_LIGHTOP_AWQ":
    lambda: _environment_flag(
        os.environ.get("VLLM_HCU_USE_LIGHTOP_AWQ", "False")
    ),
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

    # Use fused TopK + page-table transform for sparse MLA DCP candidates.
    "VLLM_HCU_USE_LIGHTOP_FAST_TOPK_TRANSFORM":
        lambda: (os.environ.get(
            "VLLM_HCU_USE_LIGHTOP_FAST_TOPK_TRANSFORM", "True"
        ).lower() in ("true", "1")),

    # If use AITER MHC impl, please set True
    "VLLM_HCU_USE_AITER_MHC":
        lambda: (os.environ.get("VLLM_HCU_USE_AITER_MHC", "True").lower() in
                    ("true", "1")),

    # If use TileLang MHC prenorm impl, please set True
    "VLLM_HCU_USE_TILELANG_MHC_PRENORM":
        lambda: (os.environ.get("VLLM_HCU_USE_TILELANG_MHC_PRENORM", "True").lower() in
                    ("true", "1")),

    # FlashMLA sparse attention for native ROCm DeepSeek-V4.
    # Both adapter paths also require VLLM_HCU_USE_CUSTOM_OPS; disabling
    # the master retains the original ROCm/AITER attention and metadata builders.
    "VLLM_HCU_DEEPSEEK_V4_ROCM_FLASHMLA_DECODE":
        lambda: os.environ.get("VLLM_HCU_DEEPSEEK_V4_ROCM_FLASHMLA_DECODE", "True").lower()
        in ("true", "1"),
    "VLLM_HCU_DEEPSEEK_V4_ROCM_FLASHMLA_PREFILL":
        lambda: os.environ.get("VLLM_HCU_DEEPSEEK_V4_ROCM_FLASHMLA_PREFILL", "True").lower()
        in ("true", "1"),
    # Disabling padding retains FlashMLA for native 64/128-head queries.
    "VLLM_HCU_DEEPSEEK_V4_ROCM_FLASHMLA_PREFILL_HEAD_PADDING":
        lambda: os.environ.get(
            "VLLM_HCU_DEEPSEEK_V4_ROCM_FLASHMLA_PREFILL_HEAD_PADDING", "True"
        ).lower() in ("true", "1"),
    # Provider selection applies only to this adapter's prefill path.
    "VLLM_HCU_DEEPSEEK_V4_ROCM_SPARSE_PREFILL_BACKEND":
        lambda: os.environ.get(
            "VLLM_HCU_DEEPSEEK_V4_ROCM_SPARSE_PREFILL_BACKEND", "flashmla"
        ).strip().lower(),

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
        lambda: (os.environ.get("VLLM_HCU_USE_LIGHTOP_MOE_ALIGN", "False").lower() in
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

    # Optional override for LightOp's fused-MoE chunk size.
    "VLLM_HCU_FUSED_MOE_CHUNK_SIZE":
        lambda: maybe_convert_int(
            os.getenv("VLLM_HCU_FUSED_MOE_CHUNK_SIZE", None)
        ),

    # Use LightOp's global MoE cache.
    "VLLM_HCU_USE_GLOBAL_MOE_CACHE":
        lambda: (os.environ.get("VLLM_HCU_USE_GLOBAL_MOE_CACHE", "False").lower() in
                 ("true", "1")),

    # If use fused rmsnorm and quant, please set True
    "VLLM_HCU_USE_FUSED_RMS_QUANT":
        lambda: (os.environ.get("VLLM_HCU_USE_FUSED_RMS_QUANT", "False").lower() in
                 ("true", "1")),

    # If set, use LightOp's fused SiLU-and-multiply kernel.
    "VLLM_HCU_USE_FUSE_SILU_AND_MUL":
        lambda: (os.environ.get("VLLM_HCU_USE_FUSE_SILU_AND_MUL", "False").lower() in
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

    # If use custom AITER_W4A16_MOE impl, please set True
    "VLLM_HCU_USE_AITER_W4A16_MOE":
        lambda: (os.environ.get("VLLM_HCU_USE_AITER_W4A16_MOE", "True").lower() in
                    ("true", "1")),

    # If set, use torch ops for EPLB logical-to-physical mapping and load
    # recording instead of the fused Triton kernel.
    "VLLM_HCU_USE_TORCH_EPLB_MAP_RECORD":
        lambda: (os.environ.get("VLLM_HCU_USE_TORCH_EPLB_MAP_RECORD", "False").lower() in
                    ("true", "1")),

    # Shuffle AITER MoE weights for any quantization mode (default True).
    "VLLM_HCU_USE_AITER_MOE_SHUFFLE": resolve_aiter_moe_shuffle,

    # Deprecated compatibility switch; unified routing always uses the config.
    "VLLM_HCU_USE_AITER_MOE_CONFIG": resolve_aiter_moe_config_compat,

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

    # DeepSeek V4 PCP: keep attention queries rank-local and gather only KV.
    "VLLM_HCU_DEEPSEEK_V4_PCP_LOCAL_QUERIES":
        lambda: os.getenv("VLLM_HCU_DEEPSEEK_V4_PCP_LOCAL_QUERIES", "True").lower()
        in ("1", "true"),

    # Zero a new KV block only in its owning cache group's layer pages.
    "VLLM_HCU_KV_ZERO_BY_GROUP":
        lambda: _environment_flag(os.getenv("VLLM_HCU_KV_ZERO_BY_GROUP", "1")),

    # DeepSeek V4 PCP local queries: run sparse prefill on contiguous local
    # Q/output slices instead of gathering Q and scattering the output.
    "VLLM_HCU_DEEPSEEK_V4_PCP_LOCAL_SLICES":
        lambda: _environment_flag(
            os.getenv("VLLM_HCU_DEEPSEEK_V4_PCP_LOCAL_SLICES", "1")),

    # DeepSeek V4 compressor KV/score projection GEMM: blaslt (cached NT
    # weight + shared BF16->FP32 BLASLt GEMM) or torch (original torch.mm).
    "VLLM_HCU_COMPRESSOR_GEMM_BACKEND":
        lambda: os.getenv("VLLM_HCU_COMPRESSOR_GEMM_BACKEND", "blaslt"),

    # C128 prefill head tiles on HCU; 0 restores the upstream CU heuristic.
    "VLLM_HCU_C128_HEAD_SPLITS":
        lambda: int(os.getenv("VLLM_HCU_C128_HEAD_SPLITS", "4")),

    # DeepSeek V4 o_proj: rotate only the 64-dim RoPE suffix of the private
    # attention output in place, instead of an out-of-place 512-dim copy.
    "VLLM_HCU_INVERSE_ROPE_SUFFIX":
        lambda: _environment_flag(os.getenv("VLLM_HCU_INVERSE_ROPE_SUFFIX", "1")),

    # Launch shared_experts on the auxiliary stream before routed MoE/DeepEP,
    # and wait only when the shared output is consumed.
    "VLLM_HCU_SHARED_EXPERTS_EARLY_LAUNCH":
        lambda: (os.environ.get("VLLM_HCU_SHARED_EXPERTS_EARLY_LAUNCH", "False").lower() in
                    ("true", "1")),

    # DeepEP auto DeepGEMM: keep one routed-expert weight copy for both the
    # HT (contiguous) and LL (masked) experts when their packers agree.
    "VLLM_HCU_DEEPEP_AUTO_SHARED_WEIGHTS":
        lambda: _environment_flag(
            os.getenv("VLLM_HCU_DEEPEP_AUTO_SHARED_WEIGHTS", "1")),

    # Round HT MoE scratch capacity to bounded buckets; 0 restores exact sizing.
    "VLLM_HCU_MOE_WORKSPACE_GROWTH_MB":
        lambda: int(os.getenv("VLLM_HCU_MOE_WORKSPACE_GROWTH_MB", "512")),

    # Size DeepEP HT expert workspaces from the physical per-expert counts
    # DeepEP returns for the received batch, instead of the M*topk upper bound.
    "VLLM_HCU_DEEPEP_HT_EXACT_LAYOUT":
        lambda: _environment_flag(os.getenv("VLLM_HCU_DEEPEP_HT_EXACT_LAYOUT", "1")),

    # Quantized DeepEP + EPLB: launch shared experts on the auxiliary stream
    # before routing and join only when the shared output is consumed.
    "VLLM_HCU_SHARED_EXPERTS_DEEPEP_EPLB_OVERLAP":
        lambda: _environment_flag(
            os.getenv("VLLM_HCU_SHARED_EXPERTS_DEEPEP_EPLB_OVERLAP", "1")),

    # Enable request-count-oriented cudagraph capture buckets up to request size 256.
    "VLLM_HCU_ENABLE_REQUEST_CUDAGRAPH_BUCKETS":
        lambda: (os.environ.get("VLLM_HCU_ENABLE_REQUEST_CUDAGRAPH_BUCKETS", "False").lower() in
                    ("true", "1")),

    # Compatibility fallback for Qwen4Exp PLE INT8 CPU offload when the target
    # VllmConfig has no EngramConfig. An explicit EngramConfig.cpu_offload value
    # takes precedence. CPU offload requires the HCU UVA operator.
    "VLLM_HCU_PLE_CPU_OFFLOAD":
        lambda: (os.environ.get("VLLM_HCU_PLE_CPU_OFFLOAD", "False").lower() in
                    ("true", "1")),

    # Overlap Qwen4Exp PLE INT8 UVA lookup with preceding model compute. This
    # is intentionally opt-in, is gated by VLLM_HCU_USE_CUSTOM_OPS, and does
    # not enable FP8 prefetch.
    "VLLM_HCU_PLE_PREFETCH_STREAM":
        ple_prefetch_enabled,

    # Quantize naive DP/EP MoE hidden-state dispatch to INT8 before
    # AllGatherv. ReduceScatterv remains in the model activation dtype.
    "VLLM_HCU_DP_INT8_ALLGATHER":
        dp_int8_allgather_enabled,

    # Run Qwen4Exp hyper-connection layers sequence-parallel (attention
    # all-gather/reduce-scatter instead of all-reduce) when sequence-parallel
    # MoE is active (DP>1, TP>1, EP). On by default, gated by
    # VLLM_HCU_USE_CUSTOM_OPS.
    "VLLM_HCU_QWEN4_EXP_HC_SP":
        qwen4_exp_hc_sp_enabled,

    # Kimi-K3 SP-MoE row-sharded forward (opt-in). The SP patch's
    # _switch_enabled delegates to this k3_sp_moe_enabled() (master AND
    # child) and the same effective result is persisted into
    # HcuFeatureConfig before VllmConfig.compute_hash, so 0->1 restarts
    # cannot reuse AOT graphs and a master-off rollback skips the SP
    # install instead of failing at model construction.
    "VLLM_HCU_K3_SP_MOE":
        k3_sp_moe_enabled,

    # gfx938 GLM5Next sparse-indexer K-cache page layout: "preshuffle"
    # (official 16x16 tiles, HCU Triton decode reader) or "normal"
    # (token-major pages, LightOp decode reader).
    "VLLM_HCU_INDEXER_KCACHE_LAYOUT":
        indexer_kcache_layout,

    # Route the RMSNorm applied after the mHC pre/fused-post-pre kernels
    # through aten's single fused rms_norm kernel instead of
    # vllm.ir.ops.rms_norm's native implementation (8 elementwise/reduce
    # kernels when no fused IR provider is registered, the case on HCU).
    # Affects every model on the native mHC fallback path (GLM5Next,
    # DeepSeek V4, ...); consulted once at patch-apply time. See
    # patch_mhc_fused_rmsnorm.py. Enabled by default; set to 0 to disable.
    "VLLM_HCU_MHC_FUSED_RMSNORM":
        lambda: (os.environ.get("VLLM_HCU_MHC_FUSED_RMSNORM", "True").lower() in
                    ("true", "1")),

    # Skip the .contiguous() copies GLM5Next's fused_recurrent_kda otherwise
    # applies to q/k/v/g/beta before the recurrent decode/spec-verify kernel,
    # by passing per-token strides into the kernel instead. On the decode and
    # spec-verify paths q/k/v are non-contiguous views split out of the merged
    # qkv causal-conv1d output, so this removes 4-5 elementwise copy kernels
    # per step. Consulted once at patch-apply time. See
    # patch_glm5next_kda_strided_recurrent.py. Enabled by default; set to 0
    # to disable.
    "VLLM_HCU_GLM5NEXT_KDA_STRIDED_INPUTS":
        lambda: (os.environ.get("VLLM_HCU_GLM5NEXT_KDA_STRIDED_INPUTS", "True").lower() in
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
