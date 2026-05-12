# SPDX-License-Identifier: Apache-2.0

import os
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    VLLM_USE_NN : bool = False
    VLLM_HCU_USE_FLASH_ATTN: bool = False
    VLLM_HCU_USE_FLASH_ATTN_UNIFIED: bool = False
    VLLM_HCU_USE_CUSTOM_FLASH_ATTN: bool = False
    VLLM_HCU_USE_FLASHMLA: bool = False
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
    VLLM_HCU_USE_KVCACHE_E5M2 : bool = False

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
    lambda: (os.environ.get("VLLM_HCU_USE_FLASH_ATTN_UNIFIED", "False").lower() in
             ("true", "1")),
    # vLLM will use custom FlashAttention (convert kv cache) Backend on hcu,  not office attention layerout blocksize 64 
    "VLLM_HCU_USE_CUSTOM_FLASH_ATTN":
    lambda: (os.environ.get("VLLM_HCU_USE_CUSTOM_FLASH_ATTN", "False").lower() in
             ("true", "1")),
    # vLLM will use FlashMLA Backend on hcu
    "VLLM_HCU_USE_FLASHMLA":
    lambda: (os.environ.get("VLLM_HCU_USE_FLASHMLA", "False").lower() in
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
    # If use kvcache fp8_e5m2, please set True (qurey not quantization)
    "VLLM_HCU_USE_KVCACHE_E5M2":
    lambda: (os.environ.get("VLLM_HCU_USE_KVCACHE_E5M2", "False").lower() in
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
