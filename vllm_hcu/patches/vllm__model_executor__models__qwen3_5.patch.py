# SPDX-License-Identifier: Apache-2.0

"""
vllm.model_executor.models.qwen3_5 get_mamba_state_dtype_from_config
"""

PATCHES = [
(
"""
logger = init_logger(__name__)
""",
"""
logger = init_logger(__name__)
import vllm_hcu.platforms.envs as henvs
""",
),
(
"""
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
            vllm_config.cache_config.mamba_ssm_cache_dtype,
        )
""",
"""
        mamba_ssm_cache_dtype = vllm_config.cache_config.mamba_ssm_cache_dtype
        if henvs.VLLM_HCU_MAMBA_SSM_CACHE_DTYPE and henvs.VLLM_HCU_USE_CUSTOM_OPS:
            mamba_ssm_cache_dtype = "auto"
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
            mamba_ssm_cache_dtype,
        )
""",
),
]
