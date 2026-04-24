# SPDX-License-Identifier: Apache-2.0

"""
vllm.model_executor.models.config verify_and_update_config
"""

PATCHES = [
(
"""
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec, MLAAttentionSpec
""",
"""
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec, MLAAttentionSpec
import vllm_hcu.platforms.envs as henvs
""",
),

(
"""
            kernel_block_alignment_size = 16
""",
"""
            if henvs.VLLM_HCU_USE_FLASH_ATTN:
                kernel_block_alignment_size = 128
            elif henvs.VLLM_HCU_USE_CUSTOM_FLASH_ATTN:
                kernel_block_alignment_size = 64
            else:
                kernel_block_alignment_size = 16
""",
),
]