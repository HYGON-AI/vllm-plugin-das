# SPDX-License-Identifier: Apache-2.0

"""
vllm.model_executor.models.config verify_and_update_config DeepseekV32ForCausalLM
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
                if henvs.VLLM_HCU_FLASH_ATTN_BLOCK_ALIGNMENT_SIZE is not None and henvs.VLLM_HCU_USE_CUSTOM_OPS:
                    value = henvs.VLLM_HCU_FLASH_ATTN_BLOCK_ALIGNMENT_SIZE
                    if value <= 0 or value % 16 != 0:
                        raise ValueError(
                            f"VLLM_HCU_FLASH_ATTN_BLOCK_ALIGNMENT_SIZE must be "
                            f"a positive multiple of 16, got {value}."
                        )
                    kernel_block_alignment_size = value
            elif henvs.VLLM_HCU_USE_CUSTOM_FLASH_ATTN:
                kernel_block_alignment_size = 64
            else:
                kernel_block_alignment_size = 16
""",
),

(
"""
from typing import TYPE_CHECKING
""",
"""
from typing import TYPE_CHECKING
import vllm_hcu.platforms.envs as henvs
""",
),

(
"""
        is_v32 = hasattr(hf_config, "index_topk")
""",
"""
        is_v32 = hasattr(hf_config, "index_topk") and not henvs.VLLM_HCU_DISABLE_DSA
""",
),

]