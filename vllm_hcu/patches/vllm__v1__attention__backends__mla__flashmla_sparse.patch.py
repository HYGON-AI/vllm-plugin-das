# SPDX-License-Identifier: Apache-2.0

"""
vllm.v1.attention.backends.mla.flashmla_sparse _fp8_flash_mla_kernel and _bf16_flash_mla_kernel
"""

PATCHES = [
(
"""
import torch
""",
"""
import torch
import vllm_hcu.platforms.envs as henvs
""",
),
   
(
"""
from vllm.v1.attention.ops.flashmla import (
    FlashMLASchedMeta,
    flash_mla_sparse_fwd,
    flash_mla_with_kvcache,
    get_mla_metadata,
)
""",
"""
from vllm_hcu.v1.attention.ops.flashmla import (
    FlashMLASchedMeta,
    flash_mla_sparse_fwd,
    flash_mla_with_kvcache,
    get_mla_metadata,
)
""",
),
     
(
"""
        fp8_use_mixed_batch = (
            self.num_heads < MIN_HEADS_FOR_BF16_PREFILL and not self.is_deepseek_v4
        )
""",
"""
        fp8_use_mixed_batch = (
            self.num_heads < MIN_HEADS_FOR_BF16_PREFILL and not self.is_deepseek_v4 and henvs.VLLM_HCU_USE_FP8_MIXED_BATCH
        )        
""",
),

(
"""
        if actual_num_heads < padded_num_heads:
            logger.warning_once(
                f"Padding num_heads from {actual_num_heads} to "
                f"{padded_num_heads} for FP8 sparse decode kernel"
            )
            q_padded = q.new_zeros((q.size(0), q.size(1), padded_num_heads, q.size(3)))
            q_padded[:, :, :actual_num_heads, :] = q
            q = q_padded
""",
"""
        if not current_platform.is_rocm():
            if actual_num_heads < padded_num_heads:
                logger.warning_once(
                    f"Padding num_heads from {actual_num_heads} to "
                    f"{padded_num_heads} for FP8 sparse decode kernel"
                )
                q_padded = q.new_zeros((q.size(0), q.size(1), padded_num_heads, q.size(3)))
                q_padded[:, :, :actual_num_heads, :] = q
                q = q_padded
""",
),

(
"""
        if actual_num_heads < padded_num_heads:
            out = out[:, :, :actual_num_heads, :]
""",
"""
        if not current_platform.is_rocm():
            if actual_num_heads < padded_num_heads:
                out = out[:, :, :actual_num_heads, :]
""",
),

(
"""
        if self.num_heads % self.prefill_padding != 0:
            assert self.prefill_padding % self.num_heads == 0
            logger.warning_once(
                f"Padding num_heads from {self.num_heads} to "
                f"{self.prefill_padding} for BF16 sparse prefill kernel"
            )
            q_padded = q.new_empty((q.shape[0], self.prefill_padding, q.shape[2]))
            q_padded[:, : self.num_heads, :] = q
            q = q_padded
""",
"""        
        if not current_platform.is_rocm():
            if self.num_heads % self.prefill_padding != 0:
                assert self.prefill_padding % self.num_heads == 0
                logger.warning_once(
                    f"Padding num_heads from {self.num_heads} to "
                    f"{self.prefill_padding} for BF16 sparse prefill kernel"
                )
                q_padded = q.new_empty((q.shape[0], self.prefill_padding, q.shape[2]))
                q_padded[:, : self.num_heads, :] = q
                q = q_padded
""",
),

################ lightly cp###########################
(
"""
    num_actual_tokens: int  # Number of tokens excluding padding.
""",
"""        
    num_actual_tokens: int  # Number of tokens excluding padding.
    num_kv_actual_tokens: int
""",
),

(
"""
            num_actual_tokens=cm.num_actual_tokens,
""",
"""        
            num_actual_tokens=cm.num_actual_tokens,
            num_kv_actual_tokens=cm.num_kv_actual_tokens,
""",
),
################ lightly cp###########################
]
