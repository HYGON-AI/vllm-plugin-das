# SPDX-License-Identifier: Apache-2.0

"""
vllm.v1.attention.backends.mla.indexer build
"""

PATCHES = [
(
"""
import torch
""",
"""
import torch
from lightop import gemmopt
""",
),

(
"""
        return [1 if current_platform.is_rocm() else 64]
""",
"""
        return [1 if not current_platform.is_rocm() else 64]
""",
),

(
"""
            if current_platform.is_cuda() and is_deep_gemm_supported():
                self.scheduler_metadata_buffer[:] = get_paged_mqa_logits_metadata(
                    seq_lens,
                    self.kv_cache_spec.block_size,
                    self.num_sms,
                )
""",
"""
            if current_platform.is_cuda() and is_deep_gemm_supported():
                self.scheduler_metadata_buffer[:] = get_paged_mqa_logits_metadata(
                    seq_lens,
                    self.kv_cache_spec.block_size,
                    self.num_sms,
                )
            else:
                self.scheduler_metadata_buffer = gemmopt.get_paged_mqa_logits_metadata(
                    seq_lens, 
                    self.kv_cache_spec.block_size, 
                    self.num_sms,
                )
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
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
""",
"""        
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            num_kv_actual_tokens=common_attn_metadata.num_kv_actual_tokens,
""",
),
################ lightly cp###########################
]
