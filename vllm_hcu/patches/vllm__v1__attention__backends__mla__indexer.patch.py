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
            if current_platform.is_cuda() and has_deep_gemm():
                self.scheduler_metadata_buffer[:] = get_paged_mqa_logits_metadata(
                    seq_lens,
                    self.kv_cache_spec.block_size,
                    self.num_sms,
                )
""",
"""
            if current_platform.is_cuda() and has_deep_gemm():
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
    num_prefill_tokens: int
""",
"""        
    num_prefill_tokens: int
    num_kv_actual_tokens: int
""",
),

(
"""
            max_seq_len=common_attn_metadata.max_seq_len,
""",
"""        
            max_seq_len=common_attn_metadata.max_seq_len,
            num_kv_actual_tokens=common_attn_metadata.num_kv_actual_tokens,
""",
),
################ lightly cp###########################
]
