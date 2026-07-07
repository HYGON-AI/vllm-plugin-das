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
                    self.kv_cache_spec.storage_block_size,
                    self.num_sms,
                )
""",
"""
            if current_platform.is_cuda() and has_deep_gemm():
                self.scheduler_metadata_buffer[:] = get_paged_mqa_logits_metadata(
                    seq_lens,
                    self.kv_cache_spec.storage_block_size,
                    self.num_sms,
                )
            else:
                self.scheduler_metadata_buffer = gemmopt.get_paged_mqa_logits_metadata(
                    seq_lens,
                    self.kv_cache_spec.storage_block_size,
                    self.num_sms,
                )
""",
),

(
"""
        req_slice = slice(start + request_offset, end + request_offset)
""",
"""
        req_slice = slice(start + request_offset, end + request_offset)
        # Zero-query rows can appear from padded requests in full-CG/ubatch metadata.
        # They produce no logits/indexer work, so skip the no-op chunk.
        if chunk_m == 0:
            continue
""",
),

(
"""
        block_table = common_attn_metadata.block_table_tensor

        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                common_attn_metadata,
                decode_threshold=self.reorder_batch_threshold,
                require_uniform=not self.use_flattening,
            )
        )
""",
"""
        block_table = common_attn_metadata.block_table_tensor
        use_is_prefilling = common_attn_metadata.is_prefilling is not None

        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                common_attn_metadata,
                decode_threshold=self.reorder_batch_threshold,
                require_uniform=not self.use_flattening,
                treat_short_extends_as_decodes=not use_is_prefilling,
            )
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
