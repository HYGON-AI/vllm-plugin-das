# SPDX-License-Identifier: Apache-2.0

"""
vllm.v1.models.spec_decode.eagle  __init__
"""

PATCHES = [
(
"""
            rocm_types = [
                TritonAttentionMetadata,
                RocmAttentionMetadata,
                ROCMAiterMLASparseMetadata,
            ]
""",
"""
            from vllm_hcu.v1.attention.backends.flash_attn import FlashAttentionMetadata
            rocm_types = [
                TritonAttentionMetadata,
                RocmAttentionMetadata,
                ROCMAiterMLASparseMetadata,
                FlashAttentionMetadata,
            ]
""",
),




]
