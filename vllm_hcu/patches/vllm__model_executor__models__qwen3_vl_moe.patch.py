# SPDX-License-Identifier: Apache-2.0

"""
vllm.model_executor.models.qwen3_vl_moe
"""

PATCHES = [
(
"""
        if self.config.tie_word_embeddings:
""",
"""
        if getattr(self.config, "tie_word_embeddings", False):
""",
),
]
