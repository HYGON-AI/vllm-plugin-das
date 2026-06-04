# SPDX-License-Identifier: Apache-2.0

"""
vllm.model_executor.models.qwen3_vl
"""

PATCHES = [
(
"""
            if config.tie_word_embeddings:
""",
"""
            if getattr(config, "tie_word_embeddings", False):
""",
),
]
