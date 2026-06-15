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
(
"""
        with self._mark_language_model(vllm_config):
            self.language_model = Qwen3MoeLLMForCausalLM(
                vllm_config=vllm_config.with_hf_config(config.text_config),
                prefix=maybe_prefix(prefix, "language_model"),
            )
""",
"""
        if getattr(config, "tie_word_embeddings", None) is not None:
            config.text_config.tie_word_embeddings = config.tie_word_embeddings

        with self._mark_language_model(vllm_config):
            self.language_model = Qwen3MoeLLMForCausalLM(
                vllm_config=vllm_config.with_hf_config(config.text_config),
                prefix=maybe_prefix(prefix, "language_model"),
            )
""",
),
]
