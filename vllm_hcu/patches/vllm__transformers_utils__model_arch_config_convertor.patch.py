# SPDX-License-Identifier: Apache-2.0

"""
vllm.transformers_utils.model_arch_config_convertor
"""

PATCHES = [
(
"""
            qk_rope_head_dim = getattr(self.hf_text_config, "qk_rope_head_dim", 0)
""",
"""
            qk_rope_head_dim = self._get_qk_rope_head_dim()
""",
),

(
'''
    def get_total_num_kv_heads(self) -> int:
''',
'''
    def _get_qk_rope_head_dim(self) -> int:
        """Get qk_rope_head_dim, fixing the transformers v5.4+ attribute_map bug."""
        cfg = self.hf_text_config
        qk_rope_head_dim = getattr(cfg, "qk_rope_head_dim", 0)
        qk_nope_head_dim = getattr(cfg, "qk_nope_head_dim", 0)

        # In valid MLA configs, qk_rope_head_dim != qk_nope_head_dim.
        if qk_rope_head_dim == 0 or qk_rope_head_dim != qk_nope_head_dim:
            return qk_rope_head_dim  # not corrupted

        # Read the correct value from raw config.json.
        from vllm.transformers_utils.repo_utils import get_hf_file_to_dict

        model_path = self.hf_config.name_or_path
        if not model_path:
            return qk_rope_head_dim
        raw = get_hf_file_to_dict("config.json", model_path)
        if raw and "qk_rope_head_dim" in raw:
            correct = raw["qk_rope_head_dim"]
            if correct != qk_rope_head_dim:
                logger.info(
                    "Fixing qk_rope_head_dim: %d -> %d "
                    "(transformers v5.4+ attribute_map bug)",
                    qk_rope_head_dim,
                    correct,
                )
                # Patch the config so downstream model layers also get
                # the correct value.
                cfg.qk_rope_head_dim = correct
                return correct
        return qk_rope_head_dim

    def get_total_num_kv_heads(self) -> int:
''',
),
]
