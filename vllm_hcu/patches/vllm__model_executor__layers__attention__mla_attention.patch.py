# SPDX-License-Identifier: Apache-2.0

"""
vllm.model_executor.layers.attention.mla_attention process_weights_after_loading
"""

PATCHES = [
(
    "import vllm.envs as envs",
    "import vllm.envs as envs\nimport vllm_hcu.platforms.envs as henvs",
),

(
"""
        kv_b_proj_weight = get_and_maybe_dequant_weights(
            self.kv_b_proj, out_dtype=act_dtype
        ).T
""",
"""        
        if henvs.VLLM_USE_NN:
            kv_b_proj_weight = get_and_maybe_dequant_weights(
                self.kv_b_proj, out_dtype=act_dtype
            )
        else:
            kv_b_proj_weight = get_and_maybe_dequant_weights(
                self.kv_b_proj, out_dtype=act_dtype
            ).T
""",
),

(
"""
        assert kv_b_proj_weight.shape == (
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
        ), (
            f"{kv_b_proj_weight.shape=}, "
            f"{self.kv_lora_rank=}, "
            f"{self.num_heads=}, "
            f"{self.qk_nope_head_dim=}, "
            f"{self.v_head_dim=}"
        )
""",
"""      
        expected_shape = (
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
        )
        if kv_b_proj_weight.shape != expected_shape:
            if kv_b_proj_weight.T.shape == expected_shape:
                kv_b_proj_weight = kv_b_proj_weight.T.contiguous()
            else:
                raise ValueError(
                    f"kv_b_proj_weight.shape={kv_b_proj_weight.shape}, "
                    f"expected={expected_shape}, "
                    f"{self.kv_lora_rank=}, "
                    f"{self.num_heads=}, "
                    f"{self.qk_nope_head_dim=}, "
                    f"{self.v_head_dim=}"
                )
""",
),

################ lightly cp###########################
(
"""
        num_actual_toks = attn_metadata.num_actual_tokens
""",
"""        
        num_actual_toks = attn_metadata.num_actual_tokens
        num_kv_actual_toks = attn_metadata.num_kv_actual_tokens
""",
),

(
"""
        k_c_normed = k_c_normed[:num_actual_toks, ...]
        k_pe = k_pe[:num_actual_toks, ...]
""",
"""        
        k_c_normed = k_c_normed[:num_kv_actual_toks, ...]
        k_pe = k_pe[:num_kv_actual_toks, ...]
""",
),
################ lightly cp###########################
]
