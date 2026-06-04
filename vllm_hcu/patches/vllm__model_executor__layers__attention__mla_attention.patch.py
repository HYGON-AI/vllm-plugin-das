# SPDX-License-Identifier: Apache-2.0

"""
vllm.model_executor.layers.attention.mla_attention process_weights_after_loading
"""

PATCHES = [
(
"""
import vllm.envs as envs
""",
"""        
import vllm.envs as envs
import vllm_hcu.platforms.envs as henvs
from vllm_hcu.platforms.hcu import on_gfx938
""",
),

(
"""
            if fp8_attention and self.impl.supports_quant_query_input:
                assert mqa_ql_nope.shape[0] == mqa_q_pe.shape[0]
                assert mqa_ql_nope.shape[1] == mqa_q_pe.shape[1]
                mqa_q = self._decode_concat_quant_fp8_op(
                    mqa_ql_nope, mqa_q_pe, self._q_scale
                )
""",
"""
            if fp8_attention and self.impl.supports_quant_query_input and on_gfx938():
                assert mqa_ql_nope.shape[0] == mqa_q_pe.shape[0]
                assert mqa_ql_nope.shape[1] == mqa_q_pe.shape[1]
                if henvs.VLLM_HCU_USE_CAT_MLA:
                    mqa_q = (mqa_ql_nope, mqa_q_pe)
                else:
                    mqa_q = self._decode_concat_quant_fp8_op(
                        mqa_ql_nope, mqa_q_pe, self._q_scale
                    )
""",
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
            num_actual_tokens=num_tokens,
""",
"""        
            num_actual_tokens=num_tokens,
            num_kv_actual_tokens=common_attn_metadata.num_kv_actual_tokens,
""",
),
################ lightly cp###########################
]
