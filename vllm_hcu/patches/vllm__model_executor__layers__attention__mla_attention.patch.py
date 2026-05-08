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

(
"""
            self._pad_v = self.vllm_flash_attn_version is None or not (
                (
                    self.vllm_flash_attn_version == 3
                    and device_capability is not None
                    and device_capability[0] == 9
                )
                or self.vllm_flash_attn_version == 4
            )
""",
"""        
            if not current_platform.is_rocm():
                self._pad_v = self.vllm_flash_attn_version is None or not (
                    (
                        self.vllm_flash_attn_version == 3
                        and device_capability is not None
                        and device_capability[0] == 9
                    )
                    or self.vllm_flash_attn_version == 4
                )
            else:
                self._pad_v = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count == 120
""",
),

# (
# """
#             maybe_padded_v = torch.nn.functional.pad(
#                 v, [0, q.shape[-1] - v.shape[-1]], value=0
#             )
# """,
# """    
#             if not current_platform.is_rocm():
#                 maybe_padded_v = torch.nn.functional.pad(
#                     v, [0, q.shape[-1] - v.shape[-1]], value=0
#                 )
#             else:
#                 maybe_padded_v = torch.nn.functional.pad(
#                     v, [0, q.shape[-1] - v.shape[-1] - 32], value=0
#                 )
#                 maybe_padded_v = maybe_padded_v[..., :-32].reshape(v.shape[0], v.shape[1],v.shape[2])
# """,
# ),
]