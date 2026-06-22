# SPDX-License-Identifier: Apache-2.0

"""
vllm.model_executor.layers.fused_moe.config: EP INT8 + FP8
"""

PATCHES = [
(
"""
    if block_shape is not None:
""",
"""
    if block_shape is not None and quant_dtype!=torch.int8 and quant_dtype!=current_platform.fp8_dtype():
""",
),
(
"""
        w_shape = GroupShape(row=block_shape[0], col=block_shape[1])
    else:
""",
"""
        w_shape = GroupShape(row=block_shape[0], col=block_shape[1])
    elif block_shape is not None and (quant_dtype == torch.int8 or quant_dtype == current_platform.fp8_dtype()):
        a_shape = GroupShape(row=block_shape[0], col=block_shape[1])
        w_shape = GroupShape(row=block_shape[0], col=block_shape[1])
    else:
""",
),
(
"""
    def __post_init__(self):
        assert not self.per_act_token_quant or self.block_shape is None, (
            "illegal quantization"
        )
""",
"""
    # def __post_init__(self):
    #     assert not self.per_act_token_quant or self.block_shape is None, (
    #         "illegal quantization"
    #     )
""",
),
(
"""
        assert quant_config.per_act_token_quant == per_act_token_quant
        assert quant_config.per_out_ch_quant == per_out_ch_quant
        assert quant_config.block_shape == block_shape
""",
"""
        if quant_dtype != torch.int8 and quant_dtype != current_platform.fp8_dtype():
            assert quant_config.per_act_token_quant == per_act_token_quant
            assert quant_config.per_out_ch_quant == per_out_ch_quant
            assert quant_config.block_shape == block_shape
""",
),
(
"""
def int8_w8a8_moe_quant_config(
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    a1_scale: torch.Tensor | None,
    a2_scale: torch.Tensor | None,
    w1_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
    per_act_token_quant: bool = False,
""",
"""
def int8_w8a8_moe_quant_config(
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    a1_scale: torch.Tensor | None,
    a2_scale: torch.Tensor | None,
    w1_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
    per_act_token_quant: bool = False,
    block_shape: list[int] | None = None,
""",
),
(
"""
        per_out_ch_quant=False,
        block_shape=None,
    )
""",
"""
        per_out_ch_quant=False,
        block_shape=block_shape,
    )
""",
),

(
"""
    @property
    def use_all2all_kernels(self):
        return self.dp_size > 1 and self.use_ep
""",
"""
    @property
    def use_all2all_kernels(self):
        return (self.dp_size > 1 or self.is_sequence_parallel) and self.use_ep
""",
),
]