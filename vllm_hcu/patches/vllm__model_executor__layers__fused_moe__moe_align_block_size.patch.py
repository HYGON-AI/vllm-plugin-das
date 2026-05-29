# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Patch for vllm.model_executor.layers.fused_moe.moe_align_block_size
"""

PATCHES = [
(
"""
from vllm.utils.math_utils import round_up
""",
"""   
from vllm.utils.math_utils import round_up
import vllm_hcu.platforms.envs as henvs
""",
),

(
'''
    ops.moe_align_block_size(
        topk_ids,
        num_experts,
        block_size,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
        expert_map if ignore_invalid_experts else None,
    )
''',
'''
    if henvs.VLLM_HCU_USE_CUSTOM_OPS and henvs.VLLM_HCU_USE_LIGHTOP_MOE_ALIGN:
        from lightop import op as op
        op.moe_align_block_size(
            topk_ids,
            num_experts,
            block_size,
            sorted_ids,
            expert_ids,
            num_tokens_post_pad,
            expert_map=None,
        )
    else:
        ops.moe_align_block_size(
            topk_ids,
            num_experts,
            block_size,
            sorted_ids,
            expert_ids,
            num_tokens_post_pad,
            expert_map if ignore_invalid_experts else None,
        )
''',
),
]
