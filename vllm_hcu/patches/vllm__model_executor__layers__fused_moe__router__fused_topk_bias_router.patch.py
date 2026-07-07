# SPDX-License-Identifier: Apache-2.0

"""
Patch for vllm/model_executor/layers/fused_moe/router/fused_topk_bias_router.py.
"""

PATCHES = [
(
"""
    ops.topk_hash_softplus_sqrt(
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
        renormalize,
        routed_scaling_factor,
        e_score_correction_bias,
        input_tokens,
        hash_indices_table,
    )
""",
"""
    if hash_indices_table is not None:
        if hash_indices_table.dtype != topk_indices.dtype:
            hash_indices_table = hash_indices_table.to(dtype=topk_indices.dtype)
        if input_tokens is not None and input_tokens.dtype != topk_indices.dtype:
            input_tokens = input_tokens.to(dtype=topk_indices.dtype)

    ops.topk_hash_softplus_sqrt(
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
        renormalize,
        routed_scaling_factor,
        e_score_correction_bias,
        input_tokens,
        hash_indices_table,
    )
""",
),
]
