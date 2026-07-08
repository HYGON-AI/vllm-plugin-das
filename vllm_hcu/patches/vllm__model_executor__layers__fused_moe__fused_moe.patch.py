# SPDX-License-Identifier: Apache-2.0

"""
Patch for vllm.model_executor.layers.fused_moe.fused_batched_moe  support triton fp8 kernel
"""

PATCHES = [
(
"""
import torch
""",
"""
import torch

import vllm_hcu.platforms.envs as henvs
try:
    if henvs.VLLM_HCU_USE_CUSTOM_OPS and henvs.VLLM_HCU_USE_AITER_W4A16_MOE:
        from aiter.moe import (
            get_aiter_moe_config,
            aiter_moe,
            MoeSolutionType,
            MoeQuantType,
        )
    else:
        raise Exception("VLLM_HCU_USE_CUSTOM_OPS or VLLM_HCU_USE_AITER_W4A16_MOE not set.")
except Exception:
    get_aiter_moe_config = None
    aiter_moe = None
    MoeQuantType = None
    print("INFO: Please install aiter if you want to infer with aiter_moe.")

from vllm_hcu.platforms.hcu import on_gfx938
""",
),

(

        "device_supports_fp8 = ",'device_supports_fp8 = on_gfx938() or ',
),

(
"""
    # Check constraints.
    if use_int4_w4a16:
""",
"""
    if henvs.VLLM_HCU_USE_CUSTOM_OPS and henvs.VLLM_HCU_USE_AITER_W4A16_MOE and use_int4_w4a16 and hidden_states.dtype == torch.bfloat16 and get_aiter_moe_config is not None and aiter_moe is not None:
        # 根据 aiter 的config 判断是否启用 aiter

        M, K = hidden_states.shape
        E, N1, _ = w1.shape
        _, N2, _ = w2.shape
        top_k_num = topk_ids.size(1)

        status, moe_config = get_aiter_moe_config(
            M=M, E=E, N1=N1, N2=N2, K=K,
            top_k=top_k_num, block_size=block_shape[1], dtype=hidden_states.dtype,
            quant_type=MoeQuantType.W4A16,
        )

        if not status:
            logger.info_once(
                f"[aiter_moe_w4a16] SKIP {M=}, {E=}, {N1=}, {N2=}, {K=}, {top_k_num=}, {block_shape=}: "
                f"no backend available"
            )
        else:
            is_inplace = inplace and not disable_inplace()
            return aiter_moe(hidden_states, w1, w2, topk_weights, topk_ids, moe_config, is_inplace, activation, w1_scale, w2_scale, w1_zp, w2_zp, a1_scale, a2_scale,
                        block_shape, global_num_experts, expert_map)
    # Check constraints.
    if use_int4_w4a16:
""",
),
]