# SPDX-License-Identifier: Apache-2.0

"""
Patch for vllm.model_executor.layers.fla.ops.chunk chunk_gated_delta_rule_fwd
"""

PATCHES = [
(
"""
from .chunk_delta_h import chunk_gated_delta_rule_fwd_h
from .chunk_o import chunk_fwd_o
""",
"""
import vllm_hcu.platforms.envs as henvs
try:
    from aiter.ops.triton.fla.chunk_delta_h import chunk_gated_delta_rule_fwd_h
    from aiter.ops.triton.fla.chunk_o import chunk_fwd_o
    
except ImportError:
    from .chunk_delta_h import chunk_gated_delta_rule_fwd_h
    from .chunk_o import chunk_fwd_o
    os.environ['VLLM_HCU_USE_CUSTOM_AITER_FLA'] = '0'
""",

),

(
"""
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
    )
    o = chunk_fwd_o(
        q=q,
        k=k,
        v=v_new,
        h=h,
        g=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
    )
""",

"""
    if henvs.VLLM_HCU_USE_CUSTOM_AITER_FLA:
        h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        use_exp2=False,
        transpose_state_layout=True,
        )
        o = chunk_fwd_o(
            q=q,
            k=k,
            v=v_new,
            h=h,
            g=g,
            scale=scale,
            cu_seqlens=cu_seqlens,
            transpose_state_layout=True,
        )
    else:
        h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        )
        o = chunk_fwd_o(
            q=q,
            k=k,
            v=v_new,
            h=h,
            g=g,
            scale=scale,
            cu_seqlens=cu_seqlens,
        )
""",
),

]

