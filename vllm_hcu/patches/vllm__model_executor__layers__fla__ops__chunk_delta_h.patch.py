# SPDX-License-Identifier: Apache-2.0

"""
Patch for vllm.model_executor.layers.fla.ops.chunk_delta_h
Add aiter launch function for chunk_gated_delta_rule_fwd_kernel_h_blockdim64
"""

PATCHES = [
(
"""
NUM_WARPS = [2, 4, 8, 16]
""",
"""
import vllm_hcu.platforms.envs as henvs
from aiter.ops.triton.fla.vllm.chunk_delta_h import launch_chunk_gated_delta_rule_fwd_kernel_h_blockdim64

NUM_WARPS = [2, 4, 8, 16]
""",
),

(
"""
    chunk_gated_delta_rule_fwd_kernel_h_blockdim64[grid](
        k=k,
        v=u,
        w=w,
        v_new=v_new,
        g=g,
        gk=gk,
        h=h,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
    )
    return h, v_new, final_state
""",
"""
    if henvs.VLLM_HCU_USE_CUSTOM_AITER_FLA and henvs.VLLM_HCU_USE_CUSTOM_OPS:
        launch_chunk_gated_delta_rule_fwd_kernel_h_blockdim64(
            k=k,
            u=u,
            w=w,
            v_new=v_new,
            g=g,
            gk=gk,
            h=h,
            initial_state=initial_state,
            initial_state_indices=None,
            final_state=final_state,
            cu_seqlens=cu_seqlens,
            chunk_offsets=chunk_offsets,
            N=N,
            T=T,
            H=H,
            Hg=Hg,
            K=K,
            V=V,
            BT=BT,
            use_exp2=False,
            transpose_state_layout=True,
            kernel_cfg=None,
        )
    else:
        chunk_gated_delta_rule_fwd_kernel_h_blockdim64[grid](
            k=k,
            v=u,
            w=w,
            v_new=v_new,
            g=g,
            gk=gk,
            h=h,
            h0=initial_state,
            ht=final_state,
            cu_seqlens=cu_seqlens,
            chunk_offsets=chunk_offsets,
            T=T,
            H=H,
            Hg=Hg,
            K=K,
            V=V,
            BT=BT,
        )
    return h, v_new, final_state
""",
),

]
