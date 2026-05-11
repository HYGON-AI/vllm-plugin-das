# SPDX-License-Identifier: Apache-2.0

"""
Patch for vllm.model_executor.layers.fla.ops.chunk_o
Add aiter launch function for chunk_fwd_kernel_o
"""

PATCHES = [
(
"""
BKV_LIST = [64, 128] if check_shared_mem() else [32, 64]
""",
"""
import vllm_hcu.platforms.envs as henvs
from aiter.ops.triton.fla.vllm.chunk_o import launch_chunk_fwd_kernel_o

BKV_LIST = [64, 128] if check_shared_mem() else [32, 64]
""",
),

(
"""
    chunk_fwd_kernel_o[grid](
        q,
        k,
        v,
        h,
        g,
        o,
        cu_seqlens,
        chunk_indices,
        scale,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
    )
    return o
""",
"""
    if henvs.VLLM_HCU_USE_CUSTOM_AITER_FLA and henvs.VLLM_HCU_USE_CUSTOM_OPS:
        launch_chunk_fwd_kernel_o(
            q=q,
            k=k,
            v=v,
            h=h,
            g=g,
            g_gamma=None,
            o=o,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            scale=scale,
            T=T,
            H=H,
            Hg=Hg,
            K=K,
            V=V,
            BT=BT,
            NT=NT,
            B=B,
            use_exp2=False,
            transpose_state_layout=True,
            kernel_cfg=None,
        )
    else:
        chunk_fwd_kernel_o[grid](
            q,
            k,
            v,
            h,
            g,
            o,
            cu_seqlens,
            chunk_indices,
            scale,
            T=T,
            H=H,
            Hg=Hg,
            K=K,
            V=V,
            BT=BT,
        )
    return o
""",
),

]
