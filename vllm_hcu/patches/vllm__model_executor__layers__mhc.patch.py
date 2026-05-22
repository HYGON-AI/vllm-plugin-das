# SPDX-License-Identifier: Apache-2.0

"""
Patch for vllm/model_executor/layers/mhc.py
"""

PATCHES = [
(
"""
with T.Kernel(num_tokens, threads=96) as i:
        T.pdl_sync()
""",
"""
with T.Kernel(num_tokens, threads=96) as i:
        # T.pdl_sync()
"""
),

(
"""
                T.copy(ol, layer_input[i, i0_h * hidden_block])
        T.pdl_trigger()
""",
"""
                T.copy(ol, layer_input[i, i0_h * hidden_block])
        # T.pdl_trigger()
"""
),

(
"""
        c_local = T.alloc_fragment(hc, T.float32)
        T.pdl_sync()
""",
"""
        c_local = T.alloc_fragment(hc, T.float32)
        # T.pdl_sync()
"""
),

(
"""
            T.copy(x_shared, x[i_n, 0, i0_h * h_blk])
        T.pdl_trigger()
""",
"""
            T.copy(x_shared, x[i_n, 0, i0_h * h_blk])
        # T.pdl_trigger()
"""
),

(
"""
        h_split_start = i_ks * h_per_split

        T.pdl_sync()           
""",
"""
        h_split_start = i_ks * h_per_split

        # T.pdl_sync()
"""
),

(
"""
                rp_out[i_ks, i_n] = v2

        T.pdl_trigger()           
""",
"""
                rp_out[i_ks, i_n] = v2

        # T.pdl_trigger()            
"""
),

(
"""
    with T.Kernel(num_tokens, threads=n_thr) as i:
        T.pdl_sync()           
""",
"""
    with T.Kernel(num_tokens, threads=n_thr) as i:
        # T.pdl_sync()            
"""
),

(
"""
            T.copy(ol, out[i, i0_h * h_block], disable_tma=True)

        T.pdl_trigger()           
""",
"""
            T.copy(ol, out[i, i0_h * h_block], disable_tma=True)

        # T.pdl_trigger()            
"""
),

(
"""
    if num_tokens <= fma_token_threshold:
        mhc_fused_tilelang(         
""",
"""
    # if num_tokens <= fma_token_threshold:
    if False:
        mhc_fused_tilelang(         
"""
),
]
