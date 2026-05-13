# SPDX-License-Identifier: Apache-2.0

"""
Patch for vllm.v1.attention to extend the GFX9 arch allowlist for HCU.
"""

PATCHES = [
(
"""
        from vllm import _custom_ops as ops

        ops.concat_and_cache_mla(
            kv_c_normed,
            k_pe.squeeze(1),
            kv_cache,
            slot_mapping.flatten(),
            kv_cache_dtype=kv_cache_dtype,
            scale=k_scale,
        )
""",
"""
        from vllm_hcu.v1.attention.backends.fa_utils import hcu_ops
        torch.ops.hcu_ops.concat_and_cache_mla(
            kv_c_normed,
            k_pe.squeeze(1),
            kv_cache,
            slot_mapping.flatten(),
            kv_cache_dtype,
            k_scale,
        )
""",
),
]
