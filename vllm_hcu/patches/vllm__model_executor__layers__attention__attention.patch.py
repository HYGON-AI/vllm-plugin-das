# SPDX-License-Identifier: Apache-2.0

"""
Patch for vllm.model_executor.layers.attention.attention unified_kv_cache_update
"""

PATCHES = [
(
    "import vllm.envs as envs",
    "import vllm.envs as envs\nimport vllm_hcu.platforms.envs as henvs",
),

(
"""
    return torch.empty(0, device=kv_cache.device, dtype=kv_cache.dtype)
""",
"""
    if henvs.VLLM_HCU_USE_FA_UNIFIED_ATTENTION:
        return torch.empty(0, device=key.device, dtype=key.dtype)
    else:
        return torch.empty(0, device=kv_cache.device, dtype=kv_cache.dtype)
""",
),
]