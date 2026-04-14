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
        "device=kv_cache.device, dtype=kv_cache.dtype",
        """device=(
            key.device if henvs.VLLM_HCU_USE_FA_UNIFIED_ATTENTION else kv_cache.device
        ),
        dtype=(
            key.dtype if henvs.VLLM_HCU_USE_FA_UNIFIED_ATTENTION else kv_cache.dtype
        )""",
    ),
]