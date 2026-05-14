# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Patch for vllm.v1.core.sched.scheduler
"""

PATCHES = [
    (
"""
                kv_cache_config=self.kv_cache_config,
""",
"""
                kv_cache_config=self.kv_cache_config,
                dp_rank=self.parallel_config.data_parallel_rank
""",              
    ),
]
