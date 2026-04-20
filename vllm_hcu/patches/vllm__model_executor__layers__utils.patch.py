# SPDX-License-Identifier: Apache-2.0

"""
vllm.model_executor.layers.utils dispatch_unquantized_gemm
"""

PATCHES = [
(
"""
import torch
""",
"""
import torch
import vllm_hcu.platforms.envs as henvs
""",
),

(
"""
        return rocm_unquantized_gemm
""",
"""
        if henvs.VLLM_HCU_USE_DEFAULT_GEMM:
            return default_unquantized_gemm
        else:
            return rocm_unquantized_gemm
            
""",
),
]