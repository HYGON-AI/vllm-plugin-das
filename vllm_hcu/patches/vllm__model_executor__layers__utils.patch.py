# SPDX-License-Identifier: Apache-2.0

"""
vllm.model_executor.layers.utils dispatch_unquantized_gemm
"""

PATCHES = [
(
"""
        return rocm_unquantized_gemm
""",
"""
        # return rocm_unquantized_gemm
        return default_unquantized_gemm
""",
),
]