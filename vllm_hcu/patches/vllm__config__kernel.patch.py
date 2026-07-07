# SPDX-License-Identifier: Apache-2.0

"""
Patch vllm.config.kernel for the DeepEP DeepGEMM MoE backend option.
"""

PATCHES = [
(
"""
    "auto",
    "triton",
    "deep_gemm",
""",
"""
    "auto",
    "triton",
    "dpsk_deep_gemm",
    "deep_gemm",
""",
),

(
"""
    - "auto": Automatically select the best backend based on model and hardware
    - "triton": Use Triton-based fused MoE kernels
    - "deep_gemm": Use DeepGEMM kernels (FP8 block-quantized only)
""",
"""
    - "auto": Automatically select the best backend based on model and hardware
    - "triton": Use Triton-based fused MoE kernels
    - "dpsk_deep_gemm": Use the DPSK V4 DeepGEMM MoE path
    - "deep_gemm": Use DeepGEMM kernels (FP8 block-quantized only)
""",
),
]
