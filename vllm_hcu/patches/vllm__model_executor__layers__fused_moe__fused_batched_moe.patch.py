# SPDX-License-Identifier: Apache-2.0

"""
Patch for vllm.model_executor.layers.fused_moe.fused_batched_moe  support triton fp8 kernel
"""

PATCHES = [
(
"""
import torch
""",
"""
import torch

from vllm_hcu.platforms.hcu import on_gfx938
""",
),

(

        "device_supports_fp8 = ",'device_supports_fp8 = on_gfx938() or ',
),
]