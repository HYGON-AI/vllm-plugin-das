# SPDX-License-Identifier: Apache-2.0

"""
Patch for vllm.model_executor.layers.fused_moe.experts.triton_moe  support triton int8 kernel
"""

PATCHES = [
(
"""
        device_supports_int8 = (
            current_platform.is_cuda()
            and current_platform.has_device_capability((7, 5))
        )
""",
"""
        device_supports_int8 = (current_platform.is_rocm() or (current_platform.is_cuda() and current_platform.has_device_capability((7, 5))))
""",
),
]