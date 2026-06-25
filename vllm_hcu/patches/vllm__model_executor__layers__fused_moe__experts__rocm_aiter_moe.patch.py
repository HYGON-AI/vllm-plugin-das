# SPDX-License-Identifier: Apache-2.0

"""
Patch for vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe
Add GELU_TANH activation support for AITER MoE backend.
"""

PATCHES = [
# 1. Add GELU_TANH to ActivationMethod enum
(
"""
class ActivationMethod(IntEnum):
    # This allows interfacing with AITER ActivationType enum
    # without importing the ActivationType enum from AITER globally.
    SILU = 0
    GELU = 1
""",
"""
class ActivationMethod(IntEnum):
    # This allows interfacing with AITER ActivationType enum
    # without importing the ActivationType enum from AITER globally.
    SILU = 0
    GELU = 1
    GELU_TANH = 3
""",
),

# 2. Add GELU_TANH case to activation mapping
(
"""
    if activation == MoEActivation.SILU:
        activation_method = ActivationMethod.SILU
    elif activation == MoEActivation.GELU:
        activation_method = ActivationMethod.GELU
    elif activation == MoEActivation.SWIGLUOAI:
""",
"""
    if activation == MoEActivation.SILU:
        activation_method = ActivationMethod.SILU
    elif activation == MoEActivation.GELU:
        activation_method = ActivationMethod.GELU
    elif activation == MoEActivation.GELU_TANH:
        activation_method = ActivationMethod.GELU_TANH
    elif activation == MoEActivation.SWIGLUOAI:
""",
),

# 3. Add GELU_TANH to _supports_activation
(
"""
        return activation in [
            MoEActivation.SILU,
            MoEActivation.GELU,
            MoEActivation.SWIGLUOAI,
        ]
""",
"""
        return activation in [
            MoEActivation.SILU,
            MoEActivation.GELU,
            MoEActivation.GELU_TANH,
            MoEActivation.SWIGLUOAI,
        ]
""",
),

]
