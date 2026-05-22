# SPDX-License-Identifier: Apache-2.0

"""
vllm.model_executor.layers.fused_moe.fused_moe_modular_method: N/K for kernel ctor, use_nn_moe on apply
"""

PATCHES = [
(
"""
                inplace=inplace,
""",
"""
                inplace=inplace,
                N=old_quant_method.N if hasattr(old_quant_method, "N") else -1,
                K=old_quant_method.K if hasattr(old_quant_method, "K") else -1,
""",
),
(
"""
    ) -> torch.Tensor:
""",
"""
        use_nn_moe: bool | None = False,
    ) -> torch.Tensor:
""",
),
]
