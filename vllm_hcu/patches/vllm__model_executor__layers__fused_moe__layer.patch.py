# SPDX-License-Identifier: Apache-2.0

"""
vllm.model_executor.layers.fused_moe.layer: one-shot debug print for FusedMoE init
"""

PATCHES = [
(
"""
# --8<-- [start:fused_moe]
""",
"""
first = True

# --8<-- [start:fused_moe]
""",
),
(
"""
        # Expert mapping used in self.load_weights
        self.expert_mapping = expert_mapping
""",
"""
        # Expert mapping used in self.load_weights
        self.expert_mapping = expert_mapping

        global first
        if first:
            print(f"###################self.global_num_experts:{self.global_num_experts}, self.logical_num_experts:{self.logical_num_experts} self.moe_parallel_config:{self.moe_parallel_config}")
            first = False
""",
),
]
