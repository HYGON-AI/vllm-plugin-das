# SPDX-License-Identifier: Apache-2.0

"""
vllm.model_executor.models.qwen3_next _forward_core
"""

PATCHES = [
(
"""
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
""",
"""
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
import vllm_hcu.platforms.envs as henvs
""",
),

(
"""
        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )
""",
"""        
        if henvs.VLLM_USE_NN:
            conv_weights = self.conv1d.weight.squeeze(1).transpose(
                0, 1).contiguous()
        else:
            conv_weights = self.conv1d.weight.view(
                self.conv1d.weight.size(0), self.conv1d.weight.size(2)
            )
""",
    ),

(
"""
from vllm.model_executor.layers.fla.ops import (
    fused_recurrent_gated_delta_rule_packed_decode,
    fused_sigmoid_gating_delta_rule_update,
)
""",
"""
try:
   
    from aiter.ops.triton.fla.fused_recurrent import (
        fused_recurrent_gated_delta_rule_packed_decode,
        fused_sigmoid_gating_delta_rule_update,
    )
except ImportError:
    from vllm.model_executor.layers.fla.ops import (
        fused_recurrent_gated_delta_rule_packed_decode,
        fused_sigmoid_gating_delta_rule_update,
    )

""",
),



]
