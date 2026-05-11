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
            mixed_qkv_non_spec = causal_conv1d_fn(
                mixed_qkv_non_spec_T,
                conv_weights,
                self.conv1d.bias,
                activation=self.activation,
                conv_states=conv_state,
                has_initial_state=has_initial_state,
                cache_indices=non_spec_state_indices_tensor,
                query_start_loc=non_spec_query_start_loc,
                metadata=attn_metadata,
            ).transpose(0, 1)
""",
"""
            if (
                henvs.VLLM_HCU_USE_CUSTOM_OPS
                and henvs.VLLM_HCU_USE_CUSTOM_CAUSAL_CONV1D
            ):
                from causal_conv1d import causal_conv1d_fn_dcu

                mixed_qkv_non_spec = causal_conv1d_fn_dcu(
                    mixed_qkv_non_spec_T,
                    conv_weights,
                    self.conv1d.bias,
                    activation=self.activation,
                    initial_states=conv_state,
                    has_initial_state=has_initial_state,
                    cache_indices=non_spec_state_indices_tensor,
                    query_start_loc=non_spec_query_start_loc,
                    seq_lens_cpu=attn_metadata.nums_dict["seqlens"],
                ).transpose(0, 1)
            else:
                mixed_qkv_non_spec = causal_conv1d_fn(
                    mixed_qkv_non_spec_T,
                    conv_weights,
                    self.conv1d.bias,
                    activation=self.activation,
                    conv_states=conv_state,
                    has_initial_state=has_initial_state,
                    cache_indices=non_spec_state_indices_tensor,
                    query_start_loc=non_spec_query_start_loc,
                    metadata=attn_metadata,
                ).transpose(0, 1)
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
    from aiter.ops.triton.fla.fused_recurrent import fused_recurrent_gated_delta_rule_packed_decode
    from aiter.ops.triton.fla.fused_sigmoid_gating import fused_sigmoid_gating_delta_rule_update
except ImportError:
    from vllm.model_executor.layers.fla.ops import (
        fused_recurrent_gated_delta_rule_packed_decode,
        fused_sigmoid_gating_delta_rule_update,
    )

""",
),



]
