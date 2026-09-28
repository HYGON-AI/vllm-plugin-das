# SPDX-License-Identifier: Apache-2.0
"""Kimi KDA state allocation shared by the model and attention layer."""
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateShapeCalculator


def kda_state_shape(tp_world_size, num_heads, head_dim, *, conv_kernel_size, num_spec=0):
    # The paired vLLM KDA calculator accepts num_spec but drops it. The GDN
    # calculator implements the same equal-Q/K/V-head layout AND reserves
    # the draft history read/written by causal_conv1d_update. Reuse its SD/DS
    # orientation so model-level cache sizing and layer views cannot diverge.
    if type(num_spec) is not int or num_spec < 0:
        raise ValueError("Kimi KDA num_spec must be a non-negative integer")
    return MambaStateShapeCalculator.gated_delta_net_state_shape(
        tp_world_size=tp_world_size,
        num_k_heads=num_heads,
        num_v_heads=num_heads,
        head_k_dim=head_dim,
        head_v_dim=head_dim,
        conv_kernel_size=conv_kernel_size,
        num_spec=num_spec,
    )
