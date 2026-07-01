# SPDX-License-Identifier: Apache-2.0

"""
Patch for vllm._aiter_ops is_aiter_found_and_supported.
"""

PATCHES = [
(
"""
import vllm.envs as envs
"""
,
"""
import vllm.envs as envs
from vllm_hcu.platforms import envs as henvs
"""
),
(
"""
        return on_mi3xx()
""",
"""
        from vllm_hcu.platforms.hcu import on_gfx93x

        return on_gfx93x() or on_mi3xx()
""",
),
(
"""
def _rocm_aiter_fused_moe_impl(
""",
"""
@functools.cache
def _get_aiter_w16a16_moe_solution_id(
    M: int,
    E: int,
    N1: int,
    N2: int,
    K: int,
    top_k: int,
    dtype: torch.dtype,
    activation: str,
    use_shuffle: int,
) -> str:
    try:
        from aiter.moe import MoeQuantType, MoeSolutionType, get_aiter_moe_config
    except Exception as exc:
        raise RuntimeError(
            "HCU AITER ASM MoE is enabled but aiter.moe.get_aiter_moe_config "
            "is unavailable."
        ) from exc

    status, moe_config = get_aiter_moe_config(
        M=M,
        E=E,
        N1=N1,
        N2=N2,
        K=K,
        top_k=top_k,
        block_size=0,
        dtype=dtype,
        quant_type=MoeQuantType.W16A16,
        activation=activation,
        use_shuffle=use_shuffle,
    )
    if not status or moe_config.solution_type != MoeSolutionType.ASM:
        raise RuntimeError(
            "AITER W16A16 MoE did not find a valid ASM backend config with "
            f"M={M}, E={E}, N1={N1}, N2={N2}, K={K}, top_k={top_k}, "
            f"dtype={dtype}, activation={activation}."
        )

    config = moe_config.config or {}
    try:
        return f"{config['SOL_ID1']}+{config['SOL_ID2']}"
    except KeyError as exc:
        raise RuntimeError(
            f"AITER W16A16 MoE ASM config is missing SOL_ID1/SOL_ID2: {config}"
        ) from exc


def _rocm_aiter_fused_moe_impl(
""",
),
(
"""
    activation = ActivationType(activation_method)
    quant_type = QuantType(quant_method)
""",
"""
    activation = (
        "gelu_tanh"
        if activation_method == 3
        else ActivationType(activation_method)
    )
    quant_type = QuantType(quant_method)
""",
),
(
"""
            "gelu": ActivationType.Gelu,
            "swiglu": ActivationType.Swiglu,
""",
"""
            "gelu": ActivationType.Gelu,
            "gelu_tanh": getattr(ActivationType, "GeluTanh", None),
            "gelu_pytorch_tanh": getattr(ActivationType, "GeluTanh", None),
            "swiglu": ActivationType.Swiglu,
""",
),
(
"""
    return fused_moe(
        hidden_states,
        w1,
        w2,
        topk_weight,
        topk_ids,
        expert_mask,
        activation,
        quant_type,
        doweight_stage1,
        w1_scale,
        w2_scale,
        a1_scale,
        a2_scale,
        num_local_tokens=num_local_tokens,
        dtype=output_dtype,
        hidden_pad=hidden_pad,
        intermediate_pad=intermediate_pad,
        bias1=bias1,
        bias2=bias2,
    )
""",
"""
    activation_name = {
        0: "silu",
        1: "gelu",
        2: "swiglu",
        3: "gelu_tanh",
    }.get(activation_method)
    use_w16a16_asm = (
        envs.VLLM_ROCM_USE_AITER
        and envs.VLLM_ROCM_USE_AITER_MOE
        and quant_type == QuantType.No
        and w1_scale is None
        and w2_scale is None
        and a1_scale is None
        and a2_scale is None
    )
    if use_w16a16_asm:
        from aiter.fused_moe_asm_wna16 import fused_experts_asm_impl

        use_shuffle = 1 if henvs.VLLM_HCU_USE_AITER_W16A16_MOE_SHUFFLE else 0
        global_num_experts = (
            expert_mask.numel() if expert_mask is not None else w1.shape[0]
        )
        if henvs.VLLM_HCU_USE_AITER_MOE_CONFIG:
            solution_id = _get_aiter_w16a16_moe_solution_id(
                M=hidden_states.shape[0],
                E=w1.shape[0],
                N1=w1.shape[1],
                N2=w2.shape[1],
                K=w1.shape[2],
                top_k=topk_ids.shape[1],
                dtype=hidden_states.dtype,
                activation=activation_name,
                use_shuffle=use_shuffle,
            )
            return fused_experts_asm_impl(
                hidden_states,
                w1,
                w2,
                topk_weight,
                topk_ids,
                output_dtype or hidden_states.dtype,
                activation=activation_name,
                global_num_experts=global_num_experts,
                expert_map=expert_mask,
                use_shuffle=use_shuffle,
                solution_id=solution_id,
            )
        else:
            return fused_experts_asm_impl(
                hidden_states,
                w1,
                w2,
                topk_weight,
                topk_ids,
                output_dtype or hidden_states.dtype,
                activation=activation_name,
                global_num_experts=global_num_experts,
                expert_map=expert_mask,
                use_shuffle=use_shuffle,
            )
    else:
        return fused_moe(
            hidden_states,
            w1,
            w2,
            topk_weight,
            topk_ids,
            expert_mask,
            activation,
            quant_type,
            doweight_stage1,
            w1_scale,
            w2_scale,
            a1_scale,
            a2_scale,
            num_local_tokens=num_local_tokens,
            dtype=output_dtype,
            hidden_pad=hidden_pad,
            intermediate_pad=intermediate_pad,
            bias1=bias1,
            bias2=bias2,
        )
""",
),
(
"""
    from aiter import topk_softmax

    topk_softmax(
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
        renormalize,
        num_shared_experts,
        shared_expert_scoring_func,
    )
""",
"""
    use_w16a16_asm = (
        envs.VLLM_ROCM_USE_AITER
        and envs.VLLM_ROCM_USE_AITER_MOE
    )
    if use_w16a16_asm:
        from aiter import topk_softmax
        topk_softmax(
            topk_weights,
            topk_indices,
            token_expert_indices,
            gating_output,
            renormalize,
            )
    else:
        from aiter import topk_softmax

        topk_softmax(
            topk_weights,
            topk_indices,
            token_expert_indices,
            gating_output,
            renormalize,
            num_shared_experts,
            shared_expert_scoring_func,
        )
""",
),
]
