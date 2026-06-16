# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect

import torch

from vllm.config import get_current_vllm_config_or_none
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.distributed import get_tp_group
from vllm.model_executor.layers.fused_moe.utils import moe_kernel_quantize_input
from vllm.platforms import current_platform


_HCU_RUNTIME_SP_ENABLED: bool | None = None


def configure_hcu_runtime_sp(enabled: bool) -> None:
    global _HCU_RUNTIME_SP_ENABLED
    _HCU_RUNTIME_SP_ENABLED = bool(enabled)


def hcu_runtime_sp_enabled() -> bool:
    tp_size = get_tensor_model_parallel_world_size()
    if tp_size == 1:
        return False
    vllm_config = get_current_vllm_config_or_none()
    if vllm_config is not None:
        enabled = bool(
            getattr(vllm_config.parallel_config, "enable_custom_sp", False)
        )
        configure_hcu_runtime_sp(enabled)
        return enabled
    return bool(_HCU_RUNTIME_SP_ENABLED)


def split_tokens_for_sp(x: torch.Tensor | None) -> torch.Tensor | None:
    if x is None or not hcu_runtime_sp_enabled():
        return x
    sp_size = get_tensor_model_parallel_world_size()
    sp_rank = get_tensor_model_parallel_rank()
    assert x.shape[0] % sp_size == 0, (
        "Runtime sequence parallelism requires the token dimension to be "
        "padded to a multiple of tensor parallel size."
    )
    local_tokens = x.shape[0] // sp_size
    return x.narrow(0, sp_rank * local_tokens, local_tokens).contiguous()


def split_positions_for_sp(positions: torch.Tensor) -> torch.Tensor:
    if not hcu_runtime_sp_enabled():
        return positions
    sp_size = get_tensor_model_parallel_world_size()
    sp_rank = get_tensor_model_parallel_rank()
    token_dim = -1 if positions.dim() > 1 else 0
    assert positions.shape[token_dim] % sp_size == 0, (
        "Runtime sequence parallelism requires positions to be padded to a "
        "multiple of tensor parallel size."
    )
    local_tokens = positions.shape[token_dim] // sp_size
    return positions.narrow(
        token_dim, sp_rank * local_tokens, local_tokens
    ).contiguous()


def gather_tokens_for_sp(x: torch.Tensor) -> torch.Tensor:
    if not hcu_runtime_sp_enabled():
        return x
    return get_tp_group().all_gather(x, dim=0).contiguous()


def reduce_scatter_tokens_for_sp(x: torch.Tensor) -> torch.Tensor:
    if not hcu_runtime_sp_enabled():
        return x
    return get_tp_group().reduce_scatter(x, dim=0).contiguous()


def quant_method_supports_sp_moe_quant_inputs(quant_method) -> bool:
    apply = getattr(quant_method, "apply", None)
    if apply is None:
        return False
    try:
        parameters = inspect.signature(apply).parameters
    except (TypeError, ValueError):
        return False
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
        return True
    return "i_q" in parameters and "i_s" in parameters


def quantize_hidden_states_for_sp_moe(
    hidden_states: torch.Tensor,
    experts,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if not hcu_runtime_sp_enabled():
        return None, None
    moe_parallel_config = getattr(experts, "moe_parallel_config", None)
    if (
        moe_parallel_config is not None
        and getattr(moe_parallel_config, "use_all2all_kernels", False)
    ):
        return None, None
    moe_quant_config = getattr(experts, "moe_quant_config", None)
    if moe_quant_config is None or not getattr(moe_quant_config, "is_quantized", False):
        return None, None
    runner = getattr(experts, "runner", None)
    quant_method = getattr(runner, "_quant_method", None)
    if getattr(quant_method, "is_monolithic", False):
        return None, None
    if not quant_method_supports_sp_moe_quant_inputs(quant_method):
        return None, None
    if moe_quant_config.block_shape is not None:
        return None, None
    if moe_quant_config.quant_dtype not in (current_platform.fp8_dtype(), torch.int8):
        return None, None

    return moe_kernel_quantize_input(
        hidden_states,
        moe_quant_config.a1_scale,
        moe_quant_config.quant_dtype,
        moe_quant_config.per_act_token_quant,
        moe_quant_config.block_shape,
        is_scale_swizzled=moe_quant_config.is_scale_swizzled,
        ocp_mx_scheme=moe_quant_config.ocp_mx_scheme,
        mx_alignment=moe_quant_config.mx_alignment,
    )


def gather_quanted_moe_inputs_for_sp(
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor | None,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
    local_tokens = hidden_states.shape[0]
    hidden_states = gather_tokens_for_sp(hidden_states)
    if (
        hidden_states_scale is not None
        and hidden_states_scale.dim() > 0
        and hidden_states_scale.shape[0] == local_tokens
    ):
        hidden_states_scale = gather_tokens_for_sp(hidden_states_scale)
    topk_weights = gather_tokens_for_sp(topk_weights)
    topk_ids = gather_tokens_for_sp(topk_ids)
    return hidden_states, hidden_states_scale, topk_weights, topk_ids


def use_sp_mlp_token_gather(reduce_results: bool) -> bool:
    return hcu_runtime_sp_enabled() and reduce_results


def sp_mlp_down_proj_reduce_results(reduce_results: bool) -> bool:
    return reduce_results and not use_sp_mlp_token_gather(reduce_results)


def prepare_mlp_inputs_for_sp(
    x: torch.Tensor,
    x_and_scale_quanted: tuple[torch.Tensor, torch.Tensor] | None,
    use_sp_token_gather: bool,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
    if not use_sp_token_gather:
        return x, x_and_scale_quanted
    if x_and_scale_quanted is None:
        return gather_tokens_for_sp(x.contiguous()), None

    x_and_scale_quanted = (
        gather_tokens_for_sp(x_and_scale_quanted[0].contiguous()),
        gather_tokens_for_sp(x_and_scale_quanted[1].contiguous()),
    )
    sp_size = get_tensor_model_parallel_world_size()
    x = torch.empty(
        (x.shape[0] * sp_size, x.shape[-1]),
        device=x.device,
        dtype=x.dtype,
    )
    return x, x_and_scale_quanted


def finalize_mlp_output_for_sp(
    x: torch.Tensor,
    use_sp_token_gather: bool,
) -> torch.Tensor:
    if not use_sp_token_gather:
        return x
    return reduce_scatter_tokens_for_sp(x)


def prepare_moe_inputs_for_sp(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    experts,
) -> tuple[
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    hidden_states_dtype = hidden_states.dtype
    quanted_hidden_states, hidden_states_scale = quantize_hidden_states_for_sp_moe(
        hidden_states, experts
    )

    if quanted_hidden_states is not None:
        topk_weights, topk_ids = experts.runner.router.select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
        )
        (
            quanted_hidden_states,
            hidden_states_scale,
            topk_weights,
            topk_ids,
        ) = gather_quanted_moe_inputs_for_sp(
            quanted_hidden_states,
            hidden_states_scale,
            topk_weights,
            topk_ids,
        )
        hidden_states = torch.empty(
            quanted_hidden_states.shape,
            device=quanted_hidden_states.device,
            dtype=hidden_states_dtype,
        )
        router_logits = None
    else:
        hidden_states = gather_tokens_for_sp(hidden_states)
        router_logits = gather_tokens_for_sp(router_logits)
        topk_weights = None
        topk_ids = None

    return (
        hidden_states,
        router_logits,
        quanted_hidden_states,
        hidden_states_scale,
        topk_weights,
        topk_ids,
    )


def finalize_moe_output_for_sp(
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    return reduce_scatter_tokens_for_sp(hidden_states)


def prepare_attention_inputs_for_sp(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    total_num_heads: int,
    total_num_kv_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return qkv_token_to_head_all2all_for_sp(
        q,
        k,
        v,
        total_num_heads,
        total_num_kv_heads,
        head_dim,
    )


def finalize_attention_output_for_sp(
    attn_output: torch.Tensor,
    total_num_heads: int,
    head_dim: int,
) -> torch.Tensor:
    return head_to_token_all2all_for_sp(attn_output, total_num_heads, head_dim)


def token_to_head_all2all_for_sp(
    x: torch.Tensor,
    total_num_heads: int,
    head_dim: int,
) -> torch.Tensor:
    if not hcu_runtime_sp_enabled():
        return x
    sp_size = get_tensor_model_parallel_world_size()
    if total_num_heads < sp_size:
        sp_rank = get_tensor_model_parallel_rank()
        local_head = sp_rank % total_num_heads
        x = x.view(x.shape[0], total_num_heads, head_dim)
        x = x[:, local_head : local_head + 1, :].reshape(x.shape[0], head_dim)
        return gather_tokens_for_sp(x)
    assert total_num_heads % sp_size == 0
    local_tokens = x.shape[0]
    heads_per_rank = total_num_heads // sp_size
    x = x.view(local_tokens, sp_size, heads_per_rank, head_dim)
    send = x.permute(1, 0, 2, 3).contiguous()
    recv = torch.empty_like(send)
    get_tp_group().all_to_all_single(recv, send)
    return recv.reshape(sp_size * local_tokens, heads_per_rank * head_dim).contiguous()


def qkv_token_to_head_all2all_for_sp(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    total_num_heads: int,
    total_num_kv_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not hcu_runtime_sp_enabled():
        return q, k, v
    sp_size = get_tensor_model_parallel_world_size()
    if total_num_heads % sp_size != 0 or total_num_kv_heads % sp_size != 0:
        return (
            token_to_head_all2all_for_sp(q, total_num_heads, head_dim),
            token_to_head_all2all_for_sp(k, total_num_kv_heads, head_dim),
            token_to_head_all2all_for_sp(v, total_num_kv_heads, head_dim),
        )
    local_tokens = q.shape[0]
    heads_per_rank = total_num_heads // sp_size
    kv_heads_per_rank = total_num_kv_heads // sp_size
    q = q.contiguous().view(local_tokens, sp_size, heads_per_rank*head_dim)
    k = k.contiguous().view(local_tokens, sp_size, kv_heads_per_rank*head_dim)
    v = v.contiguous().view(local_tokens, sp_size, kv_heads_per_rank*head_dim)
    qkv = torch.cat([q, k, v], dim=-1)
    send = qkv.permute(1, 0, 2).contiguous()
    recv = torch.empty(sp_size, local_tokens, (heads_per_rank + 2 * kv_heads_per_rank) * head_dim,
                       device=send.device, dtype=send.dtype)
    get_tp_group().all_to_all_single(recv, send)
    recv = recv.view(
        sp_size * local_tokens,
        (heads_per_rank + 2 * kv_heads_per_rank) * head_dim,
    )

    q_size = heads_per_rank * head_dim
    kv_size = kv_heads_per_rank * head_dim
    q, k, v = recv.split([q_size, kv_size, kv_size], dim=-1)
    return q.contiguous(), k.contiguous(), v.contiguous()


def head_to_token_all2all_for_sp(
    x: torch.Tensor,
    total_num_heads: int,
    head_dim: int,
) -> torch.Tensor:
    if not hcu_runtime_sp_enabled():
        return x
    sp_size = get_tensor_model_parallel_world_size()
    assert total_num_heads % sp_size == 0
    heads_per_rank = total_num_heads // sp_size
    assert x.shape[0] % sp_size == 0
    local_tokens = x.shape[0] // sp_size
    send = x.view(sp_size, local_tokens, heads_per_rank*head_dim).contiguous()
    recv = torch.empty(sp_size, local_tokens, heads_per_rank*head_dim, device=send.device, dtype = send.dtype)
    get_tp_group().all_to_all_single(recv, send)
    return recv.permute(1, 0, 2).contiguous().view(
        local_tokens, total_num_heads * head_dim
    )
