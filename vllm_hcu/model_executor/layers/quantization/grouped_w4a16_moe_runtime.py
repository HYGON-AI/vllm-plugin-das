# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""AITER dispatch for canonical symmetric GPTQ/AWQ group-wise INT4 experts."""

from __future__ import annotations

import torch
from vllm.logger import init_logger

from vllm_hcu.model_executor.layers.fused_moe.aiter_moe_dispatch import (
    AiterMoeProblem,
    execute_aiter_moe,
    prepare_aiter_moe_scales,
    prepare_aiter_moe_weights,
    select_aiter_moe_config,
)

logger = init_logger("vllm.hcu.grouped_w4a16_moe")
_SCALE_OWNER_ATTR = "_hcu_grouped_w4a16_scale_owner"


def _grouped_experts(model):
    if model is None:
        return
    for layer in model.modules():
        method = getattr(layer, "quant_method", None)
        seen = set()
        while method is not None and id(method) not in seen:
            seen.add(id(method))
            kernel = getattr(method, "moe_kernel", None)
            experts = getattr(kernel, "fused_experts", None)
            quant = getattr(experts, "quant_config", None)
            if (
                getattr(quant, "use_int4_w4a16", False)
                and not getattr(quant, "_hcu_channel_w4a16", False)
            ):
                yield layer, experts
            method = getattr(method, "old_quant_method", None)


def snapshot_grouped_w4a16_scale_owners(model) -> dict[object, object]:
    """Keep captured owners reachable if checkpoint reload rebuilds the kernel."""
    return {
        layer: getattr(experts, _SCALE_OWNER_ATTR, experts.quant_config)
        for layer, experts in _grouped_experts(model)
    }


def _bind_layer_scales(layer, experts) -> tuple[torch.Tensor, torch.Tensor]:
    """Use the Parameters restored to the layer after checkpoint processing."""
    quant = experts.quant_config
    for layer_name, expert_name, desc_name in (
        ("w13_scales", "w1_scale", "_w1"),
        ("w2_scales", "w2_scale", "_w2"),
    ):
        source = getattr(layer, layer_name, None)
        current = getattr(experts, expert_name)
        if not isinstance(source, torch.Tensor):
            continue
        if (source.shape, source.dtype, source.device) != (
            current.shape, current.dtype, current.device,
        ):
            raise RuntimeError(f"grouped W4A16 {layer_name} layout changed on reload")
        if source is current:
            continue
        descriptor = getattr(quant, desc_name, None)
        if descriptor is not None:
            descriptor.scale = source
        else:
            setattr(experts, expert_name, source)
    return experts.w1_scale, experts.w2_scale


def refresh_grouped_w4a16_scale_caches(
    model, previous_owners: dict[object, object] | None = None,
) -> None:
    """Refresh graph-owned scale allocations before weight updates can replay."""
    from vllm_hcu.model_executor.layers.fused_moe.aiter_moe_dispatch import (
        refresh_aiter_moe_scale_cache,
    )

    for layer, experts in _grouped_experts(model):
        current_owner = getattr(experts, _SCALE_OWNER_ATTR, experts.quant_config)
        owner = (previous_owners or {}).get(layer, current_owner)
        scales = _bind_layer_scales(layer, experts)
        refresh_aiter_moe_scale_cache(owner, *scales, update_generation=True)
        if current_owner is not owner:
            refresh_aiter_moe_scale_cache(
                current_owner, *scales, update_generation=True,
            )
            setattr(experts, _SCALE_OWNER_ATTR, owner)


def _symmetric_qzeros(
    quant: object,
    name: str,
    shape: tuple[int, int, int],
    device: torch.device,
) -> torch.Tensor:
    """Return GPTQ's implicit packed zero point for an AITER C kernel.

    AutoGPTQ stores symmetric GPTQ zero points implicitly and therefore leaves
    ``FusedMoEQuantConfig.w*_zp`` unset.  AITER's MOE_C ABI requires the
    packed byte tensor even for symmetric weights, where every nibble is 8.
    """
    cache_name = f"_hcu_aiter_{name}"
    cached = getattr(quant, cache_name, None)
    if (
        isinstance(cached, torch.Tensor)
        and cached.device == device
        and cached.dtype == torch.uint8
        and tuple(cached.shape) == shape
    ):
        return cached
    cached = torch.full(shape, 0x88, dtype=torch.uint8, device=device)
    try:
        setattr(quant, cache_name, cached)
    except (AttributeError, TypeError):
        pass
    return cached


def apply_grouped_w4a16_moe(
    experts, output, hidden_states, w1, w2, topk_weights, topk_ids,
    activation, global_num_experts, expert_map, a1q_scale, a2_scale,
    expert_tokens_meta, apply_router_weight_on_input,
) -> bool:
    """Return False for unsupported semantics or an explicit no-solution result.

    Keep canonical packed weights so a different batch can choose a different
    AITER solution and explicit no-solution can still use vLLM's WNA16 path.
    Kernel/ABI errors propagate instead of silently disguising a broken route.
    """
    quant = experts.quant_config
    block = getattr(experts, "block_shape", None)
    activation_config = getattr(experts, "activation_config", None)
    activation_name = str(getattr(activation, "value", activation))
    if (
        getattr(getattr(experts, "moe_config", None), "moe_backend", None) != "aiter"
        or not getattr(quant, "use_int4_w4a16", False)
        or getattr(quant, "use_int8_w8a16", False)
        or getattr(quant, "_hcu_channel_w4a16", False)
        or block is None or len(block) != 2 or block[0] != 0 or block[1] <= 0
        or activation_name != "silu"
        or getattr(activation_config, "clamp_limit", None) is not None
        or getattr(activation_config, "alpha", 1.0) != 1.0
        or getattr(activation_config, "beta", 0.0) != 0.0
        or getattr(activation_config, "activation_situ_beta", None) is not None
        or getattr(activation_config, "activation_situ_linear_beta", None) is not None
        or hidden_states.dtype not in (torch.float16, torch.bfloat16)
        or w1.dtype != torch.uint8 or w2.dtype != torch.uint8
        or getattr(quant, "w1_zp", None) is not None
        or getattr(quant, "w2_zp", None) is not None
        or expert_map is not None
        or global_num_experts not in (-1, w1.shape[0])
        or a1q_scale is not None or a2_scale is not None
        or expert_tokens_meta is not None or apply_router_weight_on_input
        or getattr(experts, "_lora_context", None) is not None
    ):
        return False
    if (
        hidden_states.ndim != 2 or w1.ndim != 3 or w2.ndim != 3
        or not hidden_states.is_contiguous()
        or w1.stride(-1) != 1 or w2.stride(-1) != 1
        or w1.shape[0] != w2.shape[0]
        or w1.shape[2] * 2 != hidden_states.shape[1]
        or w2.shape[1] != hidden_states.shape[1]
        or w2.shape[2] * 4 != w1.shape[1]
        or hidden_states.shape[1] % block[1] or (w2.shape[2] * 2) % block[1]
        or output.shape != hidden_states.shape
        or topk_ids.ndim != 2 or topk_ids.shape != topk_weights.shape
        or topk_ids.shape[0] != hidden_states.shape[0]
    ):
        return False
    from aiter.moe import MoeQuantType

    problem = AiterMoeProblem(
        M=hidden_states.shape[0], E=w1.shape[0], N1=w1.shape[1],
        N2=w2.shape[1], K=hidden_states.shape[1], top_k=topk_ids.shape[1],
        block_size=int(block[1]), dtype=hidden_states.dtype,
        device=hidden_states.device, quant_type=MoeQuantType.W4A16,
        activation=activation_name, use_shuffle=False,
    )
    config = select_aiter_moe_config(problem, cache_owner=w1)
    if config is None:
        return False
    is_moe_c = str(getattr(config, "solution_type", "")).lower() == "moe_c"
    if is_moe_c and getattr(config, "need_shuffle_scale", False):
        # Older AITER releases have no group-size field on the selected
        # config and shuffle W4A16 scales as fixed group-32 blocks. They can
        # select group-128 MOE_C but cannot prepare its scale layout safely.
        selected_group = getattr(config, "block_size", None)
        if (selected_group or 32) != block[1]:
            return False
    w1_zp = w2_zp = None
    if is_moe_c:
        w1_zp = _symmetric_qzeros(
            quant, "w1_zp",
            (w1.shape[0], w1.shape[1] // 2, hidden_states.shape[1] // block[1]),
            w1.device,
        )
        w2_zp = _symmetric_qzeros(
            quant, "w2_zp",
            (w2.shape[0], w2.shape[1] // 2, w2.shape[2] * 2 // block[1]),
            w2.device,
        )
    prepared_w1, prepared_w2 = prepare_aiter_moe_weights(
        w1, w2, config, cache_owner=w1, block_shape=block, preserve_inputs=True,
    )
    scale_owner = getattr(experts, _SCALE_OWNER_ATTR, quant)
    s1, s2 = prepare_aiter_moe_scales(
        experts.w1_scale, experts.w2_scale, config, cache_owner=scale_owner,
    )
    result = execute_aiter_moe(
        config, hidden_states=hidden_states, w1=prepared_w1, w2=prepared_w2,
        topk_weights=topk_weights.to(torch.float32), topk_ids=topk_ids.to(torch.int32),
        activation=activation_name, w1_scale=s1, w2_scale=s2,
        w1_zp=w1_zp if is_moe_c else None,
        w2_zp=w2_zp if is_moe_c else None, block_shape=block,
        global_num_experts=w1.shape[0], expert_map=None,
        use_weight_shuffle=bool(getattr(config, "need_shuffle", False)),
        output_dtype=output.dtype,
    )
    output.copy_(result)
    logger.info_once(
        "HCU grouped W4A16 MoE executes AITER solution=%s (E=%d, N1=%d, "
        "K=%d, group=%d, top_k=%d)", config.solution_type,
        problem.E, problem.N1, problem.K, problem.block_size, problem.top_k,
    )
    return True
