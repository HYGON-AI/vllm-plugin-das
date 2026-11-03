"""AITER ASM GEMMs with vLLM's dynamic INT8 and activation semantics."""

from __future__ import annotations

import torch


def unsupported_reason(arguments: dict[str, object], native: object) -> str | None:
    """Inspect metadata before launching any sorting or expert kernel."""
    if not all(
        callable(getattr(native, name, None))
        for name in ("moe_sorting_fwd", "asm_fmoe_a8")
    ):
        return "missing-stage-api"
    config = getattr(arguments["moe_config"], "config", None)
    if not isinstance(config, dict) or not all(
        isinstance(config.get(key), int) and config[key] > 0
        for key in ("SOL_ID1", "SOL_ID2", "BLOCK_SIZE_M")
    ):
        return "unsupported-stage-config"
    if config.get("PADDED_K") is not None:
        return "padded-layout"
    if arguments["activation"] != "silu":
        return "unsupported-stage-activation"
    hidden = arguments["hidden_states"]
    if (
        hidden.dtype not in (torch.bfloat16, torch.float16)
        or arguments["output_dtype"] != hidden.dtype
    ):
        return "unsupported-stage-dtype"
    if any(
        arguments.get(key) is not None
        for key in ("w1_zp", "w2_zp", "a1_scale", "a2_scale")
    ):
        return "unsupported-stage-quantization"
    for name in (
        "hidden_states",
        "w1",
        "w2",
        "topk_ids",
        "topk_weights",
        "w1_scale",
        "w2_scale",
    ):
        value = arguments.get(name)
        if (
            not isinstance(value, torch.Tensor)
            or not value.is_contiguous()
            or value.device != hidden.device
        ):
            return "unsupported-stage-tensor"
    if arguments["w1"].dtype != torch.int8 or arguments["w2"].dtype != torch.int8:
        return "unsupported-stage-weights"
    if (
        arguments["topk_ids"].dtype not in (torch.int32, torch.int64)
        or arguments["topk_weights"].dtype != torch.float32
    ):
        return "unsupported-stage-routing"
    for name, weight in (("w1_scale", arguments["w1"]), ("w2_scale", arguments["w2"])):
        scale = arguments[name]
        if scale.dtype != torch.float32 or tuple(scale.shape) != (*weight.shape[:2], 1):
            return "unsupported-stage-scales"
    mask = arguments["expert_map"]
    if mask is not None and (
        mask.dtype != torch.int32
        or mask.ndim != 1
        or not mask.is_contiguous()
        or mask.device != hidden.device
        or mask.numel() < arguments["global_num_experts"]
    ):
        return "unsupported-stage-mask"
    # Native ASM uses 32-bit buffer descriptors. Do not launch a stage whose
    # route workspace reaches that byte bound; official fallback owns it.
    routes = arguments["topk_ids"].numel()
    width = max(arguments["w1"].shape[1], arguments["w2"].shape[1])
    if routes * width * hidden.element_size() >= 2**31:
        return "stage-buffer-bound"
    return None


def execute(
    *,
    native: object,
    hidden_states,
    w1,
    w2,
    topk_weights,
    topk_ids,
    moe_config,
    w1_scale,
    w2_scale,
    expert_map,
    global_num_experts,
    output_dtype,
    **unused,
):
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.int8_utils import (
        per_token_quant_int8,
    )

    config = moe_config.config
    m = hidden_states.shape[0]
    top_k = topk_ids.shape[1]
    routes = m * top_k
    block = config["BLOCK_SIZE_M"]
    domain = global_num_experts if expert_map is None else expert_map.numel()
    # At most block-1 padding entries per expert; one extra block per expert
    # gives a conservative upper bound without reading device route counts.
    capacity = routes + domain * block
    device = hidden_states.device
    ids = torch.empty(capacity, device=device, dtype=torch.int32)
    weights = torch.empty(capacity, device=device, dtype=torch.float32)
    experts = torch.empty(
        (capacity + block - 1) // block, device=device, dtype=torch.int32
    )
    positions = torch.empty(2 * domain, device=device, dtype=torch.int32)
    count = torch.empty(1, device=device, dtype=torch.int32)
    native.moe_sorting_fwd(
        topk_ids.to(torch.int32),
        topk_weights,
        ids,
        weights,
        experts,
        positions,
        count,
        None,
        domain,
        block,
        expert_map,
    )

    # Zero remote routes before activation and reduction. Each stage writes
    # only locally owned routes, and capture retains these device-side writes.
    first = torch.zeros((routes, w1.shape[1]), dtype=output_dtype, device=device)
    hidden_q, hidden_scale = per_token_quant_int8(hidden_states)
    out_type = int(output_dtype == torch.bfloat16)
    native.asm_fmoe_a8(
        first,
        hidden_q,
        w1,
        w1,
        ids,
        topk_weights,
        experts,
        count,
        top_k,
        hidden_scale,
        w1_scale,
        None,
        0,
        config["SOL_ID1"],
        out_type,
        0,
        0,
    )
    activated = torch.empty((routes, w2.shape[2]), dtype=output_dtype, device=device)
    torch.ops._C.silu_and_mul(activated, first)
    bridge_q, bridge_scale = per_token_quant_int8(activated)
    second = torch.zeros((m, top_k, w2.shape[1]), dtype=output_dtype, device=device)
    native.asm_fmoe_a8(
        second,
        bridge_q,
        w2,
        w2,
        ids,
        topk_weights,
        experts,
        count,
        top_k,
        bridge_scale,
        w2_scale,
        None,
        1,
        config["SOL_ID2"],
        out_type,
        0,
        0,
    )
    result = torch.empty((m, w2.shape[1]), dtype=output_dtype, device=device)
    ops.moe_sum(second, result)
    return result
