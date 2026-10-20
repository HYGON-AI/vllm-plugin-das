# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Plain-matmul expert computation for the DTK2610 MoE fallback.

Routing, prepare/finalize, shared experts, TP reduction and routed scaling are
owned by the surrounding vLLM MoE pipeline. This module only replaces the
per-expert GEMM + activation step.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

_MAX_ROWS_PER_CHUNK = 2048
_GATED_ACTIVATIONS = frozenset({"silu", "gelu", "gelu_tanh"})
_RELU2_ACTIVATIONS = frozenset({"relu2", "relu_squared", "relu2_no_mul"})


def _activation_name(activation: object) -> str:
    return str(getattr(activation, "value", activation)).lower()


def _apply_gated_activation(
    x: torch.Tensor, activation: object
) -> torch.Tensor:
    name = _activation_name(activation)
    gate, up = x.chunk(2, dim=-1)
    if name == "silu":
        return F.silu(gate) * up
    if name in ("gelu", "gelu_tanh"):
        return F.gelu(gate, approximate="tanh") * up
    raise ValueError(
        f"VLLM_HCU_USE_REFERENCE_MOE does not support gated activation {name!r}"
    )


def _apply_plain_activation(
    x: torch.Tensor, activation: object
) -> torch.Tensor:
    name = _activation_name(activation)
    if name in _RELU2_ACTIVATIONS:
        return torch.square(F.relu(x))
    if name == "relu":
        return F.relu(x)
    raise ValueError(
        f"VLLM_HCU_USE_REFERENCE_MOE does not support activation {name!r}"
    )


def apply_reference_moe(
    output: torch.Tensor,
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: object,
    global_num_experts: int,
    expert_map: torch.Tensor | None,
    apply_router_weight_on_input: bool,
) -> None:
    """Write the routed expert output into ``output``.

    ``w1`` is ``[num_local_experts, 2 * I, H]`` for gated activations and
    ``[num_local_experts, I, H]`` for non-gated activations. ``w2`` is
    ``[num_local_experts, H, I]``.

    ``topk_ids`` contains global expert ids; ``expert_map`` maps them to the
    local expert index, with ``-1`` meaning "not on this rank".
    """

    if topk_weights is None or topk_ids is None:
        raise ValueError("reference MoE requires precomputed top-k routing")
    if hidden_states.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ):
        raise TypeError(
            "reference MoE only supports fp16/bf16/fp32 hidden states, "
            f"got {hidden_states.dtype}"
        )
    if w1.dtype != hidden_states.dtype or w2.dtype != hidden_states.dtype:
        raise TypeError("reference MoE requires unquantized weights")
    if topk_ids.ndim != 2 or topk_weights.shape != topk_ids.shape:
        raise ValueError("topk_weights and topk_ids must have the same 2-D shape")

    gated = w1.size(1) == 2 * w2.size(2)
    mapping = (
        None
        if expert_map is None
        else expert_map.detach().to("cpu").tolist()
    )
    num_local_experts = w1.size(0)
    num_tokens = hidden_states.size(0)

    # Accumulate in FP32 so bf16/fp16 rounding is not amplified across experts.
    out_fp32 = torch.zeros(
        (num_tokens, w2.size(1)),
        dtype=torch.float32,
        device=hidden_states.device,
    )

    for expert in torch.unique(topk_ids).tolist():
        if expert < 0:
            continue
        if mapping is None:
            if expert >= num_local_experts:
                continue
            local = int(expert)
        else:
            if expert >= len(mapping):
                continue
            local = int(mapping[expert])
            if local < 0 or local >= num_local_experts:
                continue

        rows, slots = torch.where(topk_ids == expert)
        for start in range(0, rows.numel(), _MAX_ROWS_PER_CHUNK):
            chunk_rows = rows[start : start + _MAX_ROWS_PER_CHUNK]
            chunk_slots = slots[start : start + _MAX_ROWS_PER_CHUNK]
            x = hidden_states[chunk_rows]
            weights = topk_weights[chunk_rows, chunk_slots].to(torch.float32)

            if apply_router_weight_on_input:
                x = (x.float() * weights.unsqueeze(-1)).to(
                    hidden_states.dtype
                )

            gate_up = torch.matmul(x, w1[local].T)
            if gated:
                act = _apply_gated_activation(gate_up, activation)
            else:
                act = _apply_plain_activation(gate_up, activation)

            expert_out = torch.matmul(act, w2[local].T).float()
            if not apply_router_weight_on_input:
                expert_out = expert_out * weights.unsqueeze(-1)

            out_fp32.index_add_(0, chunk_rows, expert_out)

    output.copy_(out_fp32.to(output.dtype))