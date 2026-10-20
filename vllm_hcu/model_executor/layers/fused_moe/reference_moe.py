# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Plain-matmul expert computation used as a DTK fallback.

The Triton fused MoE kernel miscomputes a small fraction of elements on
DTK2610, which degenerates generation into repeated tokens.  This module
recomputes only the per-expert GEMM plus activation with plain matmuls.

Everything else stays with its existing owner in the surrounding MoE pipeline:
router selection, prepare/finalize, shared experts, latent projections, TP
all-reduce, sequence parallelism and routed scaling.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# Rows processed per expert per accumulation step.  Tests patch this value to
# prove chunking does not change the result.
_MAX_ROWS_PER_CHUNK = 2048

_GATED_ACTIVATIONS = frozenset(
    {"silu", "gelu", "gelu_tanh", "relu2", "relu_squared"}
)
_PLAIN_RELU2_ACTIVATIONS = frozenset(
    {"relu2", "relu_squared", "relu2_no_mul", "relu2nomul"}
)
_PLAIN_ACTIVATIONS = frozenset({"relu"})


def _activation_name(activation: object) -> str:
    return str(getattr(activation, "value", activation)).lower()


def _apply_gated_activation(
    gate_up: torch.Tensor, name: str
) -> torch.Tensor:
    gate, up = gate_up.chunk(2, dim=-1)
    if name == "silu":
        return F.silu(gate) * up
    if name in ("gelu", "gelu_tanh"):
        return F.gelu(gate, approximate="tanh") * up
    if name in _PLAIN_RELU2_ACTIVATIONS:
        return torch.square(F.relu(gate)) * up
    raise ValueError(
        "VLLM_HCU_USE_REFERENCE_MOE does not support gated activation "
        f"{name!r}"
    )


def _apply_plain_activation(x: torch.Tensor, name: str) -> torch.Tensor:
    if name in _PLAIN_RELU2_ACTIVATIONS:
        return torch.square(F.relu(x))
    if name in _PLAIN_ACTIVATIONS:
        return F.relu(x)
    raise ValueError(
        f"VLLM_HCU_USE_REFERENCE_MOE does not support activation {name!r}"
    )


def _local_expert_map(
    expert_map: torch.Tensor | None,
    num_local_experts: int,
) -> dict[int, int]:
    """Resolve global expert ids to local indices in a single device sync.

    The mapping is tiny (``global_num_experts`` entries) but lives on the
    device, so it is read once per call instead of once per routed expert.
    Experts absent from this rank are omitted.
    """

    if expert_map is None:
        return {}
    values = expert_map.detach().to("cpu", torch.int64).tolist()
    return {
        global_id: local
        for global_id, local in enumerate(values)
        if 0 <= local < num_local_experts
    }


def apply_reference_moe(
    output: torch.Tensor,
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor | None,
    topk_ids: torch.Tensor | None,
    activation: object,
    global_num_experts: int,
    expert_map: torch.Tensor | None,
    apply_router_weight_on_input: bool,
) -> None:
    """Write the routed expert output into ``output`` in place.

    ``w1`` is ``[num_local_experts, 2 * I, H]`` for gated activations and
    ``[num_local_experts, I, H]`` for non-gated activations.
    ``w2`` is ``[num_local_experts, H, I]``.

    ``topk_ids`` holds global expert ids.  ``expert_map`` maps them to the
    local expert index, where ``-1`` means the expert is not on this rank.
    """

    del global_num_experts  # Routing already happened upstream.
    if topk_weights is None or topk_ids is None:
        raise ValueError(
            "reference MoE requires precomputed top-k routing "
            "(topk_weights and topk_ids)"
        )
    if topk_ids.ndim != 2 or topk_weights.shape != topk_ids.shape:
        raise ValueError(
            "topk_weights and topk_ids must have the same 2-D shape, got "
            f"{tuple(topk_weights.shape)} and {tuple(topk_ids.shape)}"
        )
    if hidden_states.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ):
        raise TypeError(
            "reference MoE only supports fp16/bf16/fp32 hidden states, got "
            f"{hidden_states.dtype}"
        )
    if w1.dtype != hidden_states.dtype or w2.dtype != hidden_states.dtype:
        raise TypeError(
            "reference MoE requires unquantized weights matching the hidden "
            f"state dtype, got {w1.dtype}/{w2.dtype} vs {hidden_states.dtype}"
        )
    if hidden_states.shape[0] == 0:
        output.zero_()
        return

    name = _activation_name(activation)
    gated = w1.shape[1] == 2 * w2.shape[2]
    if gated and name not in _GATED_ACTIVATIONS:
        raise ValueError(
            "VLLM_HCU_USE_REFERENCE_MOE does not support gated activation "
            f"{name!r}"
        )
    if not gated and name not in (
        _PLAIN_RELU2_ACTIVATIONS | _PLAIN_ACTIVATIONS
    ):
        raise ValueError(
            f"VLLM_HCU_USE_REFERENCE_MOE does not support activation {name!r}"
        )

    num_local_experts = w1.shape[0]
    num_tokens = hidden_states.shape[0]
    local_map = _local_expert_map(expert_map, num_local_experts)

    # FP32 accumulation keeps bf16/fp16 rounding from compounding across the
    # experts a single token is routed to.
    out_fp32 = torch.zeros(
        (num_tokens, w2.shape[1]),
        dtype=torch.float32,
        device=hidden_states.device,
    )

    for expert in torch.unique(topk_ids).tolist():
        expert = int(expert)
        if expert < 0:
            continue
        if expert_map is None:
            if expert >= num_local_experts:
                continue
            local = expert
        else:
            local = local_map.get(expert)
            if local is None:
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
                act = _apply_gated_activation(gate_up, name)
            else:
                act = _apply_plain_activation(gate_up, name)

            expert_out = torch.matmul(act, w2[local].T).float()
            if not apply_router_weight_on_input:
                expert_out = expert_out * weights.unsqueeze(-1)

            out_fp32.index_add_(0, chunk_rows, expert_out)

    output.copy_(out_fp32.to(output.dtype))


__all__ = ["apply_reference_moe"]
