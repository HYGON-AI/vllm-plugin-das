# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Plain-matmul expert computation used as a DTK fallback.

The Triton fused MoE kernel miscomputes a small fraction of elements on
DTK2610, which degenerates generation into repeated tokens.  This module
recomputes only the per-expert GEMM plus activation with plain matmuls.

Everything else stays with its existing owner in the surrounding MoE pipeline:
router selection, prepare/finalize, shared experts, latent projections, TP
all-reduce, sequence parallelism and routed scaling.

The plain-matmul path reproduces the parts of the fused kernel that change the
numerics, so that enabling the fallback does not change the model function:

* ``apply_router_weight_on_input`` inputs are already weighted by the prepare
  step; the routed weight must not be applied a second time.
* ``w1_bias``/``w2_bias`` are added to the FP32 accumulator of their GEMM, i.e.
  before the activation and before the routed scaling, respectively.
* the SwiGLU clamp (``gemm1_clamp_limit``/``swiglu_limit``) is applied to the
  gate and up halves before ``silu``.
* GEMM2 keeps FP32 precision until after the routed scaling, like the fused
  kernel's FP32 accumulator, so FP16 cannot overflow before the scaling.
* ``gelu`` stays exact; only ``gelu_tanh`` uses the tanh approximation.

The per-expert loop is data dependent: it enumerates ``topk_ids`` on the host
and sizes its row slices from the device.  It therefore synchronizes the device
during the forward pass and cannot run inside a CUDA graph.  Enabling
``VLLM_HCU_USE_REFERENCE_MOE`` requires eager execution; the HCU config
validation rejects every capture mode instead of replaying a stale graph.
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

# The fused kernel only applies the clamp for SILU (and the OAI variants,
# which this fallback rejects); see ``TritonExperts.activation``.
_CLAMPED_ACTIVATIONS = frozenset({"silu"})


def _activation_name(activation: object) -> str:
    return str(getattr(activation, "value", activation)).lower()


def _apply_gated_activation(
    gate_up: torch.Tensor,
    name: str,
    clamp_limit: float | None,
) -> torch.Tensor:
    gate, up = gate_up.chunk(2, dim=-1)
    if name == "silu":
        if clamp_limit is not None and clamp_limit > 0:
            # Mirrors ``swiglu_limit_func``: gate is only clamped above, up is
            # clamped to [-limit, limit].
            gate = torch.clamp(gate, max=clamp_limit)
            up = torch.clamp(up, min=-clamp_limit, max=clamp_limit)
        return F.silu(gate) * up
    if name == "gelu":
        return F.gelu(gate) * up
    if name == "gelu_tanh":
        return F.gelu(gate, approximate="tanh") * up
    if name in _PLAIN_RELU2_ACTIVATIONS:
        return torch.square(F.relu(gate)) * up
    raise ValueError(
        "VLLM_HCU_USE_REFERENCE_MOE does not support gated activation "
        f"{name!r}"
    )


def _apply_plain_activation(
    x: torch.Tensor,
    name: str,
    clamp_limit: float | None,
) -> torch.Tensor:
    del clamp_limit  # The fused kernel ignores it for non-gated activations.
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
    *,
    w1_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
    clamp_limit: float | None = None,
) -> None:
    """Write the routed expert output into ``output`` in place.

    ``w1`` is ``[num_local_experts, 2 * I, H]`` for gated activations and
    ``[num_local_experts, I, H]`` for non-gated activations.
    ``w2`` is ``[num_local_experts, H, I]``.
    ``w1_bias``/``w2_bias`` are optional ``[num_local_experts, N]`` biases.

    ``topk_ids`` holds global expert ids.  ``expert_map`` maps them to the
    local expert index, where ``-1`` means the expert is not on this rank.

    ``hidden_states`` arrives after the prepare step.  When
    ``apply_router_weight_on_input`` is set, prepare has already multiplied it
    by the routed weights, so they are not applied again here; the flag only
    suppresses the output-side scaling, exactly like the fused kernel.
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
    if clamp_limit is not None and name not in _CLAMPED_ACTIVATIONS:
        # The fused kernel ignores the clamp for every activation it would
        # route through the reference path here; keep that behaviour instead
        # of silently changing the function.
        clamp_limit = None

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

        w1_e = w1[local]
        # GEMM2 runs in FP32 so that the routed scaling happens before the
        # cast to the output dtype; converting the weight once per expert
        # keeps that out of the row-chunk loop.
        w2_e_fp32 = w2[local].to(torch.float32)
        bias1 = None if w1_bias is None else w1_bias[local]
        bias2 = None if w2_bias is None else w2_bias[local].to(torch.float32)

        rows, slots = torch.where(topk_ids == expert)
        for start in range(0, rows.numel(), _MAX_ROWS_PER_CHUNK):
            chunk_rows = rows[start : start + _MAX_ROWS_PER_CHUNK]
            chunk_slots = slots[start : start + _MAX_ROWS_PER_CHUNK]
            x = hidden_states[chunk_rows]
            weights = topk_weights[chunk_rows, chunk_slots].to(torch.float32)

            # ``x`` is the prepared input: already router-weighted when
            # ``apply_router_weight_on_input`` is set.
            if bias1 is None:
                gate_up = torch.matmul(x, w1_e.T)
            else:
                # addmm adds the bias to the FP32 accumulator before the cast,
                # which is what the fused GEMM1 does with ``B_bias``.
                gate_up = torch.addmm(bias1.to(x.dtype), x, w1_e.T)

            if gated:
                act = _apply_gated_activation(gate_up, name, clamp_limit)
            else:
                act = _apply_plain_activation(gate_up, name, clamp_limit)

            act_fp32 = act.to(torch.float32)
            if bias2 is None:
                expert_out = torch.matmul(act_fp32, w2_e_fp32.T)
            else:
                expert_out = torch.addmm(bias2, act_fp32, w2_e_fp32.T)
            if not apply_router_weight_on_input:
                expert_out = expert_out * weights.unsqueeze(-1)

            out_fp32.index_add_(0, chunk_rows, expert_out)

    output.copy_(out_fp32.to(output.dtype))


__all__ = ["apply_reference_moe"]
