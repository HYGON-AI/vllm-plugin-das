# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Route the Qwen3.5 / Qwen3-Next sparse-MoE block through the fused HCU path.

See ``vllm_hcu.ops.qwen35_moe_block``.  Unsupported layer configurations
(quantized router/shared expert, TP/EP/SP, EPLB, non-AITER W8A16 MoE) keep
the stock forward.  ``VLLM_HCU_QWEN35_MOE_BLOCK_VERIFY=N`` compares the first
N eager calls per process against the stock path and logs the difference.
"""

from __future__ import annotations

import functools
import os
from itertools import islice
from types import ModuleType

import torch

from ._common import (
    PatchCompatibilityError,
    load_exact_module,
    require_class,
    require_unpatched,
)

TARGET_MODULE = "vllm.model_executor.models.qwen3_next"
PATCH_ID = "worker.op_opt.qwen3_next.fused_moe_block"
TARGETS = (
    f"{TARGET_MODULE}.Qwen3NextSparseMoeBlock.forward",
    f"{TARGET_MODULE}.Qwen3NextModel.forward",
)
_MARKER = "_vllm_hcu_qwen35_fused_moe_block_applied"
_WRAPPER = "_vllm_hcu_qwen35_fused_moe_block_wrapper"

_verify_budget = [int(os.environ.get("VLLM_HCU_QWEN35_MOE_BLOCK_VERIFY", "0"))]


def _fused_enabled() -> bool:
    import vllm_hcu.platforms.envs as henvs

    return henvs.optional_custom_op_enabled(
        henvs.VLLM_HCU_USE_FUSED_QWEN35_MOE_BLOCK
    )


def _is_gemma_norm(norm) -> bool:
    return any(c.__name__ == "GemmaRMSNorm" for c in type(norm).__mro__)


def _layer_forward(layer, positions, hidden_states, residual, pending, fused):
    """Qwen3NextDecoderLayer.forward that consumes / produces pending MoE
    outputs.  Returns (hidden_states, residual, pending)."""
    if (
        layer.layer_scale
        or layer.use_attn_reduce_scatter_for_moe
        or not _is_gemma_norm(layer.input_layernorm)
    ):
        if pending is not None:
            hidden_states = fused.materialize(pending)
        hidden_states, residual = layer(
            positions=positions, hidden_states=hidden_states, residual=residual)
        return hidden_states, residual, None

    norm = layer.input_layernorm
    if pending is not None:
        hidden_states = fused.pending_add_rmsnorm(
            pending, residual, norm.weight, norm.variance_epsilon)
    elif residual is None:
        residual = hidden_states
        hidden_states = norm(hidden_states)
    else:
        hidden_states, residual = norm(hidden_states, residual)

    if layer.layer_type == "linear_attention":
        hidden_states = layer.linear_attn(hidden_states=hidden_states)
    else:
        hidden_states = layer.self_attn(
            hidden_states=hidden_states, positions=positions)

    hidden_states, residual = layer.post_attention_layernorm(
        hidden_states, residual)
    mlp = layer.mlp
    if type(mlp).__name__ == "Qwen3NextSparseMoeBlock" and hidden_states.shape[0]:
        state = fused.get_fused_state(mlp)
        if state is not None:
            return None, residual, fused.fused_moe_block_pending(
                mlp, state, hidden_states)
    return mlp(hidden_states), residual, None


def apply_to_module(module: ModuleType) -> bool:
    qwen = load_exact_module(TARGET_MODULE, module)
    block_cls = require_class(qwen, "Qwen3NextSparseMoeBlock", TARGETS[0])
    if getattr(qwen, _MARKER, False):
        if not getattr(block_cls.forward, _WRAPPER, False):
            raise PatchCompatibilityError(
                f"required HCU patch marker for {TARGETS[0]} is stale; "
                "restart the process"
            )
        return False
    original = require_unpatched(block_cls, "forward", TARGETS[0], _WRAPPER)

    @functools.wraps(original)
    def hcu_forward(self, hidden_states, already_sequence_parallel=False):
        import vllm_hcu.platforms.envs as henvs

        if not _fused_enabled():
            return original(self, hidden_states, already_sequence_parallel)
        from vllm_hcu.ops import qwen35_moe_block as fused

        state = fused.get_fused_state(self)
        if state is None or hidden_states.dim() != 2 or hidden_states.shape[0] == 0:
            return original(self, hidden_states, already_sequence_parallel)
        out = fused.fused_moe_block_forward(self, state, hidden_states)
        if _verify_budget[0] > 0 and not torch.cuda.is_current_stream_capturing():
            _verify_budget[0] -= 1
            ref = original(self, hidden_states.clone(), already_sequence_parallel)
            err = (out.float() - ref.float()).abs().max().item()
            scale = ref.float().abs().max().item()
            fused.logger.warning(
                "fused MoE block verify: M=%d max|diff|=%.3e max|ref|=%.3e",
                hidden_states.shape[0], err, scale)
        return out

    setattr(hcu_forward, _WRAPPER, True)
    qwen._vllm_hcu_original_qwen3_next_moe_block_forward = original
    block_cls.forward = hcu_forward

    model_cls = require_class(qwen, "Qwen3NextModel", TARGETS[1])
    original_model_forward = require_unpatched(
        model_cls, "forward", TARGETS[1], _WRAPPER)

    @functools.wraps(original_model_forward)
    def hcu_model_forward(self, input_ids, positions,
                          intermediate_tensors=None, inputs_embeds=None):
        """Pass MoE outputs between layers un-reduced so each MoE tail is
        fused into the next layer's (or the final) Gemma fused-add RMSNorm,
        removing one kernel per layer."""
        import vllm_hcu.platforms.envs as henvs
        from vllm.distributed import get_pp_group

        pp = get_pp_group()
        if (
            not _fused_enabled()
            or not henvs.VLLM_HCU_FUSED_QWEN35_MOE_NORM
            or not (pp.is_first_rank and pp.is_last_rank)
            or getattr(self, "use_sequence_parallel", False)
            or getattr(self, "aux_hidden_state_layers", ())
            or not _is_gemma_norm(self.norm)
        ):
            return original_model_forward(
                self, input_ids, positions, intermediate_tensors, inputs_embeds)
        from vllm_hcu.ops import qwen35_moe_block as fused

        hidden_states = (inputs_embeds if inputs_embeds is not None
                         else self.embed_input_ids(input_ids))
        residual = None
        pending = None
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual, pending = _layer_forward(
                layer, positions, hidden_states, residual, pending, fused)
        if pending is not None:
            return fused.pending_add_rmsnorm(
                pending, residual, self.norm.weight,
                self.norm.variance_epsilon)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    setattr(hcu_model_forward, _WRAPPER, True)
    qwen._vllm_hcu_original_qwen3_next_model_forward = original_model_forward
    model_cls.forward = hcu_model_forward
    setattr(qwen, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
