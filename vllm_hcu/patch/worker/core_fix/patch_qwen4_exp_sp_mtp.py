# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Run the Qwen4Exp MTP drafter sequence-parallel under sequence-parallel MoE.

Counterpart of ``patch_qwen4_exp_sp`` for ``Qwen4ExpMultiTokenPredictor``: the
fused embedding/backbone state is sharded across TP ranks before the HC
decoder layer, and the sampled and multi-stream outputs are all-gathered
together afterwards (upstream vLLM PR #56322, ``nvidia/mtp.py``).

``patch_qwen4_exp_mtp_pp`` must already have rewritten this forward (its
first-PP-rank guard becomes ``hidden_states is not None``); the replacement
below carries that rewrite verbatim, so it supersedes the MTP PP rebuild and
keeps that patch's wrapper marker for its idempotence check.
"""

from __future__ import annotations

import functools
import inspect
from types import ModuleType

import torch

from . import patch_qwen4_exp_mtp_pp as mtp_pp
from . import patch_qwen4_exp_sp as sp
from ._common import PatchCompatibilityError, load_exact_module, require_class
from .patch_qwen4_exp_ple_prefetch import _require_model_init_compatible

TARGET_MODULE = "vllm.models.qwen4_exp.amd.mtp"
PATCH_ID = "worker.core_fix.qwen4_exp.sequence_parallel_moe_mtp"
TARGETS = (
    f"{TARGET_MODULE}.Qwen4ExpMultiTokenPredictor.__init__",
    f"{TARGET_MODULE}.Qwen4ExpMultiTokenPredictor.forward",
)
_MARKER = "_vllm_hcu_qwen4_exp_sp_mtp_applied"
_WRAPPER = "_vllm_hcu_qwen4_exp_sp_mtp_wrapper"

# The patched module; module-level names such as ``get_pp_group`` are looked
# up there at call time, exactly as the original forward resolves them.
_target: ModuleType | None = None


def _make_init(init):
    @functools.wraps(init)
    def hcu_mtp_init(self, *args, vllm_config=None, prefix="", **kwargs):
        init(self, *args, vllm_config=vllm_config, prefix=prefix, **kwargs)
        setattr(self, sp.HC_SP_ATTR, sp.hc_sequence_parallel_enabled(vllm_config))

    return hcu_mtp_init


def forward(
    self,
    input_ids: torch.Tensor | None,
    positions: torch.Tensor,
    hidden_states: torch.Tensor | None = None,
    intermediate_tensors=None,
    inputs_embeds: torch.Tensor | None = None,
    spec_step_idx: int = 0,
):
    hc_count = self.hc_count
    hidden_size = self.hidden_size

    if hidden_states is not None:
        assert hidden_states is not None
        if inputs_embeds is None:
            assert input_ids is not None
            inputs_embeds = self.embed_input_ids(input_ids)
        # Embedding branch: pre-norm -> fc_embedding -> [T, H].
        inputs_embeds = self.pre_fc_norm_embedding(inputs_embeds)
        inputs_embeds = self.fc_embedding(inputs_embeds)

        # Backbone hidden is multi-stream [T, hc_count*H] (scheme A:
        # the main model truly emits the pre-final-mixer multi stream
        # on the first step; subsequent steps reuse the prior draft
        # step's multi stream).
        num_tokens = hidden_states.shape[0]
        hidden_states = hidden_states.view(num_tokens, hc_count, hidden_size)
        hidden_states = self.pre_fc_norm_hidden(hidden_states.flatten(-2)).view(
            num_tokens, hc_count, hidden_size
        )
        hidden_states = self.fc_hidden(hidden_states)
        # Add the embedding residual to every branch, then fold back
        # to [T, hc_count*H] (HC outer, HS inner) for the HC decoder.
        hidden_states = inputs_embeds.unsqueeze(-2) + hidden_states
        hidden_states = hidden_states.flatten(-2)
    else:
        assert intermediate_tensors is not None
        hidden_states = intermediate_tensors["hidden_states"]

    hc_sp = getattr(self, sp.HC_SP_ATTR, False)
    full_num_tokens = positions.shape[-1]
    if hc_sp:
        # The FC outputs are complete on every rank and need only a slice.
        hidden_states = sp.shard_tokens_for_hc_sp(hidden_states)

    current_step_idx = spec_step_idx % self.num_mtp_layers
    layer = self.layers[current_step_idx]
    hidden_states, block_output, injection = layer(
        hidden_states=hidden_states,
        prev_block_output=None,
        prev_injection=None,
        positions=positions,
        input_ids=None,
        query_start_loc=None,
        ngram_context=None,
    )
    if not _target.get_pp_group().is_last_rank:
        # As in the target model, PP carries a materialized tensor rather
        # than the delayed hidden/output/injection tuple.
        hidden_states = layer.mlp_hyper_connection.combine(
            hidden_states, block_output, injection
        )
        return _target.IntermediateTensors({"hidden_states": hidden_states})

    # Last PP rank finalize. Keep both:
    #   (A) sample_hidden_states [T, H]  -> single stream for the LM head
    #   (B) multi_hidden [T, hc_count*H] -> pre-final-mixer multi stream
    #       for the next draft step (zero extra compute, just kept).
    multi_hidden, sample_hidden_states, _ = (
        self.hyper_connection_mixer.combine_and_mix(
            hidden_states, block_output, injection
        )
    )
    if hc_sp:
        sample_hidden_states, multi_hidden = sp.gather_packed(
            sample_hidden_states, multi_hidden, full_num_tokens
        )
    return sample_hidden_states, multi_hidden


def apply_to_module(module: ModuleType) -> bool:
    mtp_module = load_exact_module(TARGET_MODULE, module)
    predictor = require_class(
        mtp_module,
        "Qwen4ExpMultiTokenPredictor",
        f"{TARGET_MODULE}.Qwen4ExpMultiTokenPredictor",
    )
    current_init = vars(predictor).get("__init__")
    current_forward = vars(predictor).get("forward")
    if getattr(mtp_module, _MARKER, False):
        if not (
            getattr(current_init, _WRAPPER, False)
            and getattr(current_forward, _WRAPPER, False)
        ):
            raise PatchCompatibilityError(
                f"required HCU patch marker for {TARGET_MODULE} is stale"
            )
        return False
    if getattr(current_init, _WRAPPER, False) or getattr(
        current_forward, _WRAPPER, False
    ):
        raise PatchCompatibilityError(
            "refusing a partial Qwen4Exp MTP sequence-parallel patch"
        )

    # This copy is only valid on top of the audited MTP PP rewrite.
    if not (
        getattr(mtp_module, mtp_pp._MARKER, False)
        and getattr(current_forward, mtp_pp._WRAPPER, False)
    ):
        raise PatchCompatibilityError(
            f"{TARGETS[1]} requires {mtp_pp.PATCH_ID} to be applied first"
        )
    original = getattr(predictor, mtp_pp._ORIGINAL)
    mtp_pp._rebuild_forward(original)  # re-verifies the audited source fingerprint
    if not sp.same_parameters(original, forward):
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGETS[1]} has incompatible signature "
            f"{inspect.signature(original)}"
        )
    init = _require_model_init_compatible(predictor, TARGETS[0])
    global _target
    _target = mtp_module

    hcu_init = _make_init(init)
    functools.update_wrapper(forward, original)
    setattr(hcu_init, _WRAPPER, True)
    setattr(forward, _WRAPPER, True)
    setattr(forward, mtp_pp._WRAPPER, True)
    try:
        predictor.__init__ = hcu_init
        predictor.forward = forward
        setattr(mtp_module, _MARKER, True)
    except BaseException:
        predictor.__init__ = current_init
        predictor.forward = current_forward
        if hasattr(mtp_module, _MARKER):
            delattr(mtp_module, _MARKER)
        raise
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
