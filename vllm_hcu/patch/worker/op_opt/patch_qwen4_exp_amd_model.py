# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Replace the Qwen4Exp AMD model's MTP hidden-buffer copy boundary."""

from __future__ import annotations

import functools
import inspect
from types import ModuleType

from ._common import (
    PatchCompatibilityError,
    already_applied,
    load_exact_module,
    require_callable,
    require_class,
    require_exact_signature,
)

TARGET_MODULE = "vllm.models.qwen4_exp.amd.model"
PATCH_ID = "worker.op_opt.qwen4_exp.amd.model_mtp_hidden_copy"
TARGETS = (f"{TARGET_MODULE}.Qwen4ExpModel.forward",)
_MARKER = "_vllm_hcu_qwen4_exp_model_mtp_copy_applied"
_WRAPPER = "_vllm_hcu_qwen4_exp_model_mtp_copy_wrapper"


def apply_to_module(module: ModuleType) -> bool:
    model = load_exact_module(TARGET_MODULE, module)
    model_class = require_class(
        model,
        "Qwen4ExpModel",
        f"{TARGET_MODULE}.Qwen4ExpModel",
    )
    forward = require_callable(model_class, "forward", TARGETS[0])
    wrapped = ((model_class, "forward", TARGETS[0], _WRAPPER),)
    if already_applied(model, _MARKER, wrapped):
        return False

    require_exact_signature(
        forward,
        TARGETS[0],
        positional=(
            "self",
            "input_ids",
            "positions",
            "intermediate_tensors",
            "inputs_embeds",
            "query_start_loc",
            "ngram_context",
            "deepstack_input_embeds",
        ),
        defaults={
            "intermediate_tensors": None,
            "inputs_embeds": None,
            "query_start_loc": None,
            "ngram_context": None,
            "deepstack_input_embeds": None,
        },
    )
    implementation = inspect.unwrap(forward)
    code = getattr(implementation, "__code__", None)
    if code is None or not {"_mtp_hidden_buffer", "copy_"}.issubset(code.co_names):
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGETS[0]} no longer contains the "
            "audited MTP hidden-buffer copy"
        )
    torch = getattr(model, "torch", None)
    ops = getattr(getattr(torch, "ops", None), "vllm", None)
    if ops is None or not callable(
        getattr(ops, "qwen4_exp_amd_mtp_hidden_copy", None)
    ):
        raise PatchCompatibilityError(
            "Qwen4Exp MTP hidden-copy custom op was not registered before "
            f"{TARGETS[0]}"
        )

    get_pp_group = require_callable(
        model,
        "get_pp_group",
        f"{TARGET_MODULE}.get_pp_group",
    )
    islice = require_callable(model, "islice", f"{TARGET_MODULE}.islice")

    @functools.wraps(forward)
    def hcu_forward(
        self,
        input_ids,
        positions,
        intermediate_tensors=None,
        inputs_embeds=None,
        query_start_loc=None,
        ngram_context=None,
        deepstack_input_embeds=None,
    ):
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                if input_ids is None:
                    raise ValueError("input_ids or inputs_embeds is required")
                hidden_states = self.embed_input_ids(input_ids)
            hidden_states = hidden_states.repeat(1, self.config.hc_count)
        else:
            if intermediate_tensors is None:
                raise ValueError("pipeline stage requires intermediate tensors")
            hidden_states = intermediate_tensors["hidden_states"]

        block_output = None
        injection = None
        last_layer = None
        for layer_idx, layer in islice(
            enumerate(self.layers), self.start_layer, self.end_layer
        ):
            last_layer = layer
            hidden_states, block_output, injection = layer(
                hidden_states=hidden_states,
                prev_block_output=block_output,
                prev_injection=injection,
                positions=positions,
                input_ids=input_ids,
                query_start_loc=query_start_loc,
                ngram_context=ngram_context,
            )
            if deepstack_input_embeds is not None and layer_idx < len(
                deepstack_input_embeds
            ):
                deepstack_embed = deepstack_input_embeds[
                    f"deepstack_input_embeds_{layer_idx}"
                ]
                deepstack_embed = (
                    deepstack_embed.unsqueeze(-2)
                    .expand(
                        *deepstack_embed.shape[:-1],
                        self.config.hc_count,
                        self.config.hidden_size,
                    )
                    .flatten(-2)
                )
                hidden_states = layer.mlp_hyper_connection.combine(
                    hidden_states, block_output, injection
                )
                block_output = None
                injection = None
                hidden_states = hidden_states + deepstack_embed

        if not get_pp_group().is_last_rank:
            if last_layer is not None and block_output is not None:
                hidden_states = last_layer.mlp_hyper_connection.combine(
                    hidden_states, block_output, injection
                )
            return model.IntermediateTensors({"hidden_states": hidden_states})

        final_mixer = self.hyper_connection_mixer
        assert final_mixer is not None
        multi_hidden, sample_hidden_states, _ = final_mixer.combine_and_mix(
            hidden_states, block_output, injection
        )
        if self._mtp_hidden_buffer is not None:
            torch.ops.vllm.qwen4_exp_amd_mtp_hidden_copy(
                multi_hidden,
                self._mtp_hidden_buffer,
            )
        return sample_hidden_states

    setattr(hcu_forward, _WRAPPER, True)
    setattr(model_class, "_vllm_hcu_original_forward", forward)
    setattr(model_class, "forward", hcu_forward)
    setattr(model, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
