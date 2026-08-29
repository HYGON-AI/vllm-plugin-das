# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Expose the EAGLE3 auxiliary-state interface for AMD DeepSeek-V4 DSpark."""

from __future__ import annotations

import functools
from types import ModuleType

from ._common import (
    PatchCompatibilityError,
    load_exact_module,
    require_callable,
    require_class,
    require_exact_signature,
)

TARGET_MODULE = "vllm.models.deepseek_v4.amd.model"
PATCH_ID = "worker.core_fix.deepseek_v4_amd.dspark_target_interface"
TARGET_SYMBOL = f"{TARGET_MODULE}.DeepseekV4ForCausalLM"
_MARKER = "_vllm_hcu_dspark_target_interface_applied"
_WRAPPER_MARKER = "_vllm_hcu_dspark_target_forward_wrapper"


def apply_to_module(module: ModuleType) -> bool:
    amd_model = load_exact_module(TARGET_MODULE, module)
    model_cls = require_class(amd_model, "DeepseekV4Model", TARGET_SYMBOL)
    causal_cls = require_class(amd_model, "DeepseekV4ForCausalLM", TARGET_SYMBOL)
    original_forward = require_callable(
        model_cls, "forward", f"{TARGET_MODULE}.DeepseekV4Model.forward"
    )
    if getattr(causal_cls, _MARKER, False):
        current = vars(model_cls).get("forward")
        if not getattr(current, _WRAPPER_MARKER, False):
            raise PatchCompatibilityError(
                f"required HCU patch marker for {TARGET_SYMBOL} is stale"
            )
        return False
    require_exact_signature(
        original_forward,
        f"{TARGET_MODULE}.DeepseekV4Model.forward",
        positional=(
            "self",
            "input_ids",
            "positions",
            "intermediate_tensors",
            "inputs_embeds",
        ),
        defaults={"inputs_embeds": None},
    )

    @functools.wraps(original_forward)
    def hcu_dspark_target_forward(
        self, input_ids, positions, intermediate_tensors, inputs_embeds=None
    ):
        # Keep non-speculative DeepSeek-V4 on the target vLLM implementation
        # byte-for-byte.  DSpark enables the extended path by requesting its
        # auxiliary checkpoint layers through set_aux_hidden_state_layers().
        if not self.aux_hidden_state_layers:
            return original_forward(
                self,
                input_ids,
                positions,
                intermediate_tensors,
                inputs_embeds,
            )
        if amd_model.get_pp_group().is_first_rank:
            hidden_states = (
                inputs_embeds
                if inputs_embeds is not None
                else self.embed_input_ids(input_ids)
            )
            hidden_states = hidden_states.unsqueeze(-2).repeat(1, self.hc_mult, 1)
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]

        residual, post_mix, res_mix = None, None, None
        aux_hidden_states = []
        final_aux_recon = None
        layer = None
        for idx, layer in enumerate(
            amd_model.islice(self.layers, self.start_layer, self.end_layer),
            start=self.start_layer,
        ):
            hidden_states, residual, post_mix, res_mix = layer(
                hidden_states,
                positions,
                input_ids,
                post_mix,
                res_mix,
                residual,
            )
            if idx + 1 in self.aux_hidden_state_layers:
                aux_recon = (
                    layer.hc_post(hidden_states, residual, post_mix, res_mix)
                    if layer.use_fused_mhc
                    else hidden_states
                )
                aux_hidden_states.append(aux_recon.mean(dim=1))
                final_aux_recon = aux_recon

        if layer is not None and layer.use_fused_mhc:
            hidden_states = (
                final_aux_recon
                if self.end_layer in self.aux_hidden_state_layers
                else layer.hc_post(hidden_states, residual, post_mix, res_mix)
            )

        if not amd_model.get_pp_group().is_last_rank:
            return amd_model.IntermediateTensors({"hidden_states": hidden_states})

        num_tokens = hidden_states.shape[0]
        self._mtp_hidden_buffer[:num_tokens].copy_(hidden_states.flatten(1))
        hidden_states = self.hc_head_op(
            hidden_states,
            self.hc_head_fn,
            self.hc_head_scale,
            self.hc_head_base,
            self.rms_norm_eps,
            self.hc_eps,
        )
        hidden_states = self.norm(hidden_states)
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        self.model.aux_hidden_state_layers = layers

    def get_eagle3_default_aux_hidden_state_layers(self) -> tuple[int, ...]:
        return ()

    setattr(hcu_dspark_target_forward, _WRAPPER_MARKER, True)
    setattr(model_cls, "aux_hidden_state_layers", ())
    setattr(model_cls, "_vllm_hcu_original_dspark_target_forward", original_forward)
    setattr(model_cls, "forward", hcu_dspark_target_forward)
    # SupportsEagle3 inherits these two structural protocol attributes from
    # SupportsEagleBase.  The AMD class does not inherit that protocol, so
    # publish the default values explicitly along with the DSpark methods.
    setattr(causal_cls, "has_own_lm_head", False)
    setattr(causal_cls, "has_own_embed_tokens", False)
    setattr(causal_cls, "supports_eagle3", True)
    setattr(causal_cls, "set_aux_hidden_state_layers", set_aux_hidden_state_layers)
    setattr(
        causal_cls,
        "get_eagle3_default_aux_hidden_state_layers",
        get_eagle3_default_aux_hidden_state_layers,
    )
    setattr(causal_cls, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "apply", "apply_to_module"]
