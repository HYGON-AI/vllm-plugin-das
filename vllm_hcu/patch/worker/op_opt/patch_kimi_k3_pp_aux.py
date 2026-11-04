# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Port Kimi's upstream auxiliary-state PP protocol to the AMD decoder."""
from functools import wraps
from types import ModuleType

from ._common import load_exact_module

TARGET_MODULE = "vllm.models.kimi_k3.amd.linear"
PATCH_ID = "worker.op_opt.kimi_k3.pp_aux"
TARGETS = (f"{TARGET_MODULE}.KimiLinearModel",)
_MARKER = "_hcu_kimi_pp_aux"


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    if getattr(target, _MARKER, False):
        return False
    from vllm.model_executor.models.utils import spec_decode_needs_target_embed

    torch = target.torch
    IntermediateTensors = target.IntermediateTensors
    get_pp_group = target.get_pp_group
    cdiv = target.cdiv
    _apply_attn_res = target._apply_attn_res
    cls = target.KimiLinearModel
    original_init = cls.__init__
    original_forward = cls.forward

    @wraps(original_init)
    def init(self, *, vllm_config, prefix=""):
        original_init(self, vllm_config=vllm_config, prefix=prefix)
        if (get_pp_group().is_last_rank
                and spec_decode_needs_target_embed(vllm_config)
                and isinstance(self.embed_tokens, target.PPMissingLayer)):
            self.embed_tokens = target.VocabParallelEmbedding(
                self.config.vocab_size, self.config.hidden_size,
                prefix=f"{prefix}.embed_tokens",
            )

    def pack_aux_hidden_states(self, intermediate_tensors, aux_hidden_states):
        # Preserve every incoming slot, even when this stage has no local taps.
        # Keep the global numbering and tensor references; the runner's relay
        # may also forward these same slots without changing their values.
        upstream = {}
        if self._aux_slot_base_cached:
            assert intermediate_tensors is not None
            upstream = {
                f"{self.AUX_HIDDEN_STATE_KEY}{i}": intermediate_tensors[
                    f"{self.AUX_HIDDEN_STATE_KEY}{i}"
                ]
                for i in range(self._aux_slot_base_cached)
            }
        return upstream | self.pack_local_aux_hidden_states(aux_hidden_states)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        remote_aux = self.collect_remote_aux_hidden_states(intermediate_tensors)
        aux_hidden_states = []
        if get_pp_group().is_first_rank:
            self._maybe_add_hidden_state(
                aux_hidden_states, self.start_layer, hidden_states, residual
            )

        if self.config.attn_res_block_size is None:
            for layer_idx, layer in enumerate(
                self.layers[self.start_layer : self.end_layer],
                start=self.start_layer,
            ):
                hidden_states, residual = layer(
                    positions=positions,
                    hidden_states=hidden_states,
                    residual=residual,
                )
                self._maybe_add_hidden_state(
                    aux_hidden_states, layer_idx + 1, hidden_states, residual
                )

            if not get_pp_group().is_last_rank:
                return IntermediateTensors(
                    {"hidden_states": hidden_states, "residual": residual,
                     **pack_aux_hidden_states(
                         self, intermediate_tensors, aux_hidden_states)}
                )

            # NOTE: the final norm is applied in compute_logits instead of here,
            # so the MTP draft model receives the pre-norm hidden states.
            if residual is not None:
                hidden_states = hidden_states + residual
            aux_hidden_states = remote_aux + aux_hidden_states
            if aux_hidden_states:
                return hidden_states, aux_hidden_states
            return hidden_states

        attn_res_block_num = cdiv(self.end_layer, self.config.attn_res_block_size)
        block_residual = hidden_states.new_empty(
            hidden_states.size(0), attn_res_block_num, hidden_states.size(1)
        )
        if residual is not None:
            block_residual[:, : residual.size(1), :].copy_(residual)
        residual = block_residual

        for layer_idx, layer in enumerate(
            self.layers[self.start_layer : self.end_layer],
            start=self.start_layer,
        ):
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )
            if (layer_idx + 1) in self.aux_hidden_state_layers:
                # AMD attn-res layer already returns prefix_sum + MLP delta as
                # hidden_states; the override drops the block bank in residual.
                self._maybe_add_hidden_state(
                    aux_hidden_states, layer_idx + 1, hidden_states, residual
                )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual,
                 **pack_aux_hidden_states(
                     self, intermediate_tensors, aux_hidden_states)}
            )

        hidden_states = _apply_attn_res(
            hidden_states,
            residual,
            self.output_attn_res_proj,
            self.output_attn_res_norm,
            attn_res_block_num,
        )
        # NOTE: the final norm is applied in compute_logits instead of here, so
        # the MTP draft model receives the pre-norm hidden states.
        aux_hidden_states = remote_aux + aux_hidden_states
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states

    @wraps(original_forward)
    def dispatch(self, *args, **kwargs):
        if get_pp_group().world_size < 2 or not self.aux_hidden_state_layers:
            return original_forward(self, *args, **kwargs)
        return forward(self, *args, **kwargs)

    cls.__init__ = init
    cls.forward = dispatch
    cls.supports_aux_hidden_states_over_pp = True
    setattr(target, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))
