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
_DECODER_WRAPPER_MARKER = "_vllm_hcu_deepseek_v4_boltops_mhc_wrapper"


def _boltops_mhc_pre(
    backend,
    mhc,
    residual,
    fn,
    hc_scale,
    hc_base,
    rms_eps,
    hc_pre_eps,
    hc_sinkhorn_eps,
    hc_post_mult_value,
    sinkhorn_repeat,
    n_splits=1,
    norm_weight=None,
    norm_eps=0.0,
):
    post_mix, comb_mix, layer_input = backend.mhc_pre(
        residual,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        n_splits,
    )
    return (
        post_mix,
        comb_mix,
        mhc._apply_mhc_norm(layer_input, norm_weight, norm_eps),
    )


def _boltops_mhc_post(
    backend,
    x,
    residual,
    post_layer_mix,
    comb_res_mix,
):
    return backend.mhc_post(x, residual, post_layer_mix, comb_res_mix)


def _boltops_mhc_fused_post_pre(
    backend,
    mhc,
    x,
    residual,
    post_layer_mix,
    comb_res_mix,
    fn,
    hc_scale,
    hc_base,
    rms_eps,
    hc_pre_eps,
    hc_sinkhorn_eps,
    hc_post_mult_value,
    sinkhorn_repeat,
    n_splits=1,
    tile_n=1,
    norm_weight=None,
    norm_eps=0.0,
):
    residual_cur, post_mix, comb_mix, layer_input = backend.mhc_fused_post_pre(
        x,
        residual,
        post_layer_mix,
        comb_res_mix,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        n_splits,
        tile_n,
    )
    return (
        residual_cur,
        post_mix,
        comb_mix,
        mhc._apply_mhc_norm(layer_input, norm_weight, norm_eps),
    )


def _bind_deepseek_v4_boltops_mhc(layer, mhc, backend) -> None:
    """Bind one DeepSeek-V4 target/MTP block to the audited HCU provider."""

    layer.mhc_pre._forward_method = functools.partial(
        _boltops_mhc_pre, backend, mhc
    )
    layer.mhc_post._forward_method = functools.partial(
        _boltops_mhc_post, backend
    )
    layer.mhc_fused_post_pre._forward_method = functools.partial(
        _boltops_mhc_fused_post_pre,
        backend,
        mhc,
    )


def apply_to_module(module: ModuleType) -> bool:
    amd_model = load_exact_module(TARGET_MODULE, module)
    model_cls = require_class(amd_model, "DeepseekV4Model", TARGET_SYMBOL)
    causal_cls = require_class(amd_model, "DeepseekV4ForCausalLM", TARGET_SYMBOL)
    decoder_cls = require_class(
        amd_model,
        "DeepseekV4DecoderLayer",
        f"{TARGET_MODULE}.DeepseekV4DecoderLayer",
    )
    original_forward = require_callable(
        model_cls, "forward", f"{TARGET_MODULE}.DeepseekV4Model.forward"
    )
    original_decoder_init = require_callable(
        decoder_cls,
        "__init__",
        f"{TARGET_MODULE}.DeepseekV4DecoderLayer.__init__",
    )
    if getattr(causal_cls, _MARKER, False):
        current = vars(model_cls).get("forward")
        current_decoder_init = vars(decoder_cls).get("__init__")
        if not (
            getattr(current, _WRAPPER_MARKER, False)
            and getattr(
                current_decoder_init,
                _DECODER_WRAPPER_MARKER,
                False,
            )
        ):
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
    require_exact_signature(
        original_decoder_init,
        f"{TARGET_MODULE}.DeepseekV4DecoderLayer.__init__",
        positional=(
            "self",
            "vllm_config",
            "prefix",
            "topk_indices_buffer",
            "aux_stream_list",
            "fuse_heterogeneous_shared_expert",
        ),
        defaults={
            "topk_indices_buffer": None,
            "aux_stream_list": None,
            "fuse_heterogeneous_shared_expert": False,
        },
    )

    @functools.wraps(original_decoder_init)
    def hcu_decoder_init(
        self,
        vllm_config,
        prefix,
        topk_indices_buffer=None,
        aux_stream_list=None,
        fuse_heterogeneous_shared_expert=False,
    ):
        original_decoder_init(
            self,
            vllm_config,
            prefix,
            topk_indices_buffer,
            aux_stream_list,
            fuse_heterogeneous_shared_expert,
        )
        from vllm.model_executor.layers import mhc
        from vllm_hcu.model_executor.layers import mhc as boltops_mhc

        _bind_deepseek_v4_boltops_mhc(self, mhc, boltops_mhc)

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
    setattr(hcu_decoder_init, _DECODER_WRAPPER_MARKER, True)
    setattr(
        decoder_cls,
        "_vllm_hcu_original_deepseek_v4_decoder_init",
        original_decoder_init,
    )
    setattr(decoder_cls, "__init__", hcu_decoder_init)
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
