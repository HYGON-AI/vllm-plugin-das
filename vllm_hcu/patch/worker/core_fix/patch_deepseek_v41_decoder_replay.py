# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Enable decoder-side SWA replay in the pinned AMD DeepSeek V4.1 model."""

from __future__ import annotations

import functools
from itertools import islice
from types import ModuleType
from typing import cast

import torch

from vllm.logger import init_logger
from vllm_hcu.patch.worker.core_fix._common import (
    PatchCompatibilityError,
    load_exact_module,
    require_callable,
    require_class,
    require_exact_signature,
)

from .decoder_replay import DecoderReplayLayers

TARGET_MODULE = "vllm.models.deepseek_v41.amd.model"
PATCH_ID = "worker.core_fix.deepseek_v41.decoder_replay"
TARGETS = (
    f"{TARGET_MODULE}.DeepseekV4DecoderLayer.write_kv",
    f"{TARGET_MODULE}.DeepseekV4Model.__init__",
    f"{TARGET_MODULE}.DeepseekV4Model.forward",
    f"{TARGET_MODULE}.DeepseekV41LLMForCausalLM.decoder_replay_layers",
)
_MARKER = "_vllm_hcu_dsv41_decoder_replay_applied"
_WRAPPER = "_vllm_hcu_dsv41_decoder_replay_wrapper"

logger = init_logger(__name__)


def apply_to_module(module: ModuleType) -> bool:
    mod = load_exact_module(TARGET_MODULE, module)
    model_cls = require_class(
        mod, "DeepseekV4Model", f"{TARGET_MODULE}.DeepseekV4Model"
    )
    layer_cls = require_class(
        mod, "DeepseekV4DecoderLayer", f"{TARGET_MODULE}.DeepseekV4DecoderLayer"
    )
    llm_cls = require_class(
        mod,
        "DeepseekV41LLMForCausalLM",
        f"{TARGET_MODULE}.DeepseekV41LLMForCausalLM",
    )
    if getattr(model_cls, _MARKER, False):
        if not getattr(vars(model_cls).get("__init__"), _WRAPPER, False):
            raise PatchCompatibilityError(f"stale marker for {PATCH_ID}")
        return False

    original_model_init = require_callable(model_cls, "__init__", TARGETS[1])
    require_exact_signature(
        original_model_init,
        TARGETS[1],
        positional=("self",),
        keyword_only=("vllm_config", "prefix"),
        defaults={"prefix": ""},
    )
    original_model_forward = require_callable(
        model_cls, "forward", f"{TARGET_MODULE}.DeepseekV4Model.forward"
    )
    import inspect as _inspect

    fwd_sig = _inspect.signature(original_model_forward)
    expected = (
        "self",
        "input_ids",
        "positions",
        "intermediate_tensors",
        "inputs_embeds",
        "lookback_token_ids",
    )
    if tuple(fwd_sig.parameters) != expected:
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGET_MODULE}.DeepseekV4Model.forward "
            f"has incompatible signature {fwd_sig}"
        )

    def write_kv(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None,
        pre_mix: torch.Tensor,
        post_mix: torch.Tensor,
        res_mix: torch.Tensor,
        residual: torch.Tensor,
    ) -> None:
        """Write the KV ``forward`` would, skipping the rest of the layer.

        Installed 0.29 AMD layers always run ``attn_norm`` separately (no
        ``fuse_seam_norm`` / ``norm_weight`` path).
        """
        assert self.engram is None and not self.use_sequence_parallel
        residual, post_mix, res_mix, x, _attn_pre = self.mhc_pre_delayed(
            residual,
            self.hc_attn_fn,
            self.hc_attn_scale,
            self.hc_attn_base,
            self.rms_norm_eps,
            self.hc_eps,
            self.hc_eps,
            self.hc_post_alpha,
            self.hc_sinkhorn_iters,
            pre_mix=pre_mix,
            sublayer_out=x,
            post_layer_mix=post_mix,
            comb_res_mix=res_mix,
        )
        x = self.attn_norm(x)
        self.attn.forward_kv(positions, x)

    def _decoder_replay_supported(self, vllm_config, cut: int) -> bool:
        parallel_config = vllm_config.parallel_config
        spec_config = vllm_config.speculative_config
        draft_config = spec_config.draft_model_config if spec_config else None
        draft_hf_config = getattr(draft_config, "hf_config", None)
        draft_window = getattr(draft_hf_config, "sliding_window", None)
        draft_layer_types = getattr(draft_hf_config, "layer_types", None) or ()
        window = self.config.sliding_window
        if self.start_layer > cut or self.end_layer < self.config.num_hidden_layers:
            reason = (
                "the pipeline stage holding the last KV source layer must also "
                "hold every layer after it"
            )
        elif self.use_sequence_parallel or getattr(
            parallel_config, "use_ubatching", False
        ):
            # PCP is supported with APC (#62384 / sglang CP late-tail gather):
            # each rank keeps its intersection with the global trailing window.
            reason = (
                "the replay-layer batch shrinks per rank, which sequence "
                "parallelism and microbatching cannot follow"
            )
        elif any(i >= cut for i in getattr(self.config, "engram_layer_ids", ())):
            reason = "an Engram layer sits at or after the last KV source layer"
        elif draft_config is not None and (
            draft_window is None
            or draft_window > window
            or any(t != "sliding_attention" for t in draft_layer_types)
        ):
            reason = (
                f"the drafter (sliding window {draft_window}) reads hidden states "
                f"outside the target's {window}-token window"
            )
        else:
            return True
        logger.warning_once("Decoder SWA bounded replay is off: %s.", reason)
        return False

    @functools.wraps(original_model_init)
    def model_init(self, *, vllm_config, prefix=""):
        original_model_init(self, vllm_config=vllm_config, prefix=prefix)
        self.decoder_replay_layers = None
        self.decoder_replay_start = self.end_layer
        config = self.config
        cut = max(config.kv_source_layer_ids)
        enabled = bool(
            getattr(vllm_config.cache_config, "swa_bounded_replay", True)
        )
        use_mrv2 = bool(getattr(vllm_config, "use_v2_model_runner", False))
        has_tail = cut < self.end_layer - 1
        # PP stages that do not own ``cut`` hold ``PPMissingLayer`` there.
        # Gate before any ``self.layers[cut]`` attribute access so disabling
        # bounded replay (or running on a non-owning stage) cannot AttributeError.
        owns_cut = self.start_layer <= cut < self.end_layer
        swa_bounded = False
        if owns_cut:
            cut_attn = getattr(self.layers[cut], "attn", None)
            swa_cache = getattr(cut_attn, "swa_cache_layer", None)
            swa_bounded = bool(getattr(swa_cache, "bounded_replay", False))
        if not owns_cut:
            if enabled:
                logger.warning_once(
                    "Decoder SWA bounded replay is off on this PP stage: "
                    "cut=%s start_layer=%s end_layer=%s.",
                    cut,
                    self.start_layer,
                    self.end_layer,
                )
        elif not (enabled and swa_bounded and use_mrv2 and has_tail):
            logger.warning(
                "Decoder SWA bounded replay is off: enabled=%s "
                "swa_layer.bounded_replay=%s use_v2_model_runner=%s "
                "cut=%s end_layer=%s (need cut < end_layer-1).",
                enabled,
                swa_bounded,
                use_mrv2,
                cut,
                self.end_layer,
            )
        elif not _decoder_replay_supported(self, vllm_config, cut):
            pass  # warning_once already emitted inside the helper
        else:
            assert self.layers[cut].attn.compress_ratio <= 1
            self.decoder_replay_start = cut
            metadata_prefixes: set[str] = set()
            for layer in islice(self.layers, cut, self.end_layer):
                attn = cast(layer_cls, layer).attn
                metadata_prefixes.add(attn.swa_cache_layer.prefix)
                if getattr(attn, "compressed_cache_prefix", None) is not None:
                    metadata_prefixes.add(attn.compressed_cache_prefix)
                if getattr(attn, "indexer", None) is not None:
                    metadata_prefixes.add(attn.indexer.k_cache.prefix)
            self.decoder_replay_layers = DecoderReplayLayers(
                config.sliding_window,
                self._run_replay_layers,
                self.layers[cut].write_kv,
                metadata_prefixes,
                self.layers[cut].attn.swa_cache_layer.prefix,
            )
            logger.warning(
                "Decoder SWA bounded replay ON: in eager prefill steps, layer %d "
                "writes its KV for every token, and the rest of it and layers "
                "%d-%d run on each request's last %d tokens only.",
                cut,
                cut + 1,
                self.end_layer - 1,
                config.sliding_window,
            )

    def _run_layers(
        self,
        layer_ids,
        hidden_states,
        positions,
        input_ids,
        pre_mix,
        post_mix,
        res_mix,
        residual,
        engram_hashes,
        engram_mask,
        full_num_tokens,
    ):
        from vllm.models.common.ops.sequence_parallel import sp_all_gather

        aux_hidden_states: list[torch.Tensor] = []
        final_aux_recon: torch.Tensor | None = None
        ran_layers = False
        for idx in layer_ids:
            ran_layers = True
            hidden_states, residual, post_mix, res_mix, pre_mix = self.layers[idx](
                hidden_states,
                positions,
                input_ids,
                pre_mix,
                post_mix,
                res_mix,
                residual,
                engram_hashes,
                engram_mask,
            )
            if idx + 1 in self.aux_hidden_state_layers and not (
                self.decoder_replay_layers is not None
                and idx + 1 == self.decoder_replay_start
            ):
                aux_recon = self.mhc_post(hidden_states, residual, post_mix, res_mix)
                aux_hidden_state = aux_recon.mean(dim=1)
                if self.use_sequence_parallel:
                    aux_hidden_state = sp_all_gather(aux_hidden_state)[:full_num_tokens]
                aux_hidden_states.append(aux_hidden_state)
                final_aux_recon = aux_recon
        return (
            hidden_states,
            residual,
            post_mix,
            res_mix,
            pre_mix,
            aux_hidden_states,
            final_aux_recon,
            ran_layers,
        )

    def _run_replay_layers(
        self,
        hidden_states,
        positions,
        input_ids,
        pre_mix,
        post_mix,
        res_mix,
        residual,
    ):
        cut_aux: list[torch.Tensor] = []
        if self.decoder_replay_start in self.aux_hidden_state_layers:
            cut_aux.append(
                self.mhc_post(hidden_states, residual, post_mix, res_mix).mean(dim=1)
            )
        (
            hidden_states,
            residual,
            post_mix,
            res_mix,
            pre_mix,
            aux,
            final_aux_recon,
            ran_layers,
        ) = self._run_layers(
            range(self.decoder_replay_start, self.end_layer),
            hidden_states,
            positions,
            input_ids,
            pre_mix,
            post_mix,
            res_mix,
            residual,
            None,
            None,
            positions.shape[0],
        )
        assert residual is not None and post_mix is not None
        assert res_mix is not None and pre_mix is not None
        if (
            self.end_layer in self.aux_hidden_state_layers
            and final_aux_recon is not None
        ):
            hidden_states = final_aux_recon
        elif ran_layers:
            hidden_states = self.mhc_post(hidden_states, residual, post_mix, res_mix)
        return (hidden_states, pre_mix, *cut_aux, *aux)

    @functools.wraps(original_model_forward)
    def model_forward(
        self,
        input_ids,
        positions,
        intermediate_tensors,
        inputs_embeds=None,
        lookback_token_ids=None,
    ):
        if self.decoder_replay_layers is None:
            return original_model_forward(
                self,
                input_ids,
                positions,
                intermediate_tensors,
                inputs_embeds,
                lookback_token_ids,
            )

        # Mirror the installed forward, but split at decoder_replay_start.
        import typing as _typing
        from itertools import islice as _islice

        import vllm.envs as envs
        from vllm.distributed import get_pp_group
        from vllm.forward_context import (
            get_forward_context,
            is_forward_context_available,
        )
        from vllm.models.common.ops.sequence_parallel import (
            sp_all_gather,
            sp_padding_mask,
            sp_shard,
        )
        from vllm.models.deepseek_v41.common.mm_preprocess import image_sentinel_mask
        from vllm.sequence import IntermediateTensors
        from vllm.v1.worker.ubatching import dbo_current_ubatch_id

        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]

        if self.use_mega_moe:
            input_ids = input_ids.to(torch.int64)

        engram_hashes: torch.Tensor | None = None
        engram_mask: torch.Tensor | None = None
        if (
            self.engram_hash is not None
            and input_ids is not None
            and is_forward_context_available()
        ):
            attn_metadata = get_forward_context().attn_metadata
            if isinstance(attn_metadata, list):
                attn_metadata = attn_metadata[dbo_current_ubatch_id()]
            if isinstance(attn_metadata, dict) and self.engram_hash.ensure_cache():
                assert self.engram_swa_prefix is not None
                swa_metadata = _typing.cast(
                    "DeepseekSparseSWAMetadata",
                    attn_metadata[self.engram_swa_prefix],
                )
                image_mask = image_sentinel_mask(input_ids)
                engram_mask = ~image_mask
                if lookback_token_ids is None:
                    if not self.engram_hash.use_slot_cache:
                        raise NotImplementedError(
                            "engram needs `lookback_token_ids` from the model "
                            "runner (the DBO/ubatch wrapper drops model kwargs)"
                        )
                    num_reqs = swa_metadata.num_decodes + swa_metadata.num_prefills
                    lookback_token_ids = input_ids.new_full(
                        (num_reqs, self.engram_hash.lookback_depth), -1
                    )
                engram_hashes = self.engram_hash(
                    input_ids,
                    positions,
                    swa_metadata.query_start_loc,
                    image_mask,
                    lookback_token_ids,
                    image_sentinel_mask(lookback_token_ids),
                    swa_metadata.slot_mapping,
                    swa_metadata.block_table,
                )
                for layer in _islice(self.layers, self.start_layer, self.end_layer):
                    engram = getattr(layer, "engram", None)
                    if engram is not None:
                        engram.prepare_embeddings(
                            engram_hashes[:, engram.layer_hash_index]
                        )

        full_num_tokens = positions.shape[0]
        if self.use_sequence_parallel:
            if envs.VLLM_MOE_SKIP_PADDING and is_forward_context_available():
                forward_context = get_forward_context()
                forward_context.is_padding = sp_padding_mask(
                    forward_context.is_padding, hidden_states
                )
            hidden_states = sp_shard(hidden_states)
            input_ids = sp_shard(input_ids)

        residual, post_mix, res_mix = None, None, None
        pre_mix: torch.Tensor | None = None
        if not get_pp_group().is_first_rank:
            assert intermediate_tensors is not None
            pre_mix = intermediate_tensors["pre_mix"]
        (
            hidden_states,
            residual,
            post_mix,
            res_mix,
            pre_mix,
            aux_hidden_states,
            final_aux_recon,
            ran_layers,
        ) = self._run_layers(
            range(self.start_layer, self.decoder_replay_start),
            hidden_states,
            positions,
            input_ids,
            pre_mix,
            post_mix,
            res_mix,
            residual,
            engram_hashes,
            engram_mask,
            full_num_tokens,
        )
        collapsed = False
        if self.decoder_replay_layers is not None:
            hidden_states, pre_mix, *late_aux = self.decoder_replay_layers(
                hidden_states,
                positions,
                input_ids,
                pre_mix,
                post_mix,
                res_mix,
                residual,
            )
            aux_hidden_states.extend(late_aux)
            final_aux_recon = None
            collapsed = True
        if ran_layers and not collapsed:
            if (
                self.end_layer in self.aux_hidden_state_layers
                and final_aux_recon is not None
            ):
                hidden_states = final_aux_recon
            else:
                hidden_states = self.mhc_post(
                    hidden_states, residual, post_mix, res_mix
                )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "pre_mix": pre_mix}
            )

        if self.use_sequence_parallel:
            hidden_states = sp_all_gather(hidden_states)[:full_num_tokens]
            pre_mix = sp_all_gather(pre_mix)[:full_num_tokens]

        if self._mtp_hidden_buffer is not None:
            num_tokens = hidden_states.shape[0]
            self._mtp_hidden_buffer[:num_tokens].copy_(hidden_states.flatten(1))

        assert pre_mix is not None
        hidden_states = torch.ops.vllm.hc_collapse_triton(hidden_states, pre_mix)
        hidden_states = self.norm(hidden_states)
        if len(aux_hidden_states) > 0:
            return hidden_states, aux_hidden_states
        return hidden_states

    model_cls._run_layers = _run_layers
    model_cls._run_replay_layers = _run_replay_layers
    model_cls._decoder_replay_supported = _decoder_replay_supported
    model_cls.forward = model_forward
    layer_cls.write_kv = write_kv
    model_cls.__init__ = model_init
    setattr(model_cls, _MARKER, True)
    setattr(model_cls.__init__, _WRAPPER, True)

    # Expose through the LLM wrapper so ModelState can find it.
    if not isinstance(getattr(llm_cls, "decoder_replay_layers", None), property):

        @property
        def decoder_replay_layers(self):
            return self.model.decoder_replay_layers

        llm_cls.decoder_replay_layers = decoder_replay_layers

    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGETS", "TARGET_MODULE", "apply", "apply_to_module"]
