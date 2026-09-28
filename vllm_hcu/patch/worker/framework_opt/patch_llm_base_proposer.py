# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Atomic Lightly-CP/custom-SP/multi-layer-MTP proposer migration."""

from __future__ import annotations

import functools
from types import ModuleType

from ._common import (
    PatchCompatibilityError,
    already_applied,
    load_exact_module,
    require_callable,
    require_class,
    require_exact_signature,
)

TARGET_MODULE = "vllm.v1.spec_decode.llm_base_proposer"
PATCH_ID = "worker.framework_opt.spec_decode.hcu_proposer"
TARGETS = (
    f"{TARGET_MODULE}.SpecDecodeBaseProposer.__init__",
    f"{TARGET_MODULE}.SpecDecodeBaseProposer.propose",
    f"{TARGET_MODULE}.SpecDecodeBaseProposer.prepare_inputs_padded",
    f"{TARGET_MODULE}.SpecDecodeBaseProposer._maybe_share_lm_head",
    f"{TARGET_MODULE}.SpecDecodeBaseProposer._pad_for_sequence_parallelism",
    f"{TARGET_MODULE}.SpecDecodeBaseProposer._determine_batch_execution_and_padding",
)
_MARKER = "_vllm_hcu_base_proposer_applied"
_WRAPPER = "_vllm_hcu_base_proposer_wrapper"


def apply_to_module(module: ModuleType) -> bool:
    proposer_module = load_exact_module(TARGET_MODULE, module)
    proposer_class = require_class(
        proposer_module,
        "SpecDecodeBaseProposer",
        f"{TARGET_MODULE}.SpecDecodeBaseProposer",
    )
    wrapped = (
        (proposer_class, "__init__", TARGETS[0], _WRAPPER),
        (proposer_class, "propose", TARGETS[1], _WRAPPER),
        (proposer_class, "prepare_inputs_padded", TARGETS[2], _WRAPPER),
        (proposer_class, "_maybe_share_lm_head", TARGETS[3], _WRAPPER),
        (
            proposer_class,
            "_determine_batch_execution_and_padding",
            TARGETS[5],
            _WRAPPER,
        ),
    )
    if already_applied(proposer_module, _MARKER, wrapped):
        padding = require_callable(
            proposer_class, "_pad_for_sequence_parallelism", TARGETS[4]
        )
        if not getattr(padding, _WRAPPER, False):
            raise PatchCompatibilityError(
                f"required HCU patch marker for {TARGETS[4]} is stale"
            )
        return False
    if "_pad_for_sequence_parallelism" in vars(proposer_class):
        raise PatchCompatibilityError(
            f"audited target vLLM API {TARGETS[4]} unexpectedly already exists"
        )

    original_init = require_callable(proposer_class, "__init__", TARGETS[0])
    require_exact_signature(
        original_init,
        TARGETS[0],
        positional=(
            "self",
            "vllm_config",
            "device",
            "pass_hidden_states_to_model",
            "runner",
        ),
        defaults={"runner": None},
    )
    original_propose = require_callable(proposer_class, "propose", TARGETS[1])
    require_exact_signature(
        original_propose,
        TARGETS[1],
        positional=(
            "self",
            "num_speculative_tokens",
            "target_token_ids",
            "target_positions",
            "target_hidden_states",
            "next_token_ids",
            "token_indices_to_sample",
            "common_attn_metadata",
            "sampling_metadata",
            "mm_embed_inputs",
            "num_rejected_tokens_gpu",
            "slot_mappings",
        ),
        defaults={
            "mm_embed_inputs": None,
            "num_rejected_tokens_gpu": None,
            "slot_mappings": None,
        },
    )
    original_prepare = require_callable(
        proposer_class, "prepare_inputs_padded", TARGETS[2]
    )
    require_exact_signature(
        original_prepare,
        TARGETS[2],
        positional=(
            "self",
            "common_attn_metadata",
            "spec_decode_metadata",
            "valid_sampled_tokens_count",
        ),
    )
    original_share = require_callable(
        proposer_class, "_maybe_share_lm_head", TARGETS[3]
    )
    require_exact_signature(
        original_share,
        TARGETS[3],
        positional=("self", "target_language_model"),
    )
    original_determine = require_callable(
        proposer_class, "_determine_batch_execution_and_padding", TARGETS[5]
    )
    require_exact_signature(
        original_determine,
        TARGETS[5],
        positional=("self", "num_tokens", "use_cudagraphs"),
        defaults={"use_cudagraphs": True},
    )

    from vllm_hcu.v1.spec_decode import proposer_runtime

    @functools.wraps(original_init)
    def hcu_init(
        self,
        vllm_config,
        device,
        pass_hidden_states_to_model,
        runner=None,
    ):
        # The complete official initializer runs first, including its ROCm
        # attention metadata branch.  The legacy replacement accidentally
        # removed that branch when Lightly-CP was disabled.
        original_init(
            self,
            vllm_config,
            device,
            pass_hidden_states_to_model,
            runner,
        )
        proposer_runtime.initialize_proposer(
            proposer_module, self, vllm_config, device, runner
        )

    @functools.wraps(original_propose)
    def hcu_propose(
        self,
        num_speculative_tokens,
        target_token_ids,
        target_positions,
        target_hidden_states,
        next_token_ids,
        token_indices_to_sample,
        common_attn_metadata,
        sampling_metadata,
        mm_embed_inputs=None,
        num_rejected_tokens_gpu=None,
        slot_mappings=None,
    ):
        config = getattr(self, "_hcu_feature_config", None)
        if config is None or not config.enable_lightly_cp:
            return original_propose(
                self,
                num_speculative_tokens,
                target_token_ids,
                target_positions,
                target_hidden_states,
                next_token_ids,
                token_indices_to_sample,
                common_attn_metadata,
                sampling_metadata,
                mm_embed_inputs,
                num_rejected_tokens_gpu,
                slot_mappings,
            )
        return proposer_runtime.propose(
            proposer_module,
            self,
            num_speculative_tokens,
            target_token_ids,
            target_positions,
            target_hidden_states,
            next_token_ids,
            token_indices_to_sample,
            common_attn_metadata,
            sampling_metadata,
            mm_embed_inputs,
            num_rejected_tokens_gpu,
            slot_mappings,
        )

    @functools.wraps(original_prepare)
    def hcu_prepare_inputs_padded(
        self,
        common_attn_metadata,
        spec_decode_metadata,
        valid_sampled_tokens_count,
    ):
        num_kv_actual_tokens = common_attn_metadata.num_kv_actual_tokens
        result = original_prepare(
            self,
            common_attn_metadata,
            spec_decode_metadata,
            valid_sampled_tokens_count,
        )
        result[0].num_kv_actual_tokens = num_kv_actual_tokens
        return result

    @functools.wraps(original_share)
    def hcu_share_lm_head(self, target_language_model):
        config = getattr(self, "_hcu_feature_config", None)
        if config is None or not config.enable_multi_layers_mtp:
            return original_share(self, target_language_model)
        return proposer_runtime.preserve_multi_layer_mtp_heads(
            self, target_language_model, original_share
        )

    def hcu_pad_for_sequence_parallelism(self, num_scheduled_tokens):
        return proposer_runtime.pad_for_sequence_parallelism(
            self, num_scheduled_tokens
        )

    @functools.wraps(original_determine)
    def hcu_determine(self, num_tokens, use_cudagraphs=True):
        num_tokens = self._pad_for_sequence_parallelism(num_tokens)
        return original_determine(self, num_tokens, use_cudagraphs)

    for function in (
        hcu_init,
        hcu_propose,
        hcu_prepare_inputs_padded,
        hcu_share_lm_head,
        hcu_pad_for_sequence_parallelism,
        hcu_determine,
    ):
        setattr(function, _WRAPPER, True)
    setattr(proposer_class, "_vllm_hcu_original_init", original_init)
    setattr(proposer_class, "_vllm_hcu_original_propose", original_propose)
    setattr(
        proposer_class, "_vllm_hcu_original_prepare_inputs_padded", original_prepare
    )
    setattr(proposer_class, "_vllm_hcu_original_share_lm_head", original_share)
    setattr(
        proposer_class,
        "_vllm_hcu_original_determine_batch_execution_and_padding",
        original_determine,
    )
    setattr(proposer_class, "__init__", hcu_init)
    setattr(proposer_class, "propose", hcu_propose)
    setattr(proposer_class, "prepare_inputs_padded", hcu_prepare_inputs_padded)
    setattr(proposer_class, "_maybe_share_lm_head", hcu_share_lm_head)
    setattr(
        proposer_class,
        "_pad_for_sequence_parallelism",
        hcu_pad_for_sequence_parallelism,
    )
    setattr(
        proposer_class,
        "_determine_batch_execution_and_padding",
        hcu_determine,
    )
    setattr(proposer_module, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
