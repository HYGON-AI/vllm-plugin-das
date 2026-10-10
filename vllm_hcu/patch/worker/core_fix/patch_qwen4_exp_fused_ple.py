# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Adapt the v0.28.1 Qwen4Exp model to the fused PLE implementation."""

from __future__ import annotations

import functools
from types import ModuleType

from vllm.model_executor.models.utils import WeightsMapper

from ._common import (
    PatchCompatibilityError,
    load_exact_module,
    require_callable,
    require_class,
    require_exact_signature,
)

TARGET_MODULE = "vllm.models.qwen4_exp.amd.model"
PATCH_ID = "worker.core_fix.qwen4_exp.fused_ple_model"
TARGETS = (f"{TARGET_MODULE}.Qwen4ExpDecoderLayer.forward",)
_MARKER = "_vllm_hcu_qwen4_exp_fused_ple_applied"
_FORWARD_WRAPPER = "_vllm_hcu_qwen4_exp_fused_ple_forward"
_ORIGINAL_FORWARD = "_vllm_hcu_original_qwen4_exp_decoder_forward"
_ORIGINAL_MAPPER = "_vllm_hcu_original_qwen4_exp_mapper"
_ORIGINAL_LLM_PACKED = "_vllm_hcu_original_qwen4_exp_llm_packed"
_ORIGINAL_VL_PACKED = "_vllm_hcu_original_qwen4_exp_vl_packed"
_KV_PACKED = ["key_proj", "value_proj"]
_PLE_WEIGHTS_MAPPER = WeightsMapper(
    orig_to_new_stacked={
        "ple.key_proj": ("ple.kv_proj", 0),
        "ple.value_proj": ("ple.kv_proj", 1),
    }
)


def _require_mapping(owner: type, name: str, target: str):
    value = getattr(owner, name, None)
    if not isinstance(value, dict):
        raise PatchCompatibilityError(
            f"required HCU patch target {target} must be a dict"
        )
    return value


def _require_mapper(owner: type, name: str, target: str) -> WeightsMapper:
    value = getattr(owner, name, None)
    if not isinstance(value, WeightsMapper):
        raise PatchCompatibilityError(
            f"required HCU patch target {target} must be a WeightsMapper"
        )
    return value


def _has_fused_mappings(
    model_class: type,
    llm_class: type,
    vl_class: type,
) -> bool:
    mapper = _require_mapper(
        model_class,
        "hf_to_vllm_mapper",
        f"{TARGET_MODULE}.Qwen4ExpModel.hf_to_vllm_mapper",
    )
    stacked = mapper.orig_to_new_stacked
    return bool(
        stacked.get("ple.key_proj") == ("ple.kv_proj", 0)
        and stacked.get("ple.value_proj") == ("ple.kv_proj", 1)
        and _require_mapping(
            llm_class,
            "packed_modules_mapping",
            f"{TARGET_MODULE}.Qwen4ExpForCausalLM.packed_modules_mapping",
        ).get("kv_proj")
        == _KV_PACKED
        and _require_mapping(
            vl_class,
            "packed_modules_mapping",
            f"{TARGET_MODULE}.Qwen4ExpForConditionalGeneration.packed_modules_mapping",
        ).get("kv_proj")
        == _KV_PACKED
    )


def apply_to_module(module: ModuleType) -> bool:
    owner = load_exact_module(TARGET_MODULE, module)
    decoder_class = require_class(
        owner,
        "Qwen4ExpDecoderLayer",
        f"{TARGET_MODULE}.Qwen4ExpDecoderLayer",
    )
    model_class = require_class(
        owner,
        "Qwen4ExpModel",
        f"{TARGET_MODULE}.Qwen4ExpModel",
    )
    llm_class = require_class(
        owner,
        "Qwen4ExpForCausalLM",
        f"{TARGET_MODULE}.Qwen4ExpForCausalLM",
    )
    vl_class = require_class(
        owner,
        "Qwen4ExpForConditionalGeneration",
        f"{TARGET_MODULE}.Qwen4ExpForConditionalGeneration",
    )
    current_forward = require_callable(
        decoder_class,
        "forward",
        TARGETS[0],
    )

    if getattr(owner, _MARKER, False):
        if not getattr(current_forward, _FORWARD_WRAPPER, False):
            raise PatchCompatibilityError(
                f"required HCU patch marker for {TARGETS[0]} is stale"
            )
        if not _has_fused_mappings(model_class, llm_class, vl_class):
            raise PatchCompatibilityError(
                "required HCU fused PLE mapping marker is stale"
            )
        return False
    if getattr(current_forward, _FORWARD_WRAPPER, False):
        raise PatchCompatibilityError(
            "refusing a partial Qwen4Exp fused PLE patch state"
        )

    require_exact_signature(
        current_forward,
        TARGETS[0],
        positional=(
            "self",
            "hidden_states",
            "prev_block_output",
            "prev_injection",
            "positions",
        ),
        keyword_only=("input_ids", "query_start_loc", "ngram_context"),
    )
    original_mapper = _require_mapper(
        model_class,
        "hf_to_vllm_mapper",
        f"{TARGET_MODULE}.Qwen4ExpModel.hf_to_vllm_mapper",
    )
    original_llm_packed = _require_mapping(
        llm_class,
        "packed_modules_mapping",
        f"{TARGET_MODULE}.Qwen4ExpForCausalLM.packed_modules_mapping",
    )
    original_vl_packed = _require_mapping(
        vl_class,
        "packed_modules_mapping",
        f"{TARGET_MODULE}.Qwen4ExpForConditionalGeneration.packed_modules_mapping",
    )

    @functools.wraps(current_forward)
    def hcu_decoder_forward(
        self,
        hidden_states,
        prev_block_output,
        prev_injection,
        positions,
        *,
        input_ids,
        query_start_loc,
        ngram_context,
    ):
        attn_hc = self.attn_hyper_connection
        if self.ple is not None:
            if prev_block_output is not None and prev_injection is not None:
                hidden_states = attn_hc.combine(
                    hidden_states,
                    prev_block_output,
                    prev_injection,
                )
                prev_block_output = prev_injection = None

            if input_ids is None or query_start_loc is None or ngram_context is None:
                raise RuntimeError("PLE inputs were not prepared")
            # Fused PLE owns the residual addition and returns the materialized
            # multi-stream state. Adding hidden_states again would double it.
            hidden_states = self.ple(
                hidden_states,
                input_ids,
                query_start_loc,
                ngram_context,
            )

        if prev_block_output is not None and prev_injection is not None:
            hidden_states, block_input, injection = attn_hc.combine_and_mix(
                hidden_states,
                prev_block_output,
                prev_injection,
            )
        else:
            hidden_states, block_input, injection = attn_hc.mix(hidden_states)

        if self.layer_type == "linear_attention":
            attn_out = self.linear_attn(hidden_states=block_input)
        elif self.layer_type == "full_attention":
            attn_out = self.self_attn(
                hidden_states=block_input,
                positions=positions,
            )
        else:
            raise ValueError("Invalid layer_type")

        mlp_hc = self.mlp_hyper_connection
        hidden_states, block_input, injection = mlp_hc.combine_and_mix(
            hidden_states,
            attn_out,
            injection,
        )
        mlp_out = self.mlp(block_input)
        return hidden_states, mlp_out, injection

    setattr(hcu_decoder_forward, _FORWARD_WRAPPER, True)
    fused_mapper = original_mapper | _PLE_WEIGHTS_MAPPER
    fused_llm_packed = {**original_llm_packed, "kv_proj": _KV_PACKED.copy()}
    fused_vl_packed = {**original_vl_packed, "kv_proj": _KV_PACKED.copy()}

    try:
        setattr(owner, _ORIGINAL_FORWARD, current_forward)
        setattr(owner, _ORIGINAL_MAPPER, original_mapper)
        setattr(owner, _ORIGINAL_LLM_PACKED, original_llm_packed)
        setattr(owner, _ORIGINAL_VL_PACKED, original_vl_packed)
        setattr(decoder_class, "forward", hcu_decoder_forward)
        setattr(model_class, "hf_to_vllm_mapper", fused_mapper)
        setattr(llm_class, "packed_modules_mapping", fused_llm_packed)
        setattr(vl_class, "packed_modules_mapping", fused_vl_packed)
        setattr(owner, _MARKER, True)
    except BaseException:
        setattr(decoder_class, "forward", current_forward)
        setattr(model_class, "hf_to_vllm_mapper", original_mapper)
        setattr(llm_class, "packed_modules_mapping", original_llm_packed)
        setattr(vl_class, "packed_modules_mapping", original_vl_packed)
        for name in (
            _ORIGINAL_FORWARD,
            _ORIGINAL_MAPPER,
            _ORIGINAL_LLM_PACKED,
            _ORIGINAL_VL_PACKED,
            _MARKER,
        ):
            if hasattr(owner, name):
                delattr(owner, name)
        raise

    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
