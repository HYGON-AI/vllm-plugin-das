# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Adapt DeepSeek V4 attention compressor and FP8 cache insertion for HCU."""

from __future__ import annotations

import functools
from collections.abc import Callable
from types import ModuleType
from typing import Any

import torch

from ._common import (
    PatchCompatibilityError,
    load_exact_module,
    require_callable,
    require_class,
    require_exact_signature,
)

TARGET_MODULE = "vllm.models.deepseek_v4.attention"
PATCH_ID = "worker.core_fix.deepseek_v4.attention_compressor_weight_layout"
_INIT_TARGET_SYMBOL = f"{TARGET_MODULE}.DeepseekV4Attention.__init__"
_FORWARD_TARGET_SYMBOL = f"{TARGET_MODULE}.DeepseekV4Attention.forward"
TARGET_SYMBOL = (
    f"{TARGET_MODULE}.DeepseekV4Attention._run_parallel_input_projections"
)
_INSERT_TARGET_SYMBOL = (
    f"{TARGET_MODULE}.DeepseekV4Attention._fused_qnorm_rope_kv_insert"
)
_CLASS_MARKER = "_vllm_hcu_compressor_weight_layout_applied"
_INIT_WRAPPER_MARKER = "_vllm_hcu_int8_wo_a_ignore_wrapper"
_FORWARD_WRAPPER_MARKER = "_vllm_hcu_raw_kv_caller_wrapper"
_WRAPPER_MARKER = "_vllm_hcu_compressor_weight_layout_wrapper"
_INSERT_WRAPPER_MARKER = "_vllm_hcu_fp8_ds_mla_lightop_insert_wrapper"


def _compressor_mm(hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Multiply either HCU NN ``[in, out]`` or upstream ``[out, in]`` weights."""
    rhs = weight if weight.shape[0] == hidden_states.shape[-1] else weight.T
    return torch.mm(hidden_states, rhs, out_dtype=torch.float32)


def _requires_unquantized_int8_wo_a(vllm_config: object) -> bool:
    quant_config = getattr(vllm_config, "quant_config", None)
    get_name = getattr(quant_config, "get_name", None)
    model_config = getattr(vllm_config, "model_config", None)
    hf_config = getattr(model_config, "hf_config", None)
    return bool(
        callable(get_name)
        and get_name() == "compressed-tensors"
        and getattr(quant_config, "quant_format", None) == "int-quantized"
        and getattr(hf_config, "expert_dtype", None) == "int8"
    )


def apply_to_module(module: ModuleType) -> bool:
    attention = load_exact_module(TARGET_MODULE, module)
    cls = require_class(attention, "DeepseekV4Attention", TARGET_SYMBOL)
    original = require_callable(cls, "_run_parallel_input_projections", TARGET_SYMBOL)
    if getattr(cls, _CLASS_MARKER, False):
        current_init = vars(cls).get("__init__")
        current_forward = vars(cls).get("forward")
        current = vars(cls).get("_run_parallel_input_projections")
        current_insert = vars(cls).get("_fused_qnorm_rope_kv_insert")
        if not (
            getattr(current_init, _INIT_WRAPPER_MARKER, False)
            and getattr(current_forward, _FORWARD_WRAPPER_MARKER, False)
            and getattr(current, _WRAPPER_MARKER, False)
            and getattr(current_insert, _INSERT_WRAPPER_MARKER, False)
        ):
            raise PatchCompatibilityError(
                f"required HCU patch marker for {TARGET_SYMBOL} is stale"
            )
        return False
    original_init = require_callable(cls, "__init__", _INIT_TARGET_SYMBOL)
    require_exact_signature(
        original_init,
        _INIT_TARGET_SYMBOL,
        positional=(
            "self",
            "vllm_config",
            "prefix",
            "topk_indices_buffer",
            "aux_stream_list",
        ),
        defaults={
            "topk_indices_buffer": None,
            "aux_stream_list": None,
        },
    )
    require_exact_signature(
        original,
        TARGET_SYMBOL,
        positional=("self", "hidden_states"),
    )
    original_forward = require_callable(cls, "forward", _FORWARD_TARGET_SYMBOL)
    require_exact_signature(
        original_forward,
        _FORWARD_TARGET_SYMBOL,
        positional=("self", "positions", "hidden_states", "llama_4_scaling"),
        defaults={"llama_4_scaling": None},
    )
    original_insert = require_callable(
        cls,
        "_fused_qnorm_rope_kv_insert",
        _INSERT_TARGET_SYMBOL,
    )
    require_exact_signature(
        original_insert,
        _INSERT_TARGET_SYMBOL,
        positional=("self", "q", "kv", "positions", "attn_metadata"),
    )

    execute_in_parallel = require_callable(
        attention, "execute_in_parallel", f"{TARGET_MODULE}.execute_in_parallel"
    )
    envs = attention.envs

    @functools.wraps(original_init)
    def hcu_attention_init(
        self,
        vllm_config,
        prefix,
        topk_indices_buffer=None,
        aux_stream_list=None,
    ):
        quant_config = getattr(vllm_config, "quant_config", None)
        if not _requires_unquantized_int8_wo_a(vllm_config):
            return original_init(
                self,
                vllm_config,
                prefix,
                topk_indices_buffer,
                aux_stream_list,
            )

        # This Channel-INT8 checkpoint keeps wo_a in BF16 and stores no
        # scale.  Its compressed-tensors config nevertheless targets every
        # Linear.  Exclude only this attention instance while its submodules
        # are built, then restore the shared quant config immediately.
        previous_ignore = quant_config.ignore
        quant_config.ignore = [*previous_ignore, f"{prefix}.wo_a"]
        try:
            return original_init(
                self,
                vllm_config,
                prefix,
                topk_indices_buffer,
                aux_stream_list,
            )
        finally:
            quant_config.ignore = previous_ignore

    setattr(hcu_attention_init, _INIT_WRAPPER_MARKER, True)

    @functools.wraps(original_forward)
    def hcu_attention_forward(
        self,
        positions,
        hidden_states,
        llama_4_scaling=None,
    ):
        # Upstream normalizes QR and KV together before attention_impl. The
        # uint8 LightOp insert owns KVNorm itself, so keep KV raw here and let
        # the insert wrapper normalize it only for official non-uint8 caches.
        del llama_4_scaling
        num_tokens = hidden_states.shape[0]
        o_padded = torch.empty(
            (num_tokens, self.padded_heads, self.head_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        qr_kv, kv_score, indexer_kv_score, indexer_weights = (
            self._run_parallel_input_projections(hidden_states)
        )
        qr, raw_kv = qr_kv.split(
            [self.q_lora_rank, self.head_dim], dim=-1
        )
        qr = self.q_norm(qr)

        self._prepare_and_attn_fn(
            hidden_states,
            qr,
            raw_kv,
            None,
            kv_score,
            indexer_kv_score,
            indexer_weights,
            positions,
            o_padded,
        )
        output = o_padded[:, : self.n_local_heads, :]
        return self._o_proj(output, positions)

    setattr(hcu_attention_forward, _FORWARD_WRAPPER_MARKER, True)

    @functools.wraps(original)
    def hcu_run_parallel_input_projections(self, hidden_states) -> tuple[Any, ...]:
        aux_streams = self.aux_stream_list
        if aux_streams is not None:
            assert len(aux_streams) >= 3
            aux_streams = aux_streams[:3]

        aux_fns: list[Callable[[], Any] | None] = [None, None, None]
        if self.compressor is not None:
            compressor = self.compressor

            def compressor_kv_score() -> torch.Tensor:
                return _compressor_mm(
                    hidden_states, compressor.fused_wkv_wgate.weight
                )

            aux_fns[0] = compressor_kv_score

        if self.indexer is not None:
            indexer = self.indexer

            def indexer_weights_proj() -> torch.Tensor:
                weights, _ = indexer.weights_proj(hidden_states)
                return weights

            def indexer_compressor_kv_score() -> torch.Tensor:
                return _compressor_mm(
                    hidden_states, indexer.compressor.fused_wkv_wgate.weight
                )

            aux_fns[1] = indexer_weights_proj
            aux_fns[2] = indexer_compressor_kv_score

        def fused_wqa_wkv() -> torch.Tensor:
            qr_kv, _ = self.fused_wqa_wkv(hidden_states)
            return qr_kv

        qr_kv, (kv_score, indexer_weights, indexer_kv_score) = execute_in_parallel(
            fused_wqa_wkv,
            aux_fns,
            self.ln_events[0],
            self.ln_events[1:4],
            aux_streams,
            enable=(
                hidden_states.shape[0]
                <= envs.VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD
            ),
        )
        return qr_kv, kv_score, indexer_kv_score, indexer_weights

    setattr(hcu_run_parallel_input_projections, _WRAPPER_MARKER, True)

    @functools.wraps(original_insert)
    def hcu_fused_qnorm_rope_kv_insert(
        self,
        q,
        kv,
        positions,
        attn_metadata,
    ):
        # Preserve official profiling and non-DS-MLA cache behavior. The HCU
        # LightOp is specifically the uint8 UE8M0 fp8_ds_mla implementation.
        if not isinstance(attn_metadata, dict):
            return original_insert(self, q, kv, positions, attn_metadata)

        swa_kv_cache = self.swa_cache_layer.kv_cache
        if swa_kv_cache.dtype != torch.uint8:
            normalized_kv = self.kv_norm(kv)
            return original_insert(
                self, q, normalized_kv, positions, attn_metadata
            )

        swa_metadata = attn_metadata.get(self.swa_cache_layer.prefix)
        assert swa_metadata is not None
        swa_kv_cache_2d = swa_kv_cache.view(swa_kv_cache.shape[0], -1)

        try:
            from lightop.attention import (
                fused_deepseek_v4_qnorm_rope_kvnorm_rope_quant_insert_int32,
            )
        except (ImportError, AttributeError) as exc:
            raise RuntimeError(
                "DeepSeek V4 core fix requires lightop.attention."
                "fused_deepseek_v4_qnorm_rope_kvnorm_rope_quant_insert_int32; "
                "upgrade LightOp"
            ) from exc

        swa_slot_mapping_i32 = swa_metadata.slot_mapping.to(
            dtype=torch.int32
        ).contiguous()
        fused_deepseek_v4_qnorm_rope_kvnorm_rope_quant_insert_int32(
            q,
            kv,
            self.kv_norm.weight.data,
            swa_kv_cache_2d,
            swa_slot_mapping_i32,
            positions.to(torch.int64),
            self.rotary_emb.cos_sin_cache,
            self.eps,
            swa_metadata.block_size,
        )
        return q

    setattr(hcu_fused_qnorm_rope_kv_insert, _INSERT_WRAPPER_MARKER, True)
    setattr(cls, "_vllm_hcu_original_init", original_init)
    setattr(cls, "_vllm_hcu_original_forward", original_forward)
    setattr(cls, "_vllm_hcu_original_run_parallel_input_projections", original)
    setattr(cls, "_vllm_hcu_original_fused_qnorm_rope_kv_insert", original_insert)
    setattr(cls, "__init__", hcu_attention_init)
    setattr(cls, "forward", hcu_attention_forward)
    setattr(cls, "_run_parallel_input_projections", hcu_run_parallel_input_projections)
    setattr(cls, "_fused_qnorm_rope_kv_insert", hcu_fused_qnorm_rope_kv_insert)
    setattr(cls, _CLASS_MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "apply", "apply_to_module"]
