# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Adapt DeepSeek V4 attention compressor and FP8 cache insertion for HCU."""

from __future__ import annotations

import functools
from collections.abc import Callable
from types import ModuleType
from typing import Any

import torch

from vllm_hcu.platforms.hcu import on_gfx938

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
_INDEXER_INIT_WRAPPER_MARKER = "_vllm_hcu_indexer_cache_dtype_wrapper"
_INDEXER_Q_WRAPPER_MARKER = "_vllm_hcu_bf16_indexer_query_wrapper"


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


def _bf16_indexer_q_rope(
    positions: torch.Tensor,
    q: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    weights: torch.Tensor,
    softmax_scale: float,
    head_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply interleaved GPT-J RoPE without introducing FP8 on gfx936."""
    rotary_dim = cos_sin_cache.shape[-1]
    if rotary_dim % 2 or rotary_dim > q.shape[-1]:
        raise ValueError(
            "DeepSeek V4 indexer RoPE requires an even rotary dimension no "
            f"larger than head_dim, got rotary_dim={rotary_dim}, "
            f"head_dim={q.shape[-1]}"
        )

    half_rotary_dim = rotary_dim // 2
    selected = cos_sin_cache.index_select(0, positions.to(torch.long))
    cos = selected[:, :half_rotary_dim].unsqueeze(1)
    sin = selected[:, half_rotary_dim:].unsqueeze(1)
    q_rot = q[..., -rotary_dim:].float()
    q_even = q_rot[..., 0::2]
    q_odd = q_rot[..., 1::2]
    rotated = torch.stack(
        (q_even * cos - q_odd * sin, q_odd * cos + q_even * sin),
        dim=-1,
    ).flatten(-2)
    rotated = rotated.to(q.dtype)
    if rotary_dim < q.shape[-1]:
        rotated = torch.cat((q[..., :-rotary_dim], rotated), dim=-1)

    scaled_weights = weights.float() * softmax_scale * head_scale
    return rotated, scaled_weights


def apply_to_module(module: ModuleType) -> bool:
    attention = load_exact_module(TARGET_MODULE, module)
    cls = require_class(attention, "DeepseekV4Attention", TARGET_SYMBOL)
    indexer_cls = getattr(attention, "DeepseekV4Indexer", None)
    current_indexer_q = getattr(attention, "fused_indexer_q_rope_quant", None)
    original = require_callable(cls, "_run_parallel_input_projections", TARGET_SYMBOL)
    if getattr(cls, _CLASS_MARKER, False):
        current_init = vars(cls).get("__init__")
        current_forward = vars(cls).get("forward")
        current = vars(cls).get("_run_parallel_input_projections")
        current_insert = vars(cls).get("_fused_qnorm_rope_kv_insert")
        current_indexer_init = (
            vars(indexer_cls).get("__init__") if indexer_cls is not None else None
        )
        if not (
            getattr(current_init, _INIT_WRAPPER_MARKER, False)
            and getattr(current_forward, _FORWARD_WRAPPER_MARKER, False)
            and getattr(current, _WRAPPER_MARKER, False)
            and getattr(current_insert, _INSERT_WRAPPER_MARKER, False)
            and (
                indexer_cls is None
                or getattr(
                    current_indexer_init,
                    _INDEXER_INIT_WRAPPER_MARKER,
                    False,
                )
            )
            and (
                indexer_cls is None
                or getattr(
                    current_indexer_q,
                    _INDEXER_Q_WRAPPER_MARKER,
                    False,
                )
            )
        ):
            raise PatchCompatibilityError(
                f"required HCU patch marker for {TARGET_SYMBOL} is stale"
            )
        return False
    original_init = require_callable(cls, "__init__", _INIT_TARGET_SYMBOL)
    original_indexer_init = None
    original_indexer_q = None
    if indexer_cls is not None:
        original_indexer_init = require_callable(
            indexer_cls,
            "__init__",
            f"{TARGET_MODULE}.DeepseekV4Indexer.__init__",
        )
        require_exact_signature(
            original_indexer_init,
            f"{TARGET_MODULE}.DeepseekV4Indexer.__init__",
            positional=(
                "self",
                "vllm_config",
                "config",
                "hidden_size",
                "q_lora_rank",
                "quant_config",
                "cache_config",
                "topk_indices_buffer",
                "compress_ratio",
                "prefix",
                "aux_stream",
            ),
            defaults={
                "compress_ratio": 1,
                "prefix": "",
                "aux_stream": None,
            },
        )
        original_indexer_q = require_callable(
            attention,
            "fused_indexer_q_rope_quant",
            f"{TARGET_MODULE}.fused_indexer_q_rope_quant",
        )
        require_exact_signature(
            original_indexer_q,
            f"{TARGET_MODULE}.fused_indexer_q_rope_quant",
            positional=(
                "positions",
                "index_q",
                "index_q_cos_sin_cache",
                "index_weights",
                "index_weights_softmax_scale",
                "index_weights_head_scale",
                "use_fp4",
            ),
            defaults={"use_fp4": False},
        )
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

    if original_indexer_init is not None:

        @functools.wraps(original_indexer_init)
        def hcu_indexer_init(
            self,
            vllm_config,
            config,
            hidden_size,
            q_lora_rank,
            quant_config,
            cache_config,
            topk_indices_buffer,
            compress_ratio=1,
            prefix="",
            aux_stream=None,
        ):
            original_indexer_init(
                self,
                vllm_config,
                config,
                hidden_size,
                q_lora_rank,
                quant_config,
                cache_config,
                topk_indices_buffer,
                compress_ratio,
                prefix,
                aux_stream,
            )
            # gfx938 has the quantized indexer cache reader. gfx936 selects
            # the BF16 HIPC writer/gather path, so its allocation contract
            # must contain raw BF16 keys rather than FP8 bytes plus scales.
            if not on_gfx938():
                self.k_cache.dtype = torch.bfloat16
                self.k_cache.head_dim = self.head_dim

        setattr(hcu_indexer_init, _INDEXER_INIT_WRAPPER_MARKER, True)

    if original_indexer_q is not None:

        @functools.wraps(original_indexer_q)
        def hcu_indexer_q_rope_quant(
            positions,
            index_q,
            index_q_cos_sin_cache,
            index_weights,
            index_weights_softmax_scale,
            index_weights_head_scale,
            use_fp4=False,
        ):
            if on_gfx938() or use_fp4:
                return original_indexer_q(
                    positions,
                    index_q,
                    index_q_cos_sin_cache,
                    index_weights,
                    index_weights_softmax_scale,
                    index_weights_head_scale,
                    use_fp4,
                )
            return _bf16_indexer_q_rope(
                positions,
                index_q,
                index_q_cos_sin_cache,
                index_weights,
                index_weights_softmax_scale,
                index_weights_head_scale,
            )

        setattr(hcu_indexer_q_rope_quant, _INDEXER_Q_WRAPPER_MARKER, True)

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
    if indexer_cls is not None and original_indexer_init is not None:
        setattr(indexer_cls, "_vllm_hcu_original_init", original_indexer_init)
        setattr(indexer_cls, "__init__", hcu_indexer_init)
    if original_indexer_q is not None:
        setattr(
            attention,
            "_vllm_hcu_original_fused_indexer_q_rope_quant",
            original_indexer_q,
        )
        setattr(attention, "fused_indexer_q_rope_quant", hcu_indexer_q_rope_quant)
    setattr(cls, "__init__", hcu_attention_init)
    setattr(cls, "forward", hcu_attention_forward)
    setattr(cls, "_run_parallel_input_projections", hcu_run_parallel_input_projections)
    setattr(cls, "_fused_qnorm_rope_kv_insert", hcu_fused_qnorm_rope_kv_insert)
    setattr(cls, _CLASS_MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "apply", "apply_to_module"]
