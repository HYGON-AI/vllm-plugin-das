# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Adapt DSV4.1 attention and cache publication to the HCU runtime."""

from __future__ import annotations

import functools
from collections.abc import Callable
from contextlib import contextmanager
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

TARGET_MODULE = "vllm.models.deepseek_v41.attention"
PATCH_ID = "worker.core_fix.deepseek_v41.compressor_weight_layout"
TARGETS = (
    f"{TARGET_MODULE}.DeepseekV4Attention._run_parallel_input_projections",
    f"{TARGET_MODULE}.DeepseekV4Attention._fused_qnorm_rope_kv_insert",
    f"{TARGET_MODULE}.DeepseekV4Indexer._produce_k",
    "vllm.models.deepseek_v41.compressor.DeepseekCompressor.forward",
    "vllm.models.deepseek_v41.compressor.DeepseekCompressor.insert_cache",
    f"{TARGET_MODULE}.DeepseekV4Attention.__init__",
)
_CLASS_MARKER = "_vllm_hcu_dsv41_compressor_layout_applied"
_WRAPPER_MARKER = "_vllm_hcu_dsv41_compressor_layout_wrapper"


def _compressor_mm(hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Multiply HCU ``[in, out]`` or upstream ``[out, in]`` weights."""

    rhs = weight if weight.shape[0] == hidden_states.shape[-1] else weight.T
    return torch.mm(hidden_states, rhs, out_dtype=torch.float32)


@contextmanager
def _replace_slot_mapping(metadata: object, slot_mapping: torch.Tensor):
    original = getattr(metadata, "slot_mapping")
    setattr(metadata, "slot_mapping", slot_mapping)
    try:
        yield
    finally:
        setattr(metadata, "slot_mapping", original)


def _metadata_builder_wrapper(build):
    @functools.wraps(build)
    def hcu_build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        result = build(
            self,
            common_prefix_len,
            common_attn_metadata,
            fast_build,
        )
        from vllm_hcu.model_executor.layers.attention.pcp import (
            effective_pcp_metadata_world_size,
        )

        parallel_config = getattr(
            getattr(self, "vllm_config", None), "parallel_config", None
        )
        configured_size = int(
            getattr(
                parallel_config,
                "prefill_context_parallel_size",
                getattr(common_attn_metadata, "pcp_world_size", 1),
            )
        )
        result.pcp_world_size = effective_pcp_metadata_world_size(configured_size)
        return result

    setattr(hcu_build, _WRAPPER_MARKER, True)
    return hcu_build


def _enable_pcp_backend(backend_cls: type) -> None:
    backend_cls.supports_pcp = classmethod(lambda cls: True)


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    cls = require_class(target, "DeepseekV4Attention", f"{TARGET_MODULE}.DeepseekV4Attention")
    original = require_callable(cls, "_run_parallel_input_projections", TARGETS[0])
    fused_insert = require_callable(cls, "_fused_qnorm_rope_kv_insert", TARGETS[1])
    original_init = require_callable(cls, "__init__", TARGETS[5])
    indexer_cls = require_class(
        target, "DeepseekV4Indexer", f"{TARGET_MODULE}.DeepseekV4Indexer"
    )
    produce_k = require_callable(indexer_cls, "_produce_k", TARGETS[2])
    require_exact_signature(original, TARGETS[0], positional=("self", "hidden_states"))
    require_exact_signature(
        fused_insert,
        TARGETS[1],
        positional=("self", "q", "kv", "positions", "attn_metadata"),
    )
    require_exact_signature(
        produce_k,
        TARGETS[2],
        positional=("self", "latent", "positions", "rotary_emb"),
    )

    if getattr(cls, _CLASS_MARKER, False):
        current = vars(cls).get("_run_parallel_input_projections")
        if not getattr(current, _WRAPPER_MARKER, False):
            raise PatchCompatibilityError(f"required HCU patch marker for {TARGETS[0]} is stale")
        return False

    execute_in_parallel = require_callable(
        target, "execute_in_parallel", f"{TARGET_MODULE}.execute_in_parallel"
    )
    envs = getattr(target, "envs", None)
    if envs is None:
        raise PatchCompatibilityError(f"required HCU patch target {TARGET_MODULE}.envs is missing")

    from vllm.models.deepseek_v41 import compressor as compressor_module
    from vllm.models.deepseek_v41 import sparse_mla as sparse_mla_module
    from vllm.v1.attention.backends.mla import sparse_swa as sparse_swa_module

    compressor_cls = require_class(
        compressor_module,
        "DeepseekCompressor",
        "vllm.models.deepseek_v41.compressor.DeepseekCompressor",
    )
    compressor_forward = require_callable(compressor_cls, "forward", TARGETS[3])
    compressor_insert = require_callable(compressor_cls, "insert_cache", TARGETS[4])
    require_exact_signature(
        compressor_forward,
        TARGETS[3],
        positional=("self", "kv_score", "positions"),
    )
    require_exact_signature(
        compressor_insert,
        TARGETS[4],
        positional=("self", "latent", "positions", "rotary_emb"),
    )

    @functools.wraps(original)
    def hcu_run_parallel_input_projections(self, hidden_states) -> tuple[Any, ...]:
        aux_streams = self.aux_stream_list
        if aux_streams is not None:
            aux_streams = aux_streams[:2]

        aux_fns: list[Callable[[], Any] | None] = [None, None]
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

            aux_fns[1] = indexer_weights_proj

        qr_kv, (kv_score, indexer_weights) = execute_in_parallel(
            lambda: self._fused_wqa_wkv_gemm(hidden_states),
            aux_fns,
            self.ln_events[0],
            self.ln_events[1:3],
            aux_streams,
            enable=hidden_states.shape[0]
            <= envs.VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD,
        )
        return qr_kv, kv_score, indexer_weights

    @functools.wraps(original_init)
    def hcu_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        vllm_config = args[0] if args else kwargs.get("vllm_config")
        parallel_config = getattr(vllm_config, "parallel_config", None)
        if int(getattr(parallel_config, "prefill_context_parallel_size", 1)) > 1:
            # PCP cache collectives and slot remapping execute in this region.
            self._prepare_and_attn_fn = self._prepare_and_attn_eager

    @functools.wraps(fused_insert)
    def hcu_fused_qnorm_rope_kv_insert(self, q, kv, positions, attn_metadata):
        if not isinstance(attn_metadata, dict):
            return fused_insert(self, q, kv, positions, attn_metadata)
        metadata = attn_metadata.get(self.swa_cache_layer.prefix)
        if metadata is None or int(getattr(metadata, "pcp_world_size", 1)) <= 1:
            return fused_insert(self, q, kv, positions, attn_metadata)

        from vllm_hcu.model_executor.layers.attention.pcp import (
            local_pcp_slot_mapping,
            maybe_gather_cache_inputs,
        )

        slot_mapping = metadata.slot_mapping
        (global_q, global_kv, global_positions), global_slots = (
            maybe_gather_cache_inputs(
                (q, kv, positions),
                slot_mapping,
                metadata,
            )
        )
        local_slots = local_pcp_slot_mapping(
            slot_mapping,
            q.shape[0],
            metadata,
        )
        # Publish every rank's SWA rows before local sparse attention reads them.
        with _replace_slot_mapping(metadata, global_slots):
            fused_insert(
                self,
                global_q,
                global_kv,
                global_positions,
                attn_metadata,
            )
        with _replace_slot_mapping(metadata, local_slots):
            return fused_insert(self, q, kv, positions, attn_metadata)

    @functools.wraps(compressor_forward)
    def hcu_compressor_forward(self, kv_score, positions):
        attn_metadata = target.get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            return compressor_forward(self, kv_score, positions)
        metadata_key = (
            self.state_cache.prefix
            if self.state_cache is not None
            else self.k_cache_prefix
        )
        metadata = attn_metadata[metadata_key]
        if int(getattr(metadata, "pcp_world_size", 1)) <= 1:
            return compressor_forward(self, kv_score, positions)

        from vllm_hcu.model_executor.layers.attention.pcp import (
            local_pcp_slot_mapping,
            maybe_gather_cache_inputs,
        )

        slot_mapping = metadata.slot_mapping
        local_slots = local_pcp_slot_mapping(
            slot_mapping,
            kv_score.shape[0],
            metadata,
        )
        if self.state_cache is not None:
            (global_kv_score, _), global_slots = maybe_gather_cache_inputs(
                (kv_score, positions),
                slot_mapping,
                metadata,
            )
            valid = global_slots >= 0
            self.state_cache.kv_cache.view(-1, global_kv_score.shape[-1]).index_copy_(
                0,
                global_slots[valid],
                global_kv_score[valid],
            )
        with _replace_slot_mapping(metadata, local_slots):
            return compressor_forward(self, kv_score, positions)

    @functools.wraps(compressor_insert)
    def hcu_compressor_insert_cache(self, latent, positions, rotary_emb):
        if latent is None:
            return compressor_insert(self, latent, positions, rotary_emb)
        attn_metadata = target.get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            return compressor_insert(self, latent, positions, rotary_emb)
        metadata = attn_metadata[self.k_cache_prefix]
        if int(getattr(metadata, "pcp_world_size", 1)) <= 1:
            return compressor_insert(self, latent, positions, rotary_emb)

        from vllm_hcu.model_executor.layers.attention.pcp import (
            maybe_gather_cache_inputs,
        )

        (global_latent, global_positions), global_slots = maybe_gather_cache_inputs(
            (latent, positions),
            metadata.slot_mapping,
            metadata,
        )
        with _replace_slot_mapping(metadata, global_slots):
            return compressor_insert(
                self,
                global_latent,
                global_positions,
                rotary_emb,
            )

    @functools.wraps(produce_k)
    def hcu_produce_k(self, latent, positions, rotary_emb):
        if latent is None:
            return produce_k(self, latent, positions, rotary_emb)
        attn_metadata = target.get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            return produce_k(self, latent, positions, rotary_emb)
        metadata = attn_metadata[self.k_cache.prefix]
        if int(getattr(metadata, "pcp_world_size", 1)) <= 1:
            return produce_k(self, latent, positions, rotary_emb)

        from vllm_hcu.model_executor.layers.attention.pcp import (
            maybe_gather_cache_inputs,
        )

        (global_latent, global_positions), global_slots = maybe_gather_cache_inputs(
            (latent, positions),
            metadata.slot_mapping,
            metadata,
        )
        with _replace_slot_mapping(metadata, global_slots):
            return produce_k(
                self,
                global_latent,
                global_positions,
                rotary_emb,
            )

    for function in (
        hcu_run_parallel_input_projections,
        hcu_init,
        hcu_fused_qnorm_rope_kv_insert,
        hcu_compressor_forward,
        hcu_compressor_insert_cache,
        hcu_produce_k,
    ):
        setattr(function, _WRAPPER_MARKER, True)
    setattr(cls, "_vllm_hcu_original_run_parallel_input_projections", original)
    setattr(cls, "_run_parallel_input_projections", hcu_run_parallel_input_projections)
    setattr(cls, "_vllm_hcu_original_init", original_init)
    setattr(cls, "__init__", hcu_init)
    setattr(cls, "_vllm_hcu_original_fused_qnorm_rope_kv_insert", fused_insert)
    setattr(cls, "_fused_qnorm_rope_kv_insert", hcu_fused_qnorm_rope_kv_insert)
    setattr(compressor_cls, "_vllm_hcu_original_forward", compressor_forward)
    setattr(compressor_cls, "forward", hcu_compressor_forward)
    setattr(compressor_cls, "_vllm_hcu_original_insert_cache", compressor_insert)
    setattr(compressor_cls, "insert_cache", hcu_compressor_insert_cache)
    setattr(indexer_cls, "_vllm_hcu_original_produce_k", produce_k)
    setattr(indexer_cls, "_produce_k", hcu_produce_k)

    for builder_cls in (
        sparse_mla_module.DeepseekV4SparseMLAMetadataBuilder,
        sparse_swa_module.DeepseekSparseSWAMetadataBuilder,
        compressor_module.CompressorMetadataBuilder,
    ):
        build = require_callable(builder_cls, "build", f"{builder_cls.__name__}.build")
        setattr(builder_cls, "_vllm_hcu_original_pcp_build", build)
        setattr(builder_cls, "build", _metadata_builder_wrapper(build))

    _enable_pcp_backend(sparse_mla_module.DeepseekV4SparseMLABackend)
    _enable_pcp_backend(sparse_swa_module.DeepseekSparseSWABackend)
    _enable_pcp_backend(compressor_module.CompressorBackend)
    setattr(cls, _CLASS_MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
