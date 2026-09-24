# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""DeepSeek-V4.1 PCP cache materialization for the HCU adapter.

PCP partitions each prefill across ranks, but every rank has to end the step
holding the complete caches: the sliding-window KV, the compressed KV and the
indexer K cache are read by the attention and indexer kernels as one global
sequence.  Two row layouts are in play:

``local``
    This rank's padded rows, the layout the model runs in.

``global``
    The batch's real tokens in their original order, which is also the layout
    the compressor's pooling contract assumes: a group can pair rows owned by
    different ranks, and each request's chunk is contiguous with contiguous
    positions only here.

Every cache writer therefore runs on global rows.  The manager's expanded slot
mappings are reduced to that order through its restore map, so each global row
is written exactly once on every rank, and replicated decode rows use the same
slot everywhere.  Padding rows never reach a writer.

Multi-stream overlap is disabled for these steps: the writers issue PCP
collectives, and concurrent streams would let ranks enter them in different
orders.

The adapter is armed only while ``VLLM_HCU_DSV4_PCP_EXPERIMENTAL`` is set,
matching the capability and scope gates.
"""

from __future__ import annotations

import importlib
import os
from types import ModuleType
from typing import Any

import torch

TARGET_MODULE = "vllm.models.deepseek_v41.attention"
COMPRESSOR_MODULE = "vllm.models.deepseek_v41.compressor"
PATCH_ID = "worker.core_fix.deepseek_v41.pcp_cache_materialization"
_ENV_FLAG = "VLLM_HCU_DSV4_PCP_EXPERIMENTAL"
_CLASS_MARKER = "_vllm_hcu_dsv41_pcp_materialization_applied"
_MARKER_ATTRS = (
    "_vllm_hcu_pcp_original_prepare_and_attn",
    "_vllm_hcu_pcp_original_compressor_build",
    "_vllm_hcu_pcp_original_compressor_forward",
    "_vllm_hcu_pcp_original_compressor_insert_cache",
    "_vllm_hcu_pcp_original_indexer_produce_k",
    "_vllm_hcu_pcp_original_insert",
)


def _pcp_helpers() -> Any:
    """Import the PCP layout helpers only when the adapter actually runs."""

    from vllm_hcu.model_executor.layers.attention import pcp

    return pcp


def _pcp_group() -> Any:
    from vllm.distributed.parallel_state import get_pcp_group

    return get_pcp_group()


def _globalize_slots(slots: torch.Tensor, layout: object) -> torch.Tensor:
    return _pcp_helpers().globalize_pcp_slot_mapping(slots, layout)


def _global_rows(tensor: torch.Tensor, layout: object) -> torch.Tensor:
    return _pcp_helpers().restore_pcp_rows_to_global(tensor, layout)


def _global_positions(layout: object) -> torch.Tensor:
    return _pcp_helpers().pcp_global_positions(layout)


def _global_query_start_loc(layout: object) -> torch.Tensor:
    return _pcp_helpers().pcp_global_query_start_loc(layout)


def _global_token_to_req(layout: object) -> torch.Tensor:
    return _pcp_helpers().pcp_global_token_to_req_indices(layout)


def _metadata_world_size(configured_world_size: int) -> int:
    return _pcp_helpers().effective_pcp_metadata_world_size(configured_world_size)


def _experimental_enabled() -> bool:
    return os.environ.get(_ENV_FLAG, "").lower() in ("true", "1")


def _pcp_layout(metadata: object | None) -> object | None:
    """The metadata carrying this step's partitioned-prefill layout, if any."""

    if metadata is None:
        return None
    if not getattr(metadata, "pcp_has_global_prefill", False):
        return None
    if int(getattr(metadata, "pcp_world_size", 1)) <= 1:
        return None
    return metadata


def _step_metadata() -> object | None:
    from vllm.forward_context import get_forward_context

    return get_forward_context().attn_metadata


def _step_layout(attn_metadata: object | None) -> object | None:
    """One layout per step: every metadata carries the same PCP ownership."""

    if not isinstance(attn_metadata, dict):
        return _pcp_layout(attn_metadata)
    for metadata in attn_metadata.values():
        layout = _pcp_layout(metadata)
        if layout is not None:
            return layout
    return None


def _metadata_for(attn_metadata: object, prefix: str) -> object | None:
    if isinstance(attn_metadata, dict):
        return attn_metadata.get(prefix)
    return None


def _global_slots(metadata: object, layout: object) -> torch.Tensor:
    """This cache's slots for every global token, written once per token."""

    slots = getattr(metadata, "slot_mapping", None)
    assert isinstance(slots, torch.Tensor), (
        "PCP cache materialization requires a slot mapping"
    )
    return _globalize_slots(slots, layout)


def _packed_swa_cache(cache: torch.Tensor) -> None:
    if cache.dtype != torch.uint8:
        raise NotImplementedError(
            "DeepSeek-V4.1 PCP materializes the sliding-window cache through "
            f"the packed fp8_ds_mla record; got dtype {cache.dtype}."
        )


def apply_to_module(module: ModuleType) -> bool:
    """Wrap the V4.1 cache writers for PCP cache materialization."""

    if not _experimental_enabled():
        # Fail closed: the unvalidated path stays off unless explicitly armed.
        return False
    if getattr(module, _CLASS_MARKER, False):
        return False
    for attribute in _MARKER_ATTRS:
        if hasattr(module, attribute):
            raise RuntimeError(
                f"DeepSeek-V4.1 PCP adapter marker {attribute} is stale"
            )

    from ._common import require_callable, require_class, require_exact_signature

    attention = module
    compressor_module = importlib.import_module(COMPRESSOR_MODULE)
    compressor_cls = require_class(
        compressor_module,
        "DeepseekCompressor",
        f"{COMPRESSOR_MODULE}.DeepseekCompressor",
    )
    builder_cls = require_class(
        compressor_module,
        "CompressorMetadataBuilder",
        f"{COMPRESSOR_MODULE}.CompressorMetadataBuilder",
    )
    attention_cls = require_class(
        attention,
        "DeepseekV4Attention",
        f"{TARGET_MODULE}.DeepseekV4Attention",
    )
    indexer_cls = require_class(
        attention, "DeepseekV4Indexer", f"{TARGET_MODULE}.DeepseekV4Indexer"
    )

    original_build = require_callable(
        builder_cls,
        "build",
        f"{COMPRESSOR_MODULE}.CompressorMetadataBuilder.build",
    )
    require_exact_signature(
        original_build,
        f"{COMPRESSOR_MODULE}.CompressorMetadataBuilder.build",
        positional=("self", "common_prefix_len", "common_attn_metadata", "fast_build"),
        defaults={"fast_build": False},
    )
    original_prepare = require_callable(
        attention_cls,
        "_prepare_and_attn",
        f"{TARGET_MODULE}.DeepseekV4Attention._prepare_and_attn",
    )
    require_exact_signature(
        original_prepare,
        f"{TARGET_MODULE}.DeepseekV4Attention._prepare_and_attn",
        positional=(
            "self",
            "hidden_states",
            "qr",
            "kv",
            "qr_scale",
            "kv_score",
            "indexer_weights",
            "positions",
            "attn_out",
        ),
    )
    original_forward = require_callable(
        compressor_cls,
        "forward",
        f"{COMPRESSOR_MODULE}.DeepseekCompressor.forward",
    )
    original_insert = require_callable(
        compressor_cls,
        "insert_cache",
        f"{COMPRESSOR_MODULE}.DeepseekCompressor.insert_cache",
    )
    original_produce_k = require_callable(
        indexer_cls,
        "_produce_k",
        f"{TARGET_MODULE}.DeepseekV4Indexer._produce_k",
    )
    original_insert_qkv = require_callable(
        attention_cls,
        "_fused_qnorm_rope_kv_insert",
        f"{TARGET_MODULE}.DeepseekV4Attention._fused_qnorm_rope_kv_insert",
    )

    def hcu_compressor_build(
        self: Any,
        common_prefix_len: int,
        common_attn_metadata: Any,
        fast_build: bool = False,
    ) -> Any:
        """Size the ring mapping from this rank's rows, not the PCP expansion.

        The manager's expanded slot mapping holds one segment per rank, while
        the ring kernel fills rows in the rank-local batch order from the
        rank-local positions and block table.  Handing it the expanded width
        would leave the extra rows stale, break the pooling assertion
        downstream, and can overrun the builder's own buffer.
        """

        parallel_config = self.vllm_config.parallel_config
        world_size = _metadata_world_size(
            int(parallel_config.prefill_context_parallel_size)
        )
        if world_size <= 1:
            return original_build(
                self, common_prefix_len, common_attn_metadata, fast_build
            )
        width = common_attn_metadata.slot_mapping.numel()
        if width % world_size:
            raise RuntimeError(
                "PCP compressor metadata received a slot mapping that is not "
                f"one equal segment per rank: slots={width}, world={world_size}"
            )
        local_width = width // world_size
        rank = int(_pcp_group().rank_in_group)
        assert 0 <= rank < world_size, f"invalid PCP rank {rank}/{world_size}"
        local_metadata = common_attn_metadata.replace(
            slot_mapping=common_attn_metadata.slot_mapping[
                rank * local_width : (rank + 1) * local_width
            ]
        )
        return original_build(self, common_prefix_len, local_metadata, fast_build)

    def hcu_prepare_and_attn(self: Any, *args: Any, **kwargs: Any) -> Any:
        """Run a PCP step without stream overlap.

        The cache writers issue PCP collectives, so the default and auxiliary
        streams must not interleave them across ranks.
        """

        if _step_layout(_step_metadata()) is None:
            return original_prepare(self, *args, **kwargs)
        saved_streams = self.aux_stream_list
        self.aux_stream_list = None
        try:
            return original_prepare(self, *args, **kwargs)
        finally:
            self.aux_stream_list = saved_streams

    def hcu_compressor_forward(
        self: Any, kv_score: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor | None:
        attn_metadata = _step_metadata()
        layout = _step_layout(attn_metadata)
        if layout is None or not isinstance(attn_metadata, dict):
            return original_forward(self, kv_score, positions)
        from vllm.models.deepseek_v41.common.ops.fused_compress_quant_cache import (
            fused_save_compress_norm,
        )

        state_cache: torch.Tensor | None = None
        if self.state_cache is not None:
            state_metadata = _metadata_for(
                attn_metadata, self.state_cache.prefix
            )
            assert state_metadata is not None, (
                "PCP compressor materialization requires the state cache metadata"
            )
            state_cache = self.state_cache.kv_cache
        else:
            state_metadata = _metadata_for(attn_metadata, self.k_cache_prefix)
            assert state_metadata is not None, (
                "PCP compressor materialization requires the compressed-KV metadata"
            )
        global_kv_score = _global_rows(kv_score, layout)
        global_latent = torch.empty(
            global_kv_score.shape[0],
            self.head_dim,
            dtype=torch.bfloat16,
            device=global_kv_score.device,
        )
        fused_save_compress_norm(
            global_kv_score,
            _global_positions(layout),
            state_cache,
            _global_slots(state_metadata, layout),
            _global_query_start_loc(layout) if state_cache is not None else None,
            (
                _global_token_to_req(layout) if state_cache is not None else None
            ),
            self.norm.weight,
            self.rms_norm_eps,
            self.compress_ratio,
            global_latent,
        )
        return global_latent

    def hcu_insert_cache(
        self: Any,
        latent: torch.Tensor | None,
        positions: torch.Tensor,
        rotary_emb: Any,
    ) -> None:
        attn_metadata = _step_metadata()
        layout = _step_layout(attn_metadata)
        if layout is None or latent is None or not isinstance(attn_metadata, dict):
            return original_insert(self, latent, positions, rotary_emb)
        from vllm.models.deepseek_v41.common.ops.fused_compress_quant_cache import (
            rope_quant_insert,
        )

        main_metadata = _metadata_for(attn_metadata, self.k_cache_prefix)
        assert main_metadata is not None, (
            "PCP compressed-KV insert requires the compressed-KV metadata"
        )
        k_cache_layer = self._static_forward_context[self.k_cache_prefix]
        kv_cache = k_cache_layer.kv_cache
        # Plain-row per-tensor fp8 caches carry the layer's scale; fp8_ds_mla
        # and bf16 rows need none.
        fp8_scale = (
            getattr(k_cache_layer, "_flashinfer_fp8_kv_scale", None)
            if kv_cache.dtype == torch.float8_e4m3fn
            else None
        )
        rope_quant_insert(
            latent,
            _global_positions(layout),
            rotary_emb.cos_sin_cache,
            kv_cache,
            _global_slots(main_metadata, layout),
            self.compress_ratio,
            fp8_scale=fp8_scale,
        )

    def hcu_produce_k(
        self: Any,
        latent: torch.Tensor | None,
        positions: torch.Tensor,
        rotary_emb: Any,
    ) -> None:
        attn_metadata = _step_metadata()
        layout = _step_layout(attn_metadata)
        if layout is None or latent is None or not isinstance(attn_metadata, dict):
            return original_produce_k(self, latent, positions, rotary_emb)
        from vllm.models.deepseek_v41.common.ops.indexer_k_store import (
            indexer_k_norm_rope_store,
        )

        indexer_metadata = _metadata_for(attn_metadata, self.k_cache.prefix)
        assert indexer_metadata is not None, (
            "PCP indexer-K materialization requires the indexer metadata"
        )
        assert self.owns_k
        # Rows at non-boundary tokens hold garbage latent and are skipped by
        # the store kernel's slot check.
        k_pre, _ = self.wk(latent)
        indexer_k_norm_rope_store(
            k_pre,
            _global_positions(layout),
            rotary_emb.cos_sin_cache,
            self.k_norm.weight,
            self.k_norm.variance_epsilon,
            self.k_cache.kv_cache,
            _global_slots(indexer_metadata, layout),
            self.compress_ratio,
            self.use_fp4_kv,
        )

    def hcu_fused_qnorm_rope_kv_insert(
        self: Any,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        attn_metadata: object,
    ) -> torch.Tensor:
        layout = _step_layout(attn_metadata)
        if layout is None or not isinstance(attn_metadata, dict):
            return original_insert_qkv(self, q, kv, positions, attn_metadata)
        swa_metadata = _metadata_for(attn_metadata, self.swa_cache_layer.prefix)
        assert swa_metadata is not None, (
            "PCP SWA materialization requires the sliding-window metadata"
        )
        cache = self.swa_cache_layer.kv_cache
        _packed_swa_cache(cache)
        # Q stays rank-local: the empty slot mapping keeps this launch off the
        # cache, and the KV half below writes every rank's rows.  The quant op
        # takes the cache as a block-contiguous 2D view, exactly as upstream.
        cache_2d = cache.view(cache.shape[0], -1)
        pad_to = (
            0
            if self.accepts_unnormed_unroped_query
            and self.n_local_heads == self.padded_heads
            else self.padded_heads
        )
        q_padded = torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
            q,
            kv,
            cache_2d,
            swa_metadata.slot_mapping.new_empty((0,)),
            positions,
            self.rotary_emb.cos_sin_cache,
            pad_to,
            self.eps,
            swa_metadata.block_size,
            False,  # apply_q_norm: qr is normed before wq_b
            self.kv_mxfp8,
            not self.accepts_unnormed_unroped_query,  # apply_q_rope
            self.accepts_unnormed_unroped_query,  # is_q_interleaved
        )
        torch.ops._C.fused_deepseek_v4_kv_rope_insert(
            _global_rows(kv, layout),
            cache,
            _global_slots(swa_metadata, layout),
            _global_positions(layout),
            self.rotary_emb.cos_sin_cache,
            swa_metadata.block_size,
            None,
            self.kv_mxfp8,
        )
        return q if pad_to == 0 else q_padded

    for function in (
        hcu_compressor_build,
        hcu_prepare_and_attn,
        hcu_compressor_forward,
        hcu_insert_cache,
        hcu_produce_k,
        hcu_fused_qnorm_rope_kv_insert,
    ):
        setattr(function, "_vllm_hcu_pcp_wrapper", True)
    setattr(module, "_vllm_hcu_pcp_original_prepare_and_attn", original_prepare)
    setattr(
        module, "_vllm_hcu_pcp_original_compressor_build", original_build
    )
    setattr(
        module,
        "_vllm_hcu_pcp_original_compressor_forward",
        original_forward,
    )
    setattr(
        module,
        "_vllm_hcu_pcp_original_compressor_insert_cache",
        original_insert,
    )
    setattr(module, "_vllm_hcu_pcp_original_indexer_produce_k", original_produce_k)
    setattr(module, "_vllm_hcu_pcp_original_insert", original_insert_qkv)
    setattr(builder_cls, "build", hcu_compressor_build)
    setattr(attention_cls, "_prepare_and_attn", hcu_prepare_and_attn)
    setattr(compressor_cls, "forward", hcu_compressor_forward)
    setattr(compressor_cls, "insert_cache", hcu_insert_cache)
    setattr(indexer_cls, "_produce_k", hcu_produce_k)
    setattr(attention_cls, "_fused_qnorm_rope_kv_insert", hcu_fused_qnorm_rope_kv_insert)
    setattr(module, _CLASS_MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    from ._common import load_exact_module

    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = [
    "COMPRESSOR_MODULE",
    "PATCH_ID",
    "TARGET_MODULE",
    "apply",
    "apply_to_module",
]
