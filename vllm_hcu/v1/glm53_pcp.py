# SPDX-License-Identifier: Apache-2.0
"""Eager GLM5Next PCP: sharded MLA/MoE with ordered recurrent state updates."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import lru_cache
from types import MethodType

import numpy as np
import torch

from vllm.logger import init_logger
from vllm_hcu.model_executor.layers.attention.pcp import (
    in_replicated_mtp_batch,
    logical_pcp_metadata_scope,
    replicated_mtp_batch_scope,
)
from vllm_hcu.patch.worker.framework_opt.patch_pcp_model_state import (
    _attach_pcp_cache_ownership,
    _attach_pcp_plan,
)

logger = init_logger(__name__)


@dataclass
class Glm53PCPBatch:
    manager: object
    local_batch: object
    local_indices: torch.Tensor
    _query_rows: dict = field(default_factory=dict, repr=False)

    @property
    def global_batch(self):
        return self.manager._global_batch

    def gather(self, value: torch.Tensor) -> torch.Tensor:
        return self.manager.restore_hidden_states(value)

    def localize(self, value: torch.Tensor) -> torch.Tensor:
        return value.index_select(0, self.local_indices)

    def local_query_rows(self, start: int, end: int):
        """Owned rows in global ``[start, end)`` (minus start) and their
        positions in the rank-local layout. Built from the CPU layout once per
        step, so every layer reuses the device indices without a sync."""
        key = (int(start), int(end))
        if key not in self._query_rows:
            segments = self.manager._local_segments
            owned = np.concatenate([np.arange(seg.global_slice.start, seg.global_slice.stop)
                                    for seg in segments])
            local = np.concatenate([np.arange(seg.local_slice.start,
                                              seg.local_slice.start + seg.num_actual_tokens)
                                    for seg in segments])
            mask = (owned >= key[0]) & (owned < key[1])
            device = self.local_indices.device
            self._query_rows[key] = (torch.from_numpy(owned[mask] - key[0]).to(device),
                                     torch.from_numpy(local[mask]).to(device))
        return self._query_rows[key]


_BATCH: ContextVar[Glm53PCPBatch | None] = ContextVar(
    "hcu_glm53_pcp_batch", default=None
)


def _current_pcp_batch():
    # Draft prefill/decode consumes globally restored rows. Never gather or
    # localize them again using a target-model partition left in this context.
    return None if in_replicated_mtp_batch() else _BATCH.get()


@contextmanager
def glm53_pcp_step_scope():
    token = _BATCH.set(None)
    try:
        yield
    finally:
        _BATCH.reset(token)


def bind_glm53_pcp_batch(manager, local_batch) -> None:
    width = local_batch.num_tokens_after_padding
    start = manager.pcp_rank * manager._padded_num_tokens
    indices = manager._padded_gather_idx[start : start + width]
    if indices.numel() != width:
        raise RuntimeError("GLM5Next PCP local/global token map is incomplete")
    _BATCH.set(Glm53PCPBatch(manager, local_batch, indices))


def _replicated_group(group) -> bool:
    from vllm.v1.kv_cache_interface import MambaSpec

    return isinstance(group.kv_cache_spec, MambaSpec) or group.backend.get_name() in (
        "DEEPSEEK_V32_INDEXER", "KPOOL_TAIL"
    )


def _prepare_attn(
    self, input_batch, cudagraph_mode, block_tables, slot_mappings,
    attn_groups, kv_cache_config, for_capture=False, ubatch_idx=0,
):
    original = self._hcu_glm53_original_prepare_attn
    batch = _current_pcp_batch()
    if batch is None:
        # Profiling uses replicated dummy inputs, with no real cache writes.
        with replicated_mtp_batch_scope(), logical_pcp_metadata_scope(1):
            result = original(
                input_batch, cudagraph_mode, block_tables, slot_mappings,
                attn_groups, kv_cache_config, for_capture, ubatch_idx,
            )
        _attach_pcp_cache_ownership(result, False)
        return result

    manager = batch.manager
    replicate_all = getattr(self, "_hcu_pcp_replicate_all_groups", False)
    local_groups = [
        [group for group in groups if not (replicate_all or _replicated_group(group))]
        for groups in attn_groups
    ]
    global_groups = [
        [group for group in groups if replicate_all or _replicated_group(group)]
        for groups in attn_groups
    ]
    logical_width = manager.pcp_size if manager._global_has_prefill else 1
    with logical_pcp_metadata_scope(logical_width):
        local_metadata = original(
            input_batch, cudagraph_mode, block_tables, slot_mappings,
            local_groups, kv_cache_config, for_capture, ubatch_idx,
        )
    _attach_pcp_plan(local_metadata, getattr(input_batch, "_vllm_hcu_pcp_plan", None))
    _attach_pcp_cache_ownership(
        local_metadata,
        manager._global_has_prefill,
        getattr(input_batch, "_vllm_hcu_pcp_replicated_token_mask", None),
        getattr(input_batch, "_vllm_hcu_pcp_replicated_slot_indices", None),
    )
    global_tables, global_slots = manager.prepare_global_attn()
    with replicated_mtp_batch_scope(), logical_pcp_metadata_scope(1):
        global_metadata = original(
            batch.global_batch, cudagraph_mode, global_tables, global_slots,
            global_groups, kv_cache_config, for_capture, ubatch_idx,
        )
    _attach_pcp_cache_ownership(global_metadata, False)
    overlap = local_metadata.keys() & global_metadata.keys()
    if overlap:
        raise RuntimeError(f"GLM5Next PCP metadata ownership overlaps: {overlap}")
    return {**local_metadata, **global_metadata}


def _preprocess_state(
    self, input_batch, block_tables, kv_cache_config, num_computed_tokens,
):
    batch = _current_pcp_batch()
    if batch is not None:
        input_batch = batch.global_batch
        block_tables, _ = batch.manager.prepare_global_attn()
    return self._hcu_glm53_original_preprocess_state(
        input_batch, block_tables, kv_cache_config, num_computed_tokens,
    )


def _kda_forward(self, hidden_states, positions):
    batch = _current_pcp_batch()
    original = self._hcu_glm53_original_forward
    if batch is None:
        return original(hidden_states, positions)
    # DualChunkSwap gives each rank disjoint intervals. Reconstruct their
    # causal order before updating the replicated convolution/recurrent state.
    full_hidden = batch.gather(hidden_states)
    full_positions = batch.global_batch.positions[:full_hidden.shape[0]]
    with replicated_mtp_batch_scope():
        full_output = original(full_hidden, full_positions)
    return batch.localize(full_output)


def _indexer_forward(self, hidden_states, qr, positions, rotary_emb):
    batch = _current_pcp_batch()
    original = self._hcu_glm53_original_forward
    if batch is None:
        return original(hidden_states, qr, positions, rotary_emb)
    # A compression pool can straddle PCP chunks. Update each full pool and
    # its tail exactly once per rank, then select this rank's query rows.
    full_hidden = batch.gather(hidden_states)
    full_qr = batch.gather(qr)
    full_positions = batch.global_batch.positions[:full_hidden.shape[0]]
    with replicated_mtp_batch_scope():
        result = original(full_hidden, full_qr, full_positions, rotary_emb)
    local_topk = batch.localize(result)
    self.topk_indices_buffer[:local_topk.shape[0]].copy_(local_topk)
    return self.topk_indices_buffer


@lru_cache(None)
def _replicated_backend(base):
    # PCP is provided by the surrounding GLM5Next gather/localize adapter;
    # the original backend receives ordinary, globally ordered metadata.
    class ReplicatedGlm53Backend(base):
        @classmethod
        def supports_pcp(cls):
            return True

    return ReplicatedGlm53Backend


def _install_replicated_backend(layer):
    if not hasattr(layer, "_hcu_glm53_original_backend"):
        layer._hcu_glm53_original_backend = layer.get_attn_backend
        backend = _replicated_backend(layer.get_attn_backend())
        layer.get_attn_backend = MethodType(lambda self: backend, layer)


def install_glm53_pcp(runner) -> None:
    from vllm.models.glm5next.nvidia.attention import Indexer, Glm5NextTailCache
    from vllm.models.glm5next.nvidia.kda import Glm5NextLinearAttention
    from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState

    state = runner.model_state
    if not isinstance(state, MambaHybridModelState):
        raise RuntimeError("GLM5Next PCP requires MambaHybridModelState")
    if not hasattr(state, "_hcu_glm53_original_prepare_attn"):
        state._hcu_glm53_original_prepare_attn = state.prepare_attn
        state._hcu_glm53_original_preprocess_state = state.preprocess_state
        state.prepare_attn = MethodType(_prepare_attn, state)
        state.preprocess_state = MethodType(_preprocess_state, state)

    counts = [0, 0]
    for layer in runner.model.modules():
        if isinstance(layer, Glm5NextLinearAttention):
            _install_replicated_backend(layer)
            wrapper, index = _kda_forward, 0
        elif isinstance(layer, Indexer):
            wrapper, index = _indexer_forward, 1
        elif isinstance(layer, Glm5NextTailCache):
            _install_replicated_backend(layer)
            continue
        else:
            continue
        counts[index] += 1
        if not hasattr(layer, "_hcu_glm53_original_forward"):
            layer._hcu_glm53_original_forward = layer.forward
            layer.forward = MethodType(wrapper, layer)
    if not all(counts):
        raise RuntimeError(f"GLM5Next PCP expected KDA and K-pool layers: {counts}")
    draft_model = getattr(getattr(runner, "speculator", None), "model", None)
    if draft_model is not None:
        # Draft layers also participate in the global backend compatibility
        # check, but their forwards run on a replicated global MTP batch.
        for layer in draft_model.modules():
            if isinstance(layer, (Glm5NextLinearAttention, Glm5NextTailCache)):
                _install_replicated_backend(layer)
    logger.info(
        "GLM5Next PCP eager baseline: %d replicated KDA layers, "
        "%d replicated K-pool indexers; MLA queries and MoE remain sharded",
        *counts,
    )


def install_deepseek_v4_pcp(runner) -> None:
    """Reuse the GLM5Next adapter with every DeepSeek V4 cache group global.

    V4 SWA, compressor and indexer caches all consume full KV on each rank,
    so their metadata is built from the original request rows.
    """
    state = runner.model_state
    if not hasattr(state, "_hcu_glm53_original_prepare_attn"):
        state._hcu_glm53_original_prepare_attn = state.prepare_attn
        state._hcu_pcp_replicate_all_groups = True
        state.prepare_attn = MethodType(_prepare_attn, state)
