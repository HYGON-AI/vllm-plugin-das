# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""PCP cache-input gathers for HCU MLA and sparse indexers.

Adapted from ``vllm/model_executor/layers/attention/pcp.py`` in vLLM pull
request #46570, commit ``b6ff8a2f509cc7ac9c58176f5115a836aa1e08bd``.
The HCU adapter keeps the v0.25.1 eager call graph and consumes the
rank-ordered slot mappings produced by :class:`HcuPCPManager`.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from threading import local

import torch

from vllm.distributed.parallel_state import get_pcp_group


_REPLICATED_MTP_BATCH = ContextVar(
    "vllm_hcu_replicated_mtp_batch",
    default=False,
)
_REPLICATED_MTP_GRAPH_STATE = local()


@contextmanager
def replicated_mtp_batch_scope() -> Iterator[None]:
    """Run a restored global MTP batch without reapplying PCP attention.

    The scope and any compiled forward that consumes it must execute
    synchronously on the same worker thread.  The graph-safe mirror is
    thread-local and therefore must not cross an ``await`` or thread handoff.
    ``HcuGPUModelRunnerV2.sample_tokens`` satisfies this contract by entering
    the scope immediately around its synchronous ``super().sample_tokens``.
    """

    token = _REPLICATED_MTP_BATCH.set(True)
    previous_graph_depth = getattr(_REPLICATED_MTP_GRAPH_STATE, "depth", 0)
    _REPLICATED_MTP_GRAPH_STATE.depth = previous_graph_depth + 1
    try:
        yield
    finally:
        _REPLICATED_MTP_GRAPH_STATE.depth = previous_graph_depth
        _REPLICATED_MTP_BATCH.reset(token)


def in_replicated_mtp_batch() -> bool:
    # Dynamo cannot trace ContextVar.get(). Model-runner sampling scopes and
    # their compiled forwards execute synchronously on one worker thread, so
    # mirror only the nesting depth needed while Dynamo captures that graph.
    # Eager callers retain ContextVar's task-local semantics.
    if torch.compiler.is_compiling():
        return bool(getattr(_REPLICATED_MTP_GRAPH_STATE, "depth", 0))
    return bool(_REPLICATED_MTP_BATCH.get())


def effective_pcp_world_size(configured_world_size: int) -> int:
    """Return the logical PCP width for the current model invocation."""

    # TP/DCP-only execution is compiled as a full graph.  Avoid consulting the
    # dynamic MTP scope in that common case because Dynamo cannot trace
    # ContextVar.get().  A unitary PCP group is already unaffected by replicated
    # MTP execution, so the lookup cannot change the result.
    if configured_world_size == 1:
        return 1
    if in_replicated_mtp_batch():
        return 1
    return configured_world_size


def _pcp_world_size(metadata: object | None) -> int:
    if metadata is None:
        return 1
    world_size = int(getattr(metadata, "pcp_world_size", 1))
    assert world_size >= 1, f"invalid PCP world size {world_size}"
    return world_size


def _rank_slot_slice(
    slot_mapping: torch.Tensor,
    local_num_tokens: int,
    metadata: object,
    world_size: int,
    rank: int,
) -> torch.Tensor:
    assert slot_mapping.ndim == 1, (
        "PCP cache slot mapping must be one-dimensional, got "
        f"shape={tuple(slot_mapping.shape)}"
    )
    token_counts = getattr(metadata, "pcp_token_counts", None)
    if token_counts is None:
        expected_slots = world_size * local_num_tokens
        assert slot_mapping.shape[0] == expected_slots, (
            "PCP cache inputs require one equal-width slot segment per rank: "
            f"slots={slot_mapping.shape[0]}, local_tokens={local_num_tokens}, "
            f"world_size={world_size}"
        )
        start = rank * local_num_tokens
        return slot_mapping[start : start + local_num_tokens]

    counts = tuple(int(count) for count in token_counts)
    assert len(counts) == world_size, (
        "PCP token counts must contain one entry per rank: "
        f"counts={counts}, world_size={world_size}"
    )
    assert all(count >= 0 for count in counts), (
        f"PCP token counts must be non-negative, got {counts}"
    )
    assert counts[rank] == local_num_tokens, (
        "PCP local tensor length does not match its token count: "
        f"rank={rank}, tensor={local_num_tokens}, count={counts[rank]}"
    )
    assert slot_mapping.shape[0] == sum(counts), (
        "PCP gathered slot mapping length does not match token counts: "
        f"slots={slot_mapping.shape[0]}, counts={counts}"
    )
    start = sum(counts[:rank])
    return slot_mapping[start : start + local_num_tokens]


def _decode_only_slot_mapping(
    slot_mapping: torch.Tensor,
    local_num_tokens: int,
    metadata: object,
) -> torch.Tensor:
    """Match rank-zero decode slots to local replicated cache tensors."""

    num_decode_tokens = int(getattr(metadata, "num_decode_tokens"))
    assert slot_mapping.ndim == 1, (
        "PCP cache slot mapping must be one-dimensional, got "
        f"shape={tuple(slot_mapping.shape)}"
    )
    assert 0 <= num_decode_tokens <= local_num_tokens, (
        "PCP decode token count is outside the local tensor: "
        f"decode={num_decode_tokens}, local={local_num_tokens}"
    )
    assert slot_mapping.shape[0] >= num_decode_tokens, (
        "PCP cache slot mapping is shorter than the decode segment: "
        f"slots={slot_mapping.shape[0]}, decode={num_decode_tokens}"
    )
    if slot_mapping.shape[0] == num_decode_tokens:
        return slot_mapping
    return slot_mapping[:num_decode_tokens]


def _gather_prefill_cache_inputs(
    tensors: tuple[torch.Tensor, ...],
    slot_mapping: torch.Tensor,
    metadata: object | None,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    """Keep replicated decode writes local and gather partitioned prefills."""

    world_size = _pcp_world_size(metadata)
    if world_size == 1 or metadata is None:
        return tensors, slot_mapping

    num_decode_tokens = getattr(metadata, "num_decode_tokens", None)
    if num_decode_tokens is None:
        return tensors, slot_mapping
    num_decode_tokens = int(num_decode_tokens)

    assert tensors, "PCP cache gather requires at least one tensor"
    local_num_tokens = tensors[0].shape[0]
    assert all(tensor.shape[0] == local_num_tokens for tensor in tensors), (
        "PCP cache tensors must have the same token dimension"
    )
    assert 0 <= num_decode_tokens <= local_num_tokens, (
        "PCP decode token count is outside the local tensor: "
        f"decode={num_decode_tokens}, local={local_num_tokens}"
    )
    if num_decode_tokens == local_num_tokens:
        return tensors, _decode_only_slot_mapping(
            slot_mapping,
            local_num_tokens,
            metadata,
        )

    pcp_group = get_pcp_group()
    assert int(pcp_group.world_size) == world_size, (
        "PCP metadata/process-group size mismatch: "
        f"metadata={world_size}, group={pcp_group.world_size}"
    )
    rank = int(pcp_group.rank_in_group)
    assert 0 <= rank < world_size, f"invalid PCP rank {rank}/{world_size}"
    local_slots = _rank_slot_slice(
        slot_mapping,
        local_num_tokens,
        metadata,
        world_size,
        rank,
    )

    gathered_prefills = tuple(
        pcp_group.all_gather(
            tensor[num_decode_tokens:].contiguous(),
            dim=0,
        )
        for tensor in tensors
    )
    gathered_prefill_slots = pcp_group.all_gather(
        local_slots[num_decode_tokens:].contiguous(),
        dim=0,
    )
    if num_decode_tokens == 0:
        return gathered_prefills, gathered_prefill_slots

    cache_inputs = tuple(
        torch.cat((tensor[:num_decode_tokens], gathered_prefill), dim=0)
        for tensor, gathered_prefill in zip(tensors, gathered_prefills)
    )
    # Rank zero owns the single replicated-decode write in HcuPCPManager's
    # rank-ordered slot mapping. Every rank uses that same leading segment.
    cache_slot_mapping = torch.cat(
        (
            slot_mapping[:num_decode_tokens],
            gathered_prefill_slots,
        ),
        dim=0,
    )
    return cache_inputs, cache_slot_mapping


def maybe_gather_mla_latent_cache_inputs(
    kv_c_normed: torch.Tensor,
    k_pe: torch.Tensor,
    slot_mapping: torch.Tensor,
    metadata: object | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gather MLA latent KV, RoPE K, and matching slots for cache writes."""

    world_size = _pcp_world_size(metadata)
    if world_size == 1 or metadata is None:
        return kv_c_normed, k_pe, slot_mapping
    num_decode_tokens = getattr(metadata, "num_decode_tokens", None)
    if num_decode_tokens is None:
        return kv_c_normed, k_pe, slot_mapping
    assert kv_c_normed.shape[0] == k_pe.shape[0], (
        "PCP MLA latent KV and RoPE K must have the same token dimension"
    )
    if int(num_decode_tokens) == kv_c_normed.shape[0]:
        cache_slot_mapping = _decode_only_slot_mapping(
            slot_mapping,
            kv_c_normed.shape[0],
            metadata,
        )
        return kv_c_normed, k_pe, cache_slot_mapping

    num_tokens = kv_c_normed.shape[0]
    k_pe_flat = k_pe.reshape(num_tokens, -1)
    (cache_kv_c, cache_k_pe_flat), cache_slot_mapping = (
        _gather_prefill_cache_inputs(
            (kv_c_normed, k_pe_flat),
            slot_mapping,
            metadata,
        )
    )
    cache_k_pe = cache_k_pe_flat.view(-1, *k_pe.shape[1:])
    return cache_kv_c, cache_k_pe, cache_slot_mapping


def maybe_gather_indexer_k(
    k: torch.Tensor,
    slot_mapping: torch.Tensor,
    metadata: object | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather sparse-indexer K and matching slots for its cache write."""

    world_size = _pcp_world_size(metadata)
    if world_size == 1 or metadata is None:
        return k, slot_mapping
    num_decode_tokens = getattr(metadata, "num_decode_tokens", None)
    if num_decode_tokens is None:
        return k, slot_mapping
    (cache_k,), cache_slot_mapping = _gather_prefill_cache_inputs(
        (k,),
        slot_mapping,
        metadata,
    )
    return cache_k, cache_slot_mapping


__all__ = (
    "effective_pcp_world_size",
    "in_replicated_mtp_batch",
    "maybe_gather_indexer_k",
    "maybe_gather_mla_latent_cache_inputs",
    "replicated_mtp_batch_scope",
)
