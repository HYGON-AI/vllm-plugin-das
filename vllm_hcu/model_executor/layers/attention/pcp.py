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

import torch

from vllm.distributed.parallel_state import get_pcp_group


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

    num_prefills = getattr(metadata, "num_prefills", None)
    if num_prefills == 0:
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
    num_prefills = getattr(metadata, "num_prefills", None)
    num_decode_tokens = getattr(metadata, "num_decode_tokens", None)
    if num_decode_tokens is None:
        return kv_c_normed, k_pe, slot_mapping
    if num_prefills == 0:
        cache_slot_mapping = _decode_only_slot_mapping(
            slot_mapping,
            kv_c_normed.shape[0],
            metadata,
        )
        return kv_c_normed, k_pe, cache_slot_mapping
    assert kv_c_normed.shape[0] == k_pe.shape[0], (
        "PCP MLA latent KV and RoPE K must have the same token dimension"
    )

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
    num_prefills = getattr(metadata, "num_prefills", None)
    num_decode_tokens = getattr(metadata, "num_decode_tokens", None)
    if num_decode_tokens is None:
        return k, slot_mapping
    if num_prefills == 0:
        cache_slot_mapping = _decode_only_slot_mapping(
            slot_mapping,
            k.shape[0],
            metadata,
        )
        return k, cache_slot_mapping
    (cache_k,), cache_slot_mapping = _gather_prefill_cache_inputs(
        (k,),
        slot_mapping,
        metadata,
    )
    return cache_k, cache_slot_mapping


__all__ = (
    "maybe_gather_indexer_k",
    "maybe_gather_mla_latent_cache_inputs",
)
