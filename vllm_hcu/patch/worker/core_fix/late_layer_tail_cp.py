# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""SGLang-aligned late-layer SWA tail layout under PCP (bcac6dc).

SGLang's decoder SWA bounded replay does **not** keep DualChunk/contiguous
ownership into the late layers. It:

1. Takes each request's last ``tail_len`` extend tokens (``late_layer_tail_layout``).
2. Re-owns those tokens with interleaved CP: ``owner = token_index % cp_size``
   (``_late_layer_tail_cp_layout``).
3. Pads every rank to ``max(local counts)`` and builds a gather index so
   ``cp_gather_after_forward`` can restore global tail order.

HCU PCP uses DualChunkSwap for early layers. Backends that require contiguous
query positions can use logical CP width one to replicate the complete tail,
while retaining physical PCP ownership for the early layers and final scatter.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


def late_layer_tail_layout(
    *,
    extend_lens_cpu: list[int],
    seq_lens_cpu: list[int],
    tail_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, list[int], torch.Tensor]:
    """Global extend indices for each request's last ``min(tail_len, extend)`` tokens.

    Returns ``(token_indices, per_request_tail_lens, per_row_window_floor)``.
    Mirrors SGLang ``late_layer_tail_layout``.
    """
    tail_lens_cpu = [min(tail_len, n) for n in extend_lens_cpu]
    if len(extend_lens_cpu) == 1:
        n, t, s = extend_lens_cpu[0], tail_lens_cpu[0], seq_lens_cpu[0]
        floor = torch.full((t,), s - t, dtype=torch.int32, device=device)
        return torch.arange(n - t, n, device=device, dtype=torch.long), tail_lens_cpu, floor

    lens = torch.tensor(
        [extend_lens_cpu, tail_lens_cpu, seq_lens_cpu],
        device=device,
        dtype=torch.long,
    )
    extend_lens, tail_lens, seq_lens = lens[0], lens[1], lens[2]
    total = int(sum(tail_lens_cpu))
    req = torch.repeat_interleave(
        torch.arange(len(tail_lens_cpu), device=device),
        tail_lens,
        output_size=total,
    )
    offs = (
        torch.arange(total, device=device)
        - (torch.cumsum(tail_lens, 0) - tail_lens)[req]
    )
    token_indices = (torch.cumsum(extend_lens, 0) - tail_lens)[req] + offs
    floor = (seq_lens - tail_lens)[req].to(torch.int32)
    return token_indices, tail_lens_cpu, floor


@dataclass
class LateLayerTailCPPlan:
    """Per-rank interleaved share of the global SWA tail (SGLang bcac6dc)."""

    # Global extend/token indices of the full tail (length = sum(tail_lens)).
    global_token_indices: torch.Tensor
    # Rows of ``global_token_indices`` owned by this rank (no pad).
    local_tail_rows: torch.Tensor
    # Equal-width pad so all ranks share ``max_local`` for collectives.
    max_local: int
    pad_rows: int
    # Scatter local padded outputs back into global-tail order:
    # ``gathered[gather_index]`` places each local slot into the flat tail.
    gather_index: torch.Tensor
    # Absolute positions / window floors for this rank's real rows.
    local_positions: torch.Tensor
    window_floors: torch.Tensor
    # Per-request tail lens (global).
    tail_lens_cpu: list[int]
    pcp_rank: int
    pcp_size: int

    @property
    def num_real(self) -> int:
        return int(self.local_tail_rows.numel())


def nonzero_static_flat(
    mask: torch.Tensor, *, size: int, fill_value: int = -1
) -> torch.Tensor:
    """Fixed-size nonzero indices (SGLang HCU SWA page path).

    Dynamic ``torch.nonzero`` changes output length and forces a DtoH sync on
    HCU. ``nonzero_static`` keeps a known ``size`` and pads with ``fill_value``.
    """
    out = torch.nonzero_static(
        mask.reshape(-1), size=int(size), fill_value=int(fill_value)
    )
    return out.reshape(-1)


def index_copy_masked(
    dest: torch.Tensor,
    index: torch.Tensor,
    source: torch.Tensor,
    *,
    invalid_value: int = -1,
) -> torch.Tensor:
    """``dest.index_copy_(0, index, source)`` ignoring ``invalid_value`` slots.

    Uses a +1 sink row so padded ``nonzero_static`` fills never touch real rows
    and never need a host-side length.
    """
    assert index.shape[0] == source.shape[0]
    sink = dest.shape[0]
    ext = dest.new_zeros((sink + 1, *dest.shape[1:]))
    ext[:sink].copy_(dest)
    dest_idx = torch.where(
        index != int(invalid_value), index, index.new_full((), sink)
    )
    valid = (index != int(invalid_value)).view(-1, *([1] * (source.ndim - 1)))
    ext.index_copy_(0, dest_idx, torch.where(valid, source, source.new_zeros(())))
    return ext[:sink]


def late_layer_tail_cp_layout(
    *,
    token_indices: torch.Tensor,
    tail_lens: torch.Tensor,
    positions: torch.Tensor,
    window_floors: torch.Tensor,
    pcp_rank: int,
    pcp_size: int,
    batch_size: int,
) -> LateLayerTailCPPlan:
    """Interleave-own the global tail: ``owner = token_indices % pcp_size``.

    Mirrors SGLang ``DeepseekV4AttnBackend._late_layer_tail_cp_layout``.
    ``positions`` / ``window_floors`` are indexed by ``token_indices`` order
    (same length as ``token_indices``).

    Rank counts come from a host ``bincount`` (metadata / prepare). Local rows
    use ``nonzero_static`` with that known capacity — same pattern as SGLang
    HCU SWA page selection — so the device path never runs dynamic ``nonzero``.
    """
    tail_lens_cpu = (
        [int(x) for x in tail_lens.tolist()]
        if isinstance(tail_lens, torch.Tensor)
        else [int(x) for x in tail_lens]
    )
    if pcp_size <= 1:
        n = int(token_indices.shape[0])
        return LateLayerTailCPPlan(
            global_token_indices=token_indices,
            local_tail_rows=torch.arange(n, device=token_indices.device),
            max_local=max(n, 1),
            pad_rows=0,
            gather_index=torch.arange(n, device=token_indices.device),
            local_positions=positions,
            window_floors=window_floors,
            tail_lens_cpu=tail_lens_cpu,
            pcp_rank=0,
            pcp_size=1,
        )

    device = token_indices.device
    total = int(token_indices.shape[0])
    owner_rank = token_indices.to(dtype=torch.long) % int(pcp_size)
    # Host counts only (prepare-time); feeds nonzero_static size like SGLang.
    counts = np.bincount(
        owner_rank.detach().to("cpu").numpy(), minlength=int(pcp_size)
    ).tolist()
    max_local = max(max(counts), 1)
    order = torch.argsort(owner_rank, stable=True)
    rank_starts = torch.tensor(
        [sum(counts[:r]) for r in range(pcp_size)],
        dtype=torch.int64,
        device=device,
    )
    slot = torch.empty_like(owner_rank)
    slot[order] = torch.arange(total, device=device) - rank_starts[owner_rank[order]]
    gather_index = owner_rank * max_local + slot

    # Fixed-size nonzero; trim pad (-1) for the plan's "real rows" view.
    local_tail_rows_pad = nonzero_static_flat(
        owner_rank == int(pcp_rank), size=max_local, fill_value=-1
    )
    local_tail_rows = local_tail_rows_pad[local_tail_rows_pad >= 0]
    pad_rows = max_local - counts[pcp_rank]
    local_positions = (
        positions.index_select(0, local_tail_rows)
        if local_tail_rows.numel()
        else positions[:0]
    )
    local_floors = (
        window_floors.index_select(0, local_tail_rows)
        if local_tail_rows.numel()
        else window_floors[:0]
    )
    if pad_rows:
        local_positions = torch.cat(
            [local_positions, local_positions.new_zeros(pad_rows)]
        )
        local_floors = torch.cat([local_floors, local_floors.new_zeros(pad_rows)])

    return LateLayerTailCPPlan(
        global_token_indices=token_indices,
        local_tail_rows=local_tail_rows,
        max_local=max_local,
        pad_rows=pad_rows,
        gather_index=gather_index,
        local_positions=local_positions,
        window_floors=local_floors,
        tail_lens_cpu=tail_lens_cpu,
        pcp_rank=pcp_rank,
        pcp_size=pcp_size,
    )


def pack_local_tail_rows(
    global_tail: torch.Tensor,
    plan: LateLayerTailCPPlan,
) -> torch.Tensor:
    """Select + pad this rank's interleaved share of a ``[tail, ...]`` tensor."""
    if plan.local_tail_rows.numel() == 0:
        real = global_tail.new_empty((0, *global_tail.shape[1:]))
    else:
        real = global_tail.index_select(0, plan.local_tail_rows)
    if plan.pad_rows <= 0:
        return real
    return torch.cat(
        [real, real.new_zeros((plan.pad_rows, *real.shape[1:]))],
        dim=0,
    )


def unpack_tail_from_padded_ranks(
    gathered_rank_major: torch.Tensor,
    plan: LateLayerTailCPPlan,
) -> torch.Tensor:
    """Rebuild ``[tail, ...]`` in global order from rank-major all_gather output.

    ``gathered_rank_major`` is ``[pcp_size * max_local, ...]`` (rank blocks).
    ``gather_index[g]`` is the flat slot for global-tail row ``g``.
    """
    assert gathered_rank_major.shape[0] == plan.pcp_size * plan.max_local
    return gathered_rank_major.index_select(0, plan.gather_index)


__all__ = [
    "LateLayerTailCPPlan",
    "index_copy_masked",
    "late_layer_tail_layout",
    "late_layer_tail_cp_layout",
    "nonzero_static_flat",
    "pack_local_tail_rows",
    "unpack_tail_from_padded_ranks",
]
