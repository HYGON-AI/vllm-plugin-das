# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Decoder-side SWA bounded replay: running the replay layers on their batch.

Port of upstream ``vllm/models/deepseek_v41/decoder_replay_layers.py``
(#58132 + #59039 eager + #59532 CUDA-graph). PCP late-tail follows SGLang
bcac6dc: DualChunk restore → interleaved share → run → gather → scatter.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from vllm.forward_context import (
    ForwardContext,
    get_forward_context,
    override_forward_context,
)
from vllm.model_executor.layers.fused_moe.moe_output import MoEOutput

try:
    from vllm.forward_context import in_piecewise_cudagraph
except ImportError:  # pragma: no cover

    def in_piecewise_cudagraph() -> bool:
        return False


try:
    from vllm.compilation.breakable_cudagraph import eager_break_during_capture
except ImportError:  # pragma: no cover

    def eager_break_during_capture(fn):  # type: ignore[no-untyped-def]
        return fn


@dataclass
class ReplayBatch:
    """One forward's replay batch; ``run_graph`` is set when it runs in a graph."""

    rows: torch.Tensor
    forward_context: ForwardContext
    run_graph: Callable[..., tuple[torch.Tensor, ...]] | None = None
    num_padded: int | None = None
    # SGLang-aligned interleaved late-tail (bcac6dc).
    pcp_tail: Any | None = None
    pcp_meta: Any | None = None


class DecoderReplayLayers:
    """Runs the replay layers on the step's replay batch."""

    def __init__(
        self,
        window: int,
        run_layers: Callable[..., tuple[torch.Tensor, ...]],
        write_batch_kv: Callable[..., None],
        metadata_prefixes: set[str],
        first_swa_prefix: str,
    ) -> None:
        self.window = window
        self.run_layers = run_layers
        self.write_batch_kv = write_batch_kv
        self.metadata_prefixes = metadata_prefixes
        self.first_swa_prefix = first_swa_prefix
        self.replay_batch: ReplayBatch | None = None
        self.trim_threshold: int | None = None
        self._break_outputs: list[torch.Tensor] | None = None

    def __call__(
        self, hidden_states: torch.Tensor | MoEOutput, *states: torch.Tensor | None
    ) -> tuple[torch.Tensor, ...]:
        if self.uses_replay_batch():
            self.write_batch_kv(hidden_states, *states)
            if in_piecewise_cudagraph():
                return self._run_in_graph_break(hidden_states, *states)
        return self._run(hidden_states, *states)

    def uses_replay_batch(self) -> bool:
        if in_piecewise_cudagraph():
            batch_descriptor = get_forward_context().batch_descriptor
            assert batch_descriptor is not None
            return self.graph_uses_replay_batch(batch_descriptor.num_tokens)
        return self.replay_batch is not None

    def graph_uses_replay_batch(self, num_tokens: int) -> bool:
        return self.trim_threshold is not None and num_tokens >= self.trim_threshold

    @eager_break_during_capture
    def _run_in_graph_break(
        self, hidden_states: torch.Tensor | MoEOutput, *states: torch.Tensor | None
    ) -> tuple[torch.Tensor, ...]:
        outputs = self._run(hidden_states, *states)
        if self._break_outputs is None:
            self._break_outputs = [torch.empty_like(out) for out in outputs]
        return tuple(
            buf[: out.shape[0]].copy_(out)
            for buf, out in zip(self._break_outputs, outputs)
        )

    def _run(
        self, hidden_states: torch.Tensor | MoEOutput, *states: torch.Tensor | None
    ) -> tuple[torch.Tensor, ...]:
        replay_batch = self.replay_batch
        if replay_batch is None:
            return self.run_layers(hidden_states, *states)
        assert isinstance(hidden_states, torch.Tensor)
        if replay_batch.pcp_tail is not None:
            return self._run_pcp_interleaved_tail(hidden_states, *states)

        rows = replay_batch.rows
        inputs = (hidden_states, *states)
        with override_forward_context(replay_batch.forward_context):
            if replay_batch.run_graph is not None:
                row_outputs = replay_batch.run_graph(rows, inputs)
            else:
                row_inputs = [
                    None if t is None else t.index_select(0, rows) for t in inputs
                ]
                if replay_batch.num_padded is not None:
                    padding = replay_batch.num_padded - rows.numel()
                    row_inputs = [
                        None
                        if t is None
                        else torch.cat((t, t.new_zeros((padding, *t.shape[1:]))))
                        for t in row_inputs
                    ]
                row_outputs = self.run_layers(*row_inputs)
        num_tokens = hidden_states.shape[0]
        return tuple(
            out.new_zeros((num_tokens, *out.shape[1:])).index_copy_(
                0, rows, out[: rows.numel()]
            )
            for out in row_outputs
        )

    def _run_pcp_interleaved_tail(
        self, hidden_states: torch.Tensor, *states: torch.Tensor | None
    ) -> tuple[torch.Tensor, ...]:
        """Gather the tail, run its logical PCP layout, and restore local rows."""
        from vllm.distributed.parallel_state import get_pcp_group
        from vllm_hcu.model_executor.layers.attention.pcp import (
            replicated_mtp_batch_scope,
        )
        from .late_layer_tail_cp import (
            index_copy_masked,
            nonzero_static_flat,
            pack_local_tail_rows,
            unpack_tail_from_padded_ranks,
        )

        replay_batch = self.replay_batch
        assert replay_batch is not None and replay_batch.pcp_tail is not None
        plan = replay_batch.pcp_tail
        meta = replay_batch.pcp_meta
        assert meta is not None

        local_num = int(meta.pcp_local_num_tokens)
        group = get_pcp_group()
        rank = int(group.rank_in_group)

        # DualChunk owner of every global-tail row. Gather only the tail
        # rows (O(window)), never the full sequence.
        #
        # SGLang HCU pattern: ``nonzero_static(..., size=known)`` instead of
        # dynamic ``nonzero`` / bool-index (those DtoH-sync the compute stream).
        restore_idx = meta.pcp_restore_idx
        g_tail = plan.global_token_indices
        num_tail = int(g_tail.shape[0])
        flats = restore_idx.to(dtype=torch.long).index_select(
            0, g_tail.to(dtype=torch.long)
        )
        owner = torch.div(flats, local_num, rounding_mode="floor")
        on_rank = owner == rank
        flat_lo = rank * local_num
        # size=num_tail is a static upper bound (same role as SGLang page capacity).
        tail_rows_on_rank = nonzero_static_flat(on_rank, size=num_tail, fill_value=-1)
        safe_tail = tail_rows_on_rank.clamp_min(0)
        local_rows = flats.index_select(0, safe_tail) - flat_lo
        tail_arange = torch.arange(num_tail, device=g_tail.device)

        def to_local_tail(local_tensor: torch.Tensor) -> torch.Tensor:
            n_local = int(local_tensor.shape[0])
            safe_local = local_rows.clamp(0, max(n_local - 1, 0))
            vals = local_tensor.index_select(0, safe_local)
            mine = index_copy_masked(
                local_tensor.new_zeros((num_tail, *local_tensor.shape[1:])),
                tail_rows_on_rank,
                vals,
            )
            gathered = group.all_gather(mine, dim=0).view(
                group.world_size, num_tail, *local_tensor.shape[1:]
            )
            return pack_local_tail_rows(gathered[owner, tail_arange], plan)

        local_tail = to_local_tail(hidden_states)
        state_tails = [None if t is None else to_local_tail(t) for t in states]

        # Piecewise dispatch may round ``num_padded`` above the packed tail
        # (``plan.max_local``). Metadata already spans that size; pad
        # activations to match. PCP interleaved inputs are already gathered,
        # so do not use ``run_graph`` (it index_selects DualChunk-local rows).
        packed_rows = int(local_tail.shape[0])
        num_padded = replay_batch.num_padded
        if num_padded is not None and int(num_padded) > packed_rows:
            pad_n = int(num_padded) - packed_rows

            def _pad_rows(t: torch.Tensor) -> torch.Tensor:
                return torch.cat(
                    (t, t.new_zeros((pad_n, *t.shape[1:]))), dim=0
                )

            local_tail = _pad_rows(local_tail)
            state_tails = [
                None if t is None else _pad_rows(t) for t in state_tails
            ]

        with (
            override_forward_context(replay_batch.forward_context),
            replicated_mtp_batch_scope(),
        ):
            row_outputs = self.run_layers(local_tail, *state_tails)

        def scatter_out(local_out: torch.Tensor) -> torch.Tensor:
            # Drop graph-size pad rows before DualChunk scatter / unpack.
            local_out = local_out[:packed_rows]
            if plan.pcp_size == 1:
                tail_out = local_out
            else:
                gathered = group.all_gather(local_out.contiguous(), dim=0)
                tail_out = unpack_tail_from_padded_ranks(gathered, plan)
            n_out = int(hidden_states.shape[0])
            # Invalid nonzero_static pads used safe_tail(=0) for local_rows; force
            # those destinations to -1 so index_copy_masked writes them to the sink.
            dest = torch.where(
                tail_rows_on_rank >= 0, local_rows, local_rows.new_full((), -1)
            )
            vals = tail_out.index_select(0, safe_tail)
            return index_copy_masked(
                local_out.new_zeros((n_out, *local_out.shape[1:])), dest, vals
            )

        return tuple(scatter_out(out) for out in row_outputs)


__all__ = ["DecoderReplayLayers", "ReplayBatch"]
