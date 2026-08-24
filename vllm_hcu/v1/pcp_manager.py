# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""HCU MRV2 virtual-batch prefill context parallelism.

This is a narrow HCU backport of the DualChunkSwap batch-layout algorithm from
vLLM upstream PR #46570 at commit b6ff8a2f50. Runner wiring is intentionally
owned by the later integration task.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import torch

from vllm.distributed.parallel_state import get_pcp_group
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm_hcu.patch.platform.core_fix._common import PatchCompatibilityError
from vllm_hcu.patch.platform.core_fix.patch_vllm_config import (
    _validate_hcu_pcp_scope,
)


@dataclass(frozen=True)
class RankSegment:
    """Half-open token interval assigned to one PCP rank-local row."""

    start: int
    stop: int

    @property
    def num_tokens(self) -> int:
        return self.stop - self.start


@dataclass(frozen=True)
class _BatchSegment:
    global_req_idx: int
    global_slice: slice
    local_slice: slice
    padding_tokens: int = 0

    @property
    def num_actual_tokens(self) -> int:
        return self.global_slice.stop - self.global_slice.start

    @property
    def num_tokens(self) -> int:
        return self.num_actual_tokens + self.padding_tokens


class HcuPCPManager:
    """Build rank-local virtual rows while retaining the global step batch."""

    def __init__(
        self,
        vllm_config: object,
        device: torch.device,
        req_states: object,
        block_tables: object,
        *,
        pcp_group: object | None = None,
    ) -> None:
        parallel_config = vllm_config.parallel_config
        self.pcp_size = int(parallel_config.prefill_context_parallel_size)
        assert self.pcp_size > 1
        self.device = device
        self._pcp_group = pcp_group if pcp_group is not None else get_pcp_group()
        self.pcp_rank = int(self._pcp_group.rank_in_group)
        assert int(self._pcp_group.world_size) == self.pcp_size
        assert 0 <= self.pcp_rank < self.pcp_size

        self._req_states = req_states
        self._block_tables = block_tables
        self._global_batch: InputBatch | None = None
        self._hidden_restore_idx: torch.Tensor | None = None
        self._padded_gather_idx: torch.Tensor | None = None
        self._gathered_kv_write_mask: torch.Tensor | None = None

        scheduler_config = vllm_config.scheduler_config
        # DualChunkSwap has at most two real rows per request. HCU may need one
        # extra virtual row to equalize eager-attention collective shapes.
        self._max_local_reqs = 3 * int(scheduler_config.max_num_seqs)
        self._max_local_tokens = int(scheduler_config.max_num_batched_tokens)

        # Plugin-owned buffers must never alias the saved global InputBatch.
        self._idx_mapping = torch.empty(
            self._max_local_reqs, dtype=torch.int32, device=device
        )
        self._idx_mapping_np = np.empty(self._max_local_reqs, dtype=np.int32)
        self._num_scheduled_tokens = np.empty(
            self._max_local_reqs, dtype=np.int32
        )
        self._positions = torch.empty(
            self._max_local_tokens, dtype=torch.int64, device=device
        )
        self._seq_lens = torch.empty(
            self._max_local_reqs, dtype=torch.int32, device=device
        )
        self._seq_lens_i64 = torch.empty(
            self._max_local_reqs, dtype=torch.int64, device=device
        )
        self._query_start_loc = torch.empty(
            self._max_local_reqs + 1, dtype=torch.int32, device=device
        )
        self._query_start_loc_np = np.empty(
            self._max_local_reqs + 1, dtype=np.int32
        )
        self._input_ids = torch.empty(
            self._max_local_tokens, dtype=torch.int32, device=device
        )
        self._is_padding = torch.empty(
            self._max_local_tokens, dtype=torch.bool, device=device
        )
        self._logits_indices = torch.empty(
            self._max_local_tokens, dtype=torch.int64, device=device
        )
        self._cu_num_logits = torch.empty(
            self._max_local_reqs + 1, dtype=torch.int32, device=device
        )
        self._cu_num_logits_np = np.empty(
            self._max_local_reqs + 1, dtype=np.int32
        )
        self._is_prefilling_np = np.empty(self._max_local_reqs, dtype=np.bool_)

        input_block_tables = getattr(block_tables, "input_block_tables", ())
        self._local_block_tables = tuple(
            table.new_zeros((self._max_local_reqs, table.shape[1]))
            for table in input_block_tables
        )
        self._local_block_table_ptrs = (
            torch.tensor(
                [table.data_ptr() for table in self._local_block_tables],
                dtype=torch.uint64,
                device=device,
            )
            if self._local_block_tables
            else None
        )
        num_kv_groups = int(getattr(block_tables, "num_kv_cache_groups", 0))
        self._global_slot_mappings = (
            torch.empty(
                (num_kv_groups, self._max_local_tokens),
                dtype=torch.int64,
                device=device,
            )
            if num_kv_groups
            else None
        )
        self._gathered_slot_mappings = (
            torch.empty(
                (num_kv_groups, self._max_local_tokens * self.pcp_size),
                dtype=torch.int64,
                device=device,
            )
            if num_kv_groups
            else None
        )
        self._pad_slot_id = torch.tensor(
            PAD_SLOT_ID, dtype=torch.int64, device=device
        )

    @staticmethod
    def rank_segments(
        length: int, *, pcp_rank: int, pcp_size: int
    ) -> tuple[RankSegment, RankSegment]:
        """Return the two symmetric DualChunkSwap intervals for a rank."""

        if length < 0:
            raise ValueError("token length must be non-negative")
        if pcp_size <= 0 or not 0 <= pcp_rank < pcp_size:
            raise ValueError("invalid PCP rank or size")
        chunk_count = 2 * pcp_size
        chunk_size = (length + chunk_count - 1) // chunk_count

        def segment(chunk_idx: int) -> RankSegment:
            start = min(chunk_idx * chunk_size, length)
            return RankSegment(start, min(start + chunk_size, length))

        return segment(pcp_rank), segment(chunk_count - 1 - pcp_rank)

    @staticmethod
    def _reorder_segments(
        segments: list[_BatchSegment],
        num_computed_tokens: np.ndarray,
        is_prefilling: np.ndarray,
        query_start_loc_np: np.ndarray,
    ) -> list[_BatchSegment]:
        """Keep decode/continued-prefill rows before fresh-prefill rows."""

        def is_pure_prefill(segment: _BatchSegment) -> bool:
            req_idx = segment.global_req_idx
            local_offset = segment.global_slice.start - query_start_loc_np[req_idx]
            return bool(is_prefilling[req_idx]) and int(
                num_computed_tokens[req_idx] + local_offset
            ) == 0

        segments.sort(key=is_pure_prefill)
        result = []
        local_offset = 0
        for segment in segments:
            result.append(
                replace(
                    segment,
                    local_slice=slice(
                        local_offset, local_offset + segment.num_tokens
                    ),
                )
            )
            local_offset += segment.num_tokens
        return result

    def _segments_for_rank(
        self,
        rank: int,
        input_batch: InputBatch,
    ) -> list[_BatchSegment]:
        segments: list[_BatchSegment] = []
        local_offset = 0
        for req_idx, query_len_value in enumerate(
            input_batch.num_scheduled_tokens
        ):
            query_len = int(query_len_value)
            if query_len == 0:
                continue
            global_start = int(input_batch.query_start_loc_np[req_idx])
            if bool(input_batch.is_prefilling_np[req_idx]):
                rank_segments = self.rank_segments(
                    query_len, pcp_rank=rank, pcp_size=self.pcp_size
                )
            else:
                rank_segments = (RankSegment(0, query_len),)
            for rank_segment in rank_segments:
                if rank_segment.num_tokens == 0:
                    continue
                segment = _BatchSegment(
                    global_req_idx=req_idx,
                    global_slice=slice(
                        global_start + rank_segment.start,
                        global_start + rank_segment.stop,
                    ),
                    local_slice=slice(
                        local_offset, local_offset + rank_segment.num_tokens
                    ),
                )
                segments.append(segment)
                local_offset += rank_segment.num_tokens
        return self._reorder_segments(
            segments,
            input_batch.num_computed_tokens_np,
            input_batch.is_prefilling_np,
            input_batch.query_start_loc_np,
        )

    def _build_batch_layout(
        self, input_batch: InputBatch
    ) -> tuple[list[list[_BatchSegment]], list[int]]:
        segments_by_rank = [
            self._segments_for_rank(rank, input_batch)
            for rank in range(self.pcp_size)
        ]

        # The upstream PCP layout pads the model input tensor but keeps eager
        # attention metadata unpadded. HCU MLA/EP collectives derive their
        # token shapes from that metadata, so an uneven prefill can deadlock.
        # Add isolated virtual rows until each request contributes the same
        # row count and token width on every PCP rank. These rows remain
        # padding-only and therefore own no KV slots, logits, or output tokens.
        for req_idx, query_len_value in enumerate(
            input_batch.num_scheduled_tokens
        ):
            if (
                int(query_len_value) == 0
                or not bool(input_batch.is_prefilling_np[req_idx])
            ):
                continue
            req_segments_by_rank = [
                [
                    segment
                    for segment in rank_segments
                    if segment.global_req_idx == req_idx
                ]
                for rank_segments in segments_by_rank
            ]
            actual_tokens = [
                sum(segment.num_actual_tokens for segment in req_segments)
                for req_segments in req_segments_by_rank
            ]
            target_tokens = max(actual_tokens)
            if min(actual_tokens) == target_tokens:
                continue
            row_counts = [len(req_segments) for req_segments in req_segments_by_rank]
            target_rows = max(row_counts)
            if any(
                count < target_tokens and rows == target_rows
                for count, rows in zip(actual_tokens, row_counts)
            ):
                target_rows += 1
            global_start = int(input_batch.query_start_loc_np[req_idx])
            for rank, rank_segments in enumerate(segments_by_rank):
                missing_tokens = target_tokens - actual_tokens[rank]
                missing_rows = target_rows - row_counts[rank]
                assert missing_rows >= int(missing_tokens > 0)
                for row in range(missing_rows):
                    rank_segments.append(
                        _BatchSegment(
                            global_req_idx=req_idx,
                            global_slice=slice(global_start, global_start),
                            local_slice=slice(0, 0),
                            padding_tokens=missing_tokens if row == 0 else 0,
                        )
                    )

        segments_by_rank = [
            self._reorder_segments(
                rank_segments,
                input_batch.num_computed_tokens_np,
                input_batch.is_prefilling_np,
                input_batch.query_start_loc_np,
            )
            for rank_segments in segments_by_rank
        ]
        per_rank_num_tokens = [
            sum(segment.num_tokens for segment in segments)
            for segments in segments_by_rank
        ]
        padded_num_tokens = max(per_rank_num_tokens, default=0)
        expanded_num_tokens = padded_num_tokens * self.pcp_size
        padded_gather_idx = np.zeros(expanded_num_tokens, dtype=np.int64)
        kv_write_mask = np.zeros(expanded_num_tokens, dtype=np.bool_)
        hidden_restore_idx = np.empty(input_batch.num_tokens, dtype=np.int64)

        for rank, segments in enumerate(segments_by_rank):
            rank_start = rank * padded_num_tokens
            for segment in segments:
                if segment.num_actual_tokens == 0:
                    continue
                gathered_slice = slice(
                    rank_start + segment.local_slice.start,
                    rank_start
                    + segment.local_slice.start
                    + segment.num_actual_tokens,
                )
                padded_gather_idx[gathered_slice] = np.arange(
                    segment.global_slice.start,
                    segment.global_slice.stop,
                    dtype=np.int64,
                )
                is_decode = not bool(
                    input_batch.is_prefilling_np[segment.global_req_idx]
                )
                if not is_decode or rank == 0:
                    kv_write_mask[gathered_slice] = True
                    hidden_restore_idx[segment.global_slice] = np.arange(
                        gathered_slice.start,
                        gathered_slice.stop,
                        dtype=np.int64,
                    )

        self._hidden_restore_idx = torch.from_numpy(hidden_restore_idx).to(
            self.device
        )
        self._padded_gather_idx = torch.from_numpy(padded_gather_idx).to(
            self.device
        )
        self._gathered_kv_write_mask = torch.from_numpy(kv_write_mask).to(
            self.device
        )
        return segments_by_rank, per_rank_num_tokens

    def partition_batch(self, input_batch: InputBatch) -> InputBatch:
        """Return a rank-local InputBatch without mutating the global batch."""

        if (
            input_batch.num_draft_tokens
            and input_batch.num_draft_tokens_per_req is None
        ):
            raise PatchCompatibilityError(
                "PCP spec decode requires per-request draft-token counts"
            )
        self._global_batch = input_batch
        segments_by_rank, per_rank_num_tokens = self._build_batch_layout(input_batch)
        segments = segments_by_rank[self.pcp_rank]
        if not segments:
            # Preserve one metadata row on a rank with no owned prefill tokens,
            # while keeping empty DualChunkSwap intervals out of normal batches.
            segments = [
                _BatchSegment(
                    global_req_idx=0,
                    global_slice=slice(0, 0),
                    local_slice=slice(0, 0),
                )
            ]
        num_local_reqs = len(segments)
        num_local_tokens = per_rank_num_tokens[self.pcp_rank]
        num_padded_tokens = max(per_rank_num_tokens, default=0)
        if num_local_reqs > self._max_local_reqs:
            raise RuntimeError("PCP local request count exceeds its buffer")
        if num_padded_tokens > self._max_local_tokens:
            raise RuntimeError("PCP local token count exceeds its buffer")

        global_req_indices = np.fromiter(
            (segment.global_req_idx for segment in segments),
            dtype=np.int32,
            count=num_local_reqs,
        )
        local_lengths = self._num_scheduled_tokens[:num_local_reqs]
        local_lengths[:] = [segment.num_tokens for segment in segments]
        local_idx_mapping_np = self._idx_mapping_np[:num_local_reqs]
        local_idx_mapping_np[:] = input_batch.idx_mapping_np[global_req_indices]
        local_idx_mapping = self._idx_mapping[:num_local_reqs]
        local_idx_mapping.copy_(torch.from_numpy(local_idx_mapping_np).to(self.device))

        query_start_np = self._query_start_loc_np[: num_local_reqs + 1]
        query_start_np[0] = 0
        np.cumsum(local_lengths, out=query_start_np[1:])
        query_start = self._query_start_loc[: num_local_reqs + 1]
        query_start.copy_(torch.from_numpy(query_start_np).to(self.device))

        local_start_pos_np = np.fromiter(
            (
                int(input_batch.num_computed_tokens_np[segment.global_req_idx])
                + segment.global_slice.start
                - int(input_batch.query_start_loc_np[segment.global_req_idx])
                if segment.num_actual_tokens > 0
                else 0
                for segment in segments
            ),
            dtype=np.int32,
            count=num_local_reqs,
        )
        seq_lens = self._seq_lens[:num_local_reqs]

        input_ids = self._input_ids[:num_padded_tokens]
        positions = self._positions[:num_padded_tokens]
        is_padding = self._is_padding[:num_padded_tokens]
        input_ids.zero_()
        positions.zero_()
        is_padding.fill_(True)
        for row, segment in enumerate(segments):
            local_slice = segment.local_slice
            if segment.num_actual_tokens == 0:
                continue
            actual_slice = slice(
                local_slice.start,
                local_slice.start + segment.num_actual_tokens,
            )
            input_ids[actual_slice].copy_(
                input_batch.input_ids[segment.global_slice]
            )
            # MRV2 async MTP keeps num_computed_tokens_np as an optimistic CPU
            # upper bound. Rejections are already reflected in these device
            # positions, so preserve them instead of rebuilding from the CPU
            # mirror. Derive the device seq_len from the same corrected source.
            positions[actual_slice].copy_(
                input_batch.positions[segment.global_slice]
            )
            is_padding[actual_slice] = False

        # A full request row can reuse MRV2's rejection-corrected device
        # seq_len directly. This is the common decode path and keeps the
        # async-MTP fix to one batched gather instead of one device op per row.
        full_request_rows = all(
            segment.num_actual_tokens > 0
            and segment.global_slice.start
            == int(input_batch.query_start_loc_np[segment.global_req_idx])
            and segment.global_slice.stop
            == int(input_batch.query_start_loc_np[segment.global_req_idx + 1])
            for segment in segments
        )
        seq_scratch_np = self._cu_num_logits_np[:num_local_reqs]
        seq_index = self._logits_indices[:num_local_reqs]
        if full_request_rows:
            seq_scratch_np[:] = global_req_indices
            seq_index.copy_(torch.from_numpy(seq_scratch_np).to(self.device))
            torch.index_select(
                input_batch.seq_lens,
                0,
                seq_index,
                out=seq_lens,
            )
        else:
            # Split prefill rows use the last preserved position as their local
            # sequence end. Padding-only rows retain their virtual row width.
            seq_scratch_np[:] = [segment.padding_tokens for segment in segments]
            seq_lens.copy_(torch.from_numpy(seq_scratch_np).to(self.device))
            actual_rows = [
                row
                for row, segment in enumerate(segments)
                if segment.num_actual_tokens > 0
            ]
            if actual_rows:
                num_actual_rows = len(actual_rows)
                seq_scratch_np[:num_actual_rows] = [
                    segments[row].local_slice.start
                    + segments[row].num_actual_tokens
                    - 1
                    for row in actual_rows
                ]
                actual_last_index = self._logits_indices[:num_actual_rows]
                actual_last_index.copy_(
                    torch.from_numpy(seq_scratch_np[:num_actual_rows]).to(
                        self.device
                    )
                )
                actual_seq_lens_i64 = self._seq_lens_i64[:num_actual_rows]
                torch.index_select(
                    positions,
                    0,
                    actual_last_index,
                    out=actual_seq_lens_i64,
                )
                actual_seq_lens_i64.add_(1)
                seq_scratch_np[:num_actual_rows] = actual_rows
                actual_row_index = self._logits_indices[:num_actual_rows]
                actual_row_index.copy_(
                    torch.from_numpy(seq_scratch_np[:num_actual_rows]).to(
                        self.device
                    )
                )
                seq_lens.index_copy_(
                    0,
                    actual_row_index,
                    actual_seq_lens_i64.to(dtype=seq_lens.dtype),
                )

        global_num_logits = np.diff(input_batch.cu_num_logits_np)
        local_num_logits = global_num_logits[global_req_indices].copy()
        local_actual_lengths = np.asarray(
            [segment.num_actual_tokens for segment in segments],
            dtype=np.int32,
        )
        local_num_logits[local_actual_lengths == 0] = 0
        if np.any(local_num_logits > local_actual_lengths):
            raise PatchCompatibilityError(
                "PCP local logits exceed their rank-local query lengths"
            )
        num_logits = int(local_num_logits.sum())
        logits_indices = self._logits_indices[:num_logits]
        logits_indices_np = np.empty(num_logits, dtype=np.int64)
        logits_offset = 0
        for row, count_value in enumerate(local_num_logits):
            count = int(count_value)
            if count == 0:
                continue
            query_stop = int(query_start_np[row + 1])
            logits_indices_np[logits_offset : logits_offset + count] = np.arange(
                query_stop - count,
                query_stop,
                dtype=np.int64,
            )
            logits_offset += count
        logits_indices.copy_(torch.from_numpy(logits_indices_np).to(self.device))
        cu_num_logits_np = self._cu_num_logits_np[: num_local_reqs + 1]
        cu_num_logits_np[0] = 0
        np.cumsum(local_num_logits, out=cu_num_logits_np[1:])
        cu_num_logits = self._cu_num_logits[: num_local_reqs + 1]
        cu_num_logits.copy_(torch.from_numpy(cu_num_logits_np).to(self.device))

        expanded_idx_mapping = torch.repeat_interleave(
            local_idx_mapping,
            torch.from_numpy(local_num_logits).to(self.device),
        )
        expanded_local_pos = torch.cat(
            [
                torch.arange(int(count), dtype=torch.int32, device=self.device)
                for count in local_num_logits
                if count > 0
            ],
            dim=0,
        ) if num_logits else torch.empty(0, dtype=torch.int32, device=self.device)

        local_prefill_len = input_batch.prefill_len_np[global_req_indices].copy()
        local_num_computed_prefill = np.minimum(
            local_start_pos_np, local_prefill_len
        )
        local_is_prefilling = self._is_prefilling_np[:num_local_reqs]
        local_is_prefilling[:] = (
            local_num_computed_prefill < local_prefill_len
        )
        local_num_draft_tokens_per_req = None
        local_num_draft_tokens = 0
        if input_batch.num_draft_tokens_per_req is not None:
            local_num_draft_tokens_per_req = input_batch.num_draft_tokens_per_req[
                global_req_indices
            ].copy()
            local_num_draft_tokens_per_req[local_actual_lengths == 0] = 0
            if np.any(local_num_draft_tokens_per_req[local_is_prefilling] != 0):
                raise PatchCompatibilityError(
                    "PCP+MTP draft tokens are only valid for decode requests"
                )
            local_num_draft_tokens = int(local_num_draft_tokens_per_req.sum())
        return replace(
            input_batch,
            req_ids=[input_batch.req_ids[index] for index in global_req_indices],
            num_reqs=num_local_reqs,
            num_reqs_after_padding=num_local_reqs,
            idx_mapping=local_idx_mapping,
            idx_mapping_np=local_idx_mapping_np,
            expanded_idx_mapping=expanded_idx_mapping,
            expanded_local_pos=expanded_local_pos,
            num_scheduled_tokens=local_lengths,
            num_tokens=num_local_tokens,
            num_tokens_after_padding=num_padded_tokens,
            num_draft_tokens=local_num_draft_tokens,
            num_draft_tokens_per_req=local_num_draft_tokens_per_req,
            query_start_loc=query_start,
            query_start_loc_np=query_start_np,
            seq_lens=seq_lens,
            seq_lens_cpu_upper_bound=torch.from_numpy(
                local_start_pos_np + local_lengths
            ),
            dcp_local_seq_lens=None,
            num_computed_tokens_np=local_start_pos_np,
            prefill_len_np=local_prefill_len,
            num_computed_prefill_tokens_np=local_num_computed_prefill,
            is_prefilling_np=local_is_prefilling,
            max_seq_len_np=(
                input_batch.max_seq_len_np[global_req_indices].copy()
                if input_batch.max_seq_len_np is not None
                else None
            ),
            input_ids=input_ids,
            positions=positions,
            is_padding=is_padding,
            logits_indices=logits_indices,
            cu_num_logits=cu_num_logits,
            cu_num_logits_np=cu_num_logits_np,
            prompt_lens=None,
        )

    def prepare_attn(
        self, input_batch: InputBatch
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        """Prepare local block rows and global-ownership slot mappings."""

        canonical_tables = getattr(self._block_tables, "block_tables", None)
        scratch_tables = getattr(
            self._block_tables, "input_block_tables", None
        )
        declared_groups = getattr(
            self._block_tables, "num_kv_cache_groups", None
        )
        if not isinstance(canonical_tables, list) or not canonical_tables:
            raise PatchCompatibilityError(
                "vLLM 0.25.1 BlockTables canonical block_tables storage is missing"
            )
        if not isinstance(scratch_tables, list) or not isinstance(
            declared_groups, int
        ):
            raise PatchCompatibilityError(
                "vLLM 0.25.1 BlockTables canonical block_tables contract changed"
            )
        group_counts = (
            len(canonical_tables),
            len(scratch_tables),
            len(self._local_block_tables),
            declared_groups,
        )
        if len(set(group_counts)) != 1:
            raise PatchCompatibilityError(
                "vLLM 0.25.1 BlockTables KV-group count mismatch: "
                f"canonical/scratch/local/declared={group_counts}"
            )
        source_tables = []
        for table in canonical_tables:
            source = getattr(table, "gpu", None)
            if not isinstance(source, torch.Tensor):
                raise PatchCompatibilityError(
                    "vLLM 0.25.1 BlockTables canonical block_tables GPU "
                    "storage is missing"
                )
            source_tables.append(source)
        num_reqs = input_batch.num_reqs_after_padding
        for source, destination in zip(
            source_tables, self._local_block_tables
        ):
            torch.index_select(
                source,
                0,
                input_batch.idx_mapping.to(torch.int64),
                out=destination[:num_reqs],
            )
        local_tables = tuple(
            table[:num_reqs] for table in self._local_block_tables
        )
        assert self._global_batch is not None
        assert self._global_slot_mappings is not None
        computed_slots = self._block_tables.compute_slot_mappings(
            self._global_batch.idx_mapping,
            self._global_batch.query_start_loc,
            self._global_batch.positions,
            self._global_batch.num_tokens,
        )
        global_slots = self._global_slot_mappings[
            :, : self._global_batch.num_tokens
        ]
        global_slots.copy_(computed_slots)
        return local_tables, self._convert_slot_mappings(global_slots)

    def _convert_slot_mappings(self, global_slots: torch.Tensor) -> torch.Tensor:
        assert self._padded_gather_idx is not None
        assert self._gathered_kv_write_mask is not None
        assert self._gathered_slot_mappings is not None
        num_expanded_tokens = self._padded_gather_idx.numel()
        gathered = self._gathered_slot_mappings[:, :num_expanded_tokens]
        torch.index_select(
            global_slots,
            1,
            self._padded_gather_idx,
            out=gathered,
        )
        torch.where(
            self._gathered_kv_write_mask.unsqueeze(0),
            gathered,
            self._pad_slot_id,
            out=gathered,
        )
        return gathered

    def get_dummy_slot_mappings(self, num_tokens: int) -> torch.Tensor:
        """Return invalid slots for a PCP-expanded dummy attention batch."""

        assert self._gathered_slot_mappings is not None
        expanded_tokens = num_tokens * self.pcp_size
        if expanded_tokens > self._gathered_slot_mappings.shape[1]:
            raise RuntimeError("PCP dummy slot count exceeds its buffer")
        slots = self._gathered_slot_mappings[:, :expanded_tokens]
        slots.fill_(PAD_SLOT_ID)
        return slots

    def restore_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Gather equal-width local outputs and restore global token order."""

        if self._hidden_restore_idx is None:
            return hidden_states
        gathered = self._pcp_group.all_gather(hidden_states, dim=0)
        return gathered[self._hidden_restore_idx]

    def restore_for_sampling(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, InputBatch]:
        """Restore final hidden states and the exact saved global InputBatch."""

        assert self._global_batch is not None
        return self.restore_hidden_states(hidden_states), self._global_batch


def maybe_build_pcp_manager(
    vllm_config: object,
    device: torch.device,
    req_states: object,
    block_tables: object,
) -> HcuPCPManager | None:
    """Build only for an already approved PCP>1 runtime contract."""

    if vllm_config.parallel_config.prefill_context_parallel_size == 1:
        return None
    assert _validate_hcu_pcp_scope(vllm_config)
    return HcuPCPManager(vllm_config, device, req_states, block_tables)


def maybe_partition_pcp_batch(
    manager: HcuPCPManager | None, input_batch: InputBatch
) -> InputBatch:
    if manager is None:
        return input_batch
    return manager.partition_batch(input_batch)


def maybe_restore_pcp_for_sampling(
    manager: HcuPCPManager | None,
    hidden_states: torch.Tensor,
    input_batch: InputBatch,
) -> tuple[torch.Tensor, InputBatch]:
    if manager is None:
        return hidden_states, input_batch
    return manager.restore_for_sampling(hidden_states)


__all__ = [
    "HcuPCPManager",
    "RankSegment",
    "maybe_build_pcp_manager",
    "maybe_partition_pcp_batch",
    "maybe_restore_pcp_for_sampling",
]
