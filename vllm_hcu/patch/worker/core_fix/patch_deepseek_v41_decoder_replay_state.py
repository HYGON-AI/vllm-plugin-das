# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Prepare decoder-side SWA replay + encoder APC pad for DeepSeek V4.1 on HCU.

Port of upstream ``DeepseekV41ModelState`` replay logic:

* Encoder/APC (#56227): ``replay_start`` from ``NewRequestData``, pad
  prefix-cacheable group slots for replayed positions, pass
  ``ReplayAttnMetadata`` into SWA builders — works with decoder replay OFF.
* Decoder trim (#58132 / #59039 eager + #59532 CUDA-graph): last-window
  sub-batch when ``decoder_replay_layers`` is present.

Skips DP ``_agree_across_dp``. Uses a Torch pad path (HCU-safe) that matches
the upstream Triton ``_pad_replayed_slots_kernel`` semantics.
"""

from __future__ import annotations

import functools
from dataclasses import replace
from types import ModuleType
from typing import Any

import numpy as np
import torch

from vllm.compilation.breakable_cudagraph import is_breakable_cudagraph_enabled
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed import get_pp_group
from vllm.forward_context import BatchDescriptor, create_forward_context
from vllm.logger import init_logger
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.gpu.attn_utils import (
    build_attn_metadata,
    build_slot_mappings_by_layer,
)
from vllm.v1.worker.gpu.buffer_utils import UvaBufferPool
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.model_states.interface import ModelSpecificAttnMetadata
from vllm_hcu.patch.worker.core_fix._common import (
    PatchCompatibilityError,
    load_exact_module,
    require_callable,
    require_class,
)

from .decoder_replay import ReplayBatch
from .decoder_replay_cudagraph import DecoderReplayCudaGraphManager
from .late_layer_tail_cp import (
    late_layer_tail_cp_layout,
    late_layer_tail_layout,
)

TARGET_MODULE = "vllm.models.deepseek_v41.nvidia.model_state"
PATCH_ID = "worker.core_fix.deepseek_v41.decoder_replay_state"

logger = init_logger(__name__)
TARGETS = (
    f"{TARGET_MODULE}.DeepseekV41ModelState.__init__",
    f"{TARGET_MODULE}.DeepseekV41ModelState.prepare_attn",
    f"{TARGET_MODULE}.DeepseekV41ModelState.capture_inner_cudagraphs",
)
_MARKER = "_vllm_hcu_dsv41_decoder_replay_state_applied"
_WRAPPER = "_vllm_hcu_dsv41_decoder_replay_state_wrapper"


class ReplayAttnMetadata(ModelSpecificAttnMetadata):
    """Hands the batch's replay starts to the sliding-window builders."""

    def __init__(self, replay_start: torch.Tensor) -> None:
        self.replay_start = replay_start

    def get_extra_attn_kwargs(
        self, attn_metadata_builder: Any, num_reqs: int
    ) -> dict[str, Any]:
        from vllm.v1.attention.backends.mla.sparse_swa import (
            DeepseekSparseSWAMetadataBuilder,
        )

        if isinstance(attn_metadata_builder, DeepseekSparseSWAMetadataBuilder):
            return {"replay_start": self.replay_start[:num_reqs]}
        return {}


def _pcp_local_slot_mappings(
    slot_mappings: torch.Tensor,
    pcp_size: int,
    *,
    local_num_tokens: int | None = None,
) -> tuple[torch.Tensor, bool]:
    """Slice the rank-local segment from an expanded PCP slot mapping.

    Upstream #62183 assumes ``slot_mappings`` is laid out as
    ``[num_groups, pcp_size * local_tokens]``. Decode / warmup on HCU 0.29
    often pass a rank-local or unexpanded global width that still happens to
    be divisible by ``pcp_size`` (e.g. local=16, PCP=8); treating that as
    expanded would truncate APC padding and corrupt the follow-on all_gather.
    Only chunk when the width exactly matches ``pcp_size * local_num_tokens``.
    """
    if pcp_size <= 1 or slot_mappings.ndim != 2:
        return slot_mappings, False
    if local_num_tokens is None or int(local_num_tokens) <= 0:
        return slot_mappings, False
    expected = int(pcp_size) * int(local_num_tokens)
    if slot_mappings.shape[1] != expected:
        return slot_mappings, False
    from vllm.distributed.parallel_state import get_pcp_group

    return (
        slot_mappings.chunk(pcp_size, dim=1)[get_pcp_group().rank_in_group],
        True,
    )


def _pcp_global_slots_for_replay(
    slot_mappings: torch.Tensor,
    *,
    pcp_size: int,
    local_num_tokens: int,
    global_num_tokens: int,
    restore_idx: torch.Tensor,
) -> torch.Tensor | None:
    """Map ``slot_mappings`` into global token order for PCP late-tail replay.

    Returns ``None`` when the layout cannot be restored; callers should skip
    trim and run the full batch rather than writing PAD slots for real tokens.
    """
    if slot_mappings.ndim != 2:
        return None
    width = int(slot_mappings.shape[1])
    local_num = int(local_num_tokens)
    world = int(pcp_size)
    global_num = int(global_num_tokens)
    expanded_width = world * local_num if world > 1 and local_num > 0 else -1

    # Rank-local width: gather peers then restore DualChunk order.
    if world > 1 and local_num > 0 and width == local_num:
        from vllm.distributed.parallel_state import get_pcp_group

        expanded = get_pcp_group().all_gather(slot_mappings, dim=1)
        if int(expanded.shape[1]) != expanded_width:
            return None
        return expanded.index_select(1, restore_idx)

    # Expanded rank-major: one equal-width segment per rank.
    # Prefer this over "already global" when widths coincide
    # (``pcp * local == global`` with no pad), because late-tail is only
    # entered with DualChunk restore metadata from the prefill gather path.
    if width == expanded_width:
        return slot_mappings.index_select(1, restore_idx)

    # Already global (unexpanded buffer whose width is not pcp*local).
    if global_num > 0 and width == global_num:
        return slot_mappings

    return None


def pad_replayed_slots(
    slot_mappings: torch.Tensor,
    cacheable_groups: torch.Tensor,
    query_start_loc: torch.Tensor,
    positions: torch.Tensor,
    replay_start: torch.Tensor,
    window: int,
    pad_slot_id: int = PAD_SLOT_ID,
) -> None:
    """Pad prefix-cacheable group slots for positions in ``[start, start+window)``.

    Mirrors upstream ``_pad_replayed_slots_kernel``. Mutates ``slot_mappings``
    in place. Requests with ``replay_start == 0`` are skipped (no APC replay).
    """
    if window <= 0 or cacheable_groups.numel() == 0:
        return
    num_reqs = int(replay_start.shape[0])
    for req in range(num_reqs):
        start = int(replay_start[req].item())
        if start == 0:
            continue
        begin = int(query_start_loc[req].item())
        end = int(query_start_loc[req + 1].item())
        if begin >= end:
            continue
        pos = positions[begin:end]
        replayed = (pos >= start) & (pos < start + window)
        if not bool(replayed.any()):
            continue
        for group in cacheable_groups.tolist():
            slot_mappings[int(group), begin:end].masked_fill_(replayed, pad_slot_id)


def _gather_kept_rows(
    *,
    query_start_loc_np: np.ndarray,
    kept_lens: np.ndarray,
    positions: torch.Tensor,
    slot_mappings: torch.Tensor,
    seq_lens_cpu: np.ndarray,
    window: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """CPU/PyTorch gather of each request's last ``kept_lens`` rows.

    Returns ``(rows, kept_query_start_loc, kept_positions, kept_slots, kept_kv_start)``.
    """
    num_reqs = int(kept_lens.shape[0])
    kept_query_start_loc_np = np.zeros(num_reqs + 1, dtype=np.int32)
    np.cumsum(kept_lens, out=kept_query_start_loc_np[1:])
    num_tokens = int(kept_query_start_loc_np[-1])

    rows_np = np.empty(num_tokens, dtype=np.int64)
    kept_kv_start_np = np.zeros(num_reqs, dtype=np.int32)
    cursor = 0
    for req in range(num_reqs):
        begin = int(query_start_loc_np[req])
        end = int(query_start_loc_np[req + 1])
        qlen = end - begin
        kept = int(kept_lens[req])
        first = end - kept
        rows_np[cursor : cursor + kept] = np.arange(first, end, dtype=np.int64)
        # Trimmed requests hold no replay-layer window KV below kept rows.
        kv_start = 0
        if kept < qlen:
            kv_start = max(0, int(seq_lens_cpu[req]) - window)
        kept_kv_start_np[req] = kv_start
        cursor += kept

    rows = torch.as_tensor(rows_np, device=device, dtype=torch.long)
    kept_query_start_loc = torch.as_tensor(
        kept_query_start_loc_np, device=device, dtype=torch.int32
    )
    kept_positions = positions.index_select(0, rows)
    kept_slots = slot_mappings.index_select(1, rows)
    kept_kv_start = torch.as_tensor(
        kept_kv_start_np, device=device, dtype=torch.int32
    )
    return rows, kept_query_start_loc, kept_positions, kept_slots, kept_kv_start


def _pad_kept_slots(
    kept_slots: torch.Tensor, kept_positions: torch.Tensor, num_padded: int
) -> tuple[torch.Tensor, torch.Tensor]:
    num_tokens = kept_slots.shape[1]
    if num_padded <= num_tokens:
        return kept_slots, kept_positions
    pad_n = num_padded - num_tokens
    kept_slots = torch.cat(
        [
            kept_slots,
            kept_slots.new_full((kept_slots.shape[0], pad_n), PAD_SLOT_ID),
        ],
        dim=1,
    )
    kept_positions = torch.cat(
        [kept_positions, kept_positions.new_zeros(pad_n)],
        dim=0,
    )
    return kept_slots, kept_positions


def _replay_kv_starts(
    extend_lens: list[int] | np.ndarray,
    tail_lens: list[int] | np.ndarray,
    seq_lens: list[int] | np.ndarray,
) -> np.ndarray:
    """SWA window floors for replay rows: advance only when actually trimmed."""
    n = len(extend_lens)
    out = np.zeros(n, dtype=np.int32)
    for i in range(n):
        extend_i = int(extend_lens[i])
        tail_i = int(tail_lens[i])
        if extend_i > tail_i:
            out[i] = max(0, int(seq_lens[i]) - tail_i)
    return out


def _subsequent_replay_kv_start(
    replay_start: torch.Tensor, trim_kv_start: torch.Tensor
) -> torch.Tensor:
    """Lower bound for non-first SWA layers: APC floor and trim floor.

    First SWA keeps the raw ``replay_start`` (prefix rebuild). Later layers
    must not drop below that floor when trim did not advance
    (``trim_kv_start == 0``), or a short APC chunk can read unrebuilt KV.
    """
    return torch.maximum(replay_start, trim_kv_start)


def _bump_last_seq_len_for_query_pad(
    seq_lens: list[int] | np.ndarray, *, real_tokens: int, num_padded: int
) -> list[int]:
    """Keep ``seq_len - query_len`` stable when pad hangs off the last request.

    Graph dispatch may extend ``query_start_loc[-1]`` past the real tail. SWA
    builders derive each token's absolute position from ``seq_len - query_len``,
    so the last request's sequence length must grow by the same pad delta.
    """
    out = [int(x) for x in seq_lens]
    pad_n = int(num_padded) - int(real_tokens)
    if pad_n > 0 and out:
        out[-1] += pad_n
    return out


def _copy_attn_groups(attn_groups, prefixes: set[str], vllm_config, device):
    own_groups = []
    for groups in attn_groups:
        copies = []
        for group in groups:
            if prefixes.isdisjoint(group.layer_names):
                continue
            copy = replace(group, metadata_builders=[])
            # 0.29 AttentionGroup has no kernel_block_size attr; None lets the
            # builder use the group's kv_cache_spec block size.
            copy.create_metadata_builders(vllm_config, device, None)
            copies.append(copy)
        own_groups.append(copies)
    return own_groups


def _build_subbatch_attn_metadata(
    self,
    *,
    input_batch,
    block_tables,
    slot_mappings,
    attn_groups,
    kv_cache_config,
    model_specific_attn_metadata: ModelSpecificAttnMetadata | None,
    ubatch_idx: int,
):
    """Call ``build_attn_metadata`` like DefaultModelState.prepare_attn (eager)."""
    num_reqs = input_batch.num_reqs
    num_tokens = input_batch.num_tokens
    query_start_loc_cpu = torch.from_numpy(
        input_batch.query_start_loc_np[: num_reqs + 1]
    )
    query_start_loc_gpu = input_batch.query_start_loc[: num_reqs + 1]
    max_query_len = input_batch.max_query_len
    if max_query_len is None:
        max_query_len = int(input_batch.num_scheduled_tokens.max().item())
    seq_lens_cpu_upper_bound = input_batch.seq_lens_cpu_upper_bound
    max_seq_len = int(seq_lens_cpu_upper_bound[:num_reqs].max().item())

    return build_attn_metadata(
        attn_groups=attn_groups,
        num_reqs=num_reqs,
        num_tokens=num_tokens,
        query_start_loc_gpu=query_start_loc_gpu,
        query_start_loc_cpu=query_start_loc_cpu,
        max_query_len=max_query_len,
        seq_lens=input_batch.seq_lens[:num_reqs],
        max_seq_len=max_seq_len,
        block_tables=block_tables,
        slot_mappings=slot_mappings,
        kv_cache_config=kv_cache_config,
        seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound[:num_reqs],
        positions=input_batch.positions[:num_tokens],
        # Must stay on CPU: split_decodes_and_prefills ORs this with
        # query_lens derived from query_start_loc_cpu.
        is_prefilling=torch.as_tensor(
            input_batch.is_prefilling_np[:num_reqs],
            device="cpu",
            dtype=torch.bool,
        )
        if hasattr(input_batch, "is_prefilling_np")
        else None,
        model_specific_attn_metadata=model_specific_attn_metadata,
        for_cudagraph_capture=False,
        ubatch_idx=ubatch_idx,
        fast_prefill=getattr(input_batch, "fast_prefill", None),
    )


def _ensure_replay_config(self, kv_cache_config) -> tuple[int, torch.Tensor]:
    if self._replay is None:
        specs = [group.kv_cache_spec for group in kv_cache_config.kv_cache_groups]
        self._replay = (
            max((spec.prefix_replay_tokens for spec in specs), default=0),
            torch.tensor(
                [i for i, spec in enumerate(specs) if spec.prefix_cacheable],
                dtype=torch.int32,
                device=self.device,
            ),
        )
    return self._replay


def _gather_batch_replay_start(
    self, input_batch
) -> tuple[torch.Tensor, np.ndarray]:
    num_reqs = input_batch.num_reqs
    is_prefilling = input_batch.is_prefilling_np[:num_reqs]
    replay_start_np = np.where(
        is_prefilling,
        self._replay_start_np[input_batch.idx_mapping_np[:num_reqs]],
        0,
    ).astype(np.int32)
    replay_start = self._replay_start_staging.copy_to_gpu(
        replay_start_np, out=self._replay_start[:num_reqs]
    )
    return replay_start, replay_start_np



def _gather_block_tables_for_reqs(block_tables, req_indices, device):
    """Reorder/gather per-request block tables to ``req_indices`` order."""
    idx = torch.as_tensor(req_indices, device=device, dtype=torch.long)
    if isinstance(block_tables, tuple):
        return tuple(bt.index_select(0, idx.to(device=bt.device)) for bt in block_tables)
    if torch.is_tensor(block_tables):
        return block_tables.index_select(0, idx.to(device=block_tables.device))
    # Fallback: unknown type — leave unchanged (builders may ignore extras).
    return block_tables


def _prepare_replay_batch(
    self,
    input_batch,
    cudagraph_mode,
    block_tables,
    slot_mappings,
    attn_groups,
    kv_cache_config,
    replay_start: torch.Tensor,
    capture_desc=None,
    ubatch_idx: int = 0,
) -> None:
    """Set the replay layers' batch for this forward, or none.

    Eager steps trim; captures and PIECEWISE graphs at the trim threshold or
    above break out to a replay batch; FULL and smaller PIECEWISE graphs keep
    the whole batch. PCP uses a replicated contiguous global SWA tail (logical
    CP width 1) when that shrinks the rank-local token count. Multiple logical
    requests in one step are packed as a multi-req replay batch.
    """
    layers = self.decoder_replay_layers
    assert layers is not None
    layers.replay_batch = None
    force_replay_batch = capture_desc is not None or (
        cudagraph_mode == CUDAGraphMode.PIECEWISE
        and layers.graph_uses_replay_batch(input_batch.num_tokens_after_padding)
    )
    if cudagraph_mode != CUDAGraphMode.NONE and not force_replay_batch:
        return

    logger.warning_once(
        "Decoder SWA prepare_replay: cg=%s tokens=%s pcp=%s",
        cudagraph_mode,
        int(input_batch.num_tokens_after_padding),
        type(getattr(input_batch, "pcp", None)).__name__,
    )

    num_reqs = input_batch.num_reqs
    query_lens = np.diff(input_batch.query_start_loc_np[: num_reqs + 1])
    idx_mapping_np = input_batch.idx_mapping_np[:num_reqs]
    keeps = getattr(self, "_keeps_rows_np", None)
    keeps_rows = (
        keeps[idx_mapping_np]
        if keeps is not None
        else np.zeros(num_reqs, dtype=np.bool_)
    )
    seq_lens = input_batch.seq_lens_cpu_upper_bound[:num_reqs].numpy()
    pcp = getattr(input_batch, "pcp", None)
    kv_start_np: np.ndarray | None = None
    num_padded: int | None = None

    pcp_tail_plan = None
    pcp_meta = None
    needs_replay_batch = False

    if pcp is not None:
        # Check DualChunk restore metadata *before* indexing global tails.
        # Falling back to rank-local ``positions`` with global token indices
        # OOBs when the global sequence is longer than the local batch, and
        # never reaches the missing-metadata fallback below.
        from types import SimpleNamespace

        from vllm.distributed.parallel_state import get_pcp_group
        from vllm_hcu.model_executor.layers.attention.pcp import (
            logical_pcp_metadata_scope,
        )

        restore_idx = getattr(input_batch, "_vllm_hcu_pcp_restore_idx", None)
        local_num_tokens = getattr(
            input_batch, "_vllm_hcu_pcp_local_num_tokens", None
        )
        global_num_tokens = getattr(
            input_batch, "_vllm_hcu_pcp_global_num_tokens", None
        )
        global_positions = getattr(
            input_batch, "_vllm_hcu_pcp_global_positions", None
        )
        if (
            restore_idx is None
            or local_num_tokens is None
            or global_num_tokens is None
            or global_positions is None
        ):
            logger.warning_once(
                "Decoder SWA PCP late-tail: missing restore metadata; "
                "falling back to non-PCP trim."
            )
            pcp = None  # fall through to non-PCP path below
        else:
            # One row per logical request (PCP may expand idx_mapping).
            _, first_indices = np.unique(idx_mapping_np, return_index=True)
            first_indices = np.sort(first_indices)

            extend_lens_cpu = [
                int(pcp.global_query_lens_np[i]) for i in first_indices
            ]
            seq_lens_g_cpu = [
                int(pcp.global_seq_lens_cpu_upper_bound_np[i]) for i in first_indices
            ]
            # prompt_logprobs rows must keep the full extend; others take the SWA
            # window. keeps_rows is indexed by local batch rows (== first_indices
            # representatives for each logical request).
            per_request_tail_lens = [
                int(extend_lens_cpu[j])
                if bool(keeps_rows[int(first_indices[j])])
                else min(int(layers.window), int(extend_lens_cpu[j]))
                for j in range(len(extend_lens_cpu))
            ]
            device = slot_mappings.device
            token_indices, tail_lens_cpu, window_floors = late_layer_tail_layout(
                extend_lens_cpu=extend_lens_cpu,
                seq_lens_cpu=seq_lens_g_cpu,
                tail_len=int(layers.window),
                device=device,
                per_request_tail_lens=per_request_tail_lens,
            )
            tail_positions = global_positions[: int(global_num_tokens)].index_select(
                0, token_indices
            )
            pcp_group = get_pcp_group()
            pcp_size = int(pcp_group.world_size)
            pcp_rank = int(pcp_group.rank_in_group)
            tail_lens_t = torch.tensor(tail_lens_cpu, dtype=torch.int32, device=device)
            # HCU fused SWA builders need contiguous Q rows: replicate the full
            # global tail on every rank (logical CP width 1). Physical DualChunk
            # ownership is still used in decoder_replay.to_local_tail / scatter.
            pcp_tail_plan = late_layer_tail_cp_layout(
                token_indices=token_indices,
                tail_lens=tail_lens_t,
                positions=tail_positions,
                window_floors=window_floors,
                pcp_rank=0,
                pcp_size=1,
                batch_size=len(extend_lens_cpu),
            )
            num_padded = int(pcp_tail_plan.max_local)
            num_tokens = int(pcp_tail_plan.num_real)
            needs_replay_batch = force_replay_batch or (
                num_padded < int(input_batch.num_tokens_after_padding)
            )
            pcp_meta = SimpleNamespace(
                pcp_world_size=pcp_size,
                pcp_has_global_prefill=True,
                pcp_restore_idx=restore_idx,
                pcp_local_num_tokens=int(local_num_tokens),
                pcp_global_num_tokens=int(global_num_tokens),
            )
            logger.warning_once(
                "Decoder SWA PCP replicated contiguous tail: "
                "reqs=%s global_tail=%s max_local=%s real=%s pad=%s "
                "(rank=%s/%s; multi-req enabled).",
                len(extend_lens_cpu),
                int(token_indices.numel()),
                num_padded,
                num_tokens,
                int(pcp_tail_plan.pad_rows),
                pcp_rank,
                pcp_size,
            )

    if pcp is None or pcp_tail_plan is None:
        # Non-PCP / fallback: last-window rows of the local batch.
        if pcp_tail_plan is None and getattr(input_batch, "pcp", None) is not None:
            # restore failed; use local min(query, window)
            kept_lens = np.where(
                keeps_rows, query_lens, np.minimum(query_lens, layers.window)
            ).astype(np.int32)
            needs_replay_batch = force_replay_batch or bool(
                (kept_lens < query_lens)[input_batch.is_prefilling_np[:num_reqs]].any()
            )
        elif pcp is None:
            kept_lens = np.where(
                keeps_rows, query_lens, np.minimum(query_lens, layers.window)
            ).astype(np.int32)
            is_dummy = not input_batch.is_prefilling_np[:num_reqs].any() and bool(
                (seq_lens == query_lens).all()
            )
            if is_dummy and capture_desc is None:
                kept_lens = np.zeros_like(kept_lens)
                kept_lens[0] = 1
            needs_replay_batch = force_replay_batch or bool(
                (kept_lens < query_lens)[input_batch.is_prefilling_np[:num_reqs]].any()
            )
        else:
            kept_lens = np.where(
                keeps_rows, query_lens, np.minimum(query_lens, layers.window)
            ).astype(np.int32)
            needs_replay_batch = force_replay_batch or bool(
                (kept_lens < query_lens)[input_batch.is_prefilling_np[:num_reqs]].any()
            )
        kv_start_np = None
        num_padded = None
        pcp_tail_plan = None
        pcp_meta = None

    if not needs_replay_batch:
        return

    if pcp_tail_plan is not None:
        num_padded = int(pcp_tail_plan.max_local)
        num_tokens = int(pcp_tail_plan.num_real)
        n_logical = len(pcp_tail_plan.tail_lens_cpu)
        cudagraphs, run_graph = getattr(self, "replay_cudagraphs", None), None
        cg_mode = CUDAGraphMode.NONE
        if cudagraphs is not None:
            desc = capture_desc or cudagraphs.dispatch(n_logical, num_padded, None, 0)
            if desc.cg_mode == CUDAGraphMode.PIECEWISE:
                num_padded, cg_mode = desc.num_tokens, desc.cg_mode
                run_graph = cudagraphs.run

        kept_positions = pcp_tail_plan.local_positions
        # Global slots via DualChunk restore map, then take interleaved share.
        restore_idx = pcp_meta.pcp_restore_idx
        local_num = int(pcp_meta.pcp_local_num_tokens)
        pcp_size = int(pcp_meta.pcp_world_size)
        global_num = int(getattr(pcp_meta, "pcp_global_num_tokens", 0) or 0)
        global_slots = _pcp_global_slots_for_replay(
            slot_mappings,
            pcp_size=pcp_size,
            local_num_tokens=local_num,
            global_num_tokens=global_num,
            restore_idx=restore_idx,
        )
        share = pcp_tail_plan.global_token_indices[pcp_tail_plan.local_tail_rows]
        if global_slots is None and share.numel():
            # Cannot restore real tail slots → skip trim (full execution)
            # rather than writing PAD_SLOT_ID over live late-layer KV.
            logger.warning_once(
                "Decoder SWA PCP late-tail: slot mapping width=%s does not "
                "match expanded (%s*%s), local (%s), or global (%s); "
                "skipping decoder trim for this step.",
                int(slot_mappings.shape[1]) if slot_mappings.ndim == 2 else -1,
                pcp_size,
                local_num,
                local_num,
                global_num,
            )
            layers.replay_batch = None
            return
        if global_slots is not None and share.numel():
            kept_slots = global_slots.index_select(1, share)
        else:
            kept_slots = slot_mappings.new_full(
                (slot_mappings.shape[0], 0), PAD_SLOT_ID
            )
        if pcp_tail_plan.pad_rows:
            kept_slots = torch.cat(
                [
                    kept_slots,
                    kept_slots.new_full(
                        (kept_slots.shape[0], pcp_tail_plan.pad_rows), PAD_SLOT_ID
                    ),
                ],
                dim=1,
            )
        # Piecewise graph dispatch may round ``num_padded`` above the real
        # tail (+ plan.pad_rows). Pad positions/slots to that size so metadata
        # builders never read past the provided tensors.
        kept_slots, kept_positions = _pad_kept_slots(
            kept_slots, kept_positions, num_padded
        )
        kept_query_start_loc_np = np.zeros(n_logical + 1, dtype=np.int32)
        kept_query_start_loc_np[1:] = np.cumsum(
            np.asarray(tail_lens_cpu, dtype=np.int32), dtype=np.int32
        )
        # Pad rows (if any) hang off the last request so builders see a dense
        # [0, num_padded] span without inventing an extra request.
        real_tail_tokens = int(kept_query_start_loc_np[-1])
        if num_padded > real_tail_tokens:
            kept_query_start_loc_np[-1] = int(num_padded)
        kept_query_start_loc = torch.as_tensor(
            kept_query_start_loc_np, device=device, dtype=torch.int32
        )
        rows = (
            pcp_tail_plan.local_tail_rows
            if pcp_tail_plan.local_tail_rows.numel()
            else torch.zeros(0, device=device, dtype=torch.long)
        )
        batch_num_tokens = num_padded
        max_query_len = int(max(tail_lens_cpu)) if tail_lens_cpu else num_padded
        if num_padded > max_query_len:
            max_query_len = num_padded
        # Only trimmed requests lose KV below the kept tail. Untrimmed decode
        # (or prompt_logprobs full-extend) rows keep kv_start=0 so prior SWA
        # window KV stays visible — matching the non-PCP gather path.
        kv_start_np = _replay_kv_starts(
            extend_lens_cpu, tail_lens_cpu, seq_lens_g_cpu
        )
        self._kept_kv_start[:n_logical].copy_(
            torch.as_tensor(kv_start_np, device=device, dtype=torch.int32)
        )
        kept_replay_start = replay_start[
            torch.as_tensor(first_indices, device=replay_start.device, dtype=torch.long)
        ]

        kept_block_tables = _gather_block_tables_for_reqs(
            block_tables, first_indices, device
        )

        is_prefilling_logical = np.asarray(
            [bool(input_batch.is_prefilling_np[int(i)]) for i in first_indices],
            dtype=np.bool_,
        ) if hasattr(input_batch, "is_prefilling_np") else np.ones(
            n_logical, dtype=np.bool_
        )

        seq_lens_padded = _bump_last_seq_len_for_query_pad(
            seq_lens_g_cpu, real_tokens=real_tail_tokens, num_padded=num_padded
        )
        kept_batch = replace(
            input_batch,
            num_reqs=n_logical,
            num_tokens=batch_num_tokens,
            num_tokens_after_padding=num_padded,
            query_start_loc=kept_query_start_loc,
            query_start_loc_np=kept_query_start_loc_np,
            max_query_len=max_query_len,
            seq_lens=torch.as_tensor(
                seq_lens_padded, device=device, dtype=input_batch.seq_lens.dtype
            ),
            seq_lens_cpu_upper_bound=torch.as_tensor(
                seq_lens_padded,
                device="cpu",
                dtype=input_batch.seq_lens_cpu_upper_bound.dtype,
            ),
            positions=kept_positions[:num_padded],
            fast_prefill=None,
            is_prefilling_np=is_prefilling_logical,
        )
        if hasattr(kept_batch, "pcp"):
            kept_batch.pcp = None

        if self._replay_attn_groups is None:
            self._replay_attn_groups = (
                _copy_attn_groups(
                    attn_groups,
                    layers.metadata_prefixes,
                    self.vllm_config,
                    self.device,
                ),
                _copy_attn_groups(
                    attn_groups,
                    {layers.first_swa_prefix},
                    self.vllm_config,
                    self.device,
                ),
            )
        replay_groups, first_swa_groups = self._replay_attn_groups

        first_kv_start = (
            kept_replay_start[:n_logical]
            if kept_replay_start.numel()
            else self._kept_kv_start[:n_logical]
        )
        subsequent_kv_start = _subsequent_replay_kv_start(
            first_kv_start, self._kept_kv_start[:n_logical]
        )
        with logical_pcp_metadata_scope(1):
            attn_metadata = _build_subbatch_attn_metadata(
                self,
                input_batch=kept_batch,
                block_tables=kept_block_tables,
                slot_mappings=kept_slots,
                attn_groups=replay_groups,
                kv_cache_config=kv_cache_config,
                model_specific_attn_metadata=ReplayAttnMetadata(
                    subsequent_kv_start
                ),
                ubatch_idx=ubatch_idx,
            )
            first_swa = layers.first_swa_prefix
            attn_metadata[first_swa] = _build_subbatch_attn_metadata(
                self,
                input_batch=kept_batch,
                block_tables=kept_block_tables,
                slot_mappings=kept_slots,
                attn_groups=first_swa_groups,
                kv_cache_config=kv_cache_config,
                model_specific_attn_metadata=ReplayAttnMetadata(first_kv_start),
                ubatch_idx=ubatch_idx,
            )[first_swa]

        layers.replay_batch = ReplayBatch(
            rows,
            create_forward_context(
                attn_metadata,
                self.vllm_config,
                cudagraph_runtime_mode=cg_mode,
                batch_descriptor=BatchDescriptor(num_tokens=num_padded),
                slot_mapping=build_slot_mappings_by_layer(
                    kept_slots, kv_cache_config
                ),
            ),
            run_graph,
            num_padded,
            pcp_tail=pcp_tail_plan,
            pcp_meta=pcp_meta,
        )
        return

    # ---- non-PCP local last-window path (unchanged) ----
    num_tokens = int(kept_lens.sum())
    if num_padded is None:
        num_padded = num_tokens
    cudagraphs, run_graph = getattr(self, "replay_cudagraphs", None), None
    cg_mode = CUDAGraphMode.NONE
    if cudagraphs is not None:
        desc = capture_desc or cudagraphs.dispatch(num_reqs, num_padded, None, 0)
        if desc.cg_mode == CUDAGraphMode.PIECEWISE:
            num_padded, cg_mode = desc.num_tokens, desc.cg_mode
            run_graph = cudagraphs.run

    kept_replay_start = replay_start
    if kv_start_np is not None and self._pcp_replay_start is not None:
        kept_replay_start = torch.maximum(
            replay_start,
            self._pcp_replay_start.copy_to_gpu(kv_start_np),
        )

    pcp_size = int(getattr(self, "pcp_size", 1) or 1)
    local_slot_mappings = slot_mappings
    (
        rows,
        kept_query_start_loc,
        kept_positions,
        kept_slots,
        kept_kv_start,
    ) = _gather_kept_rows(
        query_start_loc_np=input_batch.query_start_loc_np[: num_reqs + 1],
        kept_lens=kept_lens,
        positions=input_batch.positions,
        slot_mappings=local_slot_mappings,
        seq_lens_cpu=seq_lens,
        window=layers.window,
        device=slot_mappings.device,
    )
    self._kept_kv_start[:num_reqs].copy_(kept_kv_start)
    kept_slots, kept_positions = _pad_kept_slots(
        kept_slots, kept_positions, num_padded
    )

    kept_query_start_loc_np = np.zeros(num_reqs + 1, dtype=np.int32)
    np.cumsum(kept_lens, out=kept_query_start_loc_np[1:])
    batch_num_tokens = num_tokens
    seq_lens_padded = _bump_last_seq_len_for_query_pad(
        seq_lens, real_tokens=num_tokens, num_padded=num_padded
    )
    if num_padded > num_tokens:
        kept_query_start_loc_np[-1] = int(num_padded)
        batch_num_tokens = int(num_padded)
        kept_query_start_loc = torch.as_tensor(
            kept_query_start_loc_np,
            device=kept_query_start_loc.device,
            dtype=kept_query_start_loc.dtype,
        )
    max_query_len = max(int(kept_lens.max()), input_batch.max_query_len or 0)
    if batch_num_tokens > num_tokens:
        max_query_len = max(
            max_query_len, int(batch_num_tokens - kept_query_start_loc_np[-2])
        )
    kept_batch = replace(
        input_batch,
        num_tokens=batch_num_tokens,
        num_tokens_after_padding=num_padded,
        query_start_loc=kept_query_start_loc,
        query_start_loc_np=kept_query_start_loc_np,
        max_query_len=max_query_len,
        seq_lens=torch.as_tensor(
            seq_lens_padded,
            device=input_batch.seq_lens.device,
            dtype=input_batch.seq_lens.dtype,
        ),
        seq_lens_cpu_upper_bound=torch.as_tensor(
            seq_lens_padded,
            device="cpu",
            dtype=input_batch.seq_lens_cpu_upper_bound.dtype,
        ),
        positions=kept_positions,
        fast_prefill=None,
    )
    replay_start = kept_replay_start

    if self._replay_attn_groups is None:
        self._replay_attn_groups = (
            _copy_attn_groups(
                attn_groups,
                layers.metadata_prefixes,
                self.vllm_config,
                self.device,
            ),
            _copy_attn_groups(
                attn_groups,
                {layers.first_swa_prefix},
                self.vllm_config,
                self.device,
            ),
        )
    replay_groups, first_swa_groups = self._replay_attn_groups

    subsequent_kv_start = _subsequent_replay_kv_start(
        replay_start[:num_reqs], self._kept_kv_start[:num_reqs]
    )
    attn_metadata = _build_subbatch_attn_metadata(
        self,
        input_batch=kept_batch,
        block_tables=block_tables,
        slot_mappings=kept_slots,
        attn_groups=replay_groups,
        kv_cache_config=kv_cache_config,
        model_specific_attn_metadata=ReplayAttnMetadata(subsequent_kv_start),
        ubatch_idx=ubatch_idx,
    )
    first_swa = layers.first_swa_prefix
    attn_metadata[first_swa] = _build_subbatch_attn_metadata(
        self,
        input_batch=kept_batch,
        block_tables=block_tables,
        slot_mappings=kept_slots,
        attn_groups=first_swa_groups,
        kv_cache_config=kv_cache_config,
        model_specific_attn_metadata=ReplayAttnMetadata(replay_start),
        ubatch_idx=ubatch_idx,
    )[first_swa]

    layers.replay_batch = ReplayBatch(
        rows,
        create_forward_context(
            attn_metadata,
            self.vllm_config,
            cudagraph_runtime_mode=cg_mode,
            batch_descriptor=BatchDescriptor(num_tokens=num_padded),
            slot_mapping=build_slot_mappings_by_layer(
                kept_slots, kv_cache_config
            ),
        ),
        run_graph,
        None,
    )



def apply_to_module(module: ModuleType) -> bool:
    mod = load_exact_module(TARGET_MODULE, module)
    cls = require_class(
        mod, "DeepseekV41ModelState", f"{TARGET_MODULE}.DeepseekV41ModelState"
    )
    if getattr(cls, _MARKER, False):
        return False

    original_init = require_callable(cls, "__init__", TARGETS[0])
    original_prepare = require_callable(cls, "prepare_attn", TARGETS[1])

    @functools.wraps(original_init)
    def replay_init(self, vllm_config, model, encoder_cache, device):
        original_init(self, vllm_config, model, encoder_cache, device)
        self.pcp_size = vllm_config.parallel_config.prefill_context_parallel_size
        # PCP may duplicate local request segments (#62183).
        max_local_reqs = (
            2 * ((3 * self.max_num_reqs + 1) // 2)
            if self.pcp_size > 1
            else self.max_num_reqs
        )
        # Encoder/APC replay starts: always allocated so APC works with
        # decoder-side trim OFF.
        self._replay_start_np = np.zeros(max_local_reqs, dtype=np.int32)
        self._replay_start = torch.zeros(
            max_local_reqs, dtype=torch.int32, device=device
        )
        self._replay_start_staging = UvaBufferPool(max_local_reqs, torch.int32)
        self._replay: tuple[int, torch.Tensor] | None = None
        self._pcp_replay_start = (
            UvaBufferPool(max_local_reqs, torch.int32) if self.pcp_size > 1 else None
        )

        self.decoder_replay_layers = getattr(model, "decoder_replay_layers", None)
        self.replay_cudagraphs = None
        if self.decoder_replay_layers is not None:
            if self.pcp_size > 1:
                assert self.decoder_replay_layers.window >= 2 * self.pcp_size, (
                    "PCP replay requires window >= 2 * PCP size so each trimmed "
                    "request is represented on every rank."
                )
            self._keeps_rows_np = np.zeros(max_local_reqs, dtype=np.bool_)
            self._replay_attn_groups = None
            self._kept_kv_start = torch.zeros(
                max_local_reqs, dtype=torch.int32, device=device
            )
            layers = self.decoder_replay_layers
            compilation_config = vllm_config.compilation_config
            if (
                getattr(compilation_config, "cudagraph_decoder_replay", True)
                and compilation_config.cudagraph_mode.has_piecewise_cudagraphs()
                and is_breakable_cudagraph_enabled()
                and get_pp_group().is_first_rank
                and vllm_config.lora_config is None
                and not vllm_config.model_config.enable_prompt_embeds
            ):
                threshold = int(
                    getattr(
                        compilation_config, "decoder_replay_trim_threshold", 768
                    )
                )
                if threshold <= layers.window:
                    raise ValueError(
                        "decoder_replay_trim_threshold must exceed the replay "
                        f"window of {layers.window} tokens: smaller graphs have "
                        "nothing to trim"
                    )
                self.replay_cudagraphs = DecoderReplayCudaGraphManager(
                    vllm_config, device, layers
                )
                layers.trim_threshold = threshold

    original_add = getattr(cls, "add_request", None)
    if callable(original_add):

        @functools.wraps(original_add)
        def replay_add_request(self, req_index, new_req_data):
            original_add(self, req_index, new_req_data)
            self._replay_start_np[req_index] = int(
                getattr(new_req_data, "replay_start", 0) or 0
            )
            req_id = getattr(new_req_data, "req_id", None)
            if req_id is not None:
                mapping = getattr(self, "_replay_req_id_to_index", None)
                if mapping is None:
                    mapping = {}
                    self._replay_req_id_to_index = mapping
                mapping[req_id] = int(req_index)
            layers = getattr(self, "decoder_replay_layers", None)
            if layers is None:
                return
            params = getattr(new_req_data, "sampling_params", None)
            # prompt_logprobs and pooling need the full extend: trimming would
            # zero-fill omitted prompt rows and corrupt scores / embeddings.
            self._keeps_rows_np[req_index] = (
                params is not None
                and getattr(params, "prompt_logprobs", None) is not None
            ) or getattr(new_req_data, "pooling_params", None) is not None

        setattr(replay_add_request, _WRAPPER, True)
        cls.add_request = replay_add_request
    else:
        # 0.29 DefaultModelState always has add_request; fail closed if missing.
        raise PatchCompatibilityError(
            f"{TARGET_MODULE}.DeepseekV41ModelState.add_request is missing"
        )

    # Preemption / finish calls remove_request before the slot is reused.
    # Clear the APC floor so a resumed or recycled index cannot clamp SWA
    # with a stale replay_start after KV was discarded.
    original_remove = getattr(cls, "remove_request", None)
    if callable(original_remove):

        @functools.wraps(original_remove)
        def replay_remove_request(self, req_id: str) -> None:
            mapping = getattr(self, "_replay_req_id_to_index", None)
            idx = mapping.pop(req_id, None) if mapping is not None else None
            if idx is not None:
                self._replay_start_np[idx] = 0
                keeps = getattr(self, "_keeps_rows_np", None)
                if keeps is not None:
                    keeps[idx] = False
            original_remove(self, req_id)

        setattr(replay_remove_request, _WRAPPER, True)
        cls.remove_request = replay_remove_request

    @functools.wraps(original_prepare)
    def replay_prepare(
        self,
        input_batch,
        cudagraph_mode,
        block_tables,
        slot_mappings,
        attn_groups,
        kv_cache_config,
        for_capture: bool = False,
        ubatch_idx: int = 0,
    ):
        layers = getattr(self, "decoder_replay_layers", None)
        if layers is not None:
            layers.replay_batch = None

        window, cacheable_groups = _ensure_replay_config(self, kv_cache_config)
        model_specific: ModelSpecificAttnMetadata | None = None
        num_reqs = input_batch.num_reqs
        if window:
            replay_start, replay_start_np = _gather_batch_replay_start(
                self, input_batch
            )
            # Decode / warmup often pass rank-local slots; only chunk+gather
            # when input_batch.pcp is set and slots are expanded (#62183).
            # Require exact ``pcp_size * local_num_tokens`` — bare divisibility
            # false-positives on local widths like 16 under PCP=8.
            pcp = getattr(input_batch, "pcp", None)
            pcp_size = int(getattr(self, "pcp_size", 1) or 1)
            local_slot_mappings = slot_mappings
            use_pcp_slots = False
            if pcp is not None and pcp_size > 1:
                local_num_tokens = getattr(
                    input_batch, "_vllm_hcu_pcp_local_num_tokens", None
                )
                local_slot_mappings, use_pcp_slots = _pcp_local_slot_mappings(
                    slot_mappings,
                    pcp_size,
                    local_num_tokens=local_num_tokens,
                )
            if replay_start_np.any():
                pad_replayed_slots(
                    local_slot_mappings,
                    cacheable_groups,
                    input_batch.query_start_loc,
                    input_batch.positions,
                    replay_start,
                    window,
                    PAD_SLOT_ID,
                )
            if use_pcp_slots:
                from vllm.distributed.parallel_state import get_pcp_group

                # Peers may have no replay rows. Publish masked cacheable slots
                # in the same order as the gathered KV payload (#62183).
                slot_mappings[cacheable_groups] = get_pcp_group().all_gather(
                    local_slot_mappings[cacheable_groups], dim=1
                )
            model_specific = ReplayAttnMetadata(replay_start)
        else:
            # Decoder trim still needs a start tensor; zeros when APC is off.
            replay_start = self._replay_start[:num_reqs]

        # 0.29 DefaultModelState.prepare_attn does not accept
        # model_specific_attn_metadata; inject via the module global it loads.
        import vllm.v1.worker.gpu.model_states.default as default_ms

        orig_build = default_ms.build_attn_metadata
        if model_specific is not None:

            @functools.wraps(orig_build)
            def build_with_replay(*args, **kwargs):
                if kwargs.get("model_specific_attn_metadata") is None:
                    kwargs["model_specific_attn_metadata"] = model_specific
                return orig_build(*args, **kwargs)

            default_ms.build_attn_metadata = build_with_replay

        try:
            # Decoder trim only in eager / forced piecewise; APC pad above still
            # runs for FULL graphs when window>0.
            force = (
                layers is not None
                and cudagraph_mode == CUDAGraphMode.PIECEWISE
                and layers.graph_uses_replay_batch(
                    input_batch.num_tokens_after_padding
                )
            )
            if layers is None or (
                cudagraph_mode != CUDAGraphMode.NONE and not force
            ):
                return original_prepare(
                    self,
                    input_batch,
                    cudagraph_mode,
                    block_tables,
                    slot_mappings,
                    attn_groups,
                    kv_cache_config,
                    for_capture,
                    ubatch_idx,
                )

            full_metadata = original_prepare(
                self,
                input_batch,
                cudagraph_mode,
                block_tables,
                slot_mappings,
                attn_groups,
                kv_cache_config,
                for_capture,
                ubatch_idx,
            )
            assert replay_start is not None
            _prepare_replay_batch(
                self,
                input_batch,
                cudagraph_mode,
                block_tables,
                slot_mappings,
                attn_groups,
                kv_cache_config,
                replay_start[: input_batch.num_reqs],
                capture_desc=None,
                ubatch_idx=ubatch_idx,
            )
            return full_metadata
        finally:
            default_ms.build_attn_metadata = orig_build

    def capture_inner_cudagraphs(
        self,
        input_buffers,
        block_tables,
        attn_groups,
        kv_cache_config,
    ) -> None:
        if getattr(self, "replay_cudagraphs", None) is None:
            return

        def prepare(desc) -> None:
            num_reqs = min(desc.num_tokens, self.max_num_reqs)
            _prepare_replay_batch(
                self,
                InputBatch.make_dummy(num_reqs, desc.num_tokens, input_buffers),
                CUDAGraphMode.NONE,
                block_tables.get_dummy_block_tables(num_reqs),
                block_tables.get_dummy_slot_mappings(desc.num_tokens),
                attn_groups,
                kv_cache_config,
                self._replay_start[:num_reqs].zero_(),
                capture_desc=desc,
            )

        # profile_only captures use pre-KV-cache groups: drop the cache.
        self._replay_attn_groups = None
        self.replay_cudagraphs.capture_replay_graphs(prepare)

    cls.__init__ = replay_init
    cls.prepare_attn = replay_prepare
    cls.capture_inner_cudagraphs = capture_inner_cudagraphs
    setattr(cls, _MARKER, True)
    setattr(cls.__init__, _WRAPPER, True)
    setattr(cls.prepare_attn, _WRAPPER, True)
    setattr(cls.capture_inner_cudagraphs, _WRAPPER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = [
    "PATCH_ID",
    "TARGETS",
    "TARGET_MODULE",
    "ReplayAttnMetadata",
    "pad_replayed_slots",
    "apply",
    "apply_to_module",
]
