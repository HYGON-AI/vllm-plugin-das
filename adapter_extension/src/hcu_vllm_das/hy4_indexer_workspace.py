# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Share a native Indexer K prefix between one request's PCP query segments."""
from dataclasses import replace


def make_indexer(*args, **kwargs):
    import os
    from hcu_adapter.infra.kill_switch import op_disabled
    from vllm.config import get_current_vllm_config
    from vllm.models.hy_v4.nvidia.attention import Indexer
    from .environment import custom_ops_enabled

    mode = os.environ.get('VLLM_HCU_SPARSE_MLA_PREFILL_WORKSPACE_DEDUP', '0')
    if mode not in ('0', '1'):
        raise ValueError('VLLM_HCU_SPARSE_MLA_PREFILL_WORKSPACE_DEDUP must be 0 or 1')
    if (mode == '0' or not custom_ops_enabled()
            or op_disabled('hy4_prefill_workspace')):
        return Indexer(*args, **kwargs)
    cfg = get_current_vllm_config()
    if (cfg.parallel_config.prefill_context_parallel_size <= 1
            or cfg.parallel_config.decode_context_parallel_size != 1
            or cfg.scheduler_config.async_scheduling
            or cfg.attention_config.hisparse_config is not None
            or cfg.cache_config.cache_dtype not in ('fp8', 'fp8_e4m3', 'fp8_ds_mla')):
        return Indexer(*args, **kwargs)
    from .hy4_indexer_backend import HcuHY4WorkspaceIndexer
    return HcuHY4WorkspaceIndexer(*args, **kwargs)


def _request_slice(chunk, base):
    """Witness the native block-table slice without reading device contents."""
    table = chunk.block_table
    if (table.ndim != 2 or base.ndim != 2 or table.device != base.device
            or table.dtype != base.dtype or table.shape[1] != base.shape[1]
            or table.stride() != base.stride() or base.stride(0) <= 0
            or base.stride(1) != 1 or chunk.num_reqs != table.shape[0]
            or table.untyped_storage().data_ptr() != base.untyped_storage().data_ptr()):
        return None
    offset = table.storage_offset() - base.storage_offset()
    if offset < 0 or offset % base.stride(0):
        return None
    lo = offset // base.stride(0)
    hi = lo + table.shape[0]
    return (lo, hi) if hi <= base.shape[0] else None


def coalesce_indexer_prefill(builder, common, metadata):
    """Transform whole native gather/continuation chains, leaving Q order intact.

    Synchronous scheduling makes the host lengths exact, including short PCP
    extends. Preserve the native per-Q causal width instead of inferring it
    from merged Q lengths: separate PCP segments can have distant positions.
    """
    if (not builder.use_pcp or builder.dcp_world_size != 1
            or builder.compress_ratio != 1 or builder.indexer_uses_fp4
            or builder.vllm_config.scheduler_config.async_scheduling
            or builder.vllm_config.attention_config.hisparse_config is not None
            or metadata.num_decodes or metadata.num_decode_tokens
            or metadata.prefill is None or not metadata.num_prefill_tokens
            or common.req_idx is None or common.mm_req_doc_ranges):
        return metadata
    import numpy as np
    import torch
    from vllm import envs
    from vllm.utils.torch_utils import async_tensor_h2d

    ids = np.asarray(common.req_idx)
    lengths, queries = common.seq_lens_cpu_upper_bound, common.query_start_loc_cpu
    count = common.num_reqs
    if (ids.ndim != 1 or len(ids) != count or not np.issubdtype(ids.dtype, np.integer)
            or np.any(ids < 0) or metadata.num_prefills != count
            or lengths is None or lengths.device.type != 'cpu'
            or queries.device.type != 'cpu' or len(lengths) < count
            or len(queries) != count + 1):
        return metadata
    q = queries.numpy()
    lens = lengths[:count].numpy()
    runs = np.r_[True, ids[1:] != ids[:-1]]
    if (len(ids) == 0 or int(runs.sum()) == count
            or int(runs.sum()) != len(np.unique(ids)) or np.any(lens <= 0)
            or np.any(q[1:] < q[:-1]) or int(q[-1]) != common.num_actual_tokens):
        return metadata

    # Validate every member before allocating transformed device metadata. A
    # failed witness anywhere retains the entire original metadata object.
    chains = []
    for chunk in metadata.prefill.chunks:
        witness = _request_slice(chunk, common.block_table_tensor)
        if witness is None:
            return metadata
        lo, hi = witness
        rows = chunk.token_end - chunk.token_start
        if (lo == hi or chunk.pcp_deinterleave_idx is not None
                or chunk.local_cu_seq_lens is not chunk.cu_seq_lens
                or chunk.cu_seq_lens.shape != (hi-lo+1,)
                or chunk.cu_seqlen_ks.shape != (rows,)
                or chunk.cu_seqlen_ke.shape != (rows,) or rows < 0
                or chunk.total_seq_lens != int(lens[lo:hi].sum())
                or chunk.local_total_seq_lens != chunk.total_seq_lens
                or chunk.max_local_total_seq_lens != chunk.total_seq_lens
                or chunk.token_start < int(q[lo]) or chunk.token_end > int(q[hi])):
            return metadata
        if not chunk.skip_kv_gather:
            if chunk.token_start != int(q[lo]):
                return metadata
            chains.append((witness, [chunk]))
        elif (not chains or chains[-1][0] != witness
              or chains[-1][1][-1].token_end != chunk.token_start):
            return metadata
        else:
            chains[-1][1].append(chunk)

    # Native chunking budgets duplicate virtual K regions. Two separate
    # single-row gathers can fit one shared region after Q concatenation.
    # Keep continuation chains intact, and never cross a real request boundary.
    for (lo, hi), chunks in chains:
        if chunks[-1].token_end != int(q[hi]):
            return metadata
    budget = min(128, envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB) * 1024 * 1024
    grouped = []
    cursor = 0
    while cursor < len(chains):
        (lo, hi), chunks = chains[cursor]
        parts = list(chunks)
        cursor += 1
        merge = False
        if len(parts) == 1 and hi-lo == 1:
            while cursor < len(chains):
                (next_lo, next_hi), next_parts = chains[cursor]
                if (next_lo != hi or next_hi-next_lo != 1 or len(next_parts) != 1
                        or ids[next_lo] != ids[lo]):
                    break
                extent = int(lens[lo:next_hi].max())
                rows = int(q[next_hi]-q[lo])
                if (extent > builder.max_prefill_buffer_size or rows <= 0
                        or rows * extent * 4 > budget):
                    break
                parts.extend(next_parts)
                hi = next_hi
                cursor += 1
                merge = True
        grouped.append(((lo, hi), parts, merge))

    plans = []
    for (lo, hi), chunks, merge in grouped:
        bounds = np.r_[0, np.flatnonzero(ids[lo+1:hi] != ids[lo:hi-1]) + 1, hi-lo]
        if len(bounds) == hi-lo+1:
            plans.append((chunks, None, False))
            continue
        representatives = np.array([lo+a+np.argmax(lens[lo+a:lo+b])
                                    for a, b in zip(bounds[:-1], bounds[1:])], dtype=np.int64)
        sizes = lens[representatives]
        total = int(sizes.sum())
        if (total > builder.max_prefill_buffer_size
                or total > max(c.total_seq_lens for c in chunks)):
            return metadata
        offsets = np.r_[0, np.cumsum(sizes[:-1], dtype=np.int32)].astype(np.int32)
        virtual_offsets = np.repeat(offsets, np.diff(bounds))
        q_offsets = np.repeat(virtual_offsets, np.diff(q[lo:hi+1]))
        plans.append((chunks, (lo, representatives, sizes, total, q_offsets), merge))

    result = []
    changed = False
    for chunks, plan, merge in plans:
        if plan is None:
            result.extend(chunks)
            continue
        changed = True
        lo, representatives, sizes, total, q_offsets = plan
        indices = async_tensor_h2d(representatives, device=builder.device, dtype=torch.long)
        table = common.block_table_tensor.index_select(0, indices).contiguous()
        device_lengths = common.seq_lens.index_select(0, indices)
        cu = torch.zeros(len(sizes)+1, dtype=torch.int32, device=builder.device)
        torch.cumsum(device_lengths, dim=0, out=cu[1:])
        token_to_seq = torch.repeat_interleave(
            torch.arange(len(sizes), device=builder.device, dtype=torch.int32),
            device_lengths, output_size=total)
        query_starts = async_tensor_h2d(q_offsets, device=builder.device, dtype=torch.int32)
        if merge:
            # The complete native chunks cover contiguous Q rows. Concatenate
            # their exact causal widths; the merged query count is not a
            # substitute for their original (potentially distant) positions.
            widths = torch.cat([c.cu_seqlen_ke-c.cu_seqlen_ks for c in chunks])
            result.append(replace(chunks[0], block_table=table, cu_seq_lens=cu,
                local_cu_seq_lens=cu, token_to_seq=token_to_seq,
                cu_seqlen_ks=query_starts, cu_seqlen_ke=query_starts+widths,
                token_end=chunks[-1].token_end,
                max_seqlen_k=max(c.max_seqlen_k for c in chunks), num_reqs=len(sizes),
                total_seq_lens=total, local_total_seq_lens=total,
                max_local_total_seq_lens=total))
            continue
        for chunk in chunks:
            begin, end = chunk.token_start-int(q[lo]), chunk.token_end-int(q[lo])
            starts = query_starts[begin:end]
            ends = chunk.cu_seqlen_ke - chunk.cu_seqlen_ks + starts
            result.append(replace(chunk, block_table=table, cu_seq_lens=cu,
                local_cu_seq_lens=cu, token_to_seq=token_to_seq,
                cu_seqlen_ks=starts, cu_seqlen_ke=ends, num_reqs=len(sizes),
                total_seq_lens=total, local_total_seq_lens=total,
                max_local_total_seq_lens=total))
    return replace(metadata, prefill=replace(metadata.prefill, chunks=result)) if changed else metadata
