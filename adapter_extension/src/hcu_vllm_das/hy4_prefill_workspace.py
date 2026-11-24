# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""One native MLA KV region per PCP request, shared by its query segments.

This coalesces adjacent segments of the SAME request. It does not infer shared
physical pages between different requests. Native gather, index conversion,
attention/sink kernels and the forward loop remain unchanged.
"""
from dataclasses import replace


def make_attention(*args, **kwargs):
    """Select a builder on this owned model layer, leaving registry/impl alone."""
    import os
    from hcu_adapter.infra.kill_switch import op_disabled
    # Preserve the constructor's existing cached class alias, even if the
    # public attention package subsequently publishes another alias.
    from vllm.models.hy_v4.nvidia.attention import MLAAttention

    mode = os.environ.get('VLLM_HCU_SPARSE_MLA_PREFILL_WORKSPACE_DEDUP', '0')
    if mode not in ('0', '1'):
        raise ValueError('VLLM_HCU_SPARSE_MLA_PREFILL_WORKSPACE_DEDUP must be 0 or 1')
    layer = MLAAttention(*args, **kwargs)
    if mode == '0' or op_disabled('hy4_prefill_workspace'):
        return layer
    from vllm.config import get_current_vllm_config
    parallel = get_current_vllm_config().parallel_config
    if parallel.prefill_context_parallel_size <= 1 or parallel.decode_context_parallel_size != 1:
        return layer
    # Only compose with the paired Adapter's actual sparse ABI owner. The
    # selected implementation and native cache-format canonicalization stay put.
    from hcu_adapter.kernels import vllm_sparse_flashmla_dtk as adapter
    if (layer.attn_backend is not getattr(adapter, 'HcuSparseFlashMLABackend', None)
            or layer.impl.kv_cache_dtype != 'fp8_ds_mla'):
        return layer
    from .hy4_prefill_backend import HcuHY4PCPWorkspaceBackend
    layer.attn_backend = HcuHY4PCPWorkspaceBackend
    return layer


def coalesce_prefill(builder, common, extra, *, num_decode_rows=None):
    if (not builder.use_pcp or builder.pcp_dcp_kv_gather or builder.use_hisparse
            or extra.prefill is None or extra.num_prefill_tokens == 0
            or common.req_idx is None):
        return extra
    import numpy as np
    import torch
    from vllm.v1.attention.backends.mla.flashmla_sparse import (
        FlashMLASparseMetadata, get_prefill_workspace_size,
    )
    from vllm.v1.attention.backends.mla.sparse_utils import request_row_bounds
    from vllm.v1.attention.backends.utils import split_prefill_chunks
    from vllm.utils.torch_utils import async_tensor_h2d

    # Collective padding can make the row count larger than active decodes.
    begin = extra.num_decodes if num_decode_rows is None else num_decode_rows
    end = begin + extra.num_prefills
    ids = common.req_idx[begin:end]
    if len(ids) == 0 or len(ids) != extra.num_prefills:
        return extra
    # Nonadjacent repeats need query permutation; retain the native layout.
    runs = np.count_nonzero(np.r_[True, ids[1:] != ids[:-1]])
    if runs == len(ids) or runs != len(np.unique(ids)):
        return extra
    lengths = common.seq_lens_cpu_upper_bound
    queries = common.query_start_loc_cpu
    if lengths is None or lengths.device.type != 'cpu' or queries.device.type != 'cpu':
        return extra
    lengths = lengths[begin:end].numpy()
    bounds = request_row_bounds(ids)
    if any(int(queries[begin + hi]) <= int(queries[begin + lo])
           for lo, hi in zip(bounds[:-1], bounds[1:])):
        return extra
    representatives = np.array([lo + np.argmax(lengths[lo:hi])
                                for lo, hi in zip(bounds[:-1], bounds[1:])])
    rows = lengths[representatives]
    if np.any(rows <= 0):
        return extra
    capacity = get_prefill_workspace_size(builder.vllm_config)
    if np.any(rows > capacity):
        return extra
    chunk_bounds = split_prefill_chunks(torch.from_numpy(rows), capacity)
    # Small, cache-resident contexts do not justify an otherwise identical
    # launch. Still coalesce when it removes native gather/attention calls.
    if len(chunk_bounds) >= len(extra.prefill.chunks):
        cache_bytes = getattr(builder, '_hcu_workspace_l2_bytes', 0)
        native_rows = sum(int(c.chunk_tot_seqlen) for c in extra.prefill.chunks)
        if cache_bytes <= 0 or native_rows * 576 * 2 <= cache_bytes:
            return extra
    workspace_starts = np.zeros(len(rows), dtype=np.int32)
    request_ids = np.full(common.num_actual_tokens, -1, dtype=np.int32)
    for index, (lo, hi) in enumerate(zip(bounds[:-1], bounds[1:])):
        request_ids[int(queries[begin + lo]):int(queries[begin + hi])] = index
    for lo, hi in chunk_bounds:
        workspace_starts[lo + 1:hi] = np.cumsum(rows[lo:hi - 1])
    indices = async_tensor_h2d(representatives, device=builder.device, dtype=torch.long)
    tables = common.block_table_tensor[begin:end].index_select(0, indices)
    sequence_lengths = common.seq_lens[begin:end].index_select(0, indices)
    starts = async_tensor_h2d(workspace_starts, device=builder.device, dtype=torch.int32)
    FP8Meta = FlashMLASparseMetadata.FP8SeparatePrefillDecode
    chunks = [FP8Meta.Prefill.Chunk(
        tokens_slice=slice(int(queries[begin + bounds[lo]]), int(queries[begin + bounds[hi]])),
        block_table=tables[lo:hi], req_start_idx=lo, workspace_starts=starts[lo:hi],
        chunk_tot_seqlen=int(rows[lo:hi].sum()), seq_lens=sequence_lengths[lo:hi])
        for lo, hi in chunk_bounds]
    prefill = FP8Meta.Prefill(
        request_ids=async_tensor_h2d(request_ids, device=builder.device, dtype=torch.int32),
        workspace_starts=starts, chunks=chunks)
    return replace(extra, prefill=prefill)
