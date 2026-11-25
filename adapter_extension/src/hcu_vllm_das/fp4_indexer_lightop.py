# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""LightOp FP4 query/cache contract and eager reader.

Decode uses a correctness-first staging pool for allocator pages whose physical
record count is not 64. Prefill remains zero-copy through the variable-page
``dsv41_gather_index_k_fp4`` ABI.
"""
from __future__ import annotations

from dataclasses import dataclass
import inspect
from importlib import import_module

import torch
from vllm.forward_context import get_forward_context
from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerMetadata
from .fp4_workspace import workspace_bytes


class Fp4Query(tuple):
    """Native packed Q/scale pair retaining prefill's BF16 RoPE view."""
    def __new__(cls, packed, scales, bf16):
        result = super().__new__(cls, (packed, scales))
        result.bf16 = bf16
        return result


@dataclass(frozen=True)
class Providers:
    gather: object
    prefill: object
    decode: object
    native: object
    native_calls: tuple = ()
    plan: object = None
    native_plan: object = None


_PLAN_NAME = 'get_paged_mqa_logits_fp4_metadata'
_CONTRACTS = (
    ('lightop.gemmopt', 'dsv41_gather_index_k_fp4', ('cache', 'slots', 'page_size')),
    ('lightop.gemmopt', 'fp8_fp4_mqa_logits',
     ('q', 'kv', 'weights', 'cu_seq_len_k_start', 'cu_seq_len_k_end', 'clean_logits',
      'max_seqlen_k', 'candidate_blocks', 'candidate_query_lens')),
    ('lightop.attention.mqa', 'paged_mqa_logits_fp4',
     ('q', 'q_sf', 'kv_cache', 'weights', 'context_lens', 'block_table',
      'schedule_meta', 'max_context_len', 'table_block_size', 'out', 'rows_per_request')),
)
_providers = None


def load_providers():
    global _providers
    calls = []
    native_calls = []
    native = import_module('lightop.op')
    for mod, name, parameters in _CONTRACTS:
        fn = getattr(import_module(mod), name, None)
        if not callable(fn) or tuple(inspect.signature(fn).parameters) != parameters:
            raise RuntimeError('FP4 provider ABI changed: ' + mod + '.' + name)
        native_fn = getattr(native, name, None)
        if not callable(native_fn):
            raise RuntimeError('FP4 native symbol missing: ' + name)
        calls.append(fn)
        native_calls.append(native_fn)
    prefill, decode = inspect.signature(calls[1]), inspect.signature(calls[2])
    if (prefill.parameters['candidate_blocks'].default is not None
            or prefill.parameters['candidate_query_lens'].default is not None
            or decode.parameters['schedule_meta'].default is not None
            or decode.parameters['max_context_len'].default is not None
            or decode.parameters['table_block_size'].default != 64
            or decode.parameters['out'].default is not None
            or decode.parameters['rows_per_request'].default != 1):
        raise RuntimeError('FP4 provider defaults changed')
    planner = getattr(import_module('lightop.attention.mqa'), _PLAN_NAME, None)
    native_plan = getattr(native, _PLAN_NAME, None)
    if callable(planner) and callable(native_plan):
        try:
            plan_params = inspect.signature(planner).parameters
        except (TypeError, ValueError):
            plan_params = {}
        if (tuple(plan_params) != ('context_lens', 'block_kv', 'num_sms', 'rows_per_request')
                or plan_params['rows_per_request'].default != 1):
            planner = native_plan = None
    else:
        planner = native_plan = None
    prepared = Providers(*calls, native, tuple(native_calls),
                         plan=planner, native_plan=native_plan)
    if _providers is not None and prepared != _providers:
        raise RuntimeError('FP4 provider ownership changed; start a new process')
    _providers = prepared
    return prepared


def providers():
    if _providers is None:
        raise RuntimeError('FP4 providers not prepared before allocation')
    # LightOp's Python wrappers resolve these native symbols on each call.
    # Check identity before dispatch without repeating startup ABI inspection.
    for (_, name, _), expected in zip(_CONTRACTS, _providers.native_calls):
        if getattr(_providers.native, name, None) is not expected:
            raise RuntimeError('FP4 provider ownership changed; start a new process')
    if (_providers.native_plan is not None
            and getattr(_providers.native, _PLAN_NAME, None) is not _providers.native_plan):
        raise RuntimeError('FP4 planner ownership changed; start a new process')
    return _providers


def cache_view(cache):
    """Normalize an allocator view to ``[pages, records, 68]`` bytes.

    The allocator uses ``records = block_size // compress_ratio``.  LightOp
    prefill accepts that variable page size, while decode stages non-64 pages
    before calling its fixed ABI.
    """
    if cache.ndim == 4 and cache.shape[2] == 1:
        cache = cache.squeeze(2)
    elif cache.ndim == 4 and cache.shape[1] == 1:
        cache = cache.squeeze(1)
    if (cache.dtype != torch.uint8 or cache.ndim != 3 or cache.shape[2] != 68
            or cache.shape[1] not in (32, 64, 128)
            or cache.stride(1) != 68 or cache.stride(2) != 1
            or cache.stride(0) < cache.shape[1] * 68 or cache.stride(0) % 4
            or cache.storage_offset() % 4):
        raise ValueError(
            'FP4 cache requires aligned nonoverlapping [P,R,68] page-local planes'
        )
    return cache


def physical_page_records(cache):
    """Return allocator records per physical FP4 page without host reads."""
    return cache.shape[1]


# One canonical decode page is 64 records of 68 bytes (64 payload + 4 scale).
STAGING_PAGE_BYTES = 4352
# The persistent pool reserved during the startup profile, keyed by page size.
# V4.1 carries a per-layer compress_ratio, so ratio-1 and ratio-2 layers coexist
# in one model and need pools of different page geometry. A single fixed key
# would make the second layer's request a shape mismatch, which the workspace
# manager rejects outright -- a startup failure, not a slow path.
_reserved_pools: dict[int, object] = {}
_reserved_fused: set[tuple[object, int]] = set()


def decode_rows(cfg):
    """Decode rows one step can carry, including speculative tokens."""
    spec = cfg.speculative_config
    extra = spec.num_speculative_tokens if spec else 0
    requests = cfg.scheduler_config.max_num_seqs
    parallel = getattr(cfg, 'parallel_config', None)
    if spec is not None and getattr(parallel, 'pcp_shard_decode_requests', False):
        # Native PCPManager assigns each step's compact decode ordinal modulo
        # PCP width. Each owner keeps all verification tokens of its requests.
        width = parallel.prefill_context_parallel_size
        requests = (requests + width - 1) // width
    return min(cfg.scheduler_config.max_num_batched_tokens, requests * (1 + extra))


def staging_plan(cfg, max_model_len, compress_ratio, *, fused=False):
    """Shapes the decode staging needs at the configured maximum.

    Returns ``None`` when the allocator already uses 64-record pages, because
    ``stage_decode_pages`` then returns the cache zero-copy and allocates
    nothing. Otherwise returns the persistent pool shape plus the byte count of
    the transient gather/index scratch that coexists with it.
    """
    block = cfg.cache_config.block_size
    if compress_ratio <= 0 or block % compress_ratio:
        return None
    page_records = block // compress_ratio
    if page_records == 64:
        return None
    requests = decode_rows(cfg)
    pages_per_row = (max_model_len + 63) // 64
    width = pages_per_row * 64
    slots = requests * width
    # Every tensor `stage_decode_pages` holds at the same time as the pool.
    # int64 index tensors are 8 bytes each and several of them coexist:
    # offsets, the valid mask, logical_pages, its clamp_max result, the
    # physical_pages gather, its int64 copy, slots, and the zeros_like used by
    # torch.where. The gather returns int8 payload (64 bytes/slot) plus an
    # int32 scale (4 bytes/slot), and both get a contiguous byte view. `table`
    # is int32 over the staged page count. Counted per slot, then per row.
    index_tensors = 8 * 9          # int64: offsets/valid/logical/clamp/phys/slots...
    payload_bytes = 64 * 2         # int8 payload plus its uint8 view
    scale_bytes = 4 * 2            # int32 scale plus its uint8 view
    transient = slots * (index_tensors + payload_bytes + scale_bytes)
    transient += slots * 4         # int32 staging block table
    transient += width * 8         # per-row offsets before broadcasting
    return dict(requests=requests, pages_per_row=pages_per_row,
                page_records=page_records,
                transient=requests * pages_per_row * 4 if fused else transient,
                fused=fused,
                shape=(requests, pages_per_row, STAGING_PAGE_BYTES))


def staging_key(plan):
    """Workspace key for one pool geometry.

    Distinct page-record counts get distinct pools; the shape itself is also
    part of the key so a changed max_model_len cannot silently reuse a buffer
    with the wrong dimensions.
    """
    return ('fp4_indexer_decode_staging', plan['page_records'], plan['shape'])


def _staging_pool(plan, device):
    """Return the persistent staging pool, workspace-backed when available.

    Reserving this during the startup profile is what keeps the decode staging
    inside the memory budget: the profile peak then covers it, instead of the
    first long-context decode step allocating it outside the planned peak.
    """
    from vllm.v1.worker.workspace import (current_workspace_manager,
                                          is_workspace_manager_initialized)
    if is_workspace_manager_initialized():
        return current_workspace_manager().get_persistent(
            staging_key(plan), plan['shape'], torch.uint8)
    return torch.empty(plan['shape'], dtype=torch.uint8, device=device)


def reserve_decode_staging(cfg, max_model_len, compress_ratio, device):
    """Account for the decode staging during the startup profile pass.

    Two separate costs are covered, because they behave differently:

    * the staging pool is a persistent workspace buffer that decode reuses, so
      it is created here and stays allocated for the life of the process;
    * the gather return buffers and index temporaries are allocated and freed
      per step by ``stage_decode_pages``, so a same-size transient allocation
      is made here purely to raise the measured profile peak. It is released
      immediately, which is what makes the peak honest without permanently
      holding the bytes.
    """
    from hcu_adapter.kernels.fp4_staging_triton import supports_staging_device

    plan = staging_plan(cfg, max_model_len, compress_ratio,
                        fused=supports_staging_device(device))
    return reserve_staging_plan(plan, device)


def reserve_staging_plan(plan, device):
    """Reserve a frozen construction-time plan and measure its scratch peak."""
    if plan is None:
        return None
    pool = _staging_pool(plan, device)
    probe = torch.empty(plan['transient'], dtype=torch.uint8, device=device)
    del probe
    _reserved_pools[plan['page_records']] = pool
    if plan.get('fused', False):
        _reserved_fused.add((pool.device, plan['page_records']))
    return pool


def _staging_target(cache, requests, pages_per_row):
    """Return the staging pages buffer for this step.

    The buffer is the persistent workspace pool reserved during the startup
    profile, and it is a *maximum*: the pool was sized from ``max_num_seqs`` and
    ``max_model_len``, while a real decode step carries fewer rows. Requiring an
    exact shape match would therefore almost never hit and every step would fall
    back to a fresh allocation -- which is the half of the original finding
    about repeated hot-path allocation, not just about the profile peak. So the
    reserved pool is reused whenever it can hold the request, and only an
    oversized request allocates.
    """
    pool = _reserved_pools.get(physical_page_records(cache))
    if pool is not None and pool.device == cache.device:
        reserved = pool.shape
        if reserved[0] >= requests and reserved[1] >= pages_per_row:
            # A prefix view over the reserved pages. The caller writes exactly
            # the pages it needs, and the inner stride still spans a full
            # canonical page, so the pool layout is unchanged.
            return pool[:requests, :pages_per_row]
    shape = (requests, pages_per_row, STAGING_PAGE_BYTES)
    return torch.empty(shape, dtype=torch.uint8, device=cache.device)


def stage_decode_pages(cache, block_table, row_ends, max_context_len, gather, *, live_only=False):
    """Stage live non-64 allocator pages into canonical decode pages.

    ``gather`` is the audited ``dsv41_gather_index_k_fp4`` provider.  It
    returns one ``[64]`` payload and one UE8M0 ``int32`` word per requested
    slot. With live_only, the device copier visits only live pages. Within
    the final page, invalid records use slot-zero bytes. The native scorer's
    live schedule excludes later pages. The gather fallback fills every page.
    """
    page_records = physical_page_records(cache)
    if page_records == 64:
        return cache, block_table
    if row_ends.ndim != 1 or block_table.ndim != 2:
        raise ValueError('FP4 staging requires flattened row lengths and tables')
    requests = row_ends.shape[0]
    pages_per_row = (max_context_len + 63) // 64
    width = pages_per_row * 64
    if block_table.shape[0] != requests:
        raise ValueError('FP4 staging row/table count mismatch')
    if block_table.shape[1] < (width + page_records - 1) // page_records:
        raise ValueError('FP4 staging block table is too narrow for requested context')
    # Native provider identity keeps injected gather collaborators observable.
    # CPU/unsupported hardware retains the original gather and repack path.
    if cache.is_cuda and _providers is not None and gather is _providers.gather:
        from hcu_adapter.kernels.fp4_staging_triton import stage_decode_pages as stage
        pages = _staging_target(cache, requests, pages_per_row)
        table = torch.empty((requests, pages_per_row), dtype=torch.int32, device=cache.device)
        if stage(cache, block_table, row_ends, pages, table, live_only=live_only):
            flat_pool = pages.reshape(requests * pages_per_row, STAGING_PAGE_BYTES)
            pool = flat_pool.as_strided((flat_pool.shape[0], 64, 68), (4352, 68, 1))
            return pool, table
        if (cache.device, page_records) in _reserved_fused:
            raise RuntimeError('Profiled Triton FP4 staging contract changed')
    offsets = torch.arange(width, device=cache.device, dtype=torch.int64)
    valid = offsets[None, :] < row_ends.reshape(-1, 1)
    logical_pages = torch.div(offsets, page_records, rounding_mode='floor')
    logical_pages = logical_pages.clamp_max(block_table.shape[1] - 1)
    physical_pages = block_table[:, logical_pages].to(torch.int64)
    slots = physical_pages * page_records + offsets.remainder(page_records)
    slots = torch.where(valid, slots, torch.zeros_like(slots)).reshape(-1).contiguous()
    flat = cache.as_strided((cache.shape[0], page_records * 68),
                            (cache.stride(0), 1))
    payload, scales = gather(flat, slots, page_records)
    if payload.dtype != torch.int8 or payload.shape != (slots.numel(), 64):
        raise ValueError('FP4 gather payload ABI mismatch during staging')
    if scales.dtype != torch.int32 or scales.shape != (slots.numel(),):
        raise ValueError('FP4 gather scale ABI mismatch during staging')
    payload_bytes = payload.contiguous().view(torch.uint8).reshape(
        requests, pages_per_row, 4096)
    scale_bytes = scales.contiguous().view(torch.uint8).reshape(
        requests, pages_per_row, 256)
    pages = _staging_target(cache, requests, pages_per_row)
    pages[..., :4096].copy_(payload_bytes)
    pages[..., 4096:].copy_(scale_bytes)
    flat_pool = pages.reshape(requests * pages_per_row, 4352)
    pool = flat_pool.as_strided(
        (flat_pool.shape[0], 64, 68), (4352, 68, 1)
    )
    table = torch.arange(requests * pages_per_row, device=cache.device,
                         dtype=torch.int32).reshape(requests, pages_per_row)
    return pool, table


def rotate_bf16(q, positions, rope):
    if q.dtype != torch.bfloat16 or q.ndim != 3 or q.shape[-1] != 128:
        raise ValueError('FP4 query requires BF16 [T,H,128]')
    if rope.ndim != 2 or rope.shape[1] != 64:
        raise ValueError('FP4 GPT-J RoPE table requires 64 columns')
    pairs = q[..., -64:].float().reshape(*q.shape[:2], 32, 2)
    cs = rope.index_select(0, positions.long()).float().unsqueeze(1)
    even, odd = pairs[..., 0], pairs[..., 1]
    out = q.clone()
    out[..., -64:] = torch.stack((even * cs[..., :32] - odd * cs[..., 32:],
                                  odd * cs[..., :32] + even * cs[..., 32:]), -1).flatten(-2).bfloat16()
    return out


def _validate(layer, meta, hidden, cache, query, weights):
    if not isinstance(meta, DeepseekV32IndexerMetadata):
        raise TypeError('FP4 requires official DeepseekV32IndexerMetadata')
    actual = meta.num_decode_tokens + meta.num_prefill_tokens
    if not isinstance(query, Fp4Query):
        raise TypeError('FP4 reader requires packed/scales/BF16 query views')
    q, scale = query
    if (q.dtype != torch.uint8 or q.ndim != 3 or q.shape[1:] != (32, 64)
            or q.shape[0] < actual or query.bf16.dtype != torch.bfloat16
            or query.bf16.shape != (q.shape[0], 32, 128)
            or scale.dtype != torch.int32 or scale.shape != q.shape[:2]
            or weights.dtype != torch.float32 or weights.shape != q.shape[:2]):
        raise ValueError('FP4 query/weight schema mismatch')
    out = layer.topk_indices_buffer
    if (out.dtype != torch.int32 or out.ndim != 2 or out.shape[0] < hidden.shape[0]
            or out.shape[0] < actual or out.shape[1] < layer.topk_tokens):
        raise ValueError('FP4 output buffer contract mismatch')
    if any(t.device != q.device for t in (cache, weights, scale, query.bf16, out)):
        raise ValueError('FP4 tensor devices differ')
    candidates = layer.candidate_blocks
    if candidates is not None:
        if (candidates.dtype != torch.int32 or candidates.ndim != 2
                or candidates.shape[0] < actual or candidates.shape[1] < 1
                or candidates.device != q.device or layer.candidate_block_size < 1):
            raise ValueError('FP4 candidate schema mismatch')
    elif layer.candidate_write:
        raise ValueError('FP4 candidate writer requires a bank')
    if meta.num_prefill_tokens and meta.prefill is None:
        raise ValueError('missing FP4 prefill metadata')
    if meta.num_decode_tokens:
        d = meta.decode
        n = meta.num_decode_tokens
        if (d is None or d.requires_padding or d.block_table.ndim != 2
                or d.block_table.shape[0] != n or d.block_table.dtype != torch.int32
                or d.seq_lens.shape not in ((n,), (n, 1))
                or d.seq_lens.dtype != torch.int32):
            raise ValueError('FP4 requires official flattened per-token decode metadata')
        if d.seq_lens.device != q.device or d.block_table.device != q.device:
            raise ValueError('FP4 decode metadata device mismatch')
    if meta.prefill is not None:
        end_decode = meta.num_decode_tokens
        for chunk in meta.prefill.chunks:
            if (not end_decode <= chunk.token_start <= chunk.token_end <= actual
                    or chunk.block_table.dtype != torch.int32
                    or chunk.token_to_seq.dtype not in (torch.int32, torch.int64)
                    or chunk.cu_seqlen_ks.dtype != torch.int32
                    or chunk.cu_seqlen_ke.dtype != torch.int32):
                raise ValueError('FP4 prefill chunk contract mismatch')
            for tensor in (chunk.block_table, chunk.token_to_seq, chunk.cu_seq_lens,
                           chunk.cu_seqlen_ks, chunk.cu_seqlen_ke):
                if tensor.device != q.device:
                    raise ValueError('FP4 prefill metadata device mismatch')
    return actual


def execute(layer, hidden, query, weights):
    context = get_forward_context()
    if not isinstance(context.attn_metadata, dict):
        # Match the explicit temporaries of the chunked real reader. Do not
        # reserve a full prefill_rows x max_model_len dense matrix, or size
        # decode staging from the prefill token count.
        size = workspace_bytes(
            query_rows=hidden.shape[0], decode_rows=layer.fp4_decode_rows,
            max_model_len=layer.max_model_len, total_seq_lens=layer.max_total_seq_len,
            topk_tokens=layer.topk_tokens, num_query_heads=32,
            max_logits_bytes=layer.fp4_max_logits_bytes,
            physical_page_size=layer.fp4_page_records, persistent_staging=True)
        # The real decode reuses the teacher's ratio-specific reserved pool.
        # Profile only the scratch that coexists with that pool.
        reserve_staging_plan(layer.fp4_staging_plan, hidden.device)
        # These allocations are transient in execute(), not consumers of the
        # persistent WorkspaceManager. Keep the estimate transient too: Core
        # memory_profiling records allocated_bytes.all.peak after it is freed.
        temporary = torch.empty((size,), dtype=torch.uint8, device=hidden.device)
        del temporary
        return layer.topk_indices_buffer
    from .v41_indexer import _select
    cache = cache_view(layer.k_cache.kv_cache)
    meta = context.attn_metadata[layer.k_cache.prefix]
    _validate(layer, meta, hidden, cache, query, weights)
    p = providers()
    width = (layer.max_model_len + 3) // 4 * 4
    out = layer.topk_indices_buffer[:, :layer.topk_tokens]
    out[:hidden.shape[0]].fill_(-1)
    page_records = physical_page_records(cache)
    if meta.prefill is not None:
        flat = cache.as_strided(
            (cache.shape[0], page_records * 68), (cache.stride(0), 1)
        )
        for chunk in meta.prefill.chunks:
            start, end = chunk.token_start, chunk.token_end
            total = chunk.total_seq_lens
            if not total or start == end:
                continue
            positions = torch.arange(total, dtype=torch.int64, device=cache.device)
            requests = chunk.token_to_seq[:total].long()
            local = positions - chunk.cu_seq_lens[requests]
            slots = (
                chunk.block_table[requests, local // page_records].long()
                * page_records + local % page_records
            ).contiguous()
            del positions, requests, local
            k, sf = p.gather(flat, slots, page_records)
            del slots
            for row_start in range(start, end, layer.fp4_query_chunk_rows):
                row_end = min(end, row_start + layer.fp4_query_chunk_rows)
                lo, hi = row_start - start, row_end - start
                ks, ke = chunk.cu_seqlen_ks[lo:hi], chunk.cu_seqlen_ke[lo:hi]
                logits = p.prefill(
                    (query.bf16[row_start:row_end].contiguous(), None), (k, sf),
                    weights[row_start:row_end].contiguous(), ks, ke,
                    False, width, None, None)
                # LightOp columns are relative to each query's K window.
                lengths = ke - ks
                starts = torch.zeros_like(lengths)
                _select(logits, starts, lengths, out[row_start:row_end],
                        None if layer.candidate_blocks is None
                        else layer.candidate_blocks[row_start:row_end],
                        layer.candidate_block_size, layer.candidate_write)
                del logits, lengths, starts
            # Also release the previous chunk's gather before the next gather
            # or decode-staging allocation (assignment alone releases too late).
            del k, sf
    if meta.num_decode_tokens:
        n = meta.num_decode_tokens
        d = meta.decode
        lengths = d.seq_lens.reshape(-1).contiguous()
        live_schedule = p.plan is not None and cache.is_cuda
        rows_per_request = 1
        if live_schedule:
            from .fp4_rowgroup import uniform_decode_rows, stage_request_rows
            from hcu_adapter.kernels.fp4_staging_triton import supports_staging_device

            if supports_staging_device(cache.device):
                rows_per_request = uniform_decode_rows(meta, topk_tokens=layer.topk_tokens)
            decode_cache, decode_table = stage_request_rows(
                cache, d.block_table, lengths, layer.max_model_len, p.gather,
                rows_per_request=rows_per_request, page_records=page_records,
                stage=stage_decode_pages)
        else:
            decode_cache, decode_table = stage_decode_pages(
                cache, d.block_table.contiguous(), lengths, layer.max_model_len, p.gather,
                live_only=False)
        schedule = None
        if live_schedule:
            sms = torch.cuda.get_device_properties(cache.device).multi_processor_count
            schedule = p.plan(lengths, 64, sms, rows_per_request)
        logits = p.decode(query[0][:n].unsqueeze(1).contiguous(),
                          query[1][:n].unsqueeze(1).contiguous(),
                          decode_cache.unsqueeze(2), weights[:n].contiguous(),
                          lengths, decode_table.contiguous(),
                          schedule, width, 64, None, rows_per_request)
        # Unsupported planners keep the fully staged and masked reader path.
        # The bounded selector masks unwritten tails when native TopK declines.
        if not live_schedule:
            columns = torch.arange(logits.shape[1], dtype=lengths.dtype, device=logits.device)
            logits.masked_fill_(columns[None, :] >= lengths[:, None], -float('inf'))
        _select(logits, torch.zeros_like(lengths), lengths, out[:n],
                None if layer.candidate_blocks is None else layer.candidate_blocks[:n],
                layer.candidate_block_size, layer.candidate_write, bounded=live_schedule)
    return layer.topk_indices_buffer
