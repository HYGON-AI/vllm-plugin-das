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
from vllm.config import get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerMetadata


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
    native = import_module('lightop.op')
    for mod, name, parameters in _CONTRACTS:
        fn = getattr(import_module(mod), name, None)
        if not callable(fn) or tuple(inspect.signature(fn).parameters) != parameters:
            raise RuntimeError('FP4 provider ABI changed: ' + mod + '.' + name)
        if not callable(getattr(native, name, None)):
            raise RuntimeError('FP4 native symbol missing: ' + name)
        calls.append(fn)
    prefill, decode = inspect.signature(calls[1]), inspect.signature(calls[2])
    if (prefill.parameters['candidate_blocks'].default is not None
            or prefill.parameters['candidate_query_lens'].default is not None
            or decode.parameters['schedule_meta'].default is not None
            or decode.parameters['max_context_len'].default is not None
            or decode.parameters['table_block_size'].default != 64
            or decode.parameters['out'].default is not None
            or decode.parameters['rows_per_request'].default != 1):
        raise RuntimeError('FP4 provider defaults changed')
    prepared = Providers(*calls, native)
    if _providers is not None and prepared != _providers:
        raise RuntimeError('FP4 provider ownership changed; start a new process')
    _providers = prepared
    return prepared


def providers():
    if _providers is None:
        raise RuntimeError('FP4 providers not prepared before allocation')
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
_STAGING_KEY = ('fp4_indexer_decode_staging',)
# The persistent pool reserved during the startup profile, if one was reserved.
_reserved_pool = None


def decode_rows(cfg):
    """Decode rows one step can carry, including speculative tokens."""
    spec = cfg.speculative_config
    extra = spec.num_speculative_tokens if spec else 0
    return cfg.scheduler_config.max_num_seqs * (1 + extra)


def staging_plan(cfg, max_model_len, compress_ratio):
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
    # Conservative sum of everything stage_decode_pages allocates besides the
    # pool itself: offsets/valid/logical/physical/slots index tensors, the
    # gather payload and scale returns, and their contiguous byte views.
    transient = slots * (8 * 5 + 1 + 64 * 2 + 4 * 2) + width * 8
    return dict(requests=requests, pages_per_row=pages_per_row,
                page_records=page_records, transient=transient,
                shape=(requests, pages_per_row, STAGING_PAGE_BYTES))


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
            _STAGING_KEY, plan['shape'], torch.uint8)
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
    plan = staging_plan(cfg, max_model_len, compress_ratio)
    if plan is None:
        return None
    pool = _staging_pool(plan, device)
    probe = torch.empty(plan['transient'], dtype=torch.uint8, device=device)
    del probe
    global _reserved_pool
    _reserved_pool = pool
    return pool


def _staging_target(cache, requests, pages_per_row):
    """Return the staging pages buffer for this step.

    The buffer is the persistent workspace pool reserved during the startup
    profile whenever it fits, so the decode hot path stops allocating a new
    staging pool outside the planned peak. A request set larger than the
    reserved maximum (only reachable when the scheduler exceeds the configured
    bound) still gets a correct, unplanned allocation rather than a wrong
    answer.
    """
    shape = (requests, pages_per_row, STAGING_PAGE_BYTES)
    pool = _reserved_pool
    if pool is not None and pool.shape == shape and pool.device == cache.device:
        return pool
    return torch.empty(shape, dtype=torch.uint8, device=cache.device)


def stage_decode_pages(cache, block_table, row_ends, max_context_len, gather):
    """Stage non-64 allocator pages into LightOp's canonical decode pages.

    ``gather`` is the audited ``dsv41_gather_index_k_fp4`` provider.  It
    returns one ``[64]`` payload and one UE8M0 ``int32`` word per requested
    slot.  Invalid columns use slot zero only as an in-bounds read and are
    masked by the native scorer through ``row_ends``.
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
        # Profile the real dense logits bound without reading the empty cache.
        from vllm.v1.worker.workspace import current_workspace_manager
        cfg = get_current_vllm_config()
        rows = max(hidden.shape[0], decode_rows(cfg))
        current_workspace_manager().get_simultaneous(
            ((rows, (layer.max_model_len + 3) // 4 * 4), torch.float32))
        # The decode staging pool and its per-step scratch are allocated here,
        # during the profile pass, so the measured peak covers them. Reserving
        # only the FP32 logits above is what let a long-context decode allocate
        # up to ~272 MiB (131k context, 32 concurrent) outside the budget.
        reserve_decode_staging(cfg, layer.max_model_len, layer.compress_ratio,
                               hidden.device)
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
            k, sf = p.gather(flat, slots, page_records)
            logits = p.prefill((query.bf16[start:end].contiguous(), None), (k, sf),
                               weights[start:end].contiguous(), chunk.cu_seqlen_ks,
                               chunk.cu_seqlen_ke, False, width, None, None)
            lengths = chunk.cu_seqlen_ke - chunk.cu_seqlen_ks
            starts = torch.zeros_like(lengths)
            _select(logits, starts, lengths, out[start:end],
                    None if layer.candidate_blocks is None else layer.candidate_blocks[start:end],
                    layer.candidate_block_size, layer.candidate_write)
    if meta.num_decode_tokens:
        n = meta.num_decode_tokens
        d = meta.decode
        lengths = d.seq_lens.reshape(-1).contiguous()
        decode_cache, decode_table = stage_decode_pages(
            cache, d.block_table.contiguous(), lengths, layer.max_model_len, p.gather
        )
        logits = p.decode(query[0][:n].unsqueeze(1).contiguous(),
                          query[1][:n].unsqueeze(1).contiguous(),
                          decode_cache.unsqueeze(2), weights[:n].contiguous(),
                          lengths, decode_table.contiguous(),
                          None, width, 64, None, 1)
        _select(logits, torch.zeros_like(lengths), lengths, out[:n],
                None if layer.candidate_blocks is None else layer.candidate_blocks[:n],
                layer.candidate_block_size, layer.candidate_write)
    return layer.topk_indices_buffer
