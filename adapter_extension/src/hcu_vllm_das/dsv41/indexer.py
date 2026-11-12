# SPDX-License-Identifier: Apache-2.0
"""Reuse Core ROCm scoring with bounded prefill scratch and HCU SHUFFLE decode."""
import inspect
from types import FunctionType

import torch
from vllm.v1.attention.ops import rocm_aiter_mla_sparse as native

FP8_DTYPE = native.FP8_DTYPE

def _cache_flat(cache, head_dim):
    if cache.dtype != torch.uint8 or cache.ndim not in (3, 4):
        raise ValueError('FP8 indexer cache must be uint8 [pages,block,128+4]')
    if cache.shape[-1] != head_dim + 4 or (cache.ndim == 4 and cache.shape[2] != 1):
        raise ValueError('FP8 indexer cache byte width/head count mismatch')
    pages, block = cache.shape[:2]
    if head_dim != 128 or block <= 0 or (block > 1 and block % 16):
        raise ValueError('V41 FP8 indexer requires D128 and block1 or16-aligned SHUFFLE')
    try:
        return cache.view(pages, block * (head_dim + 4))
    except RuntimeError as error:
        raise ValueError('FP8 indexer cache has incompatible packed inner strides') from error


def decode_logits(q, cache, weights, context_lens, block_tables, schedule_metadata, max_model_len, *, compress_ratio=1):
    if q.ndim != 4 or q.shape[1] != 1 or q.dtype != FP8_DTYPE:
        raise ValueError('V41 reference Q must be FP8 [requests,next_n,heads,128]')
    batch, _, heads, dim = q.shape
    _cache_flat(cache, dim)
    if (weights.dtype != torch.float32 or weights.shape != (batch, heads)
            or weights.device != q.device or cache.device != q.device
            or block_tables.ndim != 2 or block_tables.shape[0] != batch
            or context_lens.shape not in ((batch,), (batch, 1))):
        raise ValueError("FP8 indexer query/weight/table/context contract mismatch")
    from .indexer_gpu import paged_logits
    return paged_logits(q[:, 0], cache, weights, context_lens.reshape(-1),
                        block_tables, max_model_len)



# Core's Torch fallback materializes [heads, queries, keys] intermediates;
# its outer logits budget only accounts for the reduced [queries, keys] result.
_PREFILL_SCORE_WORKSPACE_BYTES = 256 * 1024 * 1024


def _triton_prefill_scorer():
    # Core reaches this AITER Triton scorer only with VLLM_ROCM_USE_AITER=1,
    # which this deployment disables for other HCU-unsupported AITER paths.
    # Reuse Core's module lookup; the installed wrapper has no clean_logits.
    module = native.mqa_logits_module()
    function = getattr(module, 'fp8_mqa_logits', None)
    return function if callable(function) else None


def _torch_prefill_logits(q, kv, weights, cu_seqlen_ks, cu_seqlen_ke):
    rows, heads = q.shape[:2]
    keys = kv[0].shape[0]
    chunk_rows = max(1, _PREFILL_SCORE_WORKSPACE_BYTES // max(1, heads * keys * 4))
    if rows <= chunk_rows:
        return native.rocm_fp8_mqa_logits(q, kv, weights, cu_seqlen_ks, cu_seqlen_ke)
    logits = torch.empty((rows, keys), dtype=torch.float32, device=q.device)
    for start in range(0, rows, chunk_rows):
        end = min(start + chunk_rows, rows)
        logits[start:end].copy_(native.rocm_fp8_mqa_logits(
            q[start:end], kv, weights[start:end],
            cu_seqlen_ks[start:end], cu_seqlen_ke[start:end]))
    return logits


def prefill_logits(q, kv, weights, cu_seqlen_ks, cu_seqlen_ke):
    # Like Core's AITER path, values outside [ks, ke) are unspecified; Core's
    # candidate selection and prefill top-k read only that range.
    scorer = _PREFILL_SCORER
    heads, dim = q.shape[1:]
    if scorer is None or heads & (heads - 1) or dim & (dim - 1):
        return _torch_prefill_logits(q, kv, weights, cu_seqlen_ks, cu_seqlen_ke)
    k_fp8, scale = kv
    return scorer(q, k_fp8, scale.reshape(-1), weights, cu_seqlen_ks, cu_seqlen_ke)


_PREFILL_SCORER = _triton_prefill_scorer()


def _native_indexer():
    function = inspect.unwrap(native.rocm_aiter_sparse_attn_indexer)
    namespace = dict(function.__globals__, rocm_fp8_paged_mqa_logits=decode_logits,
                     rocm_fp8_mqa_logits=prefill_logits)
    return FunctionType(function.__code__, namespace, function.__name__,
                        function.__defaults__, function.__closure__)


_indexer = _native_indexer()


def indexer(layer, hidden_states, q, k, weights):
    if k is not None or not layer.skip_k_cache_insert or layer.use_fp4_cache:
        raise ValueError("PCP indexer requires source-owned FP8 K cache")
    if q.shape[0] == 0:
        return layer.topk_indices_buffer
    return _indexer(hidden_states, layer.k_cache.prefix, layer.k_cache.kv_cache,
        q, k, weights, layer.quant_block_size, layer.scale_fmt, layer.topk_tokens,
        layer.head_dim, layer.max_model_len, layer.max_total_seq_len,
        layer.topk_indices_buffer, skip_k_cache_insert=True,
        compress_ratio=layer.compress_ratio, candidate_blocks=layer.candidate_blocks,
        candidate_block_size=layer.candidate_block_size, candidate_write=layer.candidate_write)
