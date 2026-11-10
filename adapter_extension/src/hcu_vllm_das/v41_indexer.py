# SPDX-License-Identifier: Apache-2.0
"""Reuse Core's ROCm indexer with a SHUFFLE-aware HCU decode callback."""
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
    from .v41_indexer_gpu import paged_logits
    return paged_logits(q[:, 0], cache, weights, context_lens.reshape(-1),
                        block_tables, max_model_len)



def _native_indexer():
    function = inspect.unwrap(native.rocm_aiter_sparse_attn_indexer)
    namespace = dict(function.__globals__, rocm_fp8_paged_mqa_logits=decode_logits)
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
