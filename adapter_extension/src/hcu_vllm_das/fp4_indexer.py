# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Owned V41 FP4 classes reusing official constructors and forward methods."""
from __future__ import annotations

import hashlib
import inspect
from types import FunctionType

import torch
from hcu_adapter.fp4_indexer import validate_config
from vllm.config import get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.models.deepseek_v41 import attention as common
from vllm.models.deepseek_v41.amd import rocm
from vllm.models.deepseek_v41.common.ops import indexer_k_store as store
from vllm.v1.attention.backends.mla import indexer as metadata

from . import fp4_packer
from .fp4_indexer_lightop import Fp4Query, cache_view, load_providers, providers, rotate_bf16


# anchor-baseline(③): Core 5afd482929, 0.33.1.dev3, constructor globals are
# private per-class views. A changed official source must be reviewed first.
_HASHES = {
    'attention': '3763cdcf7175cc9938fca5b7346d58a316fd7e9252f0adfea68ace3ddc2f6e4d',
    'cache': 'e1749993e4c2664e8254f18e46c973cf4cc4d59f86ce8d6031fde21fd4dbcc2d',
    'indexer': 'de4456631b3819ff05cccec895095c5bde438d79db5f1b1aca08eb43357b9a38',
    'metadata': '0f730868e855863d4e2c1a2432e460bdf5858c7ee3dfbd3ce4bcce5f3e222565',
}
_originals = {}


def _constructor(kind, function, substitutions):
    if hashlib.sha256(inspect.getsource(function).encode()).hexdigest() != _HASHES[kind]:
        raise RuntimeError('FP4 native constructor source changed: ' + kind)
    if not all(key in function.__globals__ for key in substitutions):
        raise RuntimeError('FP4 constructor anchor missing: ' + kind)
    namespace = dict(function.__globals__)
    namespace.update(substitutions)
    result = FunctionType(function.__code__, namespace, function.__name__,
                          function.__defaults__, function.__closure__)
    result.__kwdefaults__ = function.__kwdefaults__
    result.__annotations__ = function.__annotations__
    result.__qualname__ = function.__qualname__
    _originals[kind] = function
    return result


def check_native_contract():
    current = dict(attention=common.DeepseekV4Attention.__init__,
                   cache=common.DeepseekV4IndexerCache.__init__,
                   indexer=common.DeepseekV4Indexer.__init__,
                   metadata=metadata.DeepseekV32IndexerMetadataBuilder.__init__)
    if any(current[k] is not fn for k, fn in _originals.items()):
        raise RuntimeError('FP4 native constructor identity changed')


def preflight(cfg):
    plan = validate_config(cfg)
    if plan is None:
        raise ValueError('FP4 owned model requires explicit FP4 opt-in')
    from .v41_indexer_activation import check_health
    check_health()
    fp4_packer.check_health()
    if fp4_packer._state is None or fp4_packer._state['mode'] != 'triton':
        raise RuntimeError('FP4 lowering not published before model allocation')
    check_native_contract()
    load_providers()
    return plan


def _select(cfg):
    return validate_config(cfg) is not None


class Fp4MetadataBuilder(metadata.DeepseekV32IndexerMetadataBuilder):
    __init__ = _constructor('metadata', metadata.DeepseekV32IndexerMetadataBuilder.__init__,
                            {'dsa_indexer_uses_fp4': _select,
                             '_use_flattening': lambda cfg: True,
                             '_supports_varlen_paged_mqa_logits': lambda: False})


class Fp4IndexerBackend(metadata.DeepseekV41IndexerBackend):
    @staticmethod
    def get_builder_cls():
        return Fp4MetadataBuilder

    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec=None):
        # Logical block is always 64 tokens; ratio-2 pages are 32 physical
        # records handled by decode staging, not a different logical block.
        return [64]


class Fp4IndexerCache(common.DeepseekV4IndexerCache):
    # Keep normal V41 metadata, never claim gfx950's AITER page order.
    __init__ = _constructor('cache', common.DeepseekV4IndexerCache.__init__,
                            {'dsa_indexer_uses_fp4': lambda cfg: False})

    def get_attn_backend(self):
        return Fp4IndexerBackend

    def get_kv_cache_spec(self, vllm_config):
        plan = validate_config(vllm_config)
        if (plan is None or self.head_dim != 68
                or self.compress_ratio not in plan.ratios):
            raise ValueError('FP4 cache/spec admission changed before allocation')
        return super().get_kv_cache_spec(vllm_config)

    def bind_kv_cache(self, kv_cache):
        cache_view(kv_cache.squeeze(1))
        return super().bind_kv_cache(kv_cache)


_INDEXER_INIT = _constructor('indexer', common.DeepseekV4Indexer.__init__,
                             {'dsa_indexer_uses_fp4': _select})


class Fp4Indexer(common.DeepseekV4Indexer):
    def __init__(self, *args, **kwargs):
        cfg = args[0] if args else kwargs['vllm_config']
        preflight(cfg)
        _INDEXER_INIT(self, *args, **kwargs)

    def _produce_k(self, latent, positions, rotary_emb):
        meta = get_forward_context().attn_metadata
        if not isinstance(meta, dict) or latent is None:
            return
        cache = cache_view(self.k_cache.kv_cache)
        slots = meta[self.k_cache.prefix].slot_mapping
        k_pre, _ = self.wk(latent)
        if (not self.owns_k or k_pre.dtype != torch.bfloat16 or k_pre.ndim != 2
                or k_pre.shape[1] != 128 or k_pre.stride(1) != 1
                or slots.dtype not in (torch.int32, torch.int64)
                or slots.numel() > k_pre.shape[0] or slots.numel() > positions.numel()):
            raise ValueError('FP4 producer BF16/slot contract mismatch')
        if not slots.numel():
            return
        # The official fused norm/RoPE/quant/store owns every mathematical
        # and sentinel operation. Only its existing layout parameters differ.
        kernel = store._indexer_k_norm_rope_quant_store_kernel
        kernel[(slots.numel(),)](
            k_pre, k_pre.stride(0), positions, self.k_norm.weight,
            self.k_norm.variance_epsilon, rotary_emb.cos_sin_cache,
            rotary_emb.cos_sin_cache.stride(0), cache, slots, cache.shape[1],
            HEAD_SIZE=128, ROPE_HEAD_DIM=64, COMPRESS_RATIO=self.compress_ratio,
            TOKEN_STRIDE=64, SCALE_DIM=4, KV_BLOCK_STRIDE=cache.stride(0),
            FP8_MAX=448.0, USE_FP4=True, MX_SCALE=False, SHUFFLE=False,
            BLOCK_TILE_SIZE=1, HEAD_TILE_SIZE=1, num_warps=1)

    def forward_q(self, qr, qr_scale, indexer_weights, positions, rotary_emb):
        q = self._wq_b_proj(qr, qr_scale).view(-1, self.n_head, self.head_dim)
        packed, weights = common.fused_indexer_q_rope_quant(
            positions, q, rotary_emb.cos_sin_cache, indexer_weights,
            self.softmax_scale, self.n_head**-0.5, use_fp4=True,
            weights_out_dtype=torch.float32)
        bf16 = rotate_bf16(q, positions, rotary_emb.cos_sin_cache)
        return Fp4Query(*packed, bf16), None, weights


_COMMON_INIT = _constructor('attention', common.DeepseekV4Attention.__init__,
                            {'DeepseekV4IndexerCache': Fp4IndexerCache,
                             'dsa_indexer_uses_fp4': _select})


class _Fp4CommonAttention(common.DeepseekV4Attention):
    def __init__(self, *args, **kwargs):
        cfg = args[0] if args else kwargs['vllm_config']
        preflight(cfg)
        _COMMON_INIT(self, *args, **kwargs)


class Fp4Attention(rocm.DeepseekV41ROCMAiterMLAAttention, _Fp4CommonAttention):
    # MRO routes native AMD super().__init__ to our private common constructor;
    # its post-init, attention/forward and loading stay native.
    def _indexer_cls(self, k_cache):
        return Fp4Indexer
