# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Owned V41 FP4 classes reusing official constructors and forward methods."""
from __future__ import annotations

from .adapter_api import adapter_module as _adapter_module

import hashlib
import inspect
from types import FunctionType

import torch
validate_config = _adapter_module('fp4_indexer').validate_config
from vllm.config import get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.models.deepseek_v41 import attention as common
from vllm.models.deepseek_v41.amd import rocm
from vllm.models.deepseek_v41.common.ops import indexer_k_store as store
from vllm.v1.attention.backends.mla import indexer as metadata

from . import fp4_packer, wo_a_layout
from .fp4_indexer_lightop import Fp4Query, cache_view, load_providers, providers, rotate_bf16
from .v41_pcp import HcuV41PcpAttention
from .qkv_split import qualify_qkv_split, split_qkv, supported as qkv_split_supported


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
    from .v41_indexer_activation import admit_config, check_health
    admit_config(cfg)
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
    def _wq_b_proj(self, qr, qr_scale=None):
        if isinstance(qr, torch.Tensor) and qr.dtype == torch.int8:
            from .int8_wq_b import project

            return project(self.wq_b, qr, qr_scale, indexer=True)
        return super()._wq_b_proj(qr, qr_scale)

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


def _compressor_factory(*args, **kwargs):
    cfg = args[0] if args else kwargs['vllm_config']
    if cfg.parallel_config.prefill_context_parallel_size > 1 and cfg.speculative_config is not None:
        from .pcp_dspark import HcuPcpDSparkCompressor
        return HcuPcpDSparkCompressor(*args, **kwargs)
    return common.DeepseekCompressor(*args, **kwargs)


_COMMON_INIT = _constructor('attention', common.DeepseekV4Attention.__init__,
                            {'DeepseekV4IndexerCache': Fp4IndexerCache,
                             'dsa_indexer_uses_fp4': _select,
                             'DeepseekCompressor': _compressor_factory})


class _Fp4CommonAttention(common.DeepseekV4Attention):
    def __init__(self, *args, **kwargs):
        cfg = args[0] if args else kwargs['vllm_config']
        preflight(cfg)
        _COMMON_INIT(self, *args, **kwargs)


class Fp4Attention(rocm.DeepseekV41ROCMAiterMLAAttention, _Fp4CommonAttention):
    # MRO routes native AMD super().__init__ to our private common constructor;
    # Its post-init, attention/forward and loading stay native.
    def __init__(self, *args, **kwargs):
        from .environment import _flag

        cfg = args[0] if args else kwargs['vllm_config']
        super().__init__(*args, **kwargs)
        if (not self.use_pcp and not self.kv_mxfp8 and not self.accepts_unnormed_unroped_query
                and self.n_local_heads == self.padded_heads == 64):
            qualify_qkv_split(self.rotary_emb.cos_sin_cache.device)
        self._hcu_wo_a_layout = wo_a_layout.configured(cfg, self.wo_a.weight.device)
        self._int8_wq_b_device = None
        self._int8_wq_b_direct = None
        if type(self) is Fp4Attention and _flag('HCU_DSV41_WQ_B_INT8_DIRECT', default=False):
            from .int8_wq_b import configure_direct_gemm

            self._int8_wq_b_direct = configure_direct_gemm(cfg)
        if _flag('VLLM_HCU_DSV41_INT8_WQ_B', default=True) and type(self.indexer) is Fp4Indexer:
            from .int8_wq_b import configured_device

            self._int8_wq_b_device = configured_device(self.indexer.vllm_config)

    def _o_proj(self, attn_out, positions):
        if self._hcu_wo_a_layout and wo_a_layout.supported(self, attn_out):
            return self._wo_b_after_wo_a(wo_a_layout.project(self, attn_out))
        return super()._o_proj(attn_out, positions)

    def _indexer_cls(self, k_cache):
        return Fp4Indexer

    def _split_qkv_and_norm(self, qr_kv):
        qr, qr_scale, kv = super()._split_qkv_and_norm(qr_kv)
        if (self._int8_wq_b_device is not None and type(self) is Fp4Attention
                and type(self.indexer) is Fp4Indexer
                and self._prepare_and_attn_fn == self._prepare_and_attn
                and getattr(self.indexer.forward_q, '__func__', None) is Fp4Indexer.forward_q):
            from .int8_wq_b import share

            # FULL, reindex and serial prep already carry the pair over their
            # own fork/join (or call) boundaries. It is never stored on a layer.
            qr, qr_scale = share(self.indexer.vllm_config, self._int8_wq_b_device,
                                 self, qr, qr_scale)
        return qr, qr_scale, kv

    def _wq_b_proj(self, qr, qr_scale=None):
        if self._int8_wq_b_direct is not None:
            from .int8_wq_b import project_direct

            out = project_direct(self._int8_wq_b_direct, self.wq_b, qr, qr_scale)
            if out is not None:
                return out
        if isinstance(qr, torch.Tensor) and qr.dtype == torch.int8:
            from .int8_wq_b import project

            return project(self.wq_b, qr, qr_scale, indexer=False)
        return super()._wq_b_proj(qr, qr_scale)

    def _fused_qnorm_rope_kv_insert(self, q, kv, positions, attn_metadata):
        if (isinstance(attn_metadata, dict) and not self.use_pcp
                and not self.kv_mxfp8 and not self.accepts_unnormed_unroped_query
                and self.n_local_heads == self.padded_heads == 64):
            swa = attn_metadata.get(self.swa_cache_layer.prefix)
            cache = self.swa_cache_layer.kv_cache
            if (swa is not None and swa.block_size in (32, 64)
                    and cache.dtype == torch.uint8
                    and cache.ndim > 0 and cache.shape[0] > 0
                    and cache.numel() == cache.shape[0] * swa.block_size * 584):
                cache = cache.view(cache.shape[0], swa.block_size, 584)
                cos_sin = self.rotary_emb.cos_sin_cache
                if qkv_split_supported(q, kv, cache, swa.slot_mapping, positions, cos_sin):
                    return split_qkv(q, kv, cache, swa.slot_mapping, positions, cos_sin)
        return super()._fused_qnorm_rope_kv_insert(q, kv, positions, attn_metadata)


class Fp4PcpAttention(HcuV41PcpAttention, _Fp4CommonAttention):
    """Native PCP preparation plus the FP4 private common constructor.

    MRO keeps the AMD post-init and PCP local-row/empty-rank handling. Only
    the index cache and indexer change; collectives and source-layer sharing
    remain in the unmodified common forward.
    """
    def _indexer_cls(self, k_cache):
        return Fp4Indexer


_FP4_OUTPUT_PROJECTION = Fp4Attention._o_proj
