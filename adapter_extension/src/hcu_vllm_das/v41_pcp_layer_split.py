# SPDX-License-Identifier: Apache-2.0
"""Layer-owned KV placement layered on the official Adapter PCP classes.

The base owns native PCP preparation, scoring and query extents. This extension
only owns cache write filtering, cache broadcasts and placement metadata.
"""
import copy
from functools import cache

from vllm.forward_context import get_forward_context, override_forward_context
from vllm.models.deepseek_v41.attention import DeepseekV4Attention
from hcu_adapter.kernels import pcp_fork
from .dsa_cache_layer_split import (
    _prefix_owner_rank, broadcast_kv_source_cache,
    layer_split_state, record_layer_split_state, swa_cache_owner_rank,
    owner_swa_write_metadata,
    _is_owner,
)

_base = pcp_fork.build_classes()

class LayerSplitIndexerMixin:
    """Share placement semantics across the distinct FP8 and FP4 producers."""

    def _produce_k(self, latent, positions, rotary_emb):
        if _is_owner(self.k_cache.prefix) is False:
            return
        return super()._produce_k(latent, positions, rotary_emb)

    def forward(self, *args, **kwargs):
        result = super().forward(*args, **kwargs)
        state = layer_split_state()
        if state is not None:
            # Every index-source layer reaches this boundary, including the
            # short-context top-k shortcut and ranks with zero local queries.
            broadcast_kv_source_cache(
                self.k_cache.prefix,
                _prefix_owner_rank(self.k_cache.prefix, state[1], state[0]),
                self.k_cache.kv_cache,
            )
        return result


class LayerSplitAttentionMixin:
    """Storage ownership only; math and constructors come from the selected base."""

    _compressed_kv_cache = DeepseekV4Attention._compressed_kv_cache

    def _fused_qnorm_rope_kv_insert(self, q, kv, positions, attn_metadata):
        metadata = owner_swa_write_metadata(attn_metadata, self.swa_cache_layer.prefix)
        return super()._fused_qnorm_rope_kv_insert(q, kv, positions, metadata)

    def forward_kv(self, positions, hidden_states):
        # Decoder replay first writes all rows at the cut layer, then reruns
        # the bounded sub-batch. Guard BOTH insertion routes with the same
        # ownership, retaining native gather/compressor/indexer ordering.
        context = get_forward_context()
        metadata = owner_swa_write_metadata(context.attn_metadata, self.swa_cache_layer.prefix)
        if metadata is context.attn_metadata:
            return super().forward_kv(positions, hidden_states)
        local = copy.copy(context)
        local.attn_metadata = metadata
        with override_forward_context(local):
            return super().forward_kv(positions, hidden_states)

    def _broadcast_swa_cache(self):
        state = layer_split_state()
        if state is not None:
            prefix = self.swa_cache_layer.prefix
            broadcast_kv_source_cache(prefix, swa_cache_owner_rank(prefix, state[0]),
                                      self.swa_cache_layer.kv_cache)

    def _broadcast_compressed_cache(self):
        state = layer_split_state()
        if state is not None and self.compressed_cache_prefix is not None:
            broadcast_kv_source_cache(
                self.compressed_cache_prefix,
                _prefix_owner_rank(self.compressed_cache_prefix, state[1], state[0]),
                self._compressed_kv_cache(),
            )

    def forward_mqa(self, q, kv, positions, output):
        if get_forward_context().attn_metadata is not None:
            # Empty query ranks must enter the same cache collectives.
            self._broadcast_compressed_cache()
            self._broadcast_swa_cache()
        return super().forward_mqa(q, kv, positions, output)

    def get_kv_cache_spec(self, vllm_config):
        # Every Rank reports the complete native group structure. Only the
        # builder/worker placement differs by owner, including idle owners.
        record_layer_split_state(vllm_config)
        return super().get_kv_cache_spec(vllm_config)


class HcuV41PcpIndexer(LayerSplitIndexerMixin, _base["indexer"]):
    pass


class HcuV41PcpAttention(LayerSplitAttentionMixin, _base["attention"]):
    def _indexer_cls(self, k_cache):
        if k_cache is not None and k_cache.rocm_mxfp4:
            raise ValueError('PCP adapter requires the FP8 SHUFFLE index K layout')
        return HcuV41PcpIndexer


@cache
def dspark_attention_class():
    from .pcp_dspark import HcuV41PcpDSparkAttention

    class HcuV41LayerSplitDSparkAttention(HcuV41PcpAttention, HcuV41PcpDSparkAttention):
        """Use the Adapter's speculative compressor inside layer-owned attention."""

    return HcuV41LayerSplitDSparkAttention
