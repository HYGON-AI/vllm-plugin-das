# SPDX-License-Identifier: Apache-2.0
"""Reuse Core constructors with private HCU attention/MoE class bindings.

Core owns scheduling, caches, Engram storage/lookup and weight loading.
Bridge ROCm attention, EP-local expert counts and Engram head sharding over PCP.
"""
from __future__ import annotations

import hashlib
import inspect
from types import FunctionType

import torch
from vllm.config import get_current_vllm_config
from vllm.distributed import get_pcp_group
from vllm.model_executor.kernels.mhc.triton import hc_collapse_triton
from vllm.model_executor.layers import mhc as native_mhc
from vllm.models.deepseek_v41.common import engram as native_engram
from vllm.models.deepseek_v41.amd import model as native
from vllm.models.deepseek_v41.amd import vl_model as native_vl
from ..dsv41.profile_worker import local_token_bound
from ..dsv41.mhc import mhc_post
from ..dsv41.pcp_attention import HcuDSV41PcpAttention, validate_config

_DECODER_NATIVE = native.DeepseekV4DecoderLayer.__init__
_BACKBONE_NATIVE = native.DeepseekV4Model.__init__
_OUTER_NATIVE = native_vl.DeepseekV41ForCausalLM.__init__
_REVIEWED = (
    '919c98099d1ba3e443ea5a5c5569e6bd407caf52c40945fb792e9b4e85a13a86',
    'b804d0bd4d325397b0df3a094f47456a715b2d431cb06d2c7a06c6bc8332681b',
    '41b21ccf6dc5d66c98082650c8e046b977f0e61dce927aab890e23e508bab434',
)


def _constructor(function, expected_hash, substitutions):
    actual = hashlib.sha256(inspect.getsource(function).encode()).hexdigest()
    if actual != expected_hash:
        raise RuntimeError(f'HCU V41 PCP constructor source changed: {function.__qualname__}: {actual}')
    if not substitutions or any(name not in function.__globals__ for name in substitutions):
        raise RuntimeError('HCU V41 PCP constructor class-name anchor missing')
    namespace = dict(function.__globals__)
    namespace.update(substitutions)
    clone = FunctionType(function.__code__, namespace, function.__name__,
                         function.__defaults__, function.__closure__)
    clone.__kwdefaults__ = function.__kwdefaults__
    clone.__annotations__ = function.__annotations__
    clone.__qualname__ = function.__qualname__
    return clone


def check_native_contract():
    if (native.DeepseekV4DecoderLayer.__init__ is not _DECODER_NATIVE
            or native.DeepseekV4Model.__init__ is not _BACKBONE_NATIVE
            or native_vl.DeepseekV41ForCausalLM.__init__ is not _OUTER_NATIVE):
        raise RuntimeError('HCU V41 PCP native constructors changed after preparation')


def select_attention(cfg):
    validate_config(cfg)
    return HcuDSV41PcpAttention



class HcuDSV41PcpMHCPostOp(native_mhc.MHCPostOp):
    def forward_hip(self, x, residual, post_layer_mix, comb_res_mix):
        return mhc_post(x, residual, post_layer_mix, comb_res_mix)


class HcuDSV41PcpMHCPreDelayedOp(native_mhc.MHCPreDelayedOp):
    # Core's Torch gates avoid the unsupported HCU TileLang delayed seam.
    __init__ = _constructor(
        native_mhc.MHCPreDelayedOp.__init__,
        '82cd4838e10ee5ea00ed884145d38903350dbdff35129d745c0440a892ea3552',
        {'MHCPostOp': HcuDSV41PcpMHCPostOp})

    def forward_hip(self, residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
                    hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat,
                    pre_mix=None, x=None, norm_weight=None, norm_eps=1e-6,
                    sublayer_out=None, post_layer_mix=None, comb_res_mix=None):
        residual = self._maybe_post(residual, sublayer_out, post_layer_mix, comb_res_mix)
        # Without pre_mix Core returns a free view; collapse with Core's Triton op.
        post_mix, comb_mix, layer_input, next_pre_mix = native_mhc.mhc_kernels.mhc_pre_delayed_torch(
            residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps, hc_sinkhorn_eps,
            hc_post_mult_value, sinkhorn_repeat, pre_mix=None, x=x)
        if pre_mix is not None:
            layer_input = hc_collapse_triton(residual, pre_mix)
        return (residual, post_mix, comb_mix,
                native_mhc._apply_mhc_norm(layer_input, norm_weight, norm_eps), next_pre_mix)


class HcuDSV41PcpEngramEmbedding(native_engram.ParallelEngramEmbedding):
    def _get_shard_info(self):
        if self.dp_size != 1 or self.tp_size != 1:
            raise ValueError('HCU V41 PCP Engram requires TP1/DP1')
        group = get_pcp_group()
        return group.world_size, group.rank_in_group


_PCP_GATHER_ROWS = _constructor(
    native_engram._gather_engram_rows,
    '039948803a336294f00683bd5708409441d5d538390a5f1fdf5514026c61cc1f',
    {'get_engram_dp_group': get_pcp_group})


class HcuDSV41PcpEngram(native_engram.Engram):
    _create_embedding = _constructor(
        native_engram.Engram._create_embedding,
        'ef2909d6afcc990d4a36677768da25e46f8f88bf779830469a143c436f42ae5c',
        {'ParallelEngramEmbedding': HcuDSV41PcpEngramEmbedding})

    def _init_staging(self, max_tokens, head_dim):
        pcp_size = get_pcp_group().world_size
        max_reqs = get_current_vllm_config().scheduler_config.max_num_seqs
        # The hash gather pads each rank to the largest rank-local token count.
        slot = local_token_bound(max_tokens, max_reqs, pcp_size)
        super()._init_staging(slot * pcp_size, head_dim)
        self._pcp_prepared = None

    def prepare_embeddings(self, hash_ids):
        self._pcp_prepared = None
        group = get_pcp_group()
        count = hash_ids.shape[0]
        counts = group.all_gather(hash_ids.new_tensor([count], dtype=torch.int64), dim=0)
        slot = int(counts.max().item())
        if slot * group.world_size > self.staged_rows.shape[0]:
            raise ValueError('HCU V41 PCP Engram staging capacity exceeded')
        if slot:
            padded = hash_ids.new_full((slot, hash_ids.shape[1]), native_engram.DEAD_ID)
            padded[:count].copy_(hash_ids)
            super().prepare_embeddings(group.all_gather(padded, dim=0))
        self._pcp_prepared = (count, slot)

    def _ready_rows(self, num_tokens):
        state, self._pcp_prepared = self._pcp_prepared, None
        if state is None or state[0] != num_tokens:
            raise RuntimeError('HCU V41 PCP Engram lookup requires matching prepare')
        _, slot = state
        if not slot:
            table = self.embed_tokens
            return self.staged_rows.new_empty((0, table.n_hash_cols, table.dim))
        # Core fences prefetch, exchanges heads and selects this rank's tokens.
        staged = super()._ready_rows(slot * get_pcp_group().world_size)
        return _PCP_GATHER_ROWS(staged, num_tokens)


class HcuDSV41PcpMoE(native.DeepseekV4MoE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Native ROCm reports TP-local counts; PCP also shards experts by EP.
        self.n_local_physical_experts = self.experts.moe_config.num_local_experts


_DECODER_INIT = _constructor(_DECODER_NATIVE, _REVIEWED[0],
                             {'_select_dsv4_attn_cls': select_attention,
                              'DeepseekV4MoE': HcuDSV41PcpMoE,
                              'Engram': HcuDSV41PcpEngram,
                              'MHCPreDelayedOp': HcuDSV41PcpMHCPreDelayedOp,
                              'MHCPostOp': HcuDSV41PcpMHCPostOp})


class HcuDSV41PcpDecoderLayer(native.DeepseekV4DecoderLayer):
    __init__ = _DECODER_INIT


_BACKBONE_INIT = _constructor(_BACKBONE_NATIVE, _REVIEWED[1],
                              {'DeepseekV4DecoderLayer': HcuDSV41PcpDecoderLayer,
                               'MHCPostOp': HcuDSV41PcpMHCPostOp})


class HcuDSV41PcpBackbone(native.DeepseekV4Model):
    __init__ = _BACKBONE_INIT


class HcuDSV41PcpLLM(native.DeepseekV41LLMForCausalLM):
    model_cls = HcuDSV41PcpBackbone


_OUTER_INIT = _constructor(_OUTER_NATIVE, _REVIEWED[2],
                           {'DeepseekV41LLMForCausalLM': HcuDSV41PcpLLM})


class HcuDSV41PcpForCausalLM(native_vl.DeepseekV41ForCausalLM):
    def __init__(self, *, vllm_config, prefix=''):
        validate_config(vllm_config)
        check_native_contract()
        _OUTER_INIT(self, vllm_config=vllm_config, prefix=prefix)
