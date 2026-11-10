# SPDX-License-Identifier: Apache-2.0
"""Reuse Core constructors with private HCU attention/MoE class bindings.

Core owns PCP scheduling, collectives, cache preparation, Engram and loading.
Only the ROCm attention eligibility and EP-local expert count need bridging.
"""
from __future__ import annotations

import hashlib
import inspect
from types import FunctionType

from vllm.models.deepseek_v41.amd import model as native
from vllm.models.deepseek_v41.amd import vl_model as native_vl
from ..v41_pcp import HcuV41PcpAttention, validate_config

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
    return HcuV41PcpAttention



class HcuV41PcpMoE(native.DeepseekV4MoE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Native ROCm reports TP-local counts; PCP also shards experts by EP.
        self.n_local_physical_experts = self.experts.moe_config.num_local_experts


_DECODER_INIT = _constructor(_DECODER_NATIVE, _REVIEWED[0],
                             {'_select_dsv4_attn_cls': select_attention,
                              'DeepseekV4MoE': HcuV41PcpMoE})


class HcuV41PcpDecoderLayer(native.DeepseekV4DecoderLayer):
    __init__ = _DECODER_INIT


_BACKBONE_INIT = _constructor(_BACKBONE_NATIVE, _REVIEWED[1],
                              {'DeepseekV4DecoderLayer': HcuV41PcpDecoderLayer})


class HcuV41PcpBackbone(native.DeepseekV4Model):
    __init__ = _BACKBONE_INIT


class HcuV41PcpLLM(native.DeepseekV41LLMForCausalLM):
    model_cls = HcuV41PcpBackbone


_OUTER_INIT = _constructor(_OUTER_NATIVE, _REVIEWED[2],
                           {'DeepseekV41LLMForCausalLM': HcuV41PcpLLM})


class HcuV41PcpForCausalLM(native_vl.DeepseekV41ForCausalLM):
    def __init__(self, *, vllm_config, prefix=''):
        validate_config(vllm_config)
        check_native_contract()
        _OUTER_INIT(self, vllm_config=vllm_config, prefix=prefix)
