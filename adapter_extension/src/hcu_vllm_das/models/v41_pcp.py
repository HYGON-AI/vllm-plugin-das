# SPDX-License-Identifier: Apache-2.0
"""Bounded native constructor namespaces for the HCU V41 PCP model.

Core has no decoder factory: its constructors resolve three hard-coded global
names. Reuse the exact reviewed code with private globals containing only those
name substitutions, preserving defaults/signatures/closures. This avoids copying
parameter lists or temporarily changing process-global Core symbols. Forward,
checkpoint loading and runner model-state methods stay native.
"""
from __future__ import annotations

import hashlib
import inspect
from types import FunctionType

from vllm.models.deepseek_v41.amd import model as native
from vllm.models.deepseek_v41.amd import vl_model as native_vl
from ..v41_pcp import attention_class, validate_config
from ..v41_engram import HcuV41PcpEngram
from ..v41_target_engram import HcuV41HostEngram, use_host_target, validate_host_config

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
    """Pick the attention class for one decoder layer.

    This local selector is what `_DECODER_INIT` substitutes into the reviewed
    native constructor, so it -- not `pcp_fork.select_attention()` -- is the
    real entry point the decoder walks. It therefore has to apply the same
    two gates as the adapter: the deployment must look like PCP *and* the
    operator must have selected the owned implementation. Reading the mode here
    is what makes `HCU_VLLM_PCP_IMPL=stock` (the default) and
    `HCU_DISABLE_OPS=vllm_pcp` actually mean something at construction time.
    """
    from hcu_adapter.kernels import pcp_fork

    if pcp_fork.selected_mode() == 'stock' or not validate_config(cfg):
        return native._select_dsv4_attn_cls(cfg)
    return attention_class()


_DECODER_INIT = _constructor(_DECODER_NATIVE, _REVIEWED[0],
                             {'_select_dsv4_attn_cls': select_attention,
                              'Engram': HcuV41PcpEngram})


class HcuV41PcpDecoderLayer(native.DeepseekV4DecoderLayer):
    __init__ = _DECODER_INIT


_BACKBONE_INIT = _constructor(_BACKBONE_NATIVE, _REVIEWED[1],
                              {'DeepseekV4DecoderLayer': HcuV41PcpDecoderLayer})


_REPLAY_NATIVE = native.DeepseekV4Model._decoder_replay_supported
_REPLAY_REVIEWED = '1e54ff54d365b98ece53fc86c8893e7787ae75143a8269c55b4813d6f9952a96'
if hashlib.sha256(inspect.getsource(_REPLAY_NATIVE).encode()).hexdigest() != _REPLAY_REVIEWED:
    raise RuntimeError('HCU V41 PCP decoder replay gate source changed: '
                       f'{_REPLAY_NATIVE.__qualname__}')


class HcuV41PcpBackbone(native.DeepseekV4Model):
    def __init__(self, *, vllm_config, prefix=''):
        if validate_config(vllm_config):
            _BACKBONE_INIT(self, vllm_config=vllm_config, prefix=prefix)
        else:
            super().__init__(vllm_config=vllm_config, prefix=prefix)

    def _decoder_replay_supported(self, vllm_config, cut):
        """Native gate minus its PCP refusal, for the owned PCP model only.

        Native model state pads each rank's replay batch to the global kept
        rows and rebuilds PCP metadata for it; owned PCP attention reads only
        that metadata. validate_config already pins PP/DP=1 and no drafter.
        """
        if not validate_config(vllm_config):
            return _REPLAY_NATIVE(self, vllm_config, cut)
        if self.start_layer > cut or self.end_layer < self.config.num_hidden_layers:
            reason = 'the pipeline stage must hold the last KV source and every later layer'
        elif self.use_sequence_parallel or vllm_config.parallel_config.use_ubatching:
            reason = 'sequence parallel and microbatching cannot follow the replay batch'
        elif any(i >= cut for i in getattr(self.config, 'engram_layer_ids', ())):
            reason = 'an Engram layer sits at or after the last KV source layer'
        else:
            return True
        native.logger.warning_once('Decoder SWA bounded replay is off: %s.', reason)
        return False


class HcuV41PcpLLM(native.DeepseekV41LLMForCausalLM):
    model_cls = HcuV41PcpBackbone


_OUTER_INIT = _constructor(_OUTER_NATIVE, _REVIEWED[2],
                           {'DeepseekV41LLMForCausalLM': HcuV41PcpLLM})


_HOST_DECODER_INIT = _constructor(_DECODER_NATIVE, _REVIEWED[0],
                                   {'Engram': HcuV41HostEngram})


class HcuV41HostDecoderLayer(native.DeepseekV4DecoderLayer):
    __init__ = _HOST_DECODER_INIT


_HOST_BACKBONE_INIT = _constructor(_BACKBONE_NATIVE, _REVIEWED[1],
                                    {'DeepseekV4DecoderLayer': HcuV41HostDecoderLayer})


class HcuV41HostBackbone(native.DeepseekV4Model):
    def __init__(self, *, vllm_config, prefix=''):
        validate_host_config(vllm_config)
        _HOST_BACKBONE_INIT(self, vllm_config=vllm_config, prefix=prefix)


class HcuV41HostLLM(native.DeepseekV41LLMForCausalLM):
    model_cls = HcuV41HostBackbone


_HOST_OUTER_INIT = _constructor(_OUTER_NATIVE, _REVIEWED[2],
                                 {'DeepseekV41LLMForCausalLM': HcuV41HostLLM})


class HcuV41PcpForCausalLM(native_vl.DeepseekV41ForCausalLM):
    def __init__(self, *, vllm_config, prefix=''):
        from hcu_adapter.fp4_indexer import applicable as fp4_applicable
        if fp4_applicable(vllm_config):
            from ..fp4_indexer import preflight
            preflight(vllm_config)
            from ..v41_pcp_activation import check_health
            check_health()
            check_native_contract()
            _FP4_OUTER_INIT(self, vllm_config=vllm_config, prefix=prefix)
            return
        from ..v41_pcp_activation import check_health
        check_health()
        check_native_contract()
        if vllm_config.parallel_config.prefill_context_parallel_size > 1:
            from ..v41_pcp_activation import pcp_enabled
            if not pcp_enabled():
                raise ValueError('PCP attention requires VLLM_HCU_DSV41_PCP_ATTENTION=1')
        if validate_config(vllm_config):
            _OUTER_INIT(self, vllm_config=vllm_config, prefix=prefix)
        elif use_host_target(vllm_config):
            _HOST_OUTER_INIT(self, vllm_config=vllm_config, prefix=prefix)
        else:
            # PCP1 without the explicit host route remains fully native.
            super().__init__(vllm_config=vllm_config, prefix=prefix)


def _fp4_attention(cfg):
    from ..fp4_indexer import Fp4Attention, preflight
    preflight(cfg)
    return Fp4Attention


_FP4_DECODER_INIT = _constructor(_DECODER_NATIVE, _REVIEWED[0],
                                {'_select_dsv4_attn_cls': _fp4_attention})
_FP4_HOST_DECODER_INIT = _constructor(_DECODER_NATIVE, _REVIEWED[0],
                                     {'_select_dsv4_attn_cls': _fp4_attention,
                                      'Engram': HcuV41HostEngram})


class HcuV41Fp4DecoderLayer(native.DeepseekV4DecoderLayer):
    def __init__(self, *args, **kwargs):
        cfg = args[0] if args else kwargs['vllm_config']
        init = _FP4_HOST_DECODER_INIT if use_host_target(cfg) else _FP4_DECODER_INIT
        init(self, *args, **kwargs)


_FP4_BACKBONE_INIT = _constructor(_BACKBONE_NATIVE, _REVIEWED[1],
                                 {'DeepseekV4DecoderLayer': HcuV41Fp4DecoderLayer})


class HcuV41Fp4Backbone(native.DeepseekV4Model):
    __init__ = _FP4_BACKBONE_INIT


class HcuV41Fp4LLM(native.DeepseekV41LLMForCausalLM):
    model_cls = HcuV41Fp4Backbone


_FP4_OUTER_INIT = _constructor(_OUTER_NATIVE, _REVIEWED[2],
                              {'DeepseekV41LLMForCausalLM': HcuV41Fp4LLM})
