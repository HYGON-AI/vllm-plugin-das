# SPDX-License-Identifier: Apache-2.0
"""Bounded native constructor namespaces for the HCU V41 PCP model.

Core has no decoder factory: its constructors resolve three hard-coded global
names. Reuse the exact reviewed code with private globals containing only those
name substitutions, preserving defaults/signatures/closures. This avoids copying
parameter lists or temporarily changing process-global Core symbols. Forward,
checkpoint loading and runner model-state methods stay native.
"""
from __future__ import annotations

from ..adapter_api import adapter_module as _adapter_module

import hashlib
import inspect
from contextvars import ContextVar
from types import FunctionType

from vllm.models.deepseek_v4.amd import model as native_v4
from vllm.models.deepseek_v41.amd import model as native
from vllm.models.deepseek_v41.amd import vl_model as native_vl
from .. import static_eplb_runtime as static_eplb
from ..v41_pcp import attention_class, validate_config, validate_layer_split_policy
from ..v41_engram import HcuV41PcpEngram
from ..v41_target_engram import HcuV41HostEngram, use_host_target, validate_host_config

_DECODER_NATIVE = native.DeepseekV4DecoderLayer.__init__
_BACKBONE_NATIVE = native.DeepseekV4Model.__init__
_OUTER_NATIVE = native_vl.DeepseekV41ForCausalLM.__init__
_BASE_MOE_NATIVE = native_v4.DeepseekV4MoE.__init__
_REVIEWED_BASE_MOE = '3396ecd9e48af5d6c094d392ffb62ad4271ec94cea4c42e6a037c5f398d26610'
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
            or native_vl.DeepseekV41ForCausalLM.__init__ is not _OUTER_NATIVE
            or native_v4.DeepseekV4MoE.__init__ is not _BASE_MOE_NATIVE):
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

    validate_layer_split_policy(cfg)
    admitted = pcp_fork.selected_mode() != 'stock' and validate_config(cfg)
    if not admitted:
        return native._select_dsv4_attn_cls(cfg)
    return attention_class(cfg, dspark=cfg.speculative_config is not None)


class HcuV41PcpMoE(native.DeepseekV4MoE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Model protocol fields must describe the actual EP-owned partition.
        self.n_local_physical_experts = self.experts.moe_config.num_local_experts


_DECODER_INIT = _constructor(_DECODER_NATIVE, _REVIEWED[0],
                             {'_select_dsv4_attn_cls': select_attention,
                              'DeepseekV4MoE': HcuV41PcpMoE,
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


def _draft_fits_replay(config, target_window):
    hf = config.speculative_config.draft_model_config.hf_config
    window = getattr(hf, 'sliding_window', None)
    return (type(window) is int and 0 < window <= target_window
            and all(kind == 'sliding_attention' for kind in (getattr(hf, 'layer_types', None) or ())))


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
        that metadata. validate_config pins PP/DP=1 and the supported drafter.
        """
        if not validate_config(vllm_config):
            return _REPLAY_NATIVE(self, vllm_config, cut)
        if self.start_layer > cut or self.end_layer < self.config.num_hidden_layers:
            reason = 'the pipeline stage must hold the last KV source and every later layer'
        elif self.use_sequence_parallel or vllm_config.parallel_config.use_ubatching:
            reason = 'sequence parallel and microbatching cannot follow the replay batch'
        elif any(i >= cut for i in getattr(self.config, 'engram_layer_ids', ())):
            reason = 'an Engram layer sits at or after the last KV source layer'
        elif vllm_config.speculative_config is not None and not _draft_fits_replay(
                vllm_config, self.config.sliding_window):
            reason = 'the drafter reads hidden states outside the target replay window'
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


# Static placement adds physical capacity at the existing factory seam. Keep
# native V41 initialization and forward methods, including the SWA replay gate.
_MOE_CONFIG = ContextVar('hcu_static_eplb_moe_config', default=None)


def _eplb_factory(*args, **kwargs):
    cfg = _MOE_CONFIG.get()
    if cfg is None:
        raise RuntimeError('HCU static EPLB factory used outside owned construction')
    extra = static_eplb.moe_eplb_kwargs(cfg, kwargs.get('prefix', ''))
    return native_v4.FusedMoEFactory(*args, **{**kwargs, **extra})


_EPLB_BASE_MOE_INIT = _constructor(_BASE_MOE_NATIVE, _REVIEWED_BASE_MOE,
                                    {'FusedMoEFactory': _eplb_factory})


class _HcuEplbBaseMoE(native_v4.DeepseekV4MoE):
    __init__ = _EPLB_BASE_MOE_INIT


class HcuV41EplbMoE(native.DeepseekV4MoE, _HcuEplbBaseMoE):
    def __init__(self, vllm_config, prefix='', use_sequence_parallel=False):
        token = _MOE_CONFIG.set(vllm_config)
        try:
            super().__init__(vllm_config, prefix=prefix,
                             use_sequence_parallel=use_sequence_parallel)
        finally:
            _MOE_CONFIG.reset(token)
        static_eplb.publish_physical_metadata(self, vllm_config, prefix)


_EPLB_DECODER_INIT = _constructor(_DECODER_NATIVE, _REVIEWED[0],
                                   {'_select_dsv4_attn_cls': select_attention,
                                    'Engram': HcuV41PcpEngram,
                                    'DeepseekV4MoE': HcuV41EplbMoE})


class HcuV41EplbDecoderLayer(native.DeepseekV4DecoderLayer):
    __init__ = _EPLB_DECODER_INIT


_EPLB_BACKBONE_INIT = _constructor(_BACKBONE_NATIVE, _REVIEWED[1],
                                    {'DeepseekV4DecoderLayer': HcuV41EplbDecoderLayer})


class HcuV41EplbBackbone(HcuV41PcpBackbone):
    __init__ = _EPLB_BACKBONE_INIT


class HcuV41EplbLLM(native.DeepseekV41LLMForCausalLM):
    model_cls = HcuV41EplbBackbone


_EPLB_OUTER_INIT = _constructor(_OUTER_NATIVE, _REVIEWED[2],
                                 {'DeepseekV41LLMForCausalLM': HcuV41EplbLLM})


class HcuV41PcpForCausalLM(native_vl.DeepseekV41ForCausalLM):
    def __init__(self, *, vllm_config, prefix=''):
        self._hcu_static_eplb_config = None
        uses_pcp = vllm_config.parallel_config.prefill_context_parallel_size > 1
        if uses_pcp:
            from ..dsv41.kv_gather import install as install_kv_gather

            if install_kv_gather():
                native.logger.info('DSV4.1 ROCm KV gather uses 64-bit page offsets')
        if static_eplb.enabled(vllm_config):
            static_eplb.validate_config(vllm_config)
            self._hcu_static_eplb_config = vllm_config
        fp4_applicable = _adapter_module('fp4_indexer').applicable
        if fp4_applicable(vllm_config):
            from ..fp4_indexer import preflight
            preflight(vllm_config)
            if uses_pcp:
                from ..v41_pcp_activation import pcp_enabled
                if not pcp_enabled():
                    raise ValueError('FP4 PCP attention requires VLLM_HCU_DSV41_PCP_ATTENTION=1')
                validate_config(vllm_config)
            from ..v41_pcp_activation import check_health
            check_health()
            check_native_contract()
            init = (_FP4_EPLB_OUTER_INIT if self._hcu_static_eplb_config is not None
                    else _FP4_OUTER_INIT)
            init(self, vllm_config=vllm_config, prefix=prefix)
        else:
            from ..v41_pcp_activation import check_health
            check_health()
            check_native_contract()
            if uses_pcp:
                from ..v41_pcp_activation import pcp_enabled
                if not pcp_enabled():
                    raise ValueError('PCP attention requires VLLM_HCU_DSV41_PCP_ATTENTION=1')
            if self._hcu_static_eplb_config is not None:
                _EPLB_OUTER_INIT(self, vllm_config=vllm_config, prefix=prefix)
            elif validate_config(vllm_config):
                _OUTER_INIT(self, vllm_config=vllm_config, prefix=prefix)
            elif use_host_target(vllm_config):
                _HOST_OUTER_INIT(self, vllm_config=vllm_config, prefix=prefix)
            else:
                # PCP1 without the explicit host route remains fully native.
                super().__init__(vllm_config=vllm_config, prefix=prefix)

        if uses_pcp:
            from ..inverse_rope_pcp import prepare_worker_model

            # Model construction follows Worker.init_device.
            prepare_worker_model(self)

        from ..dspark_ll import install_model_entry_reset
        install_model_entry_reset(self, vllm_config, 384)
        self._hcu_int8_lm_head = False
        from ..quantization.ct_w4a8 import HcuDSV41CTW4A8Config

        quant_config = vllm_config.quant_config
        if isinstance(quant_config, HcuDSV41CTW4A8Config):
            from ..w4a8_projections import bind_vocab_head, bind_wo_a

            bind_wo_a(self.language_model.model.layers)
            self._hcu_int8_lm_head = bind_vocab_head(
                self.language_model.lm_head, quant_config, "language_model.lm_head",
            )
            if uses_pcp and self._hcu_int8_lm_head:
                from ..dsv41.pcp_compressor import install as install_pcp_compressor

                if install_pcp_compressor():
                    native.logger.info("DSV4.1 HCU PCP compressor pointer compatibility enabled")

    def load_weights(self, weights):
        cfg = getattr(self, '_hcu_static_eplb_config', None)
        if cfg is not None:
            # Bind before native checkpoint loaders read logical expert IDs.
            static_eplb.bind_placement(self, cfg)
        if getattr(self, "_hcu_int8_lm_head", False):
            weights = (
                ("language_model.lm_head.weight_scale" if name == "head.weight_scale" else name,
                 value)
                for name, value in weights
            )
        return super().load_weights(weights)

    def process_weights_after_loading(self):
        super().process_weights_after_loading()
        cfg = getattr(self, '_hcu_static_eplb_config', None)
        if cfg is None:
            return
        # DummyModelLoader also reaches this hook without calling load_weights.
        plan = static_eplb.bind_placement(self, cfg, bind_weight_loaders=False)
        static_eplb.install_routing(self, cfg)
        if plan is not None:
            native.logger.info(
                'HCU static EPLB routing ready: physical_experts=%d redundant_experts=%d',
                plan.num_physical_experts, plan.num_redundant_experts,
            )
        self._hcu_static_eplb_session = static_eplb.record_session(self)

    def forward(self, input_ids, positions, intermediate_tensors=None,
                inputs_embeds=None, lookback_token_ids=None, **kwargs):
        session = getattr(self, '_hcu_static_eplb_session', None)
        record_step = False
        if session is not None:
            rows = positions.shape[0] if inputs_embeds is None else inputs_embeds.shape[0]
            record_step = session.prepare(rows)
        output = super().forward(input_ids, positions, intermediate_tensors,
                                 inputs_embeds,
                                 lookback_token_ids=lookback_token_ids, **kwargs)
        if record_step:
            session.step()
        return output


def _fp4_attention(cfg):
    from ..fp4_indexer import Fp4Attention, Fp4PcpAttention, preflight
    preflight(cfg)
    if validate_config(cfg):
        from ..v41_pcp import layer_split_selected
        if layer_split_selected(cfg):
            from ..dsa_cache_layer_split import apply_kv_source_write_suppression
            from ..fp4_indexer import Fp4LayerSplitPcpAttention
            apply_kv_source_write_suppression()
            return Fp4LayerSplitPcpAttention
        return Fp4PcpAttention
    return Fp4Attention


_FP4_DECODER_INIT = _constructor(_DECODER_NATIVE, _REVIEWED[0],
                                {'_select_dsv4_attn_cls': _fp4_attention})
_FP4_HOST_DECODER_INIT = _constructor(_DECODER_NATIVE, _REVIEWED[0],
                                     {'_select_dsv4_attn_cls': _fp4_attention,
                                      'Engram': HcuV41HostEngram})
_FP4_PCP_DECODER_INIT = _constructor(_DECODER_NATIVE, _REVIEWED[0],
                                    {'_select_dsv4_attn_cls': _fp4_attention,
                                     'DeepseekV4MoE': HcuV41PcpMoE,
                                     'Engram': HcuV41PcpEngram})


class HcuV41Fp4DecoderLayer(native.DeepseekV4DecoderLayer):
    def __init__(self, *args, **kwargs):
        cfg = args[0] if args else kwargs['vllm_config']
        use_pcp = validate_config(cfg)
        if use_pcp:
            init = _FP4_PCP_DECODER_INIT
        else:
            init = _FP4_HOST_DECODER_INIT if use_host_target(cfg) else _FP4_DECODER_INIT
        init(self, *args, **kwargs)
        self._hcu_paired_attention_norm = False
        if use_pcp:
            return
        from ..shared_expert_int8 import bind_shared_expert

        bind_shared_expert(cfg, self.ffn)
        from ..moe.router_stagger import bind_router

        bind_router(cfg, self.ffn)
        if not self.use_sequence_parallel and self.attn._int8_wq_b_direct is not None:
            from ..int8_wq_b import bind_decoder_direct_gemm

            bind_decoder_direct_gemm(self.attn._int8_wq_b_direct, self.attn, self.ffn)
        from ..rms_int8 import install_owned_decoder
        self._hcu_paired_attention_norm = install_owned_decoder(self)

    def forward(self, *args, **kwargs):
        if not self._hcu_paired_attention_norm:
            return super().forward(*args, **kwargs)
        from hcu_adapter.rms_int8_scope import attention_norm_scope
        with attention_norm_scope():
            return super().forward(*args, **kwargs)

    def write_kv(self, *args, **kwargs):
        if not self._hcu_paired_attention_norm:
            return super().write_kv(*args, **kwargs)
        from hcu_adapter.rms_int8_scope import attention_norm_scope
        with attention_norm_scope():
            return super().write_kv(*args, **kwargs)


_FP4_BACKBONE_INIT = _constructor(_BACKBONE_NATIVE, _REVIEWED[1],
                                 {'DeepseekV4DecoderLayer': HcuV41Fp4DecoderLayer})


class HcuV41Fp4Backbone(HcuV41PcpBackbone):
    __init__ = _FP4_BACKBONE_INIT


class HcuV41Fp4LLM(native.DeepseekV41LLMForCausalLM):
    model_cls = HcuV41Fp4Backbone


_FP4_OUTER_INIT = _constructor(_OUTER_NATIVE, _REVIEWED[2],
                              {'DeepseekV41LLMForCausalLM': HcuV41Fp4LLM})


# Compose the FP4 attention/PCP Engram selection with static physical MoE
# placement. Inherit the same backbone so decoder SWA replay remains available.
_FP4_EPLB_DECODER_INIT = _constructor(_DECODER_NATIVE, _REVIEWED[0],
                                       {'_select_dsv4_attn_cls': _fp4_attention,
                                        'DeepseekV4MoE': HcuV41EplbMoE,
                                        'Engram': HcuV41PcpEngram})


class HcuV41Fp4EplbDecoderLayer(native.DeepseekV4DecoderLayer):
    __init__ = _FP4_EPLB_DECODER_INIT


_FP4_EPLB_BACKBONE_INIT = _constructor(_BACKBONE_NATIVE, _REVIEWED[1],
                                        {'DeepseekV4DecoderLayer': HcuV41Fp4EplbDecoderLayer})


class HcuV41Fp4EplbBackbone(HcuV41PcpBackbone):
    __init__ = _FP4_EPLB_BACKBONE_INIT


class HcuV41Fp4EplbLLM(native.DeepseekV41LLMForCausalLM):
    model_cls = HcuV41Fp4EplbBackbone


_FP4_EPLB_OUTER_INIT = _constructor(_OUTER_NATIVE, _REVIEWED[2],
                                     {'DeepseekV41LLMForCausalLM': HcuV41Fp4EplbLLM})
