# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Owned static placement composed around native HY4 constructors/loaders.

No forward, attention, loader loop or compile wrapper is copied. Child
factories live in private native constructor namespaces; Core globals remain
untouched. Dynamic Core EPLB stays off while each owned MoE gets fixed slots.
"""
from vllm.config import get_current_vllm_config
from vllm.model_executor.models.interfaces import MixtureOfExperts
from vllm.models.hy_v4.amd import model as native
from vllm.models.hy_v4.nvidia import model as nvidia, moe, mtp, attention

from .. import static_eplb_runtime as runtime
from ..hy4_constructor import bind_constructor
from ..hy4_static_plan import load_hy4_plan
from ..hy4_prefill_workspace import make_attention
from .. import hy4_ihc_launch as ihc_launch
from .. import hy4_dense_fusion as dense_fusion
from .. import hy4_dp_metadata
from .hy4_ct import HcuHY4CTBackbone, HcuHY4CTMTP


def _static_factory(*args, **kwargs):
    cfg = get_current_vllm_config()
    extra = runtime.moe_eplb_kwargs(cfg, kwargs.get('prefix', ''))
    if not extra:
        raise ValueError('HY4 static factory requires an owned backbone/MTP layer')
    return moe.FusedMoEFactory(*args, **{**kwargs, **extra})


_MOE_INIT = bind_constructor(moe.HYV4MoEFused.__init__,
                            {'FusedMoEFactory': _static_factory})


class HcuHY4StaticMoE(moe.HYV4MoEFused):
    def __init__(self, config, quant_config=None, prefix='', enable_eplb=False):
        cfg = get_current_vllm_config()
        runtime.validate_config(cfg)
        _MOE_INIT(self, config, quant_config=quant_config, prefix=prefix,
                  enable_eplb=True)
        runtime.publish_physical_metadata(self, cfg, prefix)
        self.physical_expert_start = self.ep_rank * self.n_local_physical_experts
        self.physical_expert_end = self.physical_expert_start + self.n_local_physical_experts


class HcuHY4StaticAttention(attention.HYV4MLAAttention):
    __init__ = bind_constructor(attention.HYV4MLAAttention.__init__,
                               {'MLAAttention': make_attention})


class HcuHY4StaticDecoder(native.HYV4DecoderLayer):
    __init__ = bind_constructor(native.HYV4DecoderLayer.__init__,
        {'HYV4MoEFused': HcuHY4StaticMoE, 'HYV4MLAAttention': HcuHY4StaticAttention})


class HcuHY4StaticLaunchDecoder(HcuHY4StaticDecoder):
    __init__ = bind_constructor(HcuHY4StaticDecoder.__init__,
                               {'HYV4HCLayer': ihc_launch.TunedHCLayer})
    _forward_ihc = ihc_launch.TunedDecoder._forward_ihc


def _logical_expert_mapping(*args, **kwargs):
    # The static loader owns replica fan-out. Core's dynamic mapping would
    # emit a second set of physical IDs for repeated logical checkpoint keys.
    return native.fused_moe_make_expert_params_mapping(
        *args, **{**kwargs, 'num_redundant_experts': 0})


class HcuHY4StaticBackbone(HcuHY4CTBackbone):
    __init__ = bind_constructor(native.HYV4Model.__init__,
        {'HYV4DecoderLayer': HcuHY4StaticDecoder, 'HYV4MoEFused': HcuHY4StaticMoE})
    get_expert_mapping = bind_constructor(native.HYV4Model.get_expert_mapping,
        {'fused_moe_make_expert_params_mapping': _logical_expert_mapping})


class HcuHY4StaticLaunchBackbone(HcuHY4StaticBackbone):
    # Core also uses HYV4DecoderLayer in isinstance(), so keep a real class.
    __init__ = bind_constructor(native.HYV4Model.__init__,
        {'HYV4DecoderLayer': HcuHY4StaticLaunchDecoder, 'HYV4MoEFused': HcuHY4StaticMoE})


def make_static_backbone(*, vllm_config, prefix=''):
    hy4_dp_metadata.prepare(vllm_config)
    cls = HcuHY4StaticLaunchBackbone if ihc_launch.selected(vllm_config) else HcuHY4StaticBackbone
    return dense_fusion.attach(cls(vllm_config=vllm_config, prefix=prefix), vllm_config)


_OUTER_INIT = bind_constructor(native.HYV4ForCausalLM.__init__,
                              {'HYV4Model': make_static_backbone})


class HcuHY4StaticMTPDecoder(nvidia.HYV4DecoderLayer):
    __init__ = bind_constructor(nvidia.HYV4DecoderLayer.__init__,
        {'HYV4MoEFused': HcuHY4StaticMoE, 'HYV4MLAAttention': HcuHY4StaticAttention})


class HcuHY4StaticPredictorLayer(mtp.HYV4MultiTokenPredictorLayer):
    __init__ = bind_constructor(mtp.HYV4MultiTokenPredictorLayer.__init__,
                               {'HYV4DecoderLayer': HcuHY4StaticMTPDecoder})


class HcuHY4StaticPredictor(mtp.HYV4MultiTokenPredictor):
    __init__ = bind_constructor(mtp.HYV4MultiTokenPredictor.__init__,
                               {'HYV4MultiTokenPredictorLayer': HcuHY4StaticPredictorLayer})


_MTP_INIT = bind_constructor(mtp.HYV4MTP.__init__,
                            {'HYV4MultiTokenPredictor': HcuHY4StaticPredictor})


def _preflight(cfg, *, draft):
    runtime.validate_config(cfg)
    from ..hy4_pcp_mtp import prepare
    prepare(cfg)
    options = runtime.read_options(cfg)
    hf = cfg.model_config.hf_config
    if draft:
        layers = getattr(hf, 'num_nextn_predict_layers', 0)
        if type(layers) is not int or layers < 1:
            raise ValueError('HY4 static MTP requires declared draft layers')
        key = 'HYV4MTP'
    else:
        kinds = hf.mlp_layer_types
        if len(kinds) != hf.num_hidden_layers:
            raise ValueError('HY4 static MLP layer list must cover the backbone')
        layers = sum(kind != 'dense' for kind in kinds)
        key = 'HYV4ForCausalLM'
    return load_hy4_plan(options.expert_map_path, model_key=key,
        expected_model_name=cfg.model_config.model,
        expected_shape=(layers, hf.n_routed_experts + options.num_redundant_experts),
        num_logical_experts=hf.n_routed_experts,
        num_redundant_experts=options.num_redundant_experts)


def _publish_protocol(model, layers, cfg):
    if not layers:
        raise ValueError('HY4 static placement requires MoE layers')
    model.moe_mlp_layers = list(layers)
    model.moe_layers = [layer.experts for layer in layers]
    model.expert_weights = []
    model.num_moe_layers = len(layers)
    model.num_expert_groups = 1
    model.num_logical_experts = layers[0].n_logical_experts
    model.num_physical_experts = layers[0].n_physical_experts
    model.num_local_physical_experts = layers[0].n_local_physical_experts
    model.num_routed_experts = layers[0].n_routed_experts
    model.num_redundant_experts = layers[0].n_redundant_experts
    model.num_shared_experts = cfg.model_config.hf_config.n_shared_experts


class _StaticProtocol:
    set_eplb_state = MixtureOfExperts.set_eplb_state

    def update_physical_experts_metadata(self, num_physical_experts,
                                         num_local_physical_experts):
        if (num_physical_experts != self.num_physical_experts
                or num_local_physical_experts != self.num_local_physical_experts):
            raise ValueError('HY4 static physical layout cannot be resized')

    def _bind_static(self, *, bind_weight_loaders=True):
        plan = runtime.bind_placement(self, self._hcu_static_cfg,
                                     bind_weight_loaders=bind_weight_loaders)
        if plan.source_sha256 != self._hcu_static_expected_plan.source_sha256:
            raise ValueError('HY4 static placement file changed during model startup')
        return plan


class HcuHY4StaticForCausalLM(_StaticProtocol, native.HYV4ForCausalLM):
    def __init__(self, *, vllm_config, prefix=''):
        plan = _preflight(vllm_config, draft=False)
        _OUTER_INIT(self, vllm_config=vllm_config, prefix=prefix)
        self._hcu_static_cfg = vllm_config
        self._hcu_static_expected_plan = plan
        _publish_protocol(self, [layer.mlp for layer in self.model.layers
            if isinstance(getattr(layer, 'mlp', None), HcuHY4StaticMoE)], vllm_config)

    def load_weights(self, weights):
        self._bind_static()
        return super().load_weights(weights)

    def process_weights_after_loading(self):
        super().process_weights_after_loading()
        self._bind_static(bind_weight_loaders=False)
        runtime.install_routing(self, self._hcu_static_cfg)


class HcuHY4StaticMTP(_StaticProtocol, HcuHY4CTMTP):
    def __init__(self, *, vllm_config, prefix=''):
        plan = _preflight(vllm_config, draft=True)
        _MTP_INIT(self, vllm_config=vllm_config, prefix=prefix)
        dense_fusion.attach(self.model, vllm_config)
        self._hcu_static_cfg = vllm_config
        self._hcu_static_expected_plan = plan
        _publish_protocol(self, [layer.mtp_block.mlp for layer in self.model.layers.values()],
                          vllm_config)

    def load_weights(self, weights):
        self._bind_static()
        return super().load_weights(weights)

    def process_weights_after_loading(self):
        self._bind_static(bind_weight_loaders=False)
        runtime.install_routing(self, self._hcu_static_cfg)
