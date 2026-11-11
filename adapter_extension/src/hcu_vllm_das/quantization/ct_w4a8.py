# SPDX-License-Identifier: Apache-2.0
"""Distinct DSV4.1 CT W4A8 configuration, leaving official CT selection intact."""
from __future__ import annotations
import torch
from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.fused_moe import RoutedExperts, UnquantizedFusedMoEMethod
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import CompressedTensorsConfig, CompressedTensorsLinearMethod
from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8 import CompressedTensorsW8A8Fp8
from .dsv41_channel_fp8 import _is_hcu, _is_v41

NAME = 'hcu_dsv41_ct_w4a8'


def _schema(weight, activation):
    def value(obj, name, default=None):
        return obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)
    return (weight is not None and activation is not None
        and value(weight,'num_bits') == 4 and value(weight,'type') == 'int'
        and value(weight,'strategy') == 'channel' and value(weight,'symmetric',True) is True
        and value(weight,'dynamic',False) is False and value(weight,'group_size') in (None,-1)
        and value(weight,'block_structure') is None and value(activation,'block_structure') is None
        and value(activation,'num_bits') == 8 and value(activation,'type') == 'int'
        and value(activation,'strategy') == 'token' and value(activation,'symmetric',True) is True
        and value(activation,'dynamic') is True)


def validate_topology(config, moe):
    p, actual = config.parallel_config, moe.moe_parallel_config
    errors=[]
    for name in ('tensor_parallel_size','pipeline_parallel_size','decode_context_parallel_size'):
        if getattr(p,name) != 1: errors.append(name+' must be 1')
    if config.lora_config is not None or p.enable_eplb or p.enable_elastic_ep:
        errors.append('LoRA/EPLB/Elastic EP are outside the packed ownership contract')
    if p.prefill_context_parallel_size > 1 and not config.model_config.enforce_eager:
        errors.append('PCP>1 requires eager execution')
    if any(type(v) is not int or v <= 0 for v in (p.data_parallel_size,p.prefill_context_parallel_size,moe.max_num_tokens)):
        errors.append('parallel widths/capacity must be positive integers')
    if actual.all2all_backend == 'allgather_reducescatter':
        from hcu_vllm_das.dspark_agrs import _capacity_from_metadata, is_owned_draft, token_capacity
        if not is_owned_draft(config):
            errors.append('AG-RS is restricted to the owned DSpark draft configuration')
        elif token_capacity(config) != _capacity_from_metadata(config):
            errors.append('DSpark query/verify capacity changed after construction')
        if not config.model_config.enforce_eager or p.prefill_context_parallel_size != 1:
            from hcu_vllm_das.dspark_graph import native_decode_dspark, validate_graph_capacity

            if not native_decode_dspark(config):
                errors.append('DSpark AG-RS Graph requires native V41 DP8/EP8')
            else:
                validate_graph_capacity(config, moe)
    elif actual.all2all_backend not in ('deepep_high_throughput','deepep_low_latency'):
        errors.append('fixed DeepEP HT/LL or owned DSpark AG-RS is required')
    if not actual.use_ep:
        errors.append('actual expert parallel ownership is required')
    if actual.tp_size != 1 or actual.dp_size != p.data_parallel_size or actual.pcp_size != p.prefill_context_parallel_size or actual.ep_size != actual.dp_size*actual.pcp_size:
        errors.append('MoE TP/DP/PCP/EP membership mismatch')
    if moe.max_num_tokens != config.scheduler_config.max_num_batched_tokens:
        errors.append('MoE capacity must match scheduler budget')
    sizes=config.compilation_config.cudagraph_capture_sizes
    if sizes and max(sizes)>moe.max_num_tokens: errors.append('graph bucket exceeds token capacity')
    if type(actual.ep_size) is not int or actual.ep_size<=0:
        errors.append('EP size must be a positive integer')
    elif moe.num_experts % actual.ep_size or moe.num_local_experts != moe.num_experts//actual.ep_size:
        errors.append('expert count must match the actual EP group')
    if errors: raise ValueError('HCU CT W4A8 topology rejected: '+ '; '.join(errors))


class HcuDSV41CTW4A8Config(CompressedTensorsConfig):
    @classmethod
    def get_name(cls): return NAME

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant, hf_config=None):
        if user_quant not in (None,'compressed-tensors',NAME) or not _is_hcu() or not _is_v41(hf_config):
            return None
        if not isinstance(hf_quant_cfg,dict) or hf_quant_cfg.get('quant_method')!='compressed-tensors': return None
        groups=hf_quant_cfg.get('config_groups',{})
        if not isinstance(groups,dict): return None
        if any(isinstance(g,dict) and _schema(g.get('weights'),g.get('input_activations'))
               and g.get('format',hf_quant_cfg.get('format'))=='pack-quantized' for g in groups.values()):
            from hcu_vllm_das.ct_w4a8_activation import check_health
            check_health()
            return NAME
        return None

    @classmethod
    def from_config(cls, config):
        if not _is_hcu() or config.get('quant_method')!='compressed-tensors':
            raise ValueError('HCU CT W4A8 requires the native compressed-tensors checkpoint schema')
        from hcu_vllm_das.ct_w4a8_activation import check_health
        check_health()
        return super().from_config(config)

    def get_quant_method(self, layer, prefix):
        from hcu_vllm_das.ct_w4a8_activation import check_health
        check_health()
        if not isinstance(layer,RoutedExperts):
            method=super().get_quant_method(layer,prefix)
            if isinstance(layer,LinearBase) and isinstance(method,CompressedTensorsLinearMethod) and isinstance(layer.scheme,CompressedTensorsW8A8Fp8):
                from hcu_adapter.channel_fp8_empty_linear import HcuDSV41ChannelFP8LinearMethod
                return HcuDSV41ChannelFP8LinearMethod(self)
            if isinstance(layer,LinearBase) and type(method) is CompressedTensorsLinearMethod:
                from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_int8 import CompressedTensorsW8A8Int8
                if type(layer.scheme) is CompressedTensorsW8A8Int8:
                    from .dsv41_int8_linear import HcuDSV41Int8LinearMethod
                    return HcuDSV41Int8LinearMethod(self)
            return method
        # Update only this owned config's public parsed map, mirroring native
        # Linear target expansion without replacing a selector or calling its CPU constructor.
        if 'Linear' in self.target_scheme_map:
            self.target_scheme_map.setdefault('RoutedExperts',self.target_scheme_map['Linear'])
        schemes=[self.get_scheme_dict(layer,f'{prefix}.0.{n}') for n in
                 (layer.ckpt_gate_proj_name,layer.ckpt_up_proj_name,layer.ckpt_down_proj_name)]
        if not all(s==schemes[0] for s in schemes):
            raise ValueError('All W4A8 expert projections require the same scheme')
        s=schemes[0]
        if s is None: return UnquantizedFusedMoEMethod(layer.moe_config)
        if not _schema(s.get('weights'),s.get('input_activations')):
            return super().get_quant_method(layer,prefix)
        if s.get('format')!='pack-quantized': raise ValueError('HCU W4A8 requires pack-quantized weights')
        cfg=get_current_vllm_config()
        validate_topology(cfg,layer.moe_config)
        from .ct_w4a8_method import HcuCTW4A8MoEMethod
        return HcuCTW4A8MoEMethod(s['weights'],s['input_activations'],layer.moe_config,cfg)
