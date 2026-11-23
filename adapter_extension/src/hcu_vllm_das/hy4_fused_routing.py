# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Use Core's fused sigmoid router on owned HY4 runners.

Core retains select_experts, capture callbacks, EPLB mapping, index dtype,
padding masks and the actual operator. Restart when changing this policy.
"""
import math
from dataclasses import replace
from types import MethodType

import torch
from vllm import envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.forward_context import (get_forward_context, is_forward_context_available,
                                  override_forward_context)
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import fused_topk_bias
from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import GroupedTopKRouter

from .environment import _flag, custom_ops_enabled
from .hy4_ihc_launch import _device_arch
from .quantization.hy4_ct_policy import is_hy4

logger=init_logger(f'vllm.{__name__}')


def _requested():
    return custom_ops_enabled() and _flag('HCU_VLLM_HY4_FUSED_ROUTING',default=False)


def _supported_config(cfg):
    model,p=cfg.model_config,cfg.parallel_config
    hf=model.hf_config
    return (is_hy4(hf) and not model.is_multimodal_model
        and model.dtype==torch.bfloat16 and cfg.lora_config is None
        and cfg.use_v2_model_runner and cfg.compilation_config.mode==0
        and (hf.hidden_size,hf.n_routed_experts,hf.num_experts_per_tok)==(6144,256,8)
        and p.enable_expert_parallel and not p.enable_eplb and not p.enable_elastic_ep
        and not getattr(p,'enable_dbo',False) and getattr(p,'ubatch_size',0) in (0,1)
        and not envs.VLLM_BATCH_INVARIANT and not rocm_aiter_ops.is_fused_moe_enabled()
        and _device_arch()=='gfx938' and _topology(cfg) is not None)


def _topology(cfg):
    p=cfg.parallel_config
    shape=(p.tensor_parallel_size,p.pipeline_parallel_size,p.prefill_context_parallel_size,
           p.decode_context_parallel_size,p.data_parallel_size)
    if shape==(1,1,1,1,16):return 'decode'
    if shape!=(1,1,16,1,1):return None
    if (not cfg.model_config.enforce_eager or cfg.scheduler_config.async_scheduling
            or cfg.kernel_config.moe_backend!='deep_gemm'
            or p.all2all_backend!='deepep_high_throughput'):
        return None
    from .static_eplb_runtime import read_options
    options=read_options(cfg)
    if (options is not None and not options.recording
            and options.expert_map_path is not None and options.num_redundant_experts==16):
        return 'prefill'
    return None


def enabled(cfg):
    return _requested() and _supported_config(cfg)


def _routing_contract(router,physical_experts=256):
    scale=router.routed_scaling_factor
    return (router.global_num_experts==physical_experts and router.top_k==8
        and router.num_expert_group==1 and router.topk_group==1
        and router.renormalize is True and router.scoring_func=='sigmoid'
        and router.num_fused_shared_experts==0
        and type(scale) in (int,float) and math.isfinite(scale) and scale>0)


def _device_supported(device):
    # The model admission already witnessed the HCU gfx938 platform.
    return device.type=='cuda'


def _can_route(router,hidden,logits,indices_type):
    bias=router.e_score_correction_bias
    return (_routing_contract(router,getattr(router,'_hcu_routing_physical_experts',256))
        and indices_type in (None,torch.int32,torch.int64)
        and isinstance(hidden,torch.Tensor) and isinstance(logits,torch.Tensor)
        and isinstance(bias,torch.Tensor) and hidden.ndim==logits.ndim==2
        and hidden.shape==(logits.shape[0],6144) and logits.shape[0]>0
        and logits.shape[1]==256 and bias.shape==(256,)
        and hidden.dtype==torch.bfloat16 and logits.dtype==bias.dtype==torch.float32
        and hidden.device==logits.device==bias.device and _device_supported(logits.device)
        and hidden.is_contiguous() and logits.is_contiguous() and bias.is_contiguous())


def _route(self,hidden_states,router_logits,indices_type,*,input_ids=None):
    if not _requested() or not _can_route(self,hidden_states,router_logits,indices_type):
        return self._hcu_native_grouped_compute(self,hidden_states,router_logits,indices_type,
            input_ids=input_ids)
    # One group selecting that group has no filtering effect. Core still owns
    # select_experts, capture, physical mapping and the final index cast.
    kwargs=dict(hidden_states=hidden_states,gating_output=router_logits,
        scoring_func=self.scoring_func,e_score_correction_bias=self.e_score_correction_bias,
        topk=self.top_k,renormalize=self.renormalize,indices_type=indices_type,
        input_tokens=input_ids,routed_scaling_factor=self.routed_scaling_factor)
    if getattr(self,'_hcu_preserve_unmasked_padding',False) and is_forward_context_available():
        context=get_forward_context()
        if context.is_padding is not None:
            # HT's native GroupedTopKRouter has skip_padding=False. The fused
            # Core API reads the global mask regardless of that router flag.
            # Clear only the private call context, then restore it before
            # BaseRouter runs its capture callback and static EPLB mapping.
            with override_forward_context(replace(context,is_padding=None)):
                return fused_topk_bias(**kwargs)
    return fused_topk_bias(**kwargs)


def attach(model,cfg):
    if not enabled(cfg):return model
    from vllm.models.hy_v4.nvidia.moe import HYV4MoEFused
    prefill=_topology(cfg)=='prefill'
    physical_experts=272 if prefill else 256
    attached=0
    for layer in model.modules():
        if not isinstance(layer,HYV4MoEFused):continue
        runner=layer.experts;router=runner.router
        if type(router) is not GroupedTopKRouter:continue
        if hasattr(router,'_hcu_native_grouped_compute'):
            if router._compute_routing.__func__ is not _route:
                raise RuntimeError('HY4 grouped router instance ownership changed')
            continue
        parallel=runner.moe_config.moe_parallel_config
        expected=(1,1,16,16) if prefill else (1,16,1,16)
        backend='deepep_high_throughput' if prefill else 'deepep_low_latency'
        if (parallel.all2all_backend!=backend
                or (parallel.tp_size,parallel.dp_size,parallel.pcp_size,parallel.ep_size)!=expected
                or not _routing_contract(router,physical_experts)
                or router._compute_routing.__func__ is not GroupedTopKRouter._compute_routing):
            continue
        if prefill and (router.eplb_state is None or router.skip_padding
                or not layer.enable_eplb
                or (layer.n_logical_experts,layer.n_physical_experts,layer.n_redundant_experts)!=(256,272,16)):
            continue
        router._hcu_native_grouped_compute=router._compute_routing.__func__
        router._hcu_routing_physical_experts=physical_experts
        router._hcu_preserve_unmasked_padding=prefill
        router._compute_routing=MethodType(_route,router)
        attached+=1
    if attached:
        logger.info_once('HY4 native fused sigmoid routing enabled for %s',
            'PCP16/EP16 static R16 HT' if prefill else 'DP16/EP16 LL')
    return model
