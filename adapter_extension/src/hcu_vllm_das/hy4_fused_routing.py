# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Use Core's fused sigmoid router on owned HY4 DP16 low-latency runners.

Core retains select_experts, capture callbacks, EPLB mapping, index dtype,
padding masks and the actual operator. Restart when changing this policy.
"""
import math
from types import MethodType

import torch
from vllm import envs
from vllm._aiter_ops import rocm_aiter_ops
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
        and (p.tensor_parallel_size,p.pipeline_parallel_size,p.prefill_context_parallel_size,
             p.decode_context_parallel_size,p.data_parallel_size)==(1,1,1,1,16)
        and p.enable_expert_parallel and not p.enable_eplb and not p.enable_elastic_ep
        and not getattr(p,'enable_dbo',False) and getattr(p,'ubatch_size',0) in (0,1)
        and not envs.VLLM_BATCH_INVARIANT and not rocm_aiter_ops.is_fused_moe_enabled()
        and _device_arch()=='gfx938')


def enabled(cfg):
    return _requested() and _supported_config(cfg)


def _routing_contract(router):
    scale=router.routed_scaling_factor
    return (router.global_num_experts==256 and router.top_k==8
        and router.num_expert_group==1 and router.topk_group==1
        and router.renormalize is True and router.scoring_func=='sigmoid'
        and router.num_fused_shared_experts==0
        and type(scale) in (int,float) and math.isfinite(scale) and scale>0)


def _device_supported(device):
    # The model admission already witnessed the HCU gfx938 platform.
    return device.type=='cuda'


def _can_route(router,hidden,logits,indices_type):
    bias=router.e_score_correction_bias
    return (_routing_contract(router) and indices_type in (None,torch.int32,torch.int64)
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
    # One group selecting that group has no filtering effect. Keep the
    # GroupedTopKRouter type/routing_method_type; Core owns its native mask.
    return fused_topk_bias(hidden_states=hidden_states,gating_output=router_logits,
        scoring_func=self.scoring_func,e_score_correction_bias=self.e_score_correction_bias,
        topk=self.top_k,renormalize=self.renormalize,indices_type=indices_type,
        input_tokens=input_ids,routed_scaling_factor=self.routed_scaling_factor)


def attach(model,cfg):
    if not enabled(cfg):return model
    from vllm.models.hy_v4.nvidia.moe import HYV4MoEFused
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
        if (parallel.all2all_backend!='deepep_low_latency'
                or (parallel.tp_size,parallel.dp_size,parallel.pcp_size,parallel.ep_size)!=(1,16,1,16)
                or not _routing_contract(router)
                or router._compute_routing.__func__ is not GroupedTopKRouter._compute_routing):
            continue
        router._hcu_native_grouped_compute=router._compute_routing.__func__
        router._compute_routing=MethodType(_route,router)
        attached+=1
    if attached:
        logger.info_once('HY4 native fused sigmoid routing enabled for DP16/EP16 LL')
    return model
