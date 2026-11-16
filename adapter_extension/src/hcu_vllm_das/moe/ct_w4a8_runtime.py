# SPDX-License-Identifier: Apache-2.0
"""Owned HIPC experts and public modular-kernel composition for CT W4A8."""
from __future__ import annotations
from functools import lru_cache
import inspect
from types import SimpleNamespace
import torch
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.all2all_utils import maybe_make_prepare_finalize as native_make_prepare_finalize
from vllm.model_executor.layers.fused_moe.deep_gemm_utils import compute_aligned_M, deepgemm_unpermute_and_reduce, ep_scatter
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import TopKWeightAndReduceDelegate, TopKWeightAndReduceNoOP
from vllm.model_executor.layers.fused_moe.utils import _resize_cache
from vllm.model_executor.layers.quantization.utils.quant_utils import kInt4W4A8StaticChannelSym, kInt8DynamicTokenSym
from vllm.platforms import current_platform
from .ct_w4a8_prepare import HcuCTHTPrepareAndFinalize,HcuCTLLPrepareAndFinalize,DeepEPHTPrepareAndFinalize,DeepEPLLPrepareAndFinalize


@lru_cache(maxsize=1)
def get_ops():
    import deepgemm
    import lightop
    from lightop import activation
    return SimpleNamespace(pack=getattr(deepgemm,'pack_w4a8_moe_hipc_weight',None),
        view_n32=getattr(deepgemm,'view_w4a8_moe_hipc_weight_n32_layout',None),
        gemm_contiguous=getattr(deepgemm,'m_grouped_w4a8_gemm_nt_contiguous_hipc',None),
        gemm_masked=getattr(deepgemm,'m_grouped_w4a8_gemm_nt_masked_hipc',None),
        silu_quant=getattr(activation,'fuse_silu_mul_quant',None),
        silu_quant_ep=getattr(activation,'fuse_silu_mul_quant_ep',None),
        clamp_quant=getattr(lightop,'fuse_silu_mul_clamp_quant',None),
        clamp_quant_ep=getattr(lightop,'fuse_silu_mul_clamp_quant_ep',None))


def validate_ops(ops,backend,limit):
    ll=backend=='deepep_low_latency'
    names=['pack','view_n32','gemm_masked'] if ll else ['pack','gemm_contiguous']
    names += [('clamp_quant_ep' if ll else 'clamp_quant') if limit is not None and limit>0 else ('silu_quant_ep' if ll else 'silu_quant')]
    for name in names:
        if not callable(getattr(ops,name,None)): raise RuntimeError('HCU W4A8 provider is missing '+name)


def _require_abi(function,names):
    parameters=inspect.signature(function).parameters
    if any(p.kind==inspect.Parameter.VAR_KEYWORD for p in parameters.values()): return
    if not set(names)<=parameters.keys(): raise RuntimeError('HCU DeepEP provider ABI lacks '+','.join(names))


def make_prepare(moe,descriptor,routing_tables,config=None):
    backend=moe.moe_parallel_config.all2all_backend
    if backend=='allgather_reducescatter':
        from hcu_vllm_das.dspark_agrs import make_draft_prepare
        return make_draft_prepare(moe,config)
    native=native_make_prepare_finalize(moe=moe,quant_config=descriptor,routing_tables=routing_tables,
        allow_new_interface=True,use_monolithic=False,input_dtype=None)
    if backend=='deepep_low_latency' and not isinstance(native,DeepEPLLPrepareAndFinalize):
        raise RuntimeError('HCU W4A8 requested LL but native preparation chose another format')
    if backend=='deepep_high_throughput' and not isinstance(native,DeepEPHTPrepareAndFinalize):
        raise RuntimeError('HCU W4A8 requested HT but native preparation chose another format')
    if isinstance(native,DeepEPLLPrepareAndFinalize):
        if native.max_num_tokens_per_rank()!=moe.max_num_tokens or native.num_dispatchers()!=moe.moe_parallel_config.ep_size:
            raise RuntimeError('DeepEP LL capacity/group disagrees with loaded MoE')
        _require_abi(native.buffer.low_latency_dispatch,('topk_weight','quant_type','quant_group_size','return_recv_hook'))
        _require_abi(native.buffer.low_latency_combine,('out','return_recv_hook'))
        return HcuCTLLPrepareAndFinalize(native.buffer,native.max_num_tokens_per_rank(),native.num_dispatchers(),
            global_to_physical=native.global_to_physical,physical_to_global=native.physical_to_global,
            local_expert_global_ids=native.local_expert_global_ids)
    if isinstance(native,DeepEPHTPrepareAndFinalize):
        if native.num_dispatchers()!=moe.moe_parallel_config.ep_size:
            raise RuntimeError('DeepEP HT group disagrees with loaded MoE')
        return HcuCTHTPrepareAndFinalize(native.buffer,native.num_dispatchers(),native.dp_size,native.rank_expert_offset)
    raise RuntimeError('HCU W4A8 requires a fixed native HT/LL prepare object')


def deepgemm_moe_permute(*,aq,aq_scale,topk_ids,local_num_experts,expert_map,expert_tokens_meta,aq_out,block_size):
    """Owned 256-aligned permutation using the 0.33 public ep_scatter kernel."""
    if aq.ndim != 2 or aq_out is None or aq_out.shape[1] != aq.shape[1]:
        raise ValueError('W4A8 permutation requires [M,K] activation and matching workspace')
    aligned = aq_out.shape[0]
    token_scale = aq_scale
    if token_scale.ndim == 1:
        token_scale = token_scale.unsqueeze(-1)
    if not token_scale.is_contiguous():
        token_scale = token_scale.contiguous()
    counts = getattr(expert_tokens_meta, 'expert_num_tokens', None)
    if counts is None:
        from vllm.model_executor.layers.fused_moe.utils import count_expert_num_tokens
        counts = count_expert_num_tokens(topk_ids, local_num_experts, expert_map)
    if counts.dtype != torch.int32:
        counts = counts.to(torch.int32)
    if not counts.is_contiguous():
        counts = counts.contiguous()
    scale = torch.zeros((aligned, 1), dtype=torch.float32, device=aq_out.device)
    indices = torch.full((aligned,), -1, dtype=torch.int32, device=aq_out.device)
    inverse = torch.empty(topk_ids.shape, dtype=torch.int32, device=aq_out.device)
    starts = torch.empty((local_num_experts,), dtype=torch.int32, device=aq_out.device)
    ep_scatter(recv_x=aq, recv_x_scale=token_scale, recv_topk=topk_ids,
        num_recv_tokens_per_expert=counts, expert_start_loc=starts, expert_map=expert_map,
        output_tensor=aq_out, output_tensor_scale=scale, m_indices=indices,
        output_index=inverse, align_m=256, block_size=block_size,
        pack_ue8m0=False, use_psum_layout=False)
    return aq_out, scale, indices, inverse, None


class _Experts(mk.FusedMoEExpertsModular):
    @staticmethod
    def _supports_current_device(): return current_platform.is_rocm() and current_platform.device_name=='hcu'
    @staticmethod
    def _supports_no_act_and_mul(): return False
    @staticmethod
    def _supports_quant_scheme(weight_key,activation_key): return (weight_key,activation_key)==(kInt4W4A8StaticChannelSym,kInt8DynamicTokenSym)
    @staticmethod
    def _supports_activation(activation): return activation==MoEActivation.SILU
    @staticmethod
    def _supports_parallel_config(parallel): return parallel.use_ep
    def moe_problem_size(self,a1,w1,w2,topk_ids):
        return self.w1_scale.shape[0],a1.shape[-2],self.w1_scale.shape[1],self.w2_scale.shape[1],topk_ids.shape[1]
    def _check(self,x,scale,activation):
        if x.dtype!=torch.int8 or scale is None or scale.dtype!=torch.float32 or activation!=MoEActivation.SILU:
            raise ValueError('HIPC experts require INT8, FP32 token scales and gated SiLU')


class HcuCTContiguousExperts(_Experts):
    ALIGNMENT=256
    def __init__(self,moe,quant,weights,ops):
        super().__init__(moe,quant);self.weights=weights;self.ops=ops
    @staticmethod
    def activation_format(): return mk.FusedMoEActivationFormat.Standard
    def finalize_weight_and_reduce_impl(self): return TopKWeightAndReduceNoOP()
    def workspace_shapes(self,M,N,K,topk,global_num_experts,local_num_experts,expert_tokens_meta,activation):
        aligned=compute_aligned_M(M=M,num_topk=topk,local_num_experts=local_num_experts,alignment=256,expert_tokens_meta=expert_tokens_meta)
        return (aligned,max(N//2,K)),(aligned,max(N,K)),(M,K)
    def apply(self,output,hidden_states,w1,w2,topk_weights,topk_ids,activation,global_num_experts,expert_map,
              a1q_scale,a2_scale,workspace13,workspace2,expert_tokens_meta,apply_router_weight_on_input):
        if hidden_states.shape[0]==0: return
        self._check(hidden_states,a1q_scale,activation)
        e,m,n,k,topk=self.moe_problem_size(hidden_states,w1,w2,topk_ids)
        aligned=compute_aligned_M(M=m,num_topk=topk,local_num_experts=e,alignment=256,expert_tokens_meta=expert_tokens_meta)
        workspace=_resize_cache(workspace13.view(dtype=torch.int8),(aligned,k))
        data,scale,indices,inverse,_=deepgemm_moe_permute(aq=hidden_states,aq_scale=a1q_scale,
            topk_ids=topk_ids,local_num_experts=e,expert_map=expert_map,expert_tokens_meta=expert_tokens_meta,
            aq_out=workspace,block_size=k)
        indices=indices.to(dtype=torch.int32).contiguous()
        gateup=_resize_cache(workspace2,(aligned,n))
        self.ops.gemm_contiguous((data,scale),(self.weights[0],self.w1_scale),gateup,indices)
        qout=_resize_cache(workspace13.view(dtype=torch.int8),(aligned,n//2))
        limit=self.quant_config.gemm1_clamp_limit
        if limit is not None and limit>0:
            mid,ms=self.ops.clamp_quant(gateup,limit=limit,output=qout)
        else:
            mid,ms=self.ops.silu_quant(gateup,output=qout,expert_ids=indices)
        down=_resize_cache(workspace2,(aligned,k))
        self.ops.gemm_contiguous((mid,ms),(self.weights[1],self.w2_scale),down,indices)
        weights=torch.ones_like(topk_weights) if apply_router_weight_on_input else topk_weights
        deepgemm_unpermute_and_reduce(a=down,topk_ids=topk_ids,topk_weights=weights,
            inv_perm=inverse,expert_map=expert_map,output=output)


class HcuCTMaskedExperts(_Experts):
    def __init__(self,moe,quant,weights,ops,capacity,dispatchers):
        super().__init__(moe,quant,max_num_tokens=capacity,num_dispatchers=dispatchers)
        self.weights=weights;self.ops=ops
    @staticmethod
    def activation_format(): return mk.FusedMoEActivationFormat.BatchedExperts
    def finalize_weight_and_reduce_impl(self): return TopKWeightAndReduceDelegate()
    def workspace_shapes(self,M,N,K,topk,global_num_experts,local_num_experts,expert_tokens_meta,activation):
        total=self.max_num_tokens*self.num_dispatchers
        if M!=total: raise ValueError('LL received token extent differs from checked capacity')
        return (local_num_experts,total,max(N,K)),(local_num_experts,total,N//2),(local_num_experts,total,K)
    def apply(self,output,hidden_states,w1,w2,topk_weights,topk_ids,activation,global_num_experts,expert_map,
              a1q_scale,a2_scale,workspace13,workspace2,expert_tokens_meta,apply_router_weight_on_input):
        self._check(hidden_states,a1q_scale,activation)
        if expert_tokens_meta is None: raise ValueError('LL requires actual per-expert token counts')
        e,t,n,k,_=self.moe_problem_size(hidden_states,w1,w2,topk_ids)
        if t!=self.max_num_tokens*self.num_dispatchers: raise ValueError('LL input exceeds expert workspace contract')
        counts=expert_tokens_meta.expert_num_tokens
        gateup=_resize_cache(workspace13,(e,t,n))
        self.ops.gemm_masked((hidden_states,a1q_scale),(self.weights[0],self.w1_scale),gateup,counts,t)
        limit=self.quant_config.gemm1_clamp_limit
        if limit is not None and limit>0:
            mid,ms=self.ops.clamp_quant_ep(gateup,limit=limit,mask_m=counts,expect_m=t)
        else:
            mid,ms=self.ops.silu_quant_ep(gateup,tokens_per_expert=counts)
        self.ops.gemm_masked((mid,ms),(self.weights[1],self.w2_scale),output,counts,t)


def make_experts(moe,quant,weights,ops,prepare):
    if prepare.activation_format==mk.FusedMoEActivationFormat.BatchedExperts:
        return HcuCTMaskedExperts(moe,quant,weights,ops,prepare.max_num_tokens_per_rank(),prepare.num_dispatchers())
    return HcuCTContiguousExperts(moe,quant,weights,ops)
