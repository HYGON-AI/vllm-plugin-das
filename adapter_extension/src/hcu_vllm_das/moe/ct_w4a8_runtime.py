# SPDX-License-Identifier: Apache-2.0
"""Owned HIPC experts and public modular-kernel composition for CT W4A8."""
from __future__ import annotations
from functools import lru_cache
import copy
import inspect
import logging
from types import SimpleNamespace
import torch
from ..adapter_api import adapter_module

hcu_env = adapter_module('hcu_env')
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.all2all_utils import maybe_make_prepare_finalize as native_make_prepare_finalize
from vllm.model_executor.layers.fused_moe.deep_gemm_utils import compute_aligned_M, deepgemm_unpermute_and_reduce, ep_scatter
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import TopKWeightAndReduceDelegate, TopKWeightAndReduceNoOP
from vllm.model_executor.layers.fused_moe.utils import _resize_cache
from vllm.model_executor.layers.quantization.utils.quant_utils import kInt4W4A8StaticChannelSym, kInt8DynamicTokenSym
from vllm.platforms import current_platform
from .ct_w4a8_hints import expected_m, masked_gemm
from .ct_w4a8_prepare import (
    DeepEPHTPrepareAndFinalize,
    DeepEPLLPrepareAndFinalize,
    FixedHTExpertTokens,
    FixedHTInt8ExpertTokens,
    HcuCTHTPrepareAndFinalize,
    HcuCTLLPrepareAndFinalize,
)


def _accepts_device_count(clamp_quant):
    try:
        parameter = inspect.signature(clamp_quant).parameters.get("num_local_tokens_tensor")
    except (TypeError, ValueError):
        return False
    return parameter is not None and parameter.kind != inspect.Parameter.POSITIONAL_ONLY


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


def _fixed_receive_capacity(config, moe, buffer):
    from ..dsv41.profile_worker import HT_FIXED_CAPACITY_KEY, w4a8_profile_selected

    extra = getattr(config, "additional_config", None) or {}
    enabled = extra.get(HT_FIXED_CAPACITY_KEY, False)
    if enabled is False:
        return None
    if enabled is not True:
        raise ValueError(f"{HT_FIXED_CAPACITY_KEY} must be a boolean")

    from ..dsv41.w4a8_profile import routed_token_bound

    parallel = config.parallel_config
    if (not w4a8_profile_selected(config) or parallel.use_ubatching
            or parallel.prefill_context_parallel_size not in (4, 8)
            or moe.moe_parallel_config.ep_size != parallel.prefill_context_parallel_size):
        raise ValueError("Fixed-capacity W4A8 DeepEP HT requires eager PCP4/EP4 or PCP8/EP8 with rank-local profiling")
    rdma_ranks = getattr(getattr(buffer, "runtime", None), "get_num_rdma_ranks", None)
    if not callable(rdma_ranks) or rdma_ranks() != 1:
        raise ValueError("Fixed-capacity W4A8 DeepEP HT requires a verified intranode buffer")
    _require_abi(buffer.dispatch, ("num_worst_tokens",))
    spec = config.speculative_config
    speculative_tokens = (spec.num_speculative_tokens or 0) if spec else 0
    scheduler = config.scheduler_config
    return routed_token_bound(
        moe.max_num_tokens, scheduler.max_num_seqs, 1 + speculative_tokens,
        parallel.prefill_context_parallel_size, parallel.pcp_shard_decode_requests,
    )


def _fixed_sender_int8(config, capacity, buffer):
    from ..dsv41.profile_worker import HT_SENDER_INT8_KEY

    extra = getattr(config, "additional_config", None) or {}
    enabled = extra.get(HT_SENDER_INT8_KEY, False)
    if type(enabled) is not bool:
        raise ValueError(f"{HT_SENDER_INT8_KEY} must be a boolean")
    if enabled:
        if capacity is None:
            raise ValueError(f"{HT_SENDER_INT8_KEY} requires fixed-capacity DeepEP HT")
        _require_abi(buffer.dispatch, ("x", "num_worst_tokens"))
    return enabled


def _isolated_draft_ht_buffer(manager, hidden):
    import deep_ep

    options = manager._make_all2all_kwargs()
    if options["num_rdma_bytes"]:
        raise ValueError("A separate dSpark HT buffer requires single-node DeepEP")

    # The target's fixed receive path can leave host count metadata that the
    # draft's dynamic receive would otherwise read on the next dispatch.
    size_hints = (
        deep_ep.Buffer.get_dispatch_config(manager.world_size),
        deep_ep.Buffer.get_combine_config(manager.world_size),
    )
    options["num_nvl_bytes"] = max(
        config.get_nvl_buffer_size_hint(hidden * 2, manager.world_size)
        for config in size_hints
    )
    options["_buffer_owner"] = "dsv41_pcp_draft"

    def create_buffer(*, _buffer_owner, **kwargs):
        buffer = deep_ep.Buffer(**kwargs)
        logging.getLogger("vllm." + __name__).info(
            "HCU dSpark uses a separate DeepEP HT receive buffer: %d bytes",
            kwargs["num_nvl_bytes"],
        )
        return buffer

    # The EP manager owns both buffers and tears them down with the group.
    return manager.handle_cache.get_or_create(options, create_buffer)


def make_prepare(moe,descriptor,routing_tables,config=None):
    backend=moe.moe_parallel_config.all2all_backend
    if backend=='allgather_reducescatter':
        from hcu_vllm_das.dspark_agrs import make_draft_prepare
        return make_draft_prepare(moe,config)
    shared_ipc = False
    factory_args = {}
    factory_moe = moe
    if backend == 'deepep_low_latency' and config is not None:
        from hcu_vllm_das.dspark_backend import native_draft_ll_requested

        shared_ipc = native_draft_ll_requested(config)
        if shared_ipc:
            from vllm.model_executor.layers.fused_moe.all2all_utils import get_ep_all2all_manager
            from hcu_vllm_das.dspark_ll import SharedIPCManager
            from hcu_vllm_das.dspark_graph import token_capacity

            capacity = token_capacity(config).query_rows
            factory_args['all2all_manager'] = SharedIPCManager(get_ep_all2all_manager(), capacity)
            factory_moe = copy.copy(moe)
            factory_moe.max_num_tokens = capacity
    native=native_make_prepare_finalize(moe=factory_moe,quant_config=descriptor,routing_tables=routing_tables,
        allow_new_interface=True,use_monolithic=False,input_dtype=None,**factory_args)
    if backend=='deepep_low_latency' and not isinstance(native,DeepEPLLPrepareAndFinalize):
        raise RuntimeError('HCU W4A8 requested LL but native preparation chose another format')
    if backend=='deepep_high_throughput' and not isinstance(native,DeepEPHTPrepareAndFinalize):
        raise RuntimeError('HCU W4A8 requested HT but native preparation chose another format')
    if isinstance(native,DeepEPLLPrepareAndFinalize):
        if native.max_num_tokens_per_rank()!=factory_moe.max_num_tokens or native.num_dispatchers()!=moe.moe_parallel_config.ep_size:
            raise RuntimeError('DeepEP LL capacity/group disagrees with loaded MoE')
        _require_abi(native.buffer.low_latency_dispatch,('topk_weight','quant_type','quant_group_size','return_recv_hook'))
        _require_abi(native.buffer.low_latency_combine,('out','return_recv_hook'))
        return HcuCTLLPrepareAndFinalize(native.buffer,native.max_num_tokens_per_rank(),native.num_dispatchers(),
            global_to_physical=native.global_to_physical,physical_to_global=native.physical_to_global,
            local_expert_global_ids=native.local_expert_global_ids,
            reset_before_dispatch=(shared_ipc and not
                factory_args['all2all_manager'].has_model_entry_reset(moe.num_experts)),
            preserve_agrs_quantization=shared_ipc and moe.num_experts == 128)
    if isinstance(native,DeepEPHTPrepareAndFinalize):
        if native.num_dispatchers()!=moe.moe_parallel_config.ep_size:
            raise RuntimeError('DeepEP HT group disagrees with loaded MoE')
        capacity = _fixed_receive_capacity(config, moe, native.buffer)
        buffer = native.buffer
        from ..pcp_dspark_draft import needs_isolated_ht_buffer
        if needs_isolated_ht_buffer(config):
            from vllm.model_executor.layers.fused_moe.all2all_utils import get_ep_all2all_manager
            buffer = _isolated_draft_ht_buffer(get_ep_all2all_manager(), moe.hidden_dim)
        prepare = HcuCTHTPrepareAndFinalize(
            buffer, native.num_dispatchers(), native.dp_size,
            native.rank_expert_offset, fixed_receive_tokens=capacity,
            fixed_sender_int8=_fixed_sender_int8(config, capacity, buffer),
        )
        from hcu_vllm_das.hy4_ht_int8 import select_prepare
        return select_prepare(prepare, config, moe)
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
    if isinstance(expert_tokens_meta, FixedHTInt8ExpertTokens):
        from .ct_w4a8_fixed_int8_scatter import scatter_fixed_int8

        scatter_fixed_int8(aq, token_scale, topk_ids, counts, starts,
                           aq_out, scale, indices, inverse)
    else:
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


def contiguous_aligned_rows(tokens, topk, local_experts):
    alignment = HcuCTContiguousExperts.ALIGNMENT
    assignments = tokens * topk
    padding = min(assignments, local_experts) * (alignment - 1)
    return (assignments + padding + alignment - 1) // alignment * alignment


def contiguous_workspace_shapes(tokens, aligned_rows, gate_width, hidden_width):
    # workspace13 is allocated in the output dtype and viewed as INT8 for
    # both permutation and activation. Count pairs of INT8 values per slot.
    int8_width = max(hidden_width, gate_width // 2)
    packed_width = (int8_width + 1) // 2
    return (
        (aligned_rows, packed_width),
        (aligned_rows, max(gate_width, hidden_width)),
        (tokens, hidden_width),
    )


def _aligned_expert_row_count(counts, alignment):
    # ep_scatter writes a prefix of independently aligned expert segments.
    padded = torch.bitwise_and(counts + alignment - 1, -alignment)
    return padded.sum(dtype=torch.int32).view(1)


class HcuCTContiguousExperts(_Experts):
    ALIGNMENT=256
    def __init__(self,moe,quant,weights,ops):
        super().__init__(moe,quant);self.weights=weights;self.ops=ops
        self._gather = deepgemm_unpermute_and_reduce
        self._clamp_accepts_device_count = _accepts_device_count(ops.clamp_quant)
        from .clamped_int8_admission import admitted
        self._live_row_activation = admitted(ops, weights, self.w1_scale,
            self.w2_scale, quant.gemm1_clamp_limit, masked=False)
    @staticmethod
    def activation_format(): return mk.FusedMoEActivationFormat.Standard
    def finalize_weight_and_reduce_impl(self): return TopKWeightAndReduceNoOP()
    def _aligned_rows(self, tokens, topk, local_experts, metadata):
        if isinstance(metadata, FixedHTExpertTokens):
            return contiguous_aligned_rows(tokens, topk, local_experts)
        return compute_aligned_M(
            M=tokens, num_topk=topk, local_num_experts=local_experts,
            alignment=self.ALIGNMENT, expert_tokens_meta=metadata,
        )

    def workspace_shapes(self,M,N,K,topk,global_num_experts,local_num_experts,expert_tokens_meta,activation):
        aligned = self._aligned_rows(M, topk, local_num_experts, expert_tokens_meta)
        return contiguous_workspace_shapes(M, aligned, N, K)
    def apply(self,output,hidden_states,w1,w2,topk_weights,topk_ids,activation,global_num_experts,expert_map,
              a1q_scale,a2_scale,workspace13,workspace2,expert_tokens_meta,apply_router_weight_on_input):
        if hidden_states.shape[0]==0: return
        self._check(hidden_states,a1q_scale,activation)
        e,m,n,k,topk=self.moe_problem_size(hidden_states,w1,w2,topk_ids)
        aligned = self._aligned_rows(m, topk, e, expert_tokens_meta)
        if isinstance(expert_tokens_meta, FixedHTExpertTokens):
            # DeepEP returns local physical IDs after any EPLB remapping.
            expert_map = None
        workspace=_resize_cache(workspace13.view(dtype=torch.int8),(aligned,k))
        data,scale,indices,inverse,_=deepgemm_moe_permute(aq=hidden_states,aq_scale=a1q_scale,
            topk_ids=topk_ids,local_num_experts=e,expert_map=expert_map,expert_tokens_meta=expert_tokens_meta,
            aq_out=workspace,block_size=k)
        indices=indices.to(dtype=torch.int32).contiguous()
        gateup=_resize_cache(workspace2,(aligned,n))
        self.ops.gemm_contiguous((data,scale),(self.weights[0],self.w1_scale),gateup,indices)
        qout=_resize_cache(workspace13.view(dtype=torch.int8),(aligned,n//2))
        limit=self.quant_config.gemm1_clamp_limit
        if (isinstance(expert_tokens_meta, FixedHTExpertTokens)
                and limit is not None and limit > 0 and self._clamp_accepts_device_count):
            active_rows = _aligned_expert_row_count(expert_tokens_meta.expert_num_tokens, self.ALIGNMENT)
            mid, ms = self.ops.clamp_quant(
                gateup, limit=limit, output=qout, num_local_tokens_tensor=active_rows,
            )
        elif self._live_row_activation:
            from .clamped_int8_triton import clamped_int8_2304
            mid,ms=clamped_int8_2304(gateup,indices,masked=False,output=qout)
        elif limit is not None and limit>0:
            mid,ms=self.ops.clamp_quant(gateup,limit=limit,output=qout)
        else:
            mid,ms=self.ops.silu_quant(gateup,output=qout,expert_ids=indices)
        down=_resize_cache(workspace2,(aligned,k))
        self.ops.gemm_contiguous((mid,ms),(self.weights[1],self.w2_scale),down,indices)
        weights=torch.ones_like(topk_weights) if apply_router_weight_on_input else topk_weights
        self._gather(a=down,topk_ids=topk_ids,topk_weights=weights,
            inv_perm=inverse,expert_map=expert_map,output=output)


class HcuCTMaskedExperts(_Experts):
    def __init__(self,moe,quant,weights,ops,capacity,dispatchers):
        super().__init__(moe,quant,max_num_tokens=capacity,num_dispatchers=dispatchers)
        self.weights=weights;self.ops=ops
        from .clamped_int8_admission import masked_qualifier
        self._qualify_live_activation = masked_qualifier(
            ops, weights, self.w1_scale, self.w2_scale, quant.gemm1_clamp_limit)
        self._live_row_activation = self._qualify_live_activation is not None
        self._small_batch_hint = False
        hint_mode, _ = hcu_env.resolve('HCU_DSV41_W4A8_MOE_EXPECTED_M', host='vllm')
        local_experts = int(self.w1_scale.shape[0])
        if (hint_mode == '1'
                and local_experts * dispatchers in (128, 384)
                and tuple(self.w1_scale.shape[:2]) == (local_experts, 4608)
                and tuple(self.w2_scale.shape[:2]) == (local_experts, 5120)
                and quant.gemm1_clamp_limit == 10
                and weights[0].device.type == 'cuda'):
            props = torch.cuda.get_device_properties(weights[0].device)
            arch = getattr(props, 'gcnArchName', '').split(':', 1)[0]
            self._small_batch_hint = arch == 'gfx938' and props.multi_processor_count == 64
    @staticmethod
    def activation_format(): return mk.FusedMoEActivationFormat.BatchedExperts
    def finalize_weight_and_reduce_impl(self): return TopKWeightAndReduceDelegate()
    def expected_m(self, global_num_experts, capacity, topk):
        return capacity
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
        # The hint selects kernels. Device counts still bound every expert's rows.
        hint = self.expected_m(global_num_experts,t,topk_ids.shape[-1])
        use_hint = (self._small_batch_hint and e * self.num_dispatchers == global_num_experts
                    and (global_num_experts, topk_ids.shape[1]) in ((384, 6), (128, 3)))
        if use_hint:
            hint = expected_m(topk_ids.shape[0], self.num_dispatchers, topk_ids.shape[1],
                              global_num_experts, t)
        gateup=_resize_cache(workspace13,(e,t,n))
        self._masked_gemm((hidden_states, a1q_scale), (self.weights[0], self.w1_scale),
                          gateup, counts, hint, use_hint)
        limit=self.quant_config.gemm1_clamp_limit
        activation_hint=hint if use_hint else t
        if (self._live_row_activation
                and self._qualify_live_activation(t, activation_hint)):
            from .clamped_int8_triton import clamped_int8_2304
            mid,ms=clamped_int8_2304(gateup,counts,masked=True,expected_m=activation_hint)
        elif limit is not None and limit>0:
            mid,ms=self.ops.clamp_quant_ep(gateup,limit=limit,mask_m=counts,expect_m=activation_hint)
        else:
            mid,ms=self.ops.silu_quant_ep(gateup,tokens_per_expert=counts)
        self._masked_gemm((mid, ms), (self.weights[1], self.w2_scale), output, counts,
                          hint, use_hint)

    def _masked_gemm(self, inputs, weights, output, counts, hint, use_hint):
        if use_hint:
            return masked_gemm(self.ops.gemm_masked, inputs, weights, output, counts, hint)
        return self.ops.gemm_masked(inputs, weights, output, counts, hint)


def make_experts(moe,quant,weights,ops,prepare):
    if prepare.activation_format==mk.FusedMoEActivationFormat.BatchedExperts:
        from hcu_vllm_das.hy4_ll_experts import make_ll_experts
        return make_ll_experts(moe,quant,weights,ops,prepare)
    return HcuCTContiguousExperts(moe,quant,weights,ops)
