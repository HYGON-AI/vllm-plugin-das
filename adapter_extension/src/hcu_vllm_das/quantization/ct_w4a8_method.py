# SPDX-License-Identifier: Apache-2.0
"""Owned packed CT W4A8 method; native CPU CT/oracle construction is not used."""
from __future__ import annotations
import torch
from vllm.model_executor.layers.fused_moe import FusedMoEMethodBase
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.utils import set_weight_attrs
from .ct_w4a8_weights import PackedCTWeights


class HcuCTW4A8MoEMethod(FusedMoEMethodBase):
    def __init__(self, weight, activation, moe, config):
        super().__init__(moe)
        from .ct_w4a8 import _schema
        if not _schema(weight,activation):
            raise ValueError('HCU CT W4A8 requires symmetric static channel INT4 and dynamic token INT8')
        if moe.moe_backend!='deep_gemm' or moe.has_bias or moe.in_dtype!=torch.bfloat16 or not moe.is_act_and_mul or getattr(moe.activation,'value',moe.activation)!='silu':
            raise ValueError('HCU CT W4A8 requires BF16 gated SiLU, DeepGEMM and no expert bias')
        if moe.swiglu_alpha not in (None,1.0) or moe.swiglu_beta not in (None,0.0):
            raise ValueError('HCU CT W4A8 requires standard SwiGLU alpha/beta')
        self._config=config
        self._backend=moe.moe_parallel_config.all2all_backend
        self._capacity=moe.max_num_tokens
        self._packed=None
        self._failed=False
        from hcu_vllm_das.moe.ct_w4a8_runtime import get_ops, validate_ops
        self._ops=get_ops()
        validate_ops(self._ops,self._backend,moe.swiglu_limit)

    def maybe_roundup_sizes(self, hidden_size, intermediate_size_per_partition, act_dtype, moe_parallel_config):
        if hidden_size%512 or intermediate_size_per_partition%256:
            raise ValueError('HIPC requires H%512==0 and I%256==0; implicit padding is unsupported')
        if moe_parallel_config.all2all_backend=='deepep_low_latency':
            from hcu_vllm_das.moe.ct_w4a8_prepare import HcuCTLLPrepareAndFinalize
            if hidden_size not in HcuCTLLPrepareAndFinalize.SUPPORTED_HIDDEN_SIZES:
                raise ValueError('HCU DeepEP LL does not support this hidden size')
        return hidden_size,intermediate_size_per_partition

    def create_weights(self,layer,num_experts,hidden_size,intermediate_size_per_partition,params_dtype,**extra_weight_attrs):
        if self._packed is not None: raise RuntimeError('CT W4A8 parameters already created')
        self._packed=PackedCTWeights(layer,num_experts,hidden_size,intermediate_size_per_partition,extra_weight_attrs.get('weight_loader'))
        extras={k:v for k,v in extra_weight_attrs.items() if k not in ('weight_loader','is_transposed','quant_method')}
        for p in layer.parameters(recurse=False): set_weight_attrs(p,extras)

    def get_fused_moe_quant_config(self,layer):
        if self.moe_quant_config is not None: return self.moe_quant_config
        if self._packed is None or self._packed.scales is None: return None
        return FusedMoEQuantConfig.make(torch.int8,per_act_token_quant=True,per_out_ch_quant=False,
            block_shape=None,weight_dtype='int4',w1_scale=self._packed.scales[0],w2_scale=self._packed.scales[1],
            gemm1_clamp_limit=self.moe.swiglu_limit)

    def process_weights_after_loading(self,layer):
        from .ct_w4a8 import validate_topology
        from hcu_vllm_das.moe.ct_w4a8_runtime import make_prepare, make_experts
        import vllm.model_executor.layers.fused_moe.modular_kernel as mk
        if self._packed is None or self._failed: raise RuntimeError('CT W4A8 is uninitialized or failed; restart worker')
        validate_topology(self._config,self.moe)
        if self._backend!=self.moe.moe_parallel_config.all2all_backend or self._capacity!=self.moe.max_num_tokens:
            raise RuntimeError('CT W4A8 backend/capacity changed after construction')
        if self._packed.state=='ready':
            self._packed.pack(self._ops.pack,self._ops.view_n32 if self._backend=='deepep_low_latency' else None)
            if self.moe_kernel is None: raise RuntimeError('CT W4A8 has packed weights but no ready kernel')
            return
        # Build/validate communication before destructive conversion. A plain
        # INT8 descriptor is sufficient to configure its wire format/capacity.
        descriptor=FusedMoEQuantConfig.make(torch.int8,per_act_token_quant=True,weight_dtype='int4')
        prepare=make_prepare(self.moe,descriptor,layer._expert_routing_tables(),config=self._config)
        try:
            self._packed.pack(self._ops.pack,self._ops.view_n32 if self._backend=='deepep_low_latency' else None)
            self.moe_quant_config=self.get_fused_moe_quant_config(layer)
            experts=make_experts(self.moe,self.moe_quant_config,self._packed.weights,self._ops,prepare)
            self.moe_kernel=mk.FusedMoEKernel(prepare,experts)
        except BaseException:
            self._failed=True
            raise

    def apply(self,layer,x,topk_weights,topk_ids,shared_experts,shared_experts_input):
        if self._failed or self.moe_kernel is None or self._packed.state!='ready':
            raise RuntimeError('CT W4A8 weights/kernel are not ready')
        if x.shape[0]>self._capacity: raise ValueError('CT W4A8 input exceeds checked token capacity')
        return self.moe_kernel.apply(hidden_states=x,w1=layer.w13_weight_packed,w2=layer.w2_weight_packed,
            topk_weights=topk_weights,topk_ids=topk_ids,activation=layer.activation,
            global_num_experts=layer.global_num_experts,expert_map=layer.expert_map,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            shared_experts=shared_experts,shared_experts_input=shared_experts_input)
