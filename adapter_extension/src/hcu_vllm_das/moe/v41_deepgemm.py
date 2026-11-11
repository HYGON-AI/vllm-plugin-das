# SPDX-License-Identifier: Apache-2.0
"""Channel-FP8 vendor grouped GEMMs with Core DeepEP HT dispatch/combine."""
from types import FunctionType

import torch
from vllm.model_executor.layers.fused_moe import modular_kernel as mk
from vllm.model_executor.layers.fused_moe import deep_gemm_utils as core_layout
from vllm.model_executor.layers.fused_moe.utils import _resize_cache
from .v41_megamoe import HcuV41MegaMoEMethod, MegaExperts

# The HCU grouped GEMM requires 256-row expert tiles. Keep Core's scatter,
# counting, expert mapping and inverse permutation; only bind the alignment.
_permute = FunctionType(core_layout.deepgemm_moe_permute.__code__,
    dict(core_layout.deepgemm_moe_permute.__globals__,
         get_mk_alignment_for_contiguous_layout=lambda: (256, 128)),
    core_layout.deepgemm_moe_permute.__name__,
    core_layout.deepgemm_moe_permute.__defaults__)


class ChannelDeepGemmExperts(MegaExperts):
    def __init__(self, moe, quant, packed_w1, packed_w2):
        mk.FusedMoEExpertsModular.__init__(self, moe, quant)
        self.packed_w1, self.packed_w2 = packed_w1, packed_w2

    def workspace_shapes(self, M, N, K, topk, global_num_experts,
                         local_num_experts, expert_tokens_meta, activation):
        rows, _ = core_layout.compute_aligned_M_and_alignment(
            M, topk, local_num_experts, 256, expert_tokens_meta)
        return (rows, max(N // 2, K)), (rows, max(N, K)), (M, K)

    def apply(self, output, hidden_states, w1, w2, topk_weights, topk_ids,
              activation, global_num_experts, expert_map, a1q_scale, a2_scale,
              workspace13, workspace2, expert_tokens_meta,
              apply_router_weight_on_input):
        from deepgemm import m_grouped_fp8_gemm_nt_contiguous
        from lightop.quant import per_token_quant_fp8
        if apply_router_weight_on_input:
            raise ValueError("V4.1 DeepGEMM applies router weights after GEMM2")
        assert hidden_states.dtype == torch.bfloat16 and a1q_scale is None
        hidden = self.moe_config.hidden_dim
        intermediate = self.moe_config.intermediate_size_per_partition
        rows, _ = core_layout.compute_aligned_M_and_alignment(
            hidden_states.shape[0], topk_ids.shape[1],
            self.moe_config.num_local_experts, 256, expert_tokens_meta)
        aq, scales = per_token_quant_fp8(hidden_states.contiguous())
        a_perm = _resize_cache(workspace13.view(torch.float8_e4m3fn), (rows, hidden))
        aq, scales, ids, inverse, _ = _permute(
            aq, scales, topk_ids, self.moe_config.num_local_experts,
            expert_map, expert_tokens_meta, aq_out=a_perm, block_size=hidden)
        first = _resize_cache(workspace2, (rows, 2 * intermediate))
        # Call deepgemm directly: deep_gemm can be an adapter Triton shim.
        m_grouped_fp8_gemm_nt_contiguous(
            (aq, scales), (self.packed_w1, self.w1_scale.reshape(w1.shape[0], -1)),
            first, ids)
        activated = torch.empty((rows, intermediate), dtype=first.dtype,
                                device=first.device)
        self.activation(activation, activated, first)
        aq2 = _resize_cache(workspace13.view(torch.float8_e4m3fn), (rows, intermediate))
        aq2, scales2 = per_token_quant_fp8(activated, out_q=aq2)
        second = _resize_cache(workspace2, (rows, hidden))
        m_grouped_fp8_gemm_nt_contiguous(
            (aq2, scales2), (self.packed_w2, self.w2_scale.reshape(w2.shape[0], -1)),
            second, ids)
        core_layout.deepgemm_unpermute_and_reduce(
            second, topk_ids, topk_weights, inverse, expert_map, output)


class HcuV41DeepGemmMethod(HcuV41MegaMoEMethod):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.experts_cls = ChannelDeepGemmExperts

    @torch.inference_mode()
    def _process_weights(self, layer):
        from deepgemm import marlin_fp8_contiguous_weight
        if self._processed:
            raise RuntimeError("DeepGEMM packed weights cannot be processed twice")
        w1, w2 = layer.w13_weight, layer.w2_weight
        packed1 = marlin_fp8_contiguous_weight(w1)
        packed2 = marlin_fp8_contiguous_weight(w2)
        for param, packed in [(w1, packed1), (w2, packed2)]:
            if param.dtype != packed.dtype or param.numel() != packed.numel():
                raise ValueError("DeepGEMM packing changed weight dtype or size")
            param.data = packed.reshape(param.shape)
        quant = self.get_fused_moe_quant_config(layer)
        self.moe_quant_config = quant
        from vllm.model_executor.layers.fused_moe.all2all_utils import maybe_make_prepare_finalize
        prepare = maybe_make_prepare_finalize(self.moe, quant,
            routing_tables=layer._expert_routing_tables(), allow_new_interface=True)
        from vllm.model_executor.layers.fused_moe.prepare_finalize.deepep_ht import DeepEPHTPrepareAndFinalize
        if not isinstance(prepare, DeepEPHTPrepareAndFinalize):
            raise ValueError("V4.1 DeepGEMM requires Core DeepEP HT prepare/finalize")
        self.moe_kernel = mk.FusedMoEKernel(
            prepare, ChannelDeepGemmExperts(self.moe, quant, packed1, packed2))
        self._processed = True
