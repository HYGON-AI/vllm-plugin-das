# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""BF16 DeepGEMM experts for DeepEP's per-forward HT/LL dispatch.

HT consumes canonical [E, N, K] weights. LL's HIPC kernels need a separate
packed copy; install it during weight loading so memory profiling includes it.
"""

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceDelegate,
)
from vllm.model_executor.layers.fused_moe.utils import _resize_cache
from vllm.platforms import current_platform
from vllm_hcu.model_executor.layers.fused_moe.deep_gemm_utils import (
    compute_aligned_M,
    deepgemm_moe_permute,
    deepgemm_unpermute_and_reduce,
)
from vllm_hcu.model_executor.layers.fused_moe.experts.dpsk_v4_deep_gemm_moe import (
    DeepEPAutoDeepGemmExperts,
    _deepgemm_op,
    _make_deepep_auto_deepgemm_moe_kernel,
)


class DeepEPBF16ContiguousExperts(TritonExperts):
    ALIGNMENT = 256

    def workspace_shapes(
        self,
        M,
        N,
        K,
        topk,
        global_num_experts,
        local_num_experts,
        expert_tokens_meta,
        activation,
    ):
        rows = compute_aligned_M(
            M, topk, local_num_experts, self.ALIGNMENT, expert_tokens_meta
        )
        return (rows, max(N // 2, K)), (rows, max(N, K)), (M, K)

    @staticmethod
    def _supports_current_device():
        return current_platform.is_rocm()

    @staticmethod
    def _supports_no_act_and_mul():
        return False

    @staticmethod
    def _supports_quant_scheme(weight_key, activation_key):
        return weight_key is None and activation_key is None

    @staticmethod
    def _supports_activation(activation):
        return activation == MoEActivation.SILU

    @staticmethod
    def _supports_parallel_config(moe_parallel_config):
        return bool(getattr(moe_parallel_config, "use_deepep_auto_kernels", False))

    @staticmethod
    def _supports_batch_invariance():
        return False

    @staticmethod
    def supports_lora():
        return False

    @property
    def expects_unquantized_inputs(self):
        # quant_dtype=None already keeps BF16 dispatch unquantized. True
        # requests deferred input quantization, unsupported by DeepEP LL.
        return False

    def apply(
        self,
        output,
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        activation,
        global_num_experts,
        expert_map,
        a1q_scale,
        a2_scale,
        workspace13,
        workspace2,
        expert_tokens_meta,
        apply_router_weight_on_input,
    ):
        if apply_router_weight_on_input:
            raise NotImplementedError("BF16 DeepEP expects output router weighting")
        if hidden_states.size(0) == 0:
            output.zero_()
            return
        e, _, n, k, topk = self.moe_problem_size(hidden_states, w1, w2, topk_ids)
        rows = compute_aligned_M(
            topk_ids.size(0), topk, e, self.ALIGNMENT, expert_tokens_meta
        )
        permuted = _resize_cache(workspace13, (rows, k))
        # The shared scatter interface also permutes scales. BF16 GEMM ignores
        # these dummy scales; activations and weights remain BF16 throughout.
        dummy_scale = torch.ones(
            (hidden_states.size(0), 1),
            device=hidden_states.device,
            dtype=torch.float32,
        )
        x, _, ids, inv_perm, _ = deepgemm_moe_permute(
            aq=hidden_states,
            aq_scale=dummy_scale,
            topk_ids=topk_ids,
            local_num_experts=e,
            expert_map=expert_map,
            expert_tokens_meta=expert_tokens_meta,
            aq_out=permuted,
        )
        # Unlike the quantized scheduler, the BF16 contiguous kernel does
        # not accept negative expert indices for completely padded tiles.
        # Route those tiles to a valid weight; inv_perm excludes their output.
        ids.clamp_min_(0)
        gateup = _resize_cache(workspace2, (rows, n))
        gemm = _deepgemm_op("m_grouped_bf16_gemm_nt_contiguous")
        gemm(x, w1, gateup, ids)
        activated = _resize_cache(workspace13, (rows, n // 2))
        self.activation(activation, activated, gateup)
        down = _resize_cache(workspace2, (rows, k))
        gemm(activated, w2, down, ids)
        deepgemm_unpermute_and_reduce(
            a=down,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            inv_perm=inv_perm,
            expert_map=expert_map,
            output=output,
        )


class DeepEPBF16MaskedExperts(DeepEPBF16ContiguousExperts):
    def workspace_shapes(
        self,
        M,
        N,
        K,
        topk,
        global_num_experts,
        local_num_experts,
        expert_tokens_meta,
        activation,
    ):
        tokens = self.max_num_tokens * self.num_dispatchers
        output = (local_num_experts, tokens, K)
        # vLLM may alias output with workspace13. Keep both GEMM1's output
        # and the activated GEMM2 input in workspace2 to avoid in-place GEMM.
        return output, (local_num_experts, tokens, N + N // 2), output

    def __init__(self, moe_config, quant_config, max_num_tokens, num_dispatchers):
        mk.FusedMoEExpertsModular.__init__(
            self, moe_config, quant_config, max_num_tokens, num_dispatchers
        )
        self._packed_w13 = None
        self._packed_w2 = None

    @staticmethod
    def activation_format():
        return mk.FusedMoEActivationFormat.BatchedExperts

    def supports_expert_map(self):
        return False

    def finalize_weight_and_reduce_impl(self):
        return TopKWeightAndReduceDelegate()

    def process_weights_after_loading(self, layer):
        pack = _deepgemm_op("pack_bf16_masked_weight")
        for name, weight in (
            ("_packed_w13", layer.w13_weight),
            ("_packed_w2", layer.w2_weight),
        ):
            packed = pack(weight.detach())
            current = getattr(self, name)
            if current is None:
                setattr(self, name, packed)
            else:
                # Keep captured addresses stable when weights are updated.
                current.copy_(packed)

    def apply(
        self,
        output,
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        activation,
        global_num_experts,
        expert_map,
        a1q_scale,
        a2_scale,
        workspace13,
        workspace2,
        expert_tokens_meta,
        apply_router_weight_on_input,
    ):
        if apply_router_weight_on_input:
            raise NotImplementedError("BF16 DeepEP expects output router weighting")
        if expert_tokens_meta is None or expert_tokens_meta.expert_num_tokens is None:
            raise RuntimeError("BF16 DeepEP LL requires per-expert token counts")
        if self._packed_w13 is None or self._packed_w2 is None:
            raise RuntimeError("BF16 DeepEP LL weights have not been packed")
        e, m, n, k, topk = self.moe_problem_size(hidden_states, w1, w2, topk_ids)
        if global_num_experts <= 0:
            global_num_experts = e
        counts = expert_tokens_meta.expert_num_tokens.to(torch.int32).contiguous()
        expected_m = min(
            m,
            max(
                1,
                (
                    topk_ids.size(0) * self.num_dispatchers * topk
                    + global_num_experts
                    - 1
                )
                // global_num_experts,
            ),
        )
        gemm = _deepgemm_op("m_grouped_bf16_gemm_nt_masked")
        flat = workspace2.view(-1)
        gateup = flat[: e * m * n].view(e, m, n)
        # MODE 2000 uses the layout returned by pack_bf16_masked_weight.
        config = {"MODE": 2000}
        gemm(hidden_states, self._packed_w13, gateup, counts, expected_m, config=config)
        activated = flat[e * m * n : e * m * (n + n // 2)].view(e, m, n // 2)
        self.activation(activation, activated.view(-1, n // 2), gateup.view(-1, n))
        gemm(
            activated,
            self._packed_w2,
            output.view(e, m, k),
            counts,
            expected_m,
            config=config,
        )


class DeepEPAutoBF16Experts(DeepEPAutoDeepGemmExperts):
    HT_EXPERTS_CLS = DeepEPBF16ContiguousExperts
    LL_EXPERTS_CLS = DeepEPBF16MaskedExperts

    @staticmethod
    def is_supported_config(
        cls, moe_config, weight_key, activation_key, activation_format
    ):
        if moe_config.in_dtype != torch.bfloat16:
            return False, "BF16 DeepGEMM requires bfloat16 activations and weights"
        if moe_config.has_bias:
            return False, "BF16 DeepGEMM does not support expert bias"
        if (
            moe_config.hidden_dim % 256
            or moe_config.intermediate_size_per_partition % 256
        ):
            return (
                False,
                "BF16 DeepGEMM requires hidden/intermediate sizes divisible by 256",
            )
        if getattr(moe_config, "apply_router_weight_on_input", False):
            return False, "BF16 DeepGEMM requires output router weighting"
        return DeepEPBF16ContiguousExperts.is_supported_config(
            DeepEPBF16ContiguousExperts,
            moe_config,
            weight_key,
            activation_key,
            mk.FusedMoEActivationFormat.Standard,
        )

    def process_weights_after_loading(self, layer):
        if self._fixed_use_low_latency is not False:
            self.ll_experts.process_weights_after_loading(layer)


def make_deepep_auto_bf16_moe_kernel(quant_config, moe_config, routing_tables=None):
    return _make_deepep_auto_deepgemm_moe_kernel(
        quant_config, moe_config, routing_tables, experts_cls=DeepEPAutoBF16Experts
    )
