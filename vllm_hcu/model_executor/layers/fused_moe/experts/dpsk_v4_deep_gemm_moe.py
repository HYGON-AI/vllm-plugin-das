# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepEP DeepGEMM MoE expert implementations.

These backends are used for DeepEP high-throughput and low-latency DeepGEMM
paths. They expect channel-wise FP8 expert weights and per-token activation
scales.
"""

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from deepgemm import (
    m_grouped_fp8_gemm_nt_contiguous,
)
from deepgemm.m_group_gemm import (
    m_grouped_fp8_gemm_nt_masked_ll,
    pack_int8_weight_enk_to_w6_low_latency,
)

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.deep_gemm_utils import (
    compute_aligned_M,
    deepgemm_moe_permute,
    deepgemm_unpermute_and_reduce,
)
from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceDelegate,
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.fused_moe.utils import (
    _resize_cache,
    count_expert_num_tokens,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8DynamicTokenSym,
    kFp8StaticChannelSym,
)
from vllm.model_executor.utils import replace_parameter
from lightop import fuse_silu_mul_fp8_quant, fuse_silu_mul_fp8_quant_ep

logger = init_logger(__name__)


class DeepEPDeepGemmContiguousExperts(TritonExperts):
    """DeepEP HT backend backed by contiguous DeepGEMM grouped GEMM.

    `_apply_deepgemm_ht()` follows the HCU v0.21 DeepGemmExperts contiguous
    grouped GEMM flow.
    """

    # Match the HCU v0.21 DeepEP groupgemm path. DeepEP HT
    # dispatch aligns FP8 groupgemm traffic to 256 tokens per expert.
    ALIGNMENT = 256

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(moe_config, quant_config)
        self._deepgemm_w13: torch.Tensor | None = None
        self._deepgemm_w2: torch.Tensor | None = None
        logger.info_once(
            "Using DeepEPDeepGemmContiguousExperts with DeepGEMM HT path.",
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Prepare channel-wise FP8 weights for the DeepGEMM HT path.

        Packed weights replace the original runtime weight parameters so the
        original large weight storage can be released after loading.
        """
        if getattr(layer, "_dsv4_channel_fp8_deepgemm_repacked", False):
            self._deepgemm_w13 = layer.w13_weight
            self._deepgemm_w2 = layer.w2_weight
            return

        w13 = layer.w13_weight
        w2 = layer.w2_weight
        if not self._can_pack_channel_fp8(layer):
            raise RuntimeError(
                "DeepEP DeepGEMM HT requires channel-wise FP8 MoE weights, "
                f"got w13={tuple(w13.shape)} w2={tuple(w2.shape)} "
                f"w13_scale={tuple(layer.w13_weight_scale.shape)} "
                f"w2_scale={tuple(layer.w2_weight_scale.shape)}"
            )

        with torch.no_grad():
            w13_packed = pack_int8_weight_enk_to_w6_low_latency(w13).detach()
            w2_packed = pack_int8_weight_enk_to_w6_low_latency(w2).detach()

        replace_parameter(layer, "w13_weight", w13_packed)
        replace_parameter(layer, "w2_weight", w2_packed)
        self._deepgemm_w13 = layer.w13_weight
        self._deepgemm_w2 = layer.w2_weight
        layer._dsv4_channel_fp8_deepgemm_repacked = True
        del w13, w2

    def moe_problem_size(
        self,
        a1: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple[int, int, int, int, int]:
        local_num_experts, n, k = self._packed_deepgemm_problem_shape(w1, w2)
        if a1.dim() == 2:
            assert topk_ids.size(0) == a1.size(0), f"{topk_ids.size(0)} != {a1.size(0)}"
            M = a1.size(0)
        else:
            assert a1.dim() == 3
            assert a1.size(0) == local_num_experts, (
                f"{a1.size(0)} == {local_num_experts}"
            )
            M = a1.size(1)

        assert topk_ids.dim() == 2
        topk = topk_ids.size(1)
        return local_num_experts, M, n, k, topk

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        return self._apply_deepgemm_ht(
            output=output,
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=activation,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            a1q_scale=a1q_scale,
            a2_scale=a2_scale,
            workspace13=workspace13,
            workspace2=workspace2,
            expert_tokens_meta=expert_tokens_meta,
            apply_router_weight_on_input=apply_router_weight_on_input,
        )

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        m_aligned = compute_aligned_M(
            M=M,
            num_topk=topk,
            local_num_experts=local_num_experts,
            alignment=self.ALIGNMENT,
            expert_tokens_meta=expert_tokens_meta,
        )
        activation_out_dim = self.adjust_N_for_activation(N, activation)
        workspace13 = (m_aligned, max(activation_out_dim, K))
        workspace2 = (m_aligned, max(N, K))
        output = (M, K)
        return workspace13, workspace2, output

    def _apply_deepgemm_ht(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        del apply_router_weight_on_input
        if activation != MoEActivation.SILU:
            raise NotImplementedError(
                "DeepEP DeepGEMM HT path currently mirrors SGLang's silu-only "
                f"implementation, got {activation.value}."
            )
        if a1q_scale is None:
            raise RuntimeError("DeepEP DeepGEMM HT requires per-token activation scales")
        if self.w1_scale is None or self.w2_scale is None:
            raise RuntimeError("DeepEP DeepGEMM HT requires FP8 weight scales")

        local_num_experts, _, _, _, _ = self.moe_problem_size(
            hidden_states, w1, w2, topk_ids
        )
        expert_tokens_meta = self._ensure_expert_tokens_meta(
            topk_ids=topk_ids,
            local_num_experts=local_num_experts,
            expert_map=expert_map,
            expert_tokens_meta=expert_tokens_meta,
        )
        if self._deepgemm_w13 is None or self._deepgemm_w2 is None:
            raise RuntimeError(
                "DeepEP DeepGEMM HT weights were not packed. "
                "process_weights_after_loading must run before apply()."
        )
        w13_deepgemm = self._deepgemm_w13
        w2_deepgemm = self._deepgemm_w2
        if global_num_experts == -1:
            global_num_experts = local_num_experts
        _, _, N, K, _ = self.moe_problem_size(hidden_states, w1, w2, topk_ids)

        a1q_scale = self._ensure_2d_scale(a1q_scale)
        m_aligned = compute_aligned_M(
            M=topk_ids.size(0),
            num_topk=topk_ids.size(1),
            local_num_experts=local_num_experts,
            alignment=self.ALIGNMENT,
            expert_tokens_meta=expert_tokens_meta,
        )

        a1q_perm = _resize_cache(
            workspace13.view(dtype=hidden_states.dtype), (m_aligned, K)
        )
        input_tensor, input_scale, m_indices, inv_perm = deepgemm_moe_permute(
            aq=hidden_states,
            aq_scale=a1q_scale,
            topk_ids=topk_ids,
            local_num_experts=local_num_experts,
            expert_map=expert_map,
            expert_tokens_meta=expert_tokens_meta,
            aq_out=a1q_perm,
        )

        gateup_output = _resize_cache(workspace2, (m_aligned, N))
        m_grouped_fp8_gemm_nt_contiguous(
            (input_tensor, input_scale),
            (w13_deepgemm, self.w1_scale),
            gateup_output,
            m_indices,
        )
        del input_tensor
        del a2_scale

        q_activation, q_activation_scale = fuse_silu_mul_fp8_quant(
            gateup_output, fp8type=0, expert_ids=m_indices
        )
        del gateup_output

        down_output = _resize_cache(workspace2, (m_aligned, K))
        m_grouped_fp8_gemm_nt_contiguous(
            (q_activation, q_activation_scale),
            (w2_deepgemm, self.w2_scale),
            down_output,
            m_indices,
        )

        deepgemm_unpermute_and_reduce(
            a=down_output,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            inv_perm=inv_perm,
            expert_map=expert_map,
            output=output,
        )

    @staticmethod
    def _ensure_2d_scale(scale: torch.Tensor) -> torch.Tensor:
        if scale.ndim == 1:
            return scale.unsqueeze(-1)
        return scale

    @staticmethod
    def _ensure_expert_tokens_meta(
        topk_ids: torch.Tensor,
        local_num_experts: int,
        expert_map: torch.Tensor | None,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
    ) -> mk.ExpertTokensMetadata:
        if (
            expert_tokens_meta is not None
            and expert_tokens_meta.expert_num_tokens_cpu is not None
        ):
            return expert_tokens_meta

        expert_num_tokens = (
            expert_tokens_meta.expert_num_tokens
            if expert_tokens_meta is not None
            else count_expert_num_tokens(topk_ids, local_num_experts, expert_map)
        )
        return mk.ExpertTokensMetadata(
            expert_num_tokens=expert_num_tokens,
            expert_num_tokens_cpu=expert_num_tokens.cpu(),
        )

    @staticmethod
    def _can_pack_channel_fp8(layer: torch.nn.Module) -> bool:
        w13 = layer.w13_weight
        w2 = layer.w2_weight
        w13_scale = layer.w13_weight_scale
        w2_scale = layer.w2_weight_scale
        if w13.dim() != 3 or w2.dim() != 3 or w13.size(0) != w2.size(0):
            return False
        if w13_scale.ndim < 2 or w2_scale.ndim < 2:
            return False
        if getattr(layer, "weight_block_size", None) is not None:
            return False

        two_intermediate, hidden_size = w13.size(1), w13.size(2)
        if w2.size(1) != hidden_size:
            return False
        intermediate = w2.size(2)
        return two_intermediate == 2 * intermediate

    @staticmethod
    def _packed_deepgemm_problem_shape(
        w13: torch.Tensor,
        w2: torch.Tensor,
    ) -> tuple[int, int, int]:
        if w13.dim() != 6 or w2.dim() != 6:
            raise RuntimeError(
                "DeepEP DeepGEMM weights must be packed before apply(), "
                f"got w13={tuple(w13.shape)} w2={tuple(w2.shape)}"
            )
        if w13.size(0) != w2.size(0):
            raise RuntimeError(
                "DeepEP DeepGEMM packed weights have mismatched experts, "
                f"w13={tuple(w13.shape)} w2={tuple(w2.shape)}"
            )

        # pack_int8_weight_enk_to_w6_low_latency maps [E, N, K] to
        # [E, K / 64, N / 16, 4, 16, 16].
        local_num_experts = w13.size(0)
        n = w13.size(2) * w13.size(4)
        k = w13.size(1) * w13.size(3) * w13.size(5)
        w2_n = w2.size(2) * w2.size(4)
        if w2_n != k:
            raise RuntimeError(
                "DeepEP DeepGEMM packed down weight output size does not match "
                f"hidden size, w13={tuple(w13.shape)} w2={tuple(w2.shape)}"
            )
        return local_num_experts, n, k


class DeepEPDeepGemmMaskedExperts(DeepEPDeepGemmContiguousExperts):
    """DeepEP LL backend backed by low-latency masked DeepGEMM."""

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        max_num_tokens: int,
        num_dispatchers: int,
    ):
        mk.FusedMoEExpertsModular.__init__(
            self,
            moe_config=moe_config,
            quant_config=quant_config,
            max_num_tokens=max_num_tokens,
            num_dispatchers=num_dispatchers,
        )
        self._deepgemm_w13: torch.Tensor | None = None
        self._deepgemm_w2: torch.Tensor | None = None
        logger.info_once(
            "Using DeepEPDeepGemmMaskedExperts with DeepGEMM LL path.",
        )

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.BatchedExperts

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return (weight_key, activation_key) == (
            kFp8StaticChannelSym,
            kFp8DynamicTokenSym,
        )

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation == MoEActivation.SILU

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        return moe_parallel_config.use_deepep_ll_kernels

    def supports_expert_map(self) -> bool:
        return False

    def supports_packed_ue8m0_act_scales(self) -> bool:
        return False

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceDelegate()

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        del M, topk, global_num_experts, expert_tokens_meta
        assert self.max_num_tokens is not None
        assert self.num_dispatchers is not None
        max_tokens = self.max_num_tokens * self.num_dispatchers
        activation_out_dim = self.adjust_N_for_activation(N, activation)
        workspace13 = (local_num_experts, max_tokens, max(K, activation_out_dim))
        workspace2 = (local_num_experts, max_tokens, max(N, K))
        output = (local_num_experts, max_tokens, K)
        return workspace13, workspace2, output

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        del topk_weights, global_num_experts, expert_map
        del a2_scale, apply_router_weight_on_input
        del workspace13
        if activation != MoEActivation.SILU:
            raise NotImplementedError(
                "DeepEP DeepGEMM LL path currently supports silu only, "
                f"got {activation.value}."
            )
        if expert_tokens_meta is None:
            raise RuntimeError("DeepEP DeepGEMM LL requires expert token metadata")
        if a1q_scale is None:
            raise RuntimeError("DeepEP DeepGEMM LL requires activation scales")
        if self.w1_scale is None or self.w2_scale is None:
            raise RuntimeError("DeepEP DeepGEMM LL requires FP8 weight scales")
        if self._deepgemm_w13 is None or self._deepgemm_w2 is None:
            raise RuntimeError(
                "DeepEP DeepGEMM LL weights were not packed. "
                "process_weights_after_loading must run before apply()."
            )
        if hidden_states.ndim != 3:
            raise RuntimeError(
                "DeepEP DeepGEMM LL expects batched expert activations "
                f"[E, M, K], got {tuple(hidden_states.shape)}"
            )

        local_num_experts, max_tokens, _ = hidden_states.shape
        _, _, N, K, _ = self.moe_problem_size(hidden_states, w1, w2, topk_ids)
        expert_num_tokens = expert_tokens_meta.expert_num_tokens.to(
            dtype=torch.int32
        ).contiguous()
        a1q_scale = self._ensure_ll_scale(a1q_scale, local_num_experts, max_tokens)

        gateup_output = _resize_cache(workspace2, (local_num_experts, max_tokens, N))
        m_grouped_fp8_gemm_nt_masked_ll(
            (hidden_states, a1q_scale),
            (self._deepgemm_w13, self._ensure_ll_weight_scale(self.w1_scale)),
            gateup_output,
            expert_num_tokens,
            max_tokens,
        )

        q_activation, q_activation_scale = fuse_silu_mul_fp8_quant_ep(
            gateup_output,
            fp8type=0,
            tokens_per_expert=expert_num_tokens,
        )
        activation_out_dim = self.adjust_N_for_activation(N, activation)
        q_activation = q_activation.view(local_num_experts, max_tokens,
                                         activation_out_dim)
        q_activation_scale = self._ensure_ll_scale(
            q_activation_scale, local_num_experts, max_tokens
        )

        out_view = _resize_cache(output, (local_num_experts, max_tokens, K))
        m_grouped_fp8_gemm_nt_masked_ll(
            (q_activation, q_activation_scale),
            (self._deepgemm_w2, self._ensure_ll_weight_scale(self.w2_scale)),
            out_view,
            expert_num_tokens,
            max_tokens,
        )

    @staticmethod
    def _ensure_ll_scale(
        scale: torch.Tensor,
        local_num_experts: int,
        max_tokens: int,
    ) -> torch.Tensor:
        if scale.ndim == 3 and scale.size(-1) == 1:
            scale = scale.squeeze(-1)
        elif scale.ndim == 1:
            scale = scale.view(local_num_experts, max_tokens)
        elif scale.ndim == 2 and scale.shape != (local_num_experts, max_tokens):
            scale = scale.view(local_num_experts, max_tokens)
        if scale.dtype != torch.float32:
            scale = scale.to(torch.float32)
        return scale.contiguous()

    @staticmethod
    def _ensure_ll_weight_scale(scale: torch.Tensor) -> torch.Tensor:
        if scale.ndim == 3 and scale.size(-1) == 1:
            scale = scale.squeeze(-1)
        if scale.dtype != torch.float32:
            scale = scale.to(torch.float32)
        return scale.contiguous()


class DeepEPAutoDeepGemmExperts(mk.FusedMoEExpertsModular):
    """Switch DeepGEMM expert layout with the per-forward DeepEP mode."""

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        max_num_tokens: int,
        num_dispatchers: int,
    ):
        # The expert base validates one fixed activation format, while this
        # adapter intentionally owns both.  Initialize its public state using
        # the same v0.25 fields and let each concrete child validate itself.
        self.moe_config = moe_config
        self.quant_config = quant_config
        self.max_num_tokens = max_num_tokens
        self.num_dispatchers = num_dispatchers
        self._use_low_latency_snapshot = False
        self.ht_experts = DeepEPDeepGemmContiguousExperts(
            moe_config=moe_config,
            quant_config=quant_config,
        )
        self.ll_experts = DeepEPDeepGemmMaskedExperts(
            moe_config=moe_config,
            quant_config=quant_config,
            max_num_tokens=max_num_tokens,
            num_dispatchers=num_dispatchers,
        )

    def set_deepep_auto_use_low_latency(self, use_low_latency: bool) -> None:
        self._use_low_latency_snapshot = bool(use_low_latency)

    def _current(self) -> mk.FusedMoEExperts:
        return self.ll_experts if self._use_low_latency_snapshot else self.ht_experts

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self.ht_experts.process_weights_after_loading(layer)
        self.ll_experts._deepgemm_w13 = self.ht_experts._deepgemm_w13
        self.ll_experts._deepgemm_w2 = self.ht_experts._deepgemm_w2

    @staticmethod
    def _supports_current_device() -> bool:
        return DeepEPDeepGemmContiguousExperts._supports_current_device()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return DeepEPDeepGemmContiguousExperts._supports_quant_scheme(
            weight_key, activation_key
        )

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation == MoEActivation.SILU

    @staticmethod
    def _supports_parallel_config(
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> bool:
        return bool(
            getattr(moe_parallel_config, "use_deepep_auto_kernels", False)
        )

    @property
    def expects_unquantized_inputs(self) -> bool:
        return self._current().expects_unquantized_inputs

    def supports_expert_map(self) -> bool:
        return self._current().supports_expert_map()

    def supports_packed_ue8m0_act_scales(self) -> bool:
        return self._current().supports_packed_ue8m0_act_scales()

    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return self._current().activation_format()

    def moe_problem_size(
        self,
        a1: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple[int, int, int, int, int]:
        return self._current().moe_problem_size(a1, w1, w2, topk_ids)

    def workspace_dtype(self, act_dtype: torch.dtype) -> torch.dtype:
        return self._current().workspace_dtype(act_dtype)

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        return self._current().workspace_shapes(
            M,
            N,
            K,
            topk,
            global_num_experts,
            local_num_experts,
            expert_tokens_meta,
            activation,
        )

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return self._current().finalize_weight_and_reduce_impl()

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ) -> None:
        return self._current().apply(
            output=output,
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=activation,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            a1q_scale=a1q_scale,
            a2_scale=a2_scale,
            workspace13=workspace13,
            workspace2=workspace2,
            expert_tokens_meta=expert_tokens_meta,
            apply_router_weight_on_input=apply_router_weight_on_input,
        )


def make_deepep_auto_deepgemm_fp8_moe_kernel(
    moe_quant_config: FusedMoEQuantConfig,
    moe_config: FusedMoEConfig,
    routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> mk.FusedMoEKernel:
    from vllm.model_executor.layers.fused_moe.all2all_utils import (
        maybe_make_prepare_finalize,
    )

    prepare_finalize = maybe_make_prepare_finalize(
        moe=moe_config,
        quant_config=moe_quant_config,
        routing_tables=routing_tables,
        allow_new_interface=True,
        use_monolithic=False,
    )
    if prepare_finalize is None:
        raise RuntimeError("DeepEP auto prepare/finalize was not constructed")
    max_num_tokens = prepare_finalize.ll_prepare_finalize.max_num_tokens_per_rank()
    if max_num_tokens is None:
        raise RuntimeError("DeepEP auto LL token capacity is unavailable")
    experts = DeepEPAutoDeepGemmExperts(
        moe_config=moe_config,
        quant_config=moe_quant_config,
        max_num_tokens=max_num_tokens,
        num_dispatchers=prepare_finalize.num_dispatchers(),
    )
    logger.info_once("Using DeepEP auto MoE kernel with HT/LL experts.")
    return mk.FusedMoEKernel(prepare_finalize, experts)
