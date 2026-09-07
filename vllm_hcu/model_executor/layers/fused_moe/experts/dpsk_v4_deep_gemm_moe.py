# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# Modified by Hygon Information Technology Co., Ltd., 2026.
"""DeepEP DeepGEMM MoE expert implementations.

These backends are used for DeepEP high-throughput and low-latency DeepGEMM
paths. They accept channel-wise FP8 or INT8 expert weights and per-token
activation scales.
"""

import functools

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk

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
from vllm_hcu.model_executor.layers.fused_moe.deep_gemm_utils import (
    topk_weights_for_unpermute,
)
from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceDelegate,
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.fused_moe.utils import (
    _resize_cache,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8DynamicTokenSym,
    kFp8StaticChannelSym,
    kInt8DynamicTokenSym,
    kInt8StaticChannelSym,
)
from vllm.model_executor.utils import replace_parameter
from vllm_hcu.model_executor.layers.quantization.slimquant_w4a8_deepgemm_runtime import (
    DeepEPDeepGemmW4A8ContiguousExperts,
    DeepEPDeepGemmW4A8MaskedExperts,
)

# Use vLLM's configured logger hierarchy so worker-side backend evidence is
# present in the normal engine log (``vllm_hcu.*`` has no configured handler).
logger = init_logger("vllm.hcu.deepseek_v4_deepep_experts")


@functools.lru_cache(maxsize=None)
def _deepgemm_op(name: str):
    """Resolve DeepGEMM only when an HCU worker first needs an operator."""

    import deepgemm

    return getattr(deepgemm, name)


def marlin_fp8_contiguous_weight(weight: torch.Tensor) -> torch.Tensor:
    return _deepgemm_op("marlin_fp8_contiguous_weight")(weight)


def marlin_fp8_masked_weight(weight: torch.Tensor) -> torch.Tensor:
    return _deepgemm_op("marlin_fp8_masked_weight")(weight)


def marlin_i8_contiguous_weight(weight: torch.Tensor) -> torch.Tensor:
    return _deepgemm_op("marlin_i8_contiguous_weight")(weight)


def marlin_i8_masked_weight(weight: torch.Tensor) -> torch.Tensor:
    return _deepgemm_op("marlin_i8_masked_weight")(weight)


def m_grouped_fp8_gemm_nt_contiguous(*args, **kwargs):
    return _deepgemm_op("m_grouped_fp8_gemm_nt_contiguous")(*args, **kwargs)


def m_grouped_fp8_gemm_nt_masked(*args, **kwargs):
    return _deepgemm_op("m_grouped_fp8_gemm_nt_masked")(*args, **kwargs)


def m_grouped_i8_gemm_nt_contiguous(*args, **kwargs):
    return _deepgemm_op("m_grouped_i8_gemm_nt_contiguous")(*args, **kwargs)


def m_grouped_i8_gemm_nt_masked(*args, **kwargs):
    return _deepgemm_op("m_grouped_i8_gemm_nt_masked")(*args, **kwargs)


@functools.lru_cache(maxsize=None)
def _lightop_activation(name: str):
    """Resolve categorized activation kernels without eager worker imports."""

    from lightop import activation

    return getattr(activation, name)


@functools.lru_cache(maxsize=None)
def _lightop_clamp(name: str):
    """Resolve the two uncategorized LightOp clamp kernels lazily."""

    if name not in {"fuse_silu_mul_clamp_quant", "fuse_silu_mul_clamp_quant_ep"}:
        raise AttributeError(name)

    import lightop

    return getattr(lightop, name)


def fuse_silu_mul_fp8_quant(*args, **kwargs):
    return _lightop_activation("fuse_silu_mul_fp8_quant")(*args, **kwargs)


def fuse_silu_mul_fp8_quant_ep(*args, **kwargs):
    return _lightop_activation("fuse_silu_mul_fp8_quant_ep")(*args, **kwargs)


def fuse_silu_mul_quant(*args, **kwargs):
    return _lightop_activation("fuse_silu_mul_quant")(*args, **kwargs)


def fuse_silu_mul_quant_ep(*args, **kwargs):
    return _lightop_activation("fuse_silu_mul_quant_ep")(*args, **kwargs)


def fuse_silu_mul_clamp_quant(*args, **kwargs):
    return _lightop_clamp("fuse_silu_mul_clamp_quant")(*args, **kwargs)


def fuse_silu_mul_clamp_quant_ep(*args, **kwargs):
    return _lightop_clamp("fuse_silu_mul_clamp_quant_ep")(*args, **kwargs)


class DeepEPDeepGemmContiguousExperts(TritonExperts):
    """DeepEP HT backend backed by contiguous DeepGEMM grouped GEMM.

    `_apply_deepgemm_ht()` follows the HCU DeepGemmExperts contiguous
    grouped GEMM flow.
    """

    # Match the HCU DeepEP groupgemm path. DeepEP HT dispatch aligns
    # quantized groupgemm traffic to 256 tokens per expert.
    ALIGNMENT = 256
    WEIGHT_LAYOUT = "contiguous"

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return (weight_key, activation_key) in (
            (kFp8StaticChannelSym, kFp8DynamicTokenSym),
            (kInt8StaticChannelSym, kInt8DynamicTokenSym),
        )

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
        """Prepare channel-wise FP8 or INT8 weights for the DeepGEMM HT path.

        Packed weights replace the original runtime weight parameters so the
        original large weight storage can be released after loading.
        """
        packed_layout = getattr(
            layer,
            "_dsv4_channel_deepgemm_layout",
            None,
        )
        if packed_layout == self.WEIGHT_LAYOUT:
            if layer.w13_weight.dim() == 6 and layer.w2_weight.dim() == 6:
                self._deepgemm_w13 = layer.w13_weight
                self._deepgemm_w2 = layer.w2_weight
                return
            if not self._can_pack_channel_weights(layer):
                raise RuntimeError(
                    "DeepEP DeepGEMM layer has invalid reloaded weights: "
                    f"w13={tuple(layer.w13_weight.shape)} "
                    f"w2={tuple(layer.w2_weight.shape)}"
                )
        elif packed_layout is not None:
            raise RuntimeError(
                "DeepEP DeepGEMM weight layout mismatch: "
                f"experts require {self.WEIGHT_LAYOUT}, got {packed_layout}."
            )

        w13 = layer.w13_weight
        w2 = layer.w2_weight
        if not self._can_pack_channel_weights(layer):
            raise RuntimeError(
                "DeepEP DeepGEMM HT requires channel-wise FP8 or INT8 MoE "
                "weights, "
                f"got w13={tuple(w13.shape)} w2={tuple(w2.shape)} "
                f"w13_scale={tuple(layer.w13_weight_scale.shape)} "
                f"w2_scale={tuple(layer.w2_weight_scale.shape)}"
            )

        with torch.no_grad():
            w13_packed, w2_packed = self._pack_channel_weights(w13, w2)

        replace_parameter(layer, "w13_weight", w13_packed)
        replace_parameter(layer, "w2_weight", w2_packed)
        self._deepgemm_w13 = layer.w13_weight
        self._deepgemm_w2 = layer.w2_weight
        layer._dsv4_channel_deepgemm_layout = self.WEIGHT_LAYOUT
        layer._dsv4_channel_deepgemm_repacked = True
        del w13, w2

    def _pack_channel_weights(
        self,
        w13: torch.Tensor,
        w2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if getattr(getattr(self, "quant_config", None), "use_int8_w8a8", False):
            return (
                marlin_i8_contiguous_weight(w13).detach(),
                marlin_i8_contiguous_weight(w2).detach(),
            )
        return (
            marlin_fp8_contiguous_weight(w13).detach(),
            marlin_fp8_contiguous_weight(w2).detach(),
        )

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
        if activation != MoEActivation.SILU:
            raise NotImplementedError(
                "DeepEP DeepGEMM HT path currently mirrors SGLang's silu-only "
                f"implementation, got {activation.value}."
            )
        if a1q_scale is None:
            raise RuntimeError("DeepEP DeepGEMM HT requires per-token activation scales")
        if self.w1_scale is None or self.w2_scale is None:
            raise RuntimeError("DeepEP DeepGEMM HT requires weight scales")
        use_int8 = bool(self.quant_config.use_int8_w8a8)

        local_num_experts, _, _, _, _ = self.moe_problem_size(
            hidden_states, w1, w2, topk_ids
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
        input_tensor, input_scale, m_indices, inv_perm, _align_used = (
            deepgemm_moe_permute(
                aq=hidden_states,
                aq_scale=a1q_scale,
                topk_ids=topk_ids,
                local_num_experts=local_num_experts,
                expert_map=expert_map,
                expert_tokens_meta=expert_tokens_meta,
                aq_out=a1q_perm,
            )
        )

        gateup_output = _resize_cache(workspace2, (m_aligned, N))
        grouped_gemm = (
            m_grouped_i8_gemm_nt_contiguous
            if use_int8
            else m_grouped_fp8_gemm_nt_contiguous
        )
        grouped_gemm(
            (input_tensor, input_scale),
            (w13_deepgemm, self.w1_scale),
            gateup_output,
            m_indices,
        )
        del input_tensor
        del a2_scale

        if use_int8:
            clamp_limit = self.quant_config.gemm1_clamp_limit
            if clamp_limit is not None and clamp_limit > 0:
                q_activation, q_activation_scale = fuse_silu_mul_clamp_quant(
                    gateup_output,
                    limit=clamp_limit,
                )
            else:
                q_activation, q_activation_scale = fuse_silu_mul_quant(
                    gateup_output,
                    expert_ids=m_indices,
                )
        else:
            q_activation, q_activation_scale = fuse_silu_mul_fp8_quant(
                gateup_output,
                fp8type=0,
                expert_ids=m_indices,
                limit=self.quant_config.gemm1_clamp_limit,
            )
        del gateup_output

        down_output = _resize_cache(workspace2, (m_aligned, K))
        grouped_gemm(
            (q_activation, q_activation_scale),
            (w2_deepgemm, self.w2_scale),
            down_output,
            m_indices,
        )

        topk_weights = topk_weights_for_unpermute(
            topk_weights,
            apply_router_weight_on_input,
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
    def _can_pack_channel_weights(layer: torch.nn.Module) -> bool:
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

        # The gfx938 Marlin FP8/INT8 packers map [E, N, K] to
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

    WEIGHT_LAYOUT = "masked"

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
        return (weight_key, activation_key) in (
            (kFp8StaticChannelSym, kFp8DynamicTokenSym),
            (kInt8StaticChannelSym, kInt8DynamicTokenSym),
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

    def _pack_channel_weights(
        self,
        w13: torch.Tensor,
        w2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if getattr(getattr(self, "quant_config", None), "use_int8_w8a8", False):
            return (
                marlin_i8_masked_weight(w13).detach(),
                marlin_i8_masked_weight(w2).detach(),
            )
        return (
            marlin_fp8_masked_weight(w13).detach(),
            marlin_fp8_masked_weight(w2).detach(),
        )

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
            raise RuntimeError("DeepEP DeepGEMM LL requires weight scales")
        use_int8 = bool(self.quant_config.use_int8_w8a8)
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
        grouped_gemm = (
            m_grouped_i8_gemm_nt_masked
            if use_int8
            else m_grouped_fp8_gemm_nt_masked
        )
        grouped_gemm(
            (hidden_states, a1q_scale),
            (self._deepgemm_w13, self._ensure_ll_weight_scale(self.w1_scale)),
            gateup_output,
            expert_num_tokens,
            max_tokens,
        )

        if use_int8:
            clamp_limit = self.quant_config.gemm1_clamp_limit
            if clamp_limit is not None and clamp_limit > 0:
                q_activation, q_activation_scale = fuse_silu_mul_clamp_quant_ep(
                    gateup_output,
                    limit=clamp_limit,
                    mask_m=expert_num_tokens,
                    expect_m=max_tokens,
                )
            else:
                q_activation, q_activation_scale = fuse_silu_mul_quant_ep(
                    gateup_output,
                    tokens_per_expert=expert_num_tokens,
                )
        else:
            q_activation, q_activation_scale = fuse_silu_mul_fp8_quant_ep(
                gateup_output,
                fp8type=0,
                tokens_per_expert=expert_num_tokens,
                limit=self.quant_config.gemm1_clamp_limit,
            )
        activation_out_dim = self.adjust_N_for_activation(N, activation)
        q_activation = q_activation.view(local_num_experts, max_tokens,
                                         activation_out_dim)
        q_activation_scale = self._ensure_ll_scale(
            q_activation_scale, local_num_experts, max_tokens
        )

        out_view = _resize_cache(output, (local_num_experts, max_tokens, K))
        grouped_gemm(
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
    """Select a per-forward or role-fixed DeepGEMM expert layout."""

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        max_num_tokens: int,
        num_dispatchers: int,
        fixed_use_low_latency: bool | None = None,
    ):
        # The expert base validates one fixed activation format, while this
        # adapter can own both. Initialize its public state using the same
        # v0.25.1 fields and let each constructed child validate itself.
        self.moe_config = moe_config
        self.quant_config = quant_config
        self.max_num_tokens = max_num_tokens
        self.num_dispatchers = num_dispatchers
        self._fixed_use_low_latency = fixed_use_low_latency
        self._use_low_latency_snapshot = False
        if fixed_use_low_latency is not True:
            self.ht_experts = DeepEPDeepGemmContiguousExperts(
                moe_config=moe_config,
                quant_config=quant_config,
            )
        if fixed_use_low_latency is not False:
            self.ll_experts = DeepEPDeepGemmMaskedExperts(
                moe_config=moe_config,
                quant_config=quant_config,
                max_num_tokens=max_num_tokens,
                num_dispatchers=num_dispatchers,
            )
        if fixed_use_low_latency is True:
            self.ht_experts = self.ll_experts
        elif fixed_use_low_latency is False:
            self.ll_experts = self.ht_experts

    def set_deepep_auto_use_low_latency(self, use_low_latency: bool) -> None:
        if self._fixed_use_low_latency is None:
            self._use_low_latency_snapshot = bool(use_low_latency)
        else:
            self._use_low_latency_snapshot = self._fixed_use_low_latency

    def _current(self) -> mk.FusedMoEExperts:
        if self._fixed_use_low_latency is not None:
            return (
                self.ll_experts
                if self._fixed_use_low_latency
                else self.ht_experts
            )
        return self.ll_experts if self._use_low_latency_snapshot else self.ht_experts

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if getattr(self, "_fixed_use_low_latency", None) is not None:
            self._current().process_weights_after_loading(layer)
            return
        packed_layout = getattr(
            layer,
            "_dsv4_channel_deepgemm_layout",
            None,
        )
        if packed_layout == "contiguous+masked":
            ll_w13 = getattr(
                layer,
                "_dsv4_channel_deepgemm_masked_w13",
                None,
            )
            ll_w2 = getattr(
                layer,
                "_dsv4_channel_deepgemm_masked_w2",
                None,
            )
            if layer.w13_weight.dim() == 6 and layer.w2_weight.dim() == 6:
                if ll_w13 is None or ll_w2 is None:
                    raise RuntimeError(
                        "DeepEP auto DeepGEMM layer lost its masked weight layout"
                    )
                self.ht_experts._deepgemm_w13 = layer.w13_weight
                self.ht_experts._deepgemm_w2 = layer.w2_weight
                self.ll_experts._deepgemm_w13 = ll_w13
                self.ll_experts._deepgemm_w2 = ll_w2
                return
            if not self.ht_experts._can_pack_channel_weights(layer):
                raise RuntimeError(
                    "DeepEP auto DeepGEMM layer has invalid reloaded weights: "
                    f"w13={tuple(layer.w13_weight.shape)} "
                    f"w2={tuple(layer.w2_weight.shape)}"
                )
        if packed_layout is not None:
            if packed_layout != "contiguous+masked":
                raise RuntimeError(
                    "DeepEP auto DeepGEMM requires unpacked channel-wise "
                    "FP8 or INT8 weights, "
                    f"got {packed_layout} weights."
                )
        if not self.ht_experts._can_pack_channel_weights(layer):
            raise RuntimeError(
                "DeepEP auto DeepGEMM requires channel-wise FP8 or INT8 MoE "
                "weights."
            )

        w13 = layer.w13_weight
        w2 = layer.w2_weight
        with torch.no_grad():
            # Both HCU Marlin packers rearrange their input storage in place
            # and return a view of that same storage.  Preserve independent
            # layouts: contiguous owns clones while masked reuses the original
            # parameter storage that is about to be replaced on the layer.
            ht_w13, ht_w2 = self.ht_experts._pack_channel_weights(
                w13.clone(),
                w2.clone(),
            )
            ll_w13, ll_w2 = self.ll_experts._pack_channel_weights(w13, w2)

        replace_parameter(layer, "w13_weight", ht_w13)
        replace_parameter(layer, "w2_weight", ht_w2)
        self.ht_experts._deepgemm_w13 = layer.w13_weight
        self.ht_experts._deepgemm_w2 = layer.w2_weight
        self.ll_experts._deepgemm_w13 = ll_w13
        self.ll_experts._deepgemm_w2 = ll_w2
        layer._dsv4_channel_deepgemm_masked_w13 = ll_w13
        layer._dsv4_channel_deepgemm_masked_w2 = ll_w2
        layer._dsv4_channel_deepgemm_layout = "contiguous+masked"
        layer._dsv4_channel_deepgemm_repacked = True
        del w13, w2

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


class DeepEPAutoW4A8Experts(DeepEPAutoDeepGemmExperts):
    """Own one in-place W4A8 HIPC layout for strict ``deepep_auto``."""

    _LAYOUT_MARKER = "_slimquant_w4a8_deepep_auto_layout"
    _PACKED_W13 = "_slimquant_w4a8_deepep_auto_packed_w13"
    _PACKED_W2 = "_slimquant_w4a8_deepep_auto_packed_w2"
    _PACKING_LAYOUT = "packing"
    _REPACKING_LAYOUT = "repacking"

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        max_num_tokens: int,
        num_dispatchers: int,
        fixed_use_low_latency: bool | None = None,
    ):
        self.moe_config = moe_config
        self.quant_config = quant_config
        self.max_num_tokens = max_num_tokens
        self.num_dispatchers = num_dispatchers
        self._fixed_use_low_latency = fixed_use_low_latency
        self._use_low_latency_snapshot = False
        if fixed_use_low_latency is not True:
            self.ht_experts = DeepEPDeepGemmW4A8ContiguousExperts(
                moe_config=moe_config,
                quant_config=quant_config,
            )
        if fixed_use_low_latency is not False:
            self.ll_experts = DeepEPDeepGemmW4A8MaskedExperts(
                moe_config=moe_config,
                quant_config=quant_config,
                max_num_tokens=max_num_tokens,
                num_dispatchers=num_dispatchers,
            )
        if fixed_use_low_latency is True:
            self.ht_experts = self.ll_experts
        elif fixed_use_low_latency is False:
            self.ll_experts = self.ht_experts

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        from vllm_hcu.model_executor.layers.quantization import (
            slimquant_w4a8_deepgemm_runtime as runtime,
        )

        expected_layout = {
            None: "shared_hipc_auto",
            False: "shared_hipc_contiguous",
            True: "shared_hipc_n32",
        }[self._fixed_use_low_latency]
        marker = getattr(layer, self._LAYOUT_MARKER, None)
        owners = (
            getattr(layer, self._PACKED_W13, None),
            getattr(layer, self._PACKED_W2, None),
        )
        if marker is None and any(owner is not None for owner in owners):
            self._invalid_marker()
        if marker is not None and marker != expected_layout:
            self._invalid_marker()
        runtime._validate_w4a8_channel_weights(layer)
        weights = (layer.w13_weight, layer.w2_weight)
        if marker is not None:
            if any(
                not isinstance(owner, torch.Tensor)
                or owner.dtype != torch.int8
                or owner.ndim != 3
                or tuple(owner.shape) != tuple(weight.shape)
                for owner, weight in zip(owners, weights)
            ):
                self._invalid_marker()
            if all(self._same_storage(a, b) for a, b in zip(owners, weights)):
                self._bind_packed_owners(runtime, *owners)
                return

            # v0.25.1 layerwise reload processes temporary raw Parameters and
            # then restores the original kernel storage. Reuse that storage so
            # the temporary allocation cannot become a second resident owner.
            setattr(layer, self._LAYOUT_MARKER, self._REPACKING_LAYOUT)
            reloaded = self._pack_in_place(runtime, weights)
            with torch.no_grad():
                for owner, weight in zip(owners, reloaded):
                    owner.copy_(weight)
            replace_parameter(layer, "w13_weight", owners[0])
            replace_parameter(layer, "w2_weight", owners[1])
            self._bind_packed_owners(runtime, *owners)
            setattr(layer, self._LAYOUT_MARKER, expected_layout)
            return

        # The packer mutates each registered Parameter in sequence. Mark the
        # transition first so a partial failure can never be retried as raw.
        setattr(layer, self._LAYOUT_MARKER, self._PACKING_LAYOUT)
        self._pack_in_place(runtime, weights)
        owners = tuple(weight.detach() for weight in weights)
        setattr(layer, self._PACKED_W13, owners[0])
        setattr(layer, self._PACKED_W2, owners[1])
        self._bind_packed_owners(runtime, *owners)
        setattr(layer, self._LAYOUT_MARKER, expected_layout)

    @staticmethod
    def _same_storage(left: torch.Tensor, right: torch.Tensor) -> bool:
        return (
            left.untyped_storage().data_ptr()
            == right.untyped_storage().data_ptr()
        )

    def _pack_in_place(self, runtime, weights):
        with torch.no_grad():
            packed = tuple(
                runtime.pack_w4a8_moe_hipc_weight(weight).detach()
                for weight in weights
            )
        if any(
            weight.ndim != 3 or not self._same_storage(weight, source)
            for weight, source in zip(packed, weights)
        ):
            raise RuntimeError(
                "SlimQuant W4A8 deepep_auto packer must reuse rank-3 "
                "weight storage"
            )
        return packed

    @staticmethod
    def _invalid_marker() -> None:
        raise RuntimeError(
            "invalid SlimQuant W4A8 deepep_auto marker or owner shape state"
        )

    def _bind_packed_owners(self, runtime, w13, w2) -> None:
        if self._fixed_use_low_latency is not True:
            self.ht_experts._deepgemm_w13 = w13
            self.ht_experts._deepgemm_w2 = w2
        if self._fixed_use_low_latency is not False:
            self.ll_experts._deepgemm_w13 = (
                runtime.view_w4a8_moe_hipc_weight_n32_layout(w13).detach()
            )
            self.ll_experts._deepgemm_w2 = (
                runtime.view_w4a8_moe_hipc_weight_n32_layout(w2).detach()
            )

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return DeepEPDeepGemmW4A8ContiguousExperts._supports_quant_scheme(
            weight_key, activation_key
        )


def _make_deepep_auto_deepgemm_moe_kernel(
    moe_quant_config: FusedMoEQuantConfig,
    moe_config: FusedMoEConfig,
    routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    experts_cls: type[DeepEPAutoDeepGemmExperts] | None = None,
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
    from vllm.config import get_current_vllm_config_or_none
    from vllm_hcu.model_executor.layers.fused_moe.prepare_finalize.deepep_auto import (
        dspark_mooncake_pd_use_low_latency,
    )

    fixed_use_low_latency = dspark_mooncake_pd_use_low_latency(
        get_current_vllm_config_or_none()
    )
    if experts_cls is None:
        experts_cls = DeepEPAutoDeepGemmExperts
    experts = experts_cls(
        moe_config=moe_config,
        quant_config=moe_quant_config,
        max_num_tokens=max_num_tokens,
        num_dispatchers=prepare_finalize.num_dispatchers(),
        fixed_use_low_latency=fixed_use_low_latency,
    )
    if fixed_use_low_latency is None:
        logger.info_once("Using DeepEP auto MoE kernel with HT/LL experts.")
    elif fixed_use_low_latency:
        logger.info_once("Using role-fixed DeepEP auto MoE kernel with LL experts.")
    else:
        logger.info_once("Using role-fixed DeepEP auto MoE kernel with HT experts.")
    return mk.FusedMoEKernel(prepare_finalize, experts)


def make_deepep_auto_deepgemm_fp8_moe_kernel(
    moe_quant_config: FusedMoEQuantConfig,
    moe_config: FusedMoEConfig,
    routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> mk.FusedMoEKernel:
    """Build the unified DeepEP HT/LL kernel for Channel-FP8 W8A8."""

    return _make_deepep_auto_deepgemm_moe_kernel(
        moe_quant_config=moe_quant_config,
        moe_config=moe_config,
        routing_tables=routing_tables,
    )


def make_deepep_auto_deepgemm_int8_moe_kernel(
    moe_quant_config: FusedMoEQuantConfig,
    moe_config: FusedMoEConfig,
    routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> mk.FusedMoEKernel:
    """Build the unified DeepEP HT/LL kernel for Channel-INT8 W8A8."""

    if not moe_quant_config.use_int8_w8a8:
        raise ValueError("Channel-INT8 auto factory requires INT8 W8A8 quantization")
    return _make_deepep_auto_deepgemm_moe_kernel(
        moe_quant_config=moe_quant_config,
        moe_config=moe_config,
        routing_tables=routing_tables,
    )


def make_deepep_auto_deepgemm_w4a8_moe_kernel(
    moe_quant_config: FusedMoEQuantConfig,
    moe_config: FusedMoEConfig,
    routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> mk.FusedMoEKernel:
    """Build the unified DeepEP HT/LL kernel for SlimQuant W4A8."""

    if moe_quant_config.weight_quant_dtype != "int4":
        raise ValueError("SlimQuant auto factory requires INT4 W4A8 quantization")
    if (
        moe_quant_config.quant_dtype != torch.int8
        or not moe_quant_config.is_per_act_token
        or moe_quant_config.is_block_quantized
    ):
        raise ValueError(
            "SlimQuant auto factory requires dynamic per-token INT8 "
            "activation quantization"
        )
    if (
        moe_quant_config.per_out_ch_quant
        or moe_quant_config.w1_scale is None
        or moe_quant_config.w2_scale is None
    ):
        raise ValueError(
            "SlimQuant auto factory requires symmetric INT4 channel weight "
            "scales for both MoE GEMMs"
        )
    unsupported_metadata = (
        moe_quant_config.a1_scale,
        moe_quant_config.a2_scale,
        moe_quant_config.a1_gscale,
        moe_quant_config.a2_gscale,
        moe_quant_config.w1_zp,
        moe_quant_config.w2_zp,
        moe_quant_config.g1_alphas,
        moe_quant_config.g2_alphas,
        moe_quant_config.w1_bias,
        moe_quant_config.w2_bias,
        moe_quant_config.gemm1_alpha,
        moe_quant_config.gemm1_beta,
        moe_quant_config.gemm1_clamp_limit,
    )
    if any(value is not None for value in unsupported_metadata):
        raise ValueError(
            "SlimQuant auto factory requires symmetric W4A8 without "
            "auxiliary scales, zero points, biases, or clamps"
        )
    return _make_deepep_auto_deepgemm_moe_kernel(
        moe_quant_config=moe_quant_config,
        moe_config=moe_config,
        routing_tables=routing_tables,
        experts_cls=DeepEPAutoW4A8Experts,
    )
