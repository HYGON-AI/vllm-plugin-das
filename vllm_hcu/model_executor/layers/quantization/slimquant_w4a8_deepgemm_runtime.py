# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""DeepGEMM expert layouts for canonical SlimQuant W4A8 MoE weights."""

from __future__ import annotations

import functools

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.deep_gemm_utils import (
    compute_aligned_M,
    deepgemm_moe_permute,
    deepgemm_unpermute_and_reduce,
)
from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.fused_moe.utils import _resize_cache
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kInt4W4A8StaticChannelSym,
    kInt8DynamicTokenSym,
)
from vllm_hcu.model_executor.layers.fused_moe.experts.batched_deep_gemm_moe import (
    BatchedDeepGemmExperts,
)

logger = init_logger(__name__)


@functools.lru_cache(maxsize=None)
def _deepgemm_op(name: str):
    import deepgemm

    return getattr(deepgemm, name)


def pack_w4a8_moe_hipc_weight(weight: torch.Tensor) -> torch.Tensor:
    return _deepgemm_op("pack_w4a8_moe_hipc_weight")(weight)


def view_w4a8_moe_hipc_weight_n32_layout(
    weight: torch.Tensor,
) -> torch.Tensor:
    return _deepgemm_op("view_w4a8_moe_hipc_weight_n32_layout")(weight)


def m_grouped_w4a8_gemm_nt_contiguous_hipc(*args, **kwargs):
    return _deepgemm_op("m_grouped_w4a8_gemm_nt_contiguous_hipc")(
        *args, **kwargs
    )


@functools.lru_cache(maxsize=None)
def _lightop_op(name: str):
    import lightop

    return getattr(lightop, name)


def fuse_silu_mul_quant(*args, **kwargs):
    return _lightop_op("fuse_silu_mul_quant")(*args, **kwargs)


def _canonical_weight_signature(layer: torch.nn.Module) -> tuple[object, ...]:
    return (
        id(layer.w13_weight),
        layer.w13_weight._version,
        tuple(layer.w13_weight.shape),
        id(layer.w2_weight),
        layer.w2_weight._version,
        tuple(layer.w2_weight.shape),
    )


def _validate_w4a8_channel_weights(layer: torch.nn.Module) -> None:
    w13 = layer.w13_weight
    w2 = layer.w2_weight
    if (
        w13.dtype != torch.int8
        or w2.dtype != torch.int8
        or w13.ndim != 3
        or w2.ndim != 3
        or w13.size(0) != w2.size(0)
    ):
        raise RuntimeError(
            "SlimQuant W4A8 DeepGEMM requires packed INT8 rank-3 weights, "
            f"got w13={tuple(w13.shape)}/{w13.dtype} "
            f"w2={tuple(w2.shape)}/{w2.dtype}"
        )

    logical_hidden = w13.size(2) * 2
    logical_intermediate = w2.size(2) * 2
    if w2.size(1) != logical_hidden or w13.size(1) != 2 * logical_intermediate:
        raise RuntimeError(
            "SlimQuant W4A8 DeepGEMM packed weight dimensions are inconsistent: "
            f"w13={tuple(w13.shape)} w2={tuple(w2.shape)}"
        )

    local_experts = w13.size(0)
    for name, scale, output_size in (
        ("w13_weight_scale", layer.w13_weight_scale, w13.size(1)),
        ("w2_weight_scale", layer.w2_weight_scale, w2.size(1)),
    ):
        valid = (
            scale.dtype == torch.float32
            and scale.ndim in (2, 3)
            and scale.size(0) == local_experts
            and scale.size(1) == output_size
            and (scale.ndim == 2 or scale.size(2) == 1)
        )
        if not valid:
            raise RuntimeError(
                f"SlimQuant W4A8 DeepGEMM has invalid {name}: "
                f"shape={tuple(scale.shape)} dtype={scale.dtype}"
            )


class DeepEPDeepGemmW4A8ContiguousExperts(TritonExperts):
    """Contiguous HIPC W4A8 experts for DeepEP high-throughput dispatch."""

    ALIGNMENT = 256
    _CACHE_PREFIX = "_slimquant_w4a8_deepgemm_contiguous"

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ) -> None:
        super().__init__(moe_config, quant_config)
        self._deepgemm_w13: torch.Tensor | None = None
        self._deepgemm_w2: torch.Tensor | None = None
        logger.info_once(
            "Using SlimQuant W4A8 contiguous HIPC DeepGEMM experts."
        )

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return (weight_key, activation_key) == (
            kInt4W4A8StaticChannelSym,
            kInt8DynamicTokenSym,
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Pack clones once while retaining checkpoint parameters as owners."""

        _validate_w4a8_channel_weights(layer)
        signature = _canonical_weight_signature(layer)
        cache_signature_name = f"{self._CACHE_PREFIX}_source"
        cache_w13_name = f"{self._CACHE_PREFIX}_w13"
        cache_w2_name = f"{self._CACHE_PREFIX}_w2"
        if getattr(layer, cache_signature_name, None) != signature:
            with torch.no_grad():
                packed_w13 = pack_w4a8_moe_hipc_weight(
                    layer.w13_weight.detach().clone()
                ).detach()
                packed_w2 = pack_w4a8_moe_hipc_weight(
                    layer.w2_weight.detach().clone()
                ).detach()
            setattr(layer, cache_w13_name, packed_w13)
            setattr(layer, cache_w2_name, packed_w2)
            setattr(layer, cache_signature_name, signature)

        self._deepgemm_w13 = getattr(layer, cache_w13_name)
        self._deepgemm_w2 = getattr(layer, cache_w2_name)

    def moe_problem_size(
        self,
        a1: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple[int, int, int, int, int]:
        del w1, w2
        if self.w1_scale is None or self.w2_scale is None:
            raise RuntimeError("SlimQuant W4A8 DeepGEMM requires weight scales")
        return (
            self.w1_scale.size(0),
            a1.size(0),
            self.w1_scale.size(1),
            self.w2_scale.size(1),
            topk_ids.size(1),
        )

    def finalize_weight_and_reduce_impl(self):
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
        del global_num_experts
        m_aligned = compute_aligned_M(
            M=M,
            num_topk=topk,
            local_num_experts=local_num_experts,
            alignment=self.ALIGNMENT,
            expert_tokens_meta=expert_tokens_meta,
        )
        activation_out_dim = self.adjust_N_for_activation(N, activation)
        return (
            (m_aligned, max(activation_out_dim, K)),
            (m_aligned, max(N, K)),
            (M, K),
        )

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
        del global_num_experts, a2_scale, apply_router_weight_on_input
        if hidden_states.size(0) == 0:
            return
        if activation != MoEActivation.SILU:
            raise NotImplementedError(
                "SlimQuant W4A8 DeepGEMM supports only SiLU activation"
            )
        if a1q_scale is None:
            raise RuntimeError(
                "SlimQuant W4A8 DeepGEMM requires per-token activation scales"
            )
        if self.w1_scale is None or self.w2_scale is None:
            raise RuntimeError("SlimQuant W4A8 DeepGEMM requires weight scales")
        if self._deepgemm_w13 is None or self._deepgemm_w2 is None:
            raise RuntimeError(
                "SlimQuant W4A8 DeepGEMM weights were not packed before apply"
            )

        local_num_experts, _, N, K, _ = self.moe_problem_size(
            hidden_states, w1, w2, topk_ids
        )
        m_aligned = compute_aligned_M(
            M=topk_ids.size(0),
            num_topk=topk_ids.size(1),
            local_num_experts=local_num_experts,
            alignment=self.ALIGNMENT,
            expert_tokens_meta=expert_tokens_meta,
        )
        input_workspace = _resize_cache(
            workspace13.view(dtype=hidden_states.dtype), (m_aligned, K)
        )
        input_tensor, input_scale, m_indices, inv_perm, _ = (
            deepgemm_moe_permute(
                aq=hidden_states,
                aq_scale=self._ensure_2d_scale(a1q_scale),
                topk_ids=topk_ids,
                local_num_experts=local_num_experts,
                expert_map=expert_map,
                expert_tokens_meta=expert_tokens_meta,
                aq_out=input_workspace,
            )
        )
        m_indices = m_indices.to(dtype=torch.int32).contiguous()

        gateup_output = _resize_cache(workspace2, (m_aligned, N))
        m_grouped_w4a8_gemm_nt_contiguous_hipc(
            (input_tensor, input_scale),
            (self._deepgemm_w13, self.w1_scale),
            gateup_output,
            m_indices,
        )
        activation_out_dim = self.adjust_N_for_activation(N, activation)
        quant_output = _resize_cache(
            workspace13.view(dtype=torch.int8),
            (m_aligned, activation_out_dim),
        )
        q_activation, q_activation_scale = fuse_silu_mul_quant(
            gateup_output,
            output=quant_output,
            expert_ids=m_indices,
        )
        down_output = _resize_cache(workspace2, (m_aligned, K))
        m_grouped_w4a8_gemm_nt_contiguous_hipc(
            (q_activation, q_activation_scale),
            (self._deepgemm_w2, self.w2_scale),
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
        return scale.unsqueeze(-1) if scale.ndim == 1 else scale


class DeepEPDeepGemmW4A8BatchedExperts(BatchedDeepGemmExperts):
    """N32 HIPC W4A8 experts for DeepEP low-latency dispatch."""

    _CACHE_PREFIX = "_slimquant_w4a8_deepgemm_masked"

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        max_num_tokens: int,
        num_dispatchers: int,
    ) -> None:
        super().__init__(
            moe_config=moe_config,
            quant_config=quant_config,
            max_num_tokens=max_num_tokens,
            num_dispatchers=num_dispatchers,
        )
        self._deepgemm_w13: torch.Tensor | None = None
        self._deepgemm_w2: torch.Tensor | None = None
        logger.info_once("Using SlimQuant W4A8 masked N32 HIPC DeepGEMM experts.")

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Cache N32 views derived from canonical checkpoint parameters."""

        _validate_w4a8_channel_weights(layer)
        signature = _canonical_weight_signature(layer)
        cache_signature_name = f"{self._CACHE_PREFIX}_source"
        cache_w13_name = f"{self._CACHE_PREFIX}_w13"
        cache_w2_name = f"{self._CACHE_PREFIX}_w2"
        if getattr(layer, cache_signature_name, None) != signature:
            with torch.no_grad():
                packed_w13 = pack_w4a8_moe_hipc_weight(
                    layer.w13_weight.detach().clone()
                )
                packed_w2 = pack_w4a8_moe_hipc_weight(
                    layer.w2_weight.detach().clone()
                )
                masked_w13 = view_w4a8_moe_hipc_weight_n32_layout(
                    packed_w13
                ).detach()
                masked_w2 = view_w4a8_moe_hipc_weight_n32_layout(
                    packed_w2
                ).detach()
            setattr(layer, cache_w13_name, masked_w13)
            setattr(layer, cache_w2_name, masked_w2)
            setattr(layer, cache_signature_name, signature)

        self._deepgemm_w13 = getattr(layer, cache_w13_name)
        self._deepgemm_w2 = getattr(layer, cache_w2_name)


DeepEPDeepGemmW4A8MaskedExperts = DeepEPDeepGemmW4A8BatchedExperts


__all__ = [
    "DeepEPDeepGemmW4A8BatchedExperts",
    "DeepEPDeepGemmW4A8ContiguousExperts",
    "DeepEPDeepGemmW4A8MaskedExperts",
]
