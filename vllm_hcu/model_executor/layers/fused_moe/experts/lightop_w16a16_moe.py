# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""vLLM modular-expert adapter for LightOp BF16 Marlin MoE."""

from __future__ import annotations

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import FusedMoEParallelConfig
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import QuantKey
from vllm.platforms import current_platform
from vllm_hcu.model_executor.layers.fused_moe.lightop_w16a16_runtime import (
    is_lightop_w16a16_available,
    lightop_w16a16_layout,
    run_lightop_w16a16,
)


class LightopW16A16Experts(mk.FusedMoEExpertsModular):
    """Own both GEMMs, SiLU multiplication, route weighting and reduction."""

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        # Keep optional-package probing out of the generic modular-kernel
        # checks.  Strict local eligibility is evaluated first below.
        return current_platform.is_cuda_alike()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return weight_key is None and activation_key is None

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation == MoEActivation.SILU

    @staticmethod
    def _supports_parallel_config(
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> bool:
        # Packing is local-expert specific.  Keep the initial integration on
        # pure tensor parallelism until EP/EPLB layout updates are proven.
        return not (
            bool(getattr(moe_parallel_config, "use_ep", False))
            or bool(getattr(moe_parallel_config, "enable_eplb", False))
            or int(getattr(moe_parallel_config, "dp_size", 1)) != 1
            or int(getattr(moe_parallel_config, "pcp_size", 1)) != 1
            or int(getattr(moe_parallel_config, "sp_size", 1)) != 1
        )

    @staticmethod
    def _supports_shape(hidden_dim: int) -> bool:
        return hidden_dim > 0 and hidden_dim % 32 == 0

    @staticmethod
    def is_supported_config(
        cls,
        moe_config,
        weight_key,
        activation_key,
        activation_format,
    ):
        supported, reason = super().is_supported_config(
            cls,
            moe_config,
            weight_key,
            activation_key,
            activation_format,
        )
        if not supported:
            return supported, reason
        if getattr(moe_config, "in_dtype", None) != torch.bfloat16:
            return False, "kernel supports BF16 activations and weights only"
        if bool(getattr(moe_config, "has_bias", False)):
            return False, "kernel does not support expert bias"
        if bool(getattr(moe_config, "aiter_fmoe_shared_expert_enabled", False)):
            return False, "kernel does not support AITER fused shared experts"
        swiglu_limit = getattr(moe_config, "swiglu_limit", None)
        swiglu_alpha = getattr(moe_config, "swiglu_alpha", None)
        swiglu_beta = getattr(moe_config, "swiglu_beta", None)
        if (
            swiglu_limit is not None
            or swiglu_alpha not in (None, 1.0)
            or swiglu_beta not in (None, 0.0)
        ):
            return False, "kernel supports only unmodified SwiGLU activation"
        intermediate = int(
            getattr(moe_config, "intermediate_size_per_partition", -1)
        )
        if intermediate <= 0 or intermediate % 16:
            return False, "kernel requires intermediate size divisible by 16"
        topk = int(getattr(moe_config, "experts_per_token", -1))
        if topk <= 0 or topk > 16:
            return False, "kernel requires 1 <= top-k <= 16"
        problem = (
            int(getattr(moe_config, "num_experts", -1)),
            int(getattr(moe_config, "hidden_dim", -1)),
            intermediate,
            topk,
        )
        if problem != (256, 2048, 512, 8):
            return False, (
                "kernel performance is validated only for "
                "E=256,K=2048,N=512,top-k=8"
            )
        max_num_tokens = int(getattr(moe_config, "max_num_tokens", -1))
        if max_num_tokens <= 0 or max_num_tokens > 16:
            return False, (
                "kernel performance requires 1 <= max_num_tokens <= 16"
            )
        if not is_lightop_w16a16_available():
            return False, "required categorized LightOp W16A16 exports are unavailable"
        return True, None

    def moe_problem_size(
        self,
        a1: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple[int, int, int, int, int]:
        layout = lightop_w16a16_layout(w1, w2)
        experts, two_n, hidden = layout.logical_w13_shape
        if a1.dim() != 2 or a1.shape[1] != hidden:
            raise RuntimeError("LightOp W16A16 expects standard [M,K] activations")
        if topk_ids.dim() != 2 or topk_ids.shape[0] != a1.shape[0]:
            raise RuntimeError(
                "LightOp W16A16 routing shape does not match activations"
            )
        return experts, int(a1.shape[0]), two_n, hidden, int(topk_ids.shape[1])

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
        del global_num_experts, local_num_experts
        if expert_tokens_meta is not None or activation != MoEActivation.SILU:
            raise RuntimeError("LightOp W16A16 supports standard TP SiLU MoE only")
        intermediate = N // 2
        return (
            (M * topk, max(N, K)),
            (M * topk, intermediate),
            (M, K),
        )

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

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
        if activation != MoEActivation.SILU:
            raise RuntimeError("LightOp W16A16 supports SiLU activation only")
        if expert_map is not None or expert_tokens_meta is not None:
            raise RuntimeError("LightOp W16A16 supports TP-only expert routing")
        if a1q_scale is not None or a2_scale is not None:
            raise RuntimeError("LightOp W16A16 does not consume quantization scales")
        if apply_router_weight_on_input:
            raise RuntimeError(
                "LightOp W16A16 applies routing weights in GEMM2 and cannot "
                "apply them on the input"
            )
        run_lightop_w16a16(
            output=output,
            hidden_states=hidden_states,
            w13=w1,
            w2=w2,
            topk_weights=topk_weights.to(dtype=torch.float32),
            topk_ids=topk_ids.to(dtype=torch.int32),
            workspace13=workspace13,
            workspace2=workspace2,
            global_num_experts=global_num_experts,
        )


__all__ = ["LightopW16A16Experts"]
