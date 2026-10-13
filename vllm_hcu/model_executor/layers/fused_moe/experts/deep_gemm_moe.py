# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# Modified by Hygon Information Technology Co., Ltd., 2026.

import os
from contextlib import nullcontext

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
    compute_aligned_M_and_alignment,
    deepgemm_moe_permute,
    deepgemm_unpermute_and_reduce,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.fused_moe.utils import _resize_cache
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
    per_token_group_quant_fp8_packed_for_deepgemm,
    silu_mul_per_token_group_quant_fp8_colmajor,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    silu_mul_quant_fp8_packed_triton as fused_silu_mul_fp8_quant_packed,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8Dynamic128Sym,
    kFp8DynamicTokenSym,
    kFp8Static128BlockSym,
    kFp8StaticChannelSym,
    kInt8DynamicTokenSym,
    kInt8StaticChannelSym,
    kMxfp4Static,
    kMxfp8Dynamic,
    kMxfp8Static,
)
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import (
    DeepGemmQuantScaleFMT,
    get_mk_alignment_for_contiguous_layout,
    is_deep_gemm_supported,
    m_grouped_fp8_fp4_gemm_nt_contiguous,
    m_grouped_fp8_gemm_nt_contiguous,
    mk_alignment_scope,
)
from vllm.utils.import_utils import has_deep_gemm

logger = init_logger(__name__)


# DeepEP high-throughput kernels on HCU require 256-row expert slices.
_HCU_HT_EP_TOKEN_ALIGNMENT = 256

# Opt-in: run Channel-FP8 EP MoE through ep_fused_moe (expert-sorted Triton GEMMs,
# work bounded by local routes) instead of DeepGEMM. patch_fp8_oracle then keeps
# the weights as plain [E, N, K].
_EP_FUSED_MOE = os.environ.get("VLLM_HCU_EP_FUSED_MOE", "0") == "1"


def _valid_deep_gemm_shape(M: int, N: int, K: int) -> bool:
    if current_platform.is_rocm():
        # HCU pads every expert to _HCU_HT_EP_TOKEN_ALIGNMENT rows, so the
        # DeepGemm workspace must be used even for small M; the Triton one
        # would be undersized when DeepGemm is selected at apply() time.
        return True
    align = get_mk_alignment_for_contiguous_layout()[0]
    return align <= M and N % align == 0 and K % align == 0


def _valid_deep_gemm(
    hidden_states: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor
) -> bool:
    """
    Check if the given problem size is supported by the DeepGemm grouped
    gemm kernel.  All of M, N, K and the quantization block_shape must be
    aligned by `dg.get_m_alignment_for_contiguous_layout()`.
    """
    if not has_deep_gemm():
        logger.debug_once("DeepGemm disabled: deep_gemm not available.")
        return False

    # HCU Marlin-packed weights [E, K/64, N/16, 4, 16, 16] are consumable
    # only by the HCU DeepGEMM kernels; Triton cannot read this layout.
    if w1.dim() == 6 and w2.dim() == 6:
        return True

    M = hidden_states.size(0)
    _, K, N = w2.size()

    align = get_mk_alignment_for_contiguous_layout()[0]

    if not _valid_deep_gemm_shape(M, N, K):
        logger.debug_once(
            "DeepGemm disabled due to unaligned problem size. "
            "M: %s, N: %s, K: %s. M should >= %s "
            "and N and K must be multiples of %s. "
            "This is not an error and we will fall back to triton.",
            M,
            N,
            K,
            align,
            align,
        )
        return False
    elif N <= 512:
        logger.debug_once(
            "DeepGemm disabled for N <= 512. M: %s, N: %s, K: %s. "
            "This means we will fallback to triton "
            "for this specific shape for further speed up.",
            M,
            N,
            K,
        )
        return False

    if w1.dtype != torch.float8_e4m3fn or w2.dtype != torch.float8_e4m3fn:
        logger.debug_once(
            "DeepGemm disabled: invalid weight dtype(s). w1.dtype: %s, w2.dtype: %s",
            w1.dtype,
            w2.dtype,
        )
        return False

    if (
        not hidden_states.is_contiguous()
        or not w1.is_contiguous()
        or not w2.is_contiguous()
    ):
        logger.debug_once(
            "DeepGemm disabled: weights or activations not contiguous. "
            "hidden_states.is_contiguous(): %s, w1.is_contiguous(): %s, "
            "w2.is_contiguous(): %s",
            hidden_states.is_contiguous(),
            w1.is_contiguous(),
            w2.is_contiguous(),
        )
        return False

    return True


class DeepGemmExperts(mk.FusedMoEExpertsModular):
    """DeepGemm-based fused MoE expert implementation."""

    def __init__(self, moe_config: FusedMoEConfig, quant_config: FusedMoEQuantConfig):
        super().__init__(moe_config=moe_config, quant_config=quant_config)
        # MXFP8: FP8 e4m3 values + UE8M0 1x32 block scales (Blackwell). Reuses
        # the same grouped GEMM (aliased to fp8_fp4) with recipe (1, 32).
        self.mxfp8 = quant_config.block_shape == [1, 32]
        # NOTE: Relaxed assertions for HCU w8a8 to support channel-level FP8
        # quantization (as in vLLM 0.21). HCU DeepGemm can handle per-channel
        # quantized weights with per-token activation scales.
        if current_platform.is_rocm() and (
            quant_config.use_int8_w8a8 or quant_config.use_fp8_w8a8
        ):
            # HCU w8a8: skip block_shape and per_out_ch_quant assertions
            pass
        else:
            if self.mxfp8:
                assert quant_config.quant_dtype == "mxfp8"
            else:
                assert (
                    quant_config.block_shape
                    == get_mk_alignment_for_contiguous_layout()
                )
                assert quant_config.quant_dtype == torch.float8_e4m3fn
            assert not quant_config.per_act_token_quant
            assert not quant_config.per_out_ch_quant

        self.gemm1_clamp_limit = quant_config.gemm1_clamp_limit
        # Gated-activation params: silu == swigluoai with alpha=1, beta=0.
        # FP8 (silu) configs leave these None, reproducing plain silu.
        self.gemm1_alpha = (
            quant_config.gemm1_alpha if quant_config.gemm1_alpha is not None else 1.0
        )
        self.gemm1_beta = (
            quant_config.gemm1_beta if quant_config.gemm1_beta is not None else 0.0
        )
        # HCU's repacked Channel weights can expose physical N/K that differ
        # from the model dimensions. Keep the override private and derive it
        # from v0.25.1's unpadded config instead of extending this constructor.
        self._hcu_logical_n = 2 * int(
            moe_config.intermediate_size_per_partition_unpadded
        )
        self._hcu_logical_k = int(moe_config.hidden_dim_unpadded)

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        return current_platform.is_rocm() or is_deep_gemm_supported()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        if (weight_key, activation_key) == (kFp8Static128BlockSym, kFp8Dynamic128Sym):
            return True
        if (weight_key, activation_key) == (
            kInt8StaticChannelSym,
            kInt8DynamicTokenSym,
        ):
            return current_platform.is_rocm()
        if (weight_key, activation_key) == (
            kFp8StaticChannelSym,
            kFp8DynamicTokenSym,
        ):
            return current_platform.is_rocm()
        # MXFP8 1x32 uses the fp8_fp4 grouped GEMM with recipe (1, 32) — only
        # available on Blackwell (SM100).
        if (weight_key, activation_key) == (kMxfp8Static, kMxfp8Dynamic):
            return current_platform.is_device_capability_family(100)
        return False

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        # silu/swigluoai go through the fused alpha/beta kernel; swiglustep
        # uses the unfused activation path. The fused kernel reads packed w13
        # (gate = first half, up = second half), so it implements the
        # *uninterleaved* SwiGLU-OAI variant.
        return activation in [
            MoEActivation.SILU,
            MoEActivation.SWIGLUSTEP,
            MoEActivation.SWIGLUOAI_UNINTERLEAVE,
        ]

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        # NOTE(rob): discovered an IMA with this combination. Needs investigation.
        return not (
            moe_parallel_config.use_fi_nvl_two_sided_kernels
            or moe_parallel_config.use_fi_nvl_one_sided_kernels
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
        assert self.block_shape is not None or (
            current_platform.is_rocm()
            and (
                self.quant_config.use_int8_w8a8
                or self.quant_config.use_fp8_w8a8
            )
        )
        # Use the contiguous-layout M alignment (matches apply()); block_shape[0]
        # is the quant block (1 for MXFP8) and would under-size the workspace.
        block_m = (
            _HCU_HT_EP_TOKEN_ALIGNMENT
            if current_platform.is_rocm()
            else get_mk_alignment_for_contiguous_layout()[0]
        )
        M_sum, align_used = compute_aligned_M_and_alignment(
            M, topk, local_num_experts, block_m, expert_tokens_meta
        )
        assert M_sum % align_used == 0

        activation_out_dim = self.adjust_N_for_activation(N, activation)
        workspace1 = (M_sum, max(activation_out_dim, K))
        workspace2 = (M_sum, max(N, K))
        output = (M, K)
        return (workspace1, workspace2, output)

    def _act_mul_quant(
        self, input: torch.Tensor, output: torch.Tensor, activation: MoEActivation
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.block_shape is not None
        block_k = self.block_shape[1]
        scale_fmt = DeepGemmQuantScaleFMT.from_oracle()

        M_sum, N = input.size()
        activation_out_dim = self.adjust_N_for_activation(N, activation)

        # silu and swigluoai are both expressible by the fused gated kernel via
        # (alpha, beta): silu uses alpha=1, beta=0; swigluoai uses config values.
        # The fused kernel reads packed w13, hence SWIGLUOAI_UNINTERLEAVE.
        fused_gated = activation in (
            MoEActivation.SILU,
            MoEActivation.SWIGLUOAI_UNINTERLEAVE,
        )

        # 1. DeepGemm UE8M0: fused gate+mul+clamp+quant+pack
        if scale_fmt == DeepGemmQuantScaleFMT.UE8M0:
            if fused_gated:
                return fused_silu_mul_fp8_quant_packed(
                    input=input,
                    output_q=output,
                    group_size=block_k,
                    clamp_limit=self.gemm1_clamp_limit,
                    alpha=self.gemm1_alpha,
                    beta=self.gemm1_beta,
                )
            act_out = torch.empty(
                (M_sum, activation_out_dim), dtype=input.dtype, device=input.device
            )
            self.activation(activation, act_out, input)
            a2q, a2q_scale = per_token_group_quant_fp8_packed_for_deepgemm(
                act_out,
                block_k,
                out_q=output,
            )
            return a2q, a2q_scale

        # 2. Hopper / non‑E8M0: prefer the fused gate+mul+quant kernel
        if fused_gated:
            use_ue8m0 = scale_fmt == DeepGemmQuantScaleFMT.FLOAT32_CEIL_UE8M0
            return silu_mul_per_token_group_quant_fp8_colmajor(
                input=input,
                output=output,
                use_ue8m0=use_ue8m0,
                clamp_limit=self.gemm1_clamp_limit,
                group_size=block_k,
                alpha=self.gemm1_alpha,
                beta=self.gemm1_beta,
            )

        # 3. fallback path for non-SiLU activations in non‑UE8M0 cases.
        act_out = torch.empty(
            (M_sum, activation_out_dim), dtype=input.dtype, device=input.device
        )
        self.activation(activation, act_out, input)
        return per_token_group_quant_fp8(
            act_out, block_k, column_major_scales=True, out_q=output
        )

    def _apply_ep_fused(
        self, output, a1q, w1, w2, topk_weights, topk_ids, activation,
        global_num_experts, expert_map, a1q_scale, workspace13, workspace2,
        apply_router_weight_on_input,
    ):
        from vllm_hcu.model_executor.layers.fused_moe.ep_fused_moe_support import (
            supports_channel_ep_fused,
        )

        if not (
            self.quant_config.use_fp8_w8a8
            and self.quant_config.per_out_ch_quant
            and self.quant_config.per_act_token_quant
            and supports_channel_ep_fused(
                w1, w2, self.w1_scale, self.w2_scale,
                block_shape=self.block_shape,
                activation=activation,
                expert_map=expert_map,
                apply_router_weight_on_input=apply_router_weight_on_input,
            )
        ):
            raise ValueError(
                "EP fused MoE requires supported channel-FP8 weights, dynamic "
                "per-token activations and SWIGLUSTEP with expert parallelism"
            )
        if (
            a1q_scale is None
            or a1q_scale.shape not in ((a1q.shape[0],), (a1q.shape[0], 1))
            or a1q_scale.device != a1q.device
        ):
            raise ValueError("EP fused MoE requires one activation scale per token")
        from vllm_hcu.model_executor.layers.fused_moe.ep_fused_moe import (
            ep_fused_experts,
        )

        mask = getattr(self, "_ep_fused_mask", None)
        if mask is None or self._ep_fused_mask_src is not expert_map:
            mask = (expert_map >= 0).to(torch.int32)
            self._ep_fused_mask, self._ep_fused_mask_src = mask, expert_map
        ep_fused_experts(
            a1q,
            w1,
            w2,
            topk_weights.to(torch.float32),
            topk_ids.to(torch.int32).contiguous(),
            expert_map,
            self.w1_scale,
            self.w2_scale,
            a1_scale=a1q_scale,
            global_num_experts=int(global_num_experts),
            output_dtype=output.dtype,
            gemm1_limit=float(self.gemm1_clamp_limit or 7.0),
            expert_mask=mask,
            out=output,
            workspace13=workspace13,
            workspace2=workspace2,
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
    ):
        assert a1q_scale is not None
        assert a2_scale is None
        assert self.block_shape is not None or (
            current_platform.is_rocm()
            and (
                self.quant_config.use_int8_w8a8
                or self.quant_config.use_fp8_w8a8
            )
        )
        assert self.w1_scale is not None
        assert self.w2_scale is not None

        a1q = hidden_states
        local_num_experts = w1.size(0)
        if global_num_experts == -1:
            global_num_experts = local_num_experts

        if (
            _EP_FUSED_MOE
            and self.block_shape is None
            and self.quant_config.use_fp8_w8a8
            and w1.dim() == 3
        ):
            self._apply_ep_fused(
                output, a1q, w1, w2, topk_weights, topk_ids, activation,
                global_num_experts, expert_map, a1q_scale, workspace13, workspace2,
                apply_router_weight_on_input,
            )
            return

        if current_platform.is_rocm():
            N = self._hcu_logical_n
            K = self._hcu_logical_k
        else:
            _, N, K = w1.size()
            assert w2.size(1) == K

        M_sum, _ = compute_aligned_M_and_alignment(
            M=topk_ids.size(0),
            num_topk=topk_ids.size(1),
            local_num_experts=local_num_experts,
            alignment=(
                _HCU_HT_EP_TOKEN_ALIGNMENT
                if current_platform.is_rocm()
                else get_mk_alignment_for_contiguous_layout()[0]
            ),
            expert_tokens_meta=expert_tokens_meta,
        )

        permuted_dtype = (
            a1q.dtype
            if self.quant_config.use_int8_w8a8
            else torch.float8_e4m3fn
        )
        a1q_perm = _resize_cache(workspace13.view(dtype=permuted_dtype), (M_sum, K))
        a1q, a1q_scale, expert_ids, inv_perm, align_used = deepgemm_moe_permute(
            aq=a1q,
            aq_scale=a1q_scale,
            topk_ids=topk_ids,
            local_num_experts=local_num_experts,
            expert_map=expert_map,
            expert_tokens_meta=expert_tokens_meta,
            aq_out=a1q_perm,
            # MXFP8 uses a 32-element activation-scale group (block_shape[1]);
            # FP8-block keeps the default (128) alignment.
            block_size=self.block_shape[1] if self.mxfp8 else None,
        )
        assert a1q.size(0) == M_sum

        # MXFP8 (1x32) drives the fp8_fp4-aliased grouped GEMM with recipe
        # (1, 32); the FP8 block path keeps the default (128) recipe.
        gemm_kwargs = (
            {"recipe_a": (1, self.block_shape[1]), "recipe_b": (1, self.block_shape[1])}
            if self.mxfp8
            else {}
        )

        # Cap DG's BLOCK_M heuristic at the workspace's per-expert alignment;
        # otherwise the scheduler can pick the wrong expert id from m_indices
        # under cudagraph replay. HCU's INT8 front end consumes its fixed
        # 256-row layout directly and has no upstream alignment context.
        alignment_scope = (
            nullcontext()
            if current_platform.is_rocm() and self.quant_config.use_int8_w8a8
            else mk_alignment_scope(align_used)
        )
        with alignment_scope:
            mm1_out = _resize_cache(workspace2, (M_sum, N))
            if self.quant_config.use_int8_w8a8:
                if activation != MoEActivation.SILU:
                    raise ValueError(
                        "HCU Channel INT8 DeepGEMM supports only SiLU activation"
                    )
                from deepgemm import m_grouped_i8_gemm_nt_contiguous
                from lightop.activation import fuse_silu_mul_quant

                m_grouped_i8_gemm_nt_contiguous(
                    (a1q, a1q_scale),
                    (w1, self.w1_scale),
                    mm1_out,
                    expert_ids,
                )
                activation_out_dim = self.adjust_N_for_activation(N, activation)
                quant_out = _resize_cache(
                    workspace13.view(dtype=torch.int8),
                    (M_sum, activation_out_dim),
                )
                a2q, a2q_scale = fuse_silu_mul_quant(
                    mm1_out,
                    output=quant_out,
                    expert_ids=expert_ids,
                )
                mm2_out = _resize_cache(workspace2, (M_sum, K))
                m_grouped_i8_gemm_nt_contiguous(
                    (a2q, a2q_scale),
                    (w2, self.w2_scale),
                    mm2_out,
                    expert_ids,
                )
            else:
                m_grouped_fp8_gemm_nt_contiguous(
                    (a1q, a1q_scale),
                    (w1, self.w1_scale),
                    mm1_out,
                    expert_ids,
                    **gemm_kwargs,
                )

                activation_out_dim = self.adjust_N_for_activation(N, activation)
                quant_out = _resize_cache(
                    workspace13.view(dtype=torch.float8_e4m3fn),
                    (M_sum, activation_out_dim),
                )
                if current_platform.is_rocm() and activation in (
                    MoEActivation.SILU,
                    MoEActivation.SWIGLUSTEP,
                ):
                    from lightop.activation import fuse_silu_mul_fp8_quant

                    # lightop's `limit` computes min(silu(g), L) * clamp(u, -L, L),
                    # i.e. swiglustep; -1 disables the clamp (plain SiLU).
                    limit = -1.0
                    if activation == MoEActivation.SWIGLUSTEP:
                        limit = float(self.gemm1_clamp_limit or 7.0)
                    a2q, a2q_scale = fuse_silu_mul_fp8_quant(
                        mm1_out,
                        fp8type=0,
                        output=quant_out,
                        expert_ids=expert_ids,
                        limit=limit,
                    )
                else:
                    a2q, a2q_scale = self._act_mul_quant(
                        input=mm1_out.view(-1, N),
                        output=quant_out,
                        activation=activation,
                    )

                mm2_out = _resize_cache(workspace2, (M_sum, K))
                m_grouped_fp8_gemm_nt_contiguous(
                    (a2q, a2q_scale),
                    (w2, self.w2_scale),
                    mm2_out,
                    expert_ids,
                    **gemm_kwargs,
                )

        if apply_router_weight_on_input:
            topk_weights = torch.ones_like(topk_weights)

        deepgemm_unpermute_and_reduce(
            a=mm2_out,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            inv_perm=inv_perm,
            expert_map=expert_map,
            output=output,
        )


class DeepGemmFP4Experts(mk.FusedMoEExpertsModular):
    """DeepGemm-based fused MoE expert implementation for FP4 weights.

    Uses m_grouped_fp8_fp4_gemm_nt_contiguous with FP8 activations and
    MXFP4 (FP4 E2M1 packed as uint8) weights. Requires Blackwell-family
    GPUs (SM100 datacenter or SM120 consumer).
    """

    # FP8 activation block size (hardcoded since mxfp4_w4a8 quant config
    # does not set a block_shape on the activation descriptor).
    _ACT_BLOCK_K = 128
    # FP4 weight block size
    _WEIGHT_BLOCK_K = 32

    def __init__(self, moe_config: FusedMoEConfig, quant_config: FusedMoEQuantConfig):
        super().__init__(moe_config=moe_config, quant_config=quant_config)
        assert quant_config.weight_quant_dtype == "mxfp4"
        assert not quant_config.per_act_token_quant
        assert not quant_config.per_out_ch_quant

        self.gemm1_clamp_limit = quant_config.gemm1_clamp_limit

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        from vllm.platforms import current_platform

        return is_deep_gemm_supported() and (
            current_platform.is_device_capability_family(100)
            or current_platform.is_device_capability_family(120)
        )

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        SUPPORTED_W_A = [
            (kMxfp4Static, kFp8Dynamic128Sym),
        ]
        return (weight_key, activation_key) in SUPPORTED_W_A

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in [MoEActivation.SILU, MoEActivation.SWIGLUSTEP]

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        return not (
            moe_parallel_config.use_fi_nvl_two_sided_kernels
            or moe_parallel_config.use_fi_nvl_one_sided_kernels
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
        block_m = get_mk_alignment_for_contiguous_layout()[0]
        M_sum, align_used = compute_aligned_M_and_alignment(
            M, topk, local_num_experts, block_m, expert_tokens_meta
        )
        assert M_sum % align_used == 0

        activation_out_dim = self.adjust_N_for_activation(N, activation)
        workspace1 = (M_sum, max(activation_out_dim, K))
        workspace2 = (M_sum, max(N, K))
        output = (M, K)
        return (workspace1, workspace2, output)

    def _act_mul_quant(
        self, input: torch.Tensor, output: torch.Tensor, activation: MoEActivation
    ) -> tuple[torch.Tensor, torch.Tensor]:
        block_k = self._ACT_BLOCK_K
        scale_fmt = DeepGemmQuantScaleFMT.from_oracle()

        M_sum, N = input.size()
        activation_out_dim = self.adjust_N_for_activation(N, activation)

        if scale_fmt == DeepGemmQuantScaleFMT.UE8M0:
            assert activation == MoEActivation.SILU
            return fused_silu_mul_fp8_quant_packed(
                input=input,
                output_q=output,
                group_size=block_k,
                clamp_limit=self.gemm1_clamp_limit,
            )

        if activation == MoEActivation.SILU:
            use_ue8m0 = scale_fmt == DeepGemmQuantScaleFMT.FLOAT32_CEIL_UE8M0
            return silu_mul_per_token_group_quant_fp8_colmajor(
                input=input,
                output=output,
                use_ue8m0=use_ue8m0,
                clamp_limit=self.gemm1_clamp_limit,
            )

        act_out = torch.empty(
            (M_sum, activation_out_dim), dtype=input.dtype, device=input.device
        )
        self.activation(activation, act_out, input)
        return per_token_group_quant_fp8(
            act_out, block_k, column_major_scales=True, out_q=output
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
    ):
        assert a1q_scale is not None
        assert a2_scale is None
        assert self.w1_scale is not None
        assert self.w2_scale is not None

        a1q = hidden_states
        _, N, _ = w1.size()
        # K comes from activations (full hidden dim), not from w1 which is
        # packed FP4 (E, N, K//2).
        K = a1q.size(1)

        local_num_experts = w1.size(0)
        if global_num_experts == -1:
            global_num_experts = local_num_experts

        M_sum, _ = compute_aligned_M_and_alignment(
            M=topk_ids.size(0),
            num_topk=topk_ids.size(1),
            local_num_experts=local_num_experts,
            alignment=get_mk_alignment_for_contiguous_layout()[0],
            expert_tokens_meta=expert_tokens_meta,
        )

        a1q_perm = _resize_cache(
            workspace13.view(dtype=torch.float8_e4m3fn), (M_sum, K)
        )
        a1q, a1q_scale, expert_ids, inv_perm, align_used = deepgemm_moe_permute(
            aq=a1q,
            aq_scale=a1q_scale,
            topk_ids=topk_ids,
            local_num_experts=local_num_experts,
            expert_map=expert_map,
            expert_tokens_meta=expert_tokens_meta,
            aq_out=a1q_perm,
        )
        assert a1q.size(0) == M_sum

        # Cap DG's BLOCK_M heuristic at the workspace's per-expert alignment;
        # see DeepGemmExperts.apply for rationale.
        with mk_alignment_scope(align_used):
            # FC1: FP8 activations x FP4 weights
            # DeepGEMM 2.4.2 requires FP4-packed weights as int8 (kPackedFP4).
            mm1_out = _resize_cache(workspace2, (M_sum, N))
            m_grouped_fp8_fp4_gemm_nt_contiguous(
                (a1q, a1q_scale),
                (w1.view(torch.int8), self.w1_scale),
                mm1_out,
                expert_ids,
                recipe_a=(1, self._ACT_BLOCK_K),
                recipe_b=(1, self._WEIGHT_BLOCK_K),
            )

            # SwiGLU activation + FP8 requant
            activation_out_dim = self.adjust_N_for_activation(N, activation)
            quant_out = _resize_cache(
                workspace13.view(dtype=torch.float8_e4m3fn), (M_sum, activation_out_dim)
            )
            a2q, a2q_scale = self._act_mul_quant(
                input=mm1_out.view(-1, N), output=quant_out, activation=activation
            )

            # FC2: FP8 activations x FP4 weights
            mm2_out = _resize_cache(workspace2, (M_sum, K))
            m_grouped_fp8_fp4_gemm_nt_contiguous(
                (a2q, a2q_scale),
                (w2.view(torch.int8), self.w2_scale),
                mm2_out,
                expert_ids,
                recipe_a=(1, self._ACT_BLOCK_K),
                recipe_b=(1, self._WEIGHT_BLOCK_K),
            )

        if apply_router_weight_on_input:
            topk_weights = torch.ones_like(topk_weights)

        deepgemm_unpermute_and_reduce(
            a=mm2_out,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            inv_perm=inv_perm,
            expert_map=expert_map,
            output=output,
        )
