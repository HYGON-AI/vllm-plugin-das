# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3 low-latency W4A8 DeepEP expert specialization."""

from __future__ import annotations

import torch
from vllm.model_executor.layers.fused_moe.utils import _resize_cache

from .slimquant_w4a8_deepgemm_runtime import (
    DeepEPDeepGemmW4A8BatchedExperts,
    _validate_w4a8_channel_weights,
    pack_w4a8_moe_hipc_weight,
    view_w4a8_moe_hipc_weight_n32_layout,
)


class KimiK3LLExperts(DeepEPDeepGemmW4A8BatchedExperts):
    """Masked N32 HIPC experts with Kimi's SiTU activation and ownership."""

    _PACKED_IDS = "_kimi_k3_ll_packed_ids"

    def __init__(self, moe_config, quant_config, max_num_tokens, num_dispatchers):
        super().__init__(moe_config, quant_config, max_num_tokens, num_dispatchers)
        self._situ_beta = getattr(moe_config, "activation_situ_beta", None)
        self._situ_linear_beta = getattr(
            moe_config, "activation_situ_linear_beta", None
        )
        if self._situ_beta is None or self._situ_linear_beta is None:
            raise ValueError("Kimi-K3 LL requires both SiTU beta values")

    def process_weights_after_loading(self, layer) -> None:
        """Replace canonical weights once: Kimi LL never consumes them again."""

        # With EP enabled each rank owns complete experts. The config's
        # intermediate_size_per_partition_unpadded may still describe TP
        # sharding; per-channel scales describe the actual loaded matrices.
        self._hcu_logical_n = layer.w13_weight_scale.size(1)
        self._hcu_logical_k = layer.w2_weight_scale.size(1)
        installed = getattr(layer, self._PACKED_IDS, None)
        if installed is None:
            _validate_w4a8_channel_weights(layer)
            with torch.no_grad():
                w13 = view_w4a8_moe_hipc_weight_n32_layout(
                    pack_w4a8_moe_hipc_weight(layer.w13_weight.detach())
                )
                w2 = view_w4a8_moe_hipc_weight_n32_layout(
                    pack_w4a8_moe_hipc_weight(layer.w2_weight.detach())
                )
            layer.w13_weight = torch.nn.Parameter(w13, requires_grad=False)
            layer.w2_weight = torch.nn.Parameter(w2, requires_grad=False)
            installed = (id(layer.w13_weight), id(layer.w2_weight))
            setattr(layer, self._PACKED_IDS, installed)
        elif installed != (id(layer.w13_weight), id(layer.w2_weight)):
            raise RuntimeError("Kimi LL weights replaced after HIPC packing; reload unsupported")
        self._deepgemm_w13 = layer.w13_weight
        self._deepgemm_w2 = layer.w2_weight

    def apply(self, *args, **kwargs) -> None:
        """Run masked HIPC GEMMs with native EP SiTU and one scale correction."""

        names = (
            "output", "hidden_states", "w1", "w2", "topk_weights", "topk_ids",
            "activation", "global_num_experts", "expert_map", "a1q_scale",
            "a2_scale", "workspace13", "workspace2", "expert_tokens_meta",
            "apply_router_weight_on_input",
        )
        values = dict(zip(names, args))
        values.update(kwargs)
        activation = values["activation"]
        if getattr(activation, "value", activation) != "situ":
            raise ValueError("Kimi-K3 LL requires SiTU activation")
        expert_tokens_meta = values["expert_tokens_meta"]
        if expert_tokens_meta is None:
            raise RuntimeError("Kimi-K3 LL requires expert token metadata")
        hidden_states = values["hidden_states"]
        if hidden_states.ndim != 3:
            raise ValueError("Kimi-K3 LL expects [experts, tokens, hidden] input")
        if hidden_states.size(1) == 0:
            return
        if self._deepgemm_w13 is None or self._deepgemm_w2 is None:
            raise RuntimeError("Kimi-K3 LL weights were not packed before apply")
        if self.w1_scale is None or self.w2_scale is None:
            raise RuntimeError("Kimi-K3 LL requires W4A8 weight scales")

        from deepgemm import m_grouped_w4a8_gemm_nt_masked_hipc
        from lightop.activation import fuse_situ_mul_quant_ep

        topk_ids = values["topk_ids"]
        experts, max_tokens, logical_n, _, _ = self.moe_problem_size(
            hidden_states, values["w1"], values["w2"], topk_ids
        )
        logical_n = self._hcu_logical_n
        workspace = values["workspace13"]
        workspace1 = _resize_cache(workspace, (experts, max_tokens, logical_n))
        expected_m = self.estimate_expected_m(
            values["global_num_experts"], max_tokens, topk_ids.size(-1)
        )
        counts = expert_tokens_meta.expert_num_tokens
        m_grouped_w4a8_gemm_nt_masked_hipc(
            (hidden_states, values["a1q_scale"]),
            (self._deepgemm_w13, self.w1_scale * 16.0),
            workspace1,
            counts,
            expected_m,
        )
        a2q, a2q_scale = fuse_situ_mul_quant_ep(
            workspace1,
            counts,
            situ_beta=float(self._situ_beta),
            situ_linear_beta=float(self._situ_linear_beta),
            expect_m=expected_m,
        )
        m_grouped_w4a8_gemm_nt_masked_hipc(
            (a2q, a2q_scale),
            (self._deepgemm_w2, self.w2_scale * 16.0),
            values["output"],
            counts,
            expected_m,
        )
