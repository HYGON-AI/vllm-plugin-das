# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3 SiTU specialization of the HCU contiguous HIPC expert."""
import torch
from .slimquant_w4a8_deepgemm_runtime import (
    DeepEPDeepGemmW4A8ContiguousExperts,
    _validate_w4a8_channel_weights,
    pack_w4a8_moe_hipc_weight,
)


class KimiK3HTExperts(DeepEPDeepGemmW4A8ContiguousExperts):
    _CACHE_PREFIX = "_kimi_k3_w4a8_ht"

    def process_weights_after_loading(self, layer):
        # Kimi uses only HIPC after loading. Retaining canonical weights plus
        # packed clones doubles 93 layers of routed weights and exceeds HBM.
        installed = getattr(layer, "_kimi_ht_packed_ids", None)
        if installed is None:
            _validate_w4a8_channel_weights(layer)
            with torch.no_grad():
                w13 = pack_w4a8_moe_hipc_weight(layer.w13_weight.detach())
                w2 = pack_w4a8_moe_hipc_weight(layer.w2_weight.detach())
            layer.w13_weight = torch.nn.Parameter(w13, requires_grad=False)
            layer.w2_weight = torch.nn.Parameter(w2, requires_grad=False)
            layer._kimi_ht_packed_ids = (id(layer.w13_weight), id(layer.w2_weight))
        elif installed != (id(layer.w13_weight), id(layer.w2_weight)):
            raise RuntimeError("Kimi HT weights replaced after HIPC packing; reload unsupported")
        self._deepgemm_w13 = layer.w13_weight
        self._deepgemm_w2 = layer.w2_weight

    def __init__(self, moe_config, quant_config):
        super().__init__(moe_config, quant_config)
        self._situ_beta = getattr(moe_config, "activation_situ_beta", None)
        self._situ_linear_beta = getattr(moe_config, "activation_situ_linear_beta", None)
        if self._situ_beta is None or self._situ_linear_beta is None:
            raise ValueError("Kimi-K3 HT requires both SiTU beta and linear_beta")
        # Resolve required APIs before allocating packed expert weights.
        from .kimi_k3_situ_quant import resolve_situ_quant
        from vllm.logger import init_logger

        self._situ_backend, fuse_situ_mul_quant = resolve_situ_quant()
        from deepgemm import pack_w4a8_moe_hipc_weight
        from deepgemm import m_grouped_w4a8_gemm_nt_contiguous_hipc

        if not all(callable(op) for op in (
            fuse_situ_mul_quant, pack_w4a8_moe_hipc_weight,
            m_grouped_w4a8_gemm_nt_contiguous_hipc,
        )):
            raise RuntimeError("Kimi-K3 HT requires callable DeepGEMM/LightOp APIs")
        self._situ_quant = fuse_situ_mul_quant
        init_logger(__name__).info_once(
            "Kimi-K3 HT SiTU quant backend=%s (DeepEP/DeepGEMM unchanged)",
            self._situ_backend,
        )

    def _validate_activation(self, activation):
        if getattr(activation, "value", activation) != "situ":
            raise ValueError("Kimi-K3 HT requires SiTU activation")

    @staticmethod
    def adjust_N_for_activation(N, activation):
        if getattr(activation, "value", activation) != "situ":
            raise ValueError("Kimi-K3 HT requires SiTU activation")
        return N // 2

    def _quantize_activation(self, gateup_output, quant_output, m_indices):
        return self._situ_quant(
            gateup_output, beta=float(self._situ_beta),
            linear_beta=float(self._situ_linear_beta),
        )

    def _permute_scale_kwargs(self, hidden_size):
        # One INT8 scale per token, not one scale per 128 hidden elements.
        return {"block_size": hidden_size}
