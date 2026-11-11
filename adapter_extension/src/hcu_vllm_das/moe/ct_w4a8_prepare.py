# SPDX-License-Identifier: Apache-2.0
"""Owned public DeepEP prepare adapters; official classes remain unchanged."""
from __future__ import annotations
import torch
from vllm.model_executor.layers.fused_moe import modular_kernel as mk
from vllm.model_executor.layers.fused_moe.prepare_finalize.deepep_ht import DeepEPHTPrepareAndFinalize
from vllm.model_executor.layers.fused_moe.prepare_finalize.deepep_ll import DeepEPLLPrepareAndFinalize
from vllm.model_executor.layers.fused_moe.utils import moe_kernel_quantize_input, normalize_batched_scales_shape
from vllm.v1.worker.ubatching import dbo_current_ubatch_id


def _check_quant(config):
    if config.quant_dtype != torch.int8 or not config.per_act_token_quant or config.block_shape is not None:
        raise ValueError('HCU CT W4A8 requires dynamic per-token INT8 preparation')


class HcuCTHTPrepareAndFinalize(DeepEPHTPrepareAndFinalize):
    """Inherit native transport/finalize; quantize tokens after dispatch."""
    def prepare_async(self, a1, topk_weights, topk_ids, num_experts, expert_map,
                      apply_router_weight_on_input, quant_config, defer_input_quant=False):
        _check_quant(quant_config)
        if defer_input_quant:
            raise ValueError('HCU CT experts consume quantized inputs')
        receiver = super().prepare_async(a1, topk_weights, topk_ids, num_experts,
            expert_map, apply_router_weight_on_input, quant_config, defer_input_quant=True)
        def receive():
            x, _, meta, ids, weights = receiver()
            if x.shape[0] == 0:
                q, scale = x.to(torch.int8), torch.empty((0, 1), dtype=torch.float32, device=x.device)
            else:
                q, scale = moe_kernel_quantize_input(x, None, quant_dtype=torch.int8,
                    per_act_token_quant=True, block_shape=None)
            if scale.ndim == 1:
                scale = scale.unsqueeze(-1)
            if q.dtype != torch.int8 or scale.dtype != torch.float32 or scale.device != q.device or tuple(scale.shape) != (q.shape[0], 1):
                raise RuntimeError('HCU HT did not produce INT8 and per-token scales')
            return q, scale, meta, ids, weights
        return receive


class HcuCTLLPrepareAndFinalize(DeepEPLLPrepareAndFinalize):
    """Inherit native constructor/finalize; speak the actual HCU INT8 wire ABI."""
    use_int8_dispatch = True

    def prepare_async(self, a1, topk_weights, topk_ids, num_experts, expert_map,
                      apply_router_weight_on_input, quant_config, defer_input_quant=False):
        _check_quant(quant_config)
        if defer_input_quant or self.use_fp8_dispatch or self.use_ue8m0_dispatch:
            raise ValueError('HCU W4A8 LL requires explicit INT8-only dispatch')
        if a1.ndim != 2 or a1.dtype != torch.bfloat16 or a1.shape[1] not in self.SUPPORTED_HIDDEN_SIZES:
            raise ValueError('HCU LL requires an admitted BF16 hidden size')
        if a1.shape[0] > self.max_tokens_per_rank:
            raise ValueError('HCU LL input exceeds checked token capacity')
        # v0.33 core routers emit int32 topk_ids; the legacy v0.29 plugin
        # path used int64. Accept both, reject anything else.
        if (topk_ids.dtype not in (torch.int32, torch.int64)
                or topk_weights.dtype != torch.float32
                or topk_ids.shape != topk_weights.shape
                or topk_ids.shape[0] != a1.shape[0]):
            raise ValueError('HCU LL routing IDs/weights have incompatible shape or dtype')
        weights = topk_weights
        if apply_router_weight_on_input:
            if topk_ids.shape[1] != 1:
                raise ValueError('router weighting on input requires topk=1')
            a1 = a1 * topk_weights.to(a1.dtype)
            weights = torch.ones_like(topk_weights)
        ids = topk_ids if self.global_to_physical is None else self.global_to_physical[topk_ids]
        received, counts, handle, _, hook = self.buffer.low_latency_dispatch(
            a1, ids, weights, self.max_tokens_per_rank, num_experts,
            quant_type=1, quant_group_size=0, fp8_round_scale=False,
            async_finish=False, return_recv_hook=True)
        if not callable(hook) or handle is None or not isinstance(received, tuple) or len(received) != 2:
            raise RuntimeError('HCU LL did not return quantized data, handle and recv hook')
        q, scale = received
        expected_e = num_experts // self.num_dispatchers()
        expected_t = self.max_tokens_per_rank * self.num_dispatchers()
        if q.dtype != torch.int8 or tuple(q.shape) != (expected_e, expected_t, a1.shape[1]) or q.device != a1.device:
            raise RuntimeError('HCU LL INT8 receive layout disagrees with EP capacity')
        scale = normalize_batched_scales_shape(scale, expected_e)
        if scale.dtype != torch.float32 or tuple(scale.shape) != (expected_e, expected_t, 1) or scale.device != q.device:
            raise RuntimeError('HCU LL requires FP32 batched per-token scales')
        if counts.dtype != torch.int32 or tuple(counts.shape) != (expected_e,) or counts.device != q.device:
            raise RuntimeError('HCU LL expert counts have incompatible layout')
        self.handles[dbo_current_ubatch_id()] = handle
        return hook, lambda: (q, scale, mk.ExpertTokensMetadata(counts, None), None, None)
