# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Optional HY4 HIPC scheduling and the v0.25 FP32 clamp-quant math.

The Adapter owns the activation kernel. This policy changes only an owned
expert instance; Core still owns DP agreement and its CPU expected-M estimate.
Restart the service when changing this policy.
"""
from copy import copy
import math

import torch
from vllm.config import CUDAGraphMode, get_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.platforms import current_platform

from .adapter_api import adapter_module
from .environment import _flag, custom_ops_enabled
from .moe.ct_w4a8_runtime import HcuCTMaskedExperts
from .quantization.hy4_ct_policy import is_hy4

logger = init_logger(f'vllm.{__name__}')


def _device_arch():
    if current_platform.device_name != 'hcu':
        return None
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return props.gcnArchName.split(':')[0]


def _requested():
    return custom_ops_enabled() and _flag('HCU_VLLM_HY4_LL_TUNING', default=False)


def _supported(cfg, moe, quant, prepare):
    model, p = cfg.model_config, cfg.parallel_config
    actual = moe.moe_parallel_config
    limit = quant.gemm1_clamp_limit
    return (is_hy4(model.hf_config)
        and not model.is_multimodal_model and model.dtype == torch.bfloat16
        and not model.enforce_eager and cfg.lora_config is None
        and cfg.compilation_config.mode == 0
        and cfg.compilation_config.cudagraph_mode == CUDAGraphMode.FULL
        and (p.tensor_parallel_size, p.pipeline_parallel_size,
             p.prefill_context_parallel_size, p.decode_context_parallel_size,
             p.data_parallel_size) == (1, 1, 1, 1, 16)
        and p.enable_expert_parallel and not p.enable_eplb and not p.enable_elastic_ep
        and (actual.tp_size, actual.dp_size, actual.pcp_size, actual.ep_size) == (1, 16, 1, 16)
        and actual.use_ep and actual.all2all_backend == 'deepep_low_latency'
        and (moe.num_experts, moe.num_logical_experts, moe.num_local_experts,
             moe.experts_per_token, moe.hidden_dim,
             moe.intermediate_size_per_partition) == (256, 256, 16, 8, 6144, 2048)
        and moe.in_dtype == torch.bfloat16 and moe.activation == MoEActivation.SILU
        and not moe.has_bias and not moe.is_lora_enabled
        and not moe.apply_router_weight_on_input
        and prepare.max_num_tokens_per_rank() == 32 and prepare.num_dispatchers() == 16
        and type(limit) in (int, float) and math.isfinite(limit) and limit > 0
        and _device_arch() == 'gfx938')


def enabled(cfg, moe, quant, prepare):
    return _requested() and _supported(cfg, moe, quant, prepare)


def tuned_ops(ops):
    provider = adapter_module('ct_w4a8_masked_activation')
    quantize = provider.persistent_masked_m_silu_mul_clamp_int8_quant

    def clamp_quant_ep(value, *, limit, mask_m, expect_m):
        # The grid is fixed; GPU counts control every valid row, including hot
        # experts above expected_m. No data is read back to the CPU.
        return quantize(value, mask_m, limit=limit, parallel_tokens=64)

    result = copy(ops)
    result.clamp_quant_ep = clamp_quant_ep
    return result


class TunedMaskedExperts(HcuCTMaskedExperts):
    def expected_m(self, global_num_experts, capacity, topk):
        if (global_num_experts, capacity, topk) != (256, 512, 8):
            return capacity
        from vllm.model_executor.layers.fused_moe.experts.batched_deep_gemm_moe import (
            BatchedDeepGemmExperts,
        )
        # Reuse the complete upstream fallback and CPU DP metadata semantics.
        estimate = BatchedDeepGemmExperts.estimate_expected_m(
            self, global_num_experts, capacity, topk)
        # A large floor selects oversized HIPC tiles for small decode batches.
        # Device counts still cover every live row, including hot experts.
        return min(capacity, max(32, estimate))


def make_ll_experts(moe, quant, weights, ops, prepare):
    capacity, dispatchers = prepare.max_num_tokens_per_rank(), prepare.num_dispatchers()
    if not _requested():
        return HcuCTMaskedExperts(moe, quant, weights, ops, capacity, dispatchers)
    cfg = get_current_vllm_config()
    if not _supported(cfg, moe, quant, prepare):
        return HcuCTMaskedExperts(moe, quant, weights, ops, capacity, dispatchers)
    private_ops = tuned_ops(ops)
    logger.info_once('HY4 LL HIPC tuning enabled: expected-M floor32, '
                     'persistent64, v0.25 FP32 clamp-quant math')
    return TunedMaskedExperts(moe, quant, weights, private_ops, capacity, dispatchers)
