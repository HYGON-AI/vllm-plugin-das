# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Optional per-token INT8 transport for large HY4 PCP16 W4A8 prefills.

Core owns dispatch, receive metadata, handles and finalize. Quantize each
source row once, before replication, instead of quantizing every BF16 copy.
Small inputs retain the existing BF16 transport. Restart after changing policy.
"""
import torch
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.utils import moe_kernel_quantize_input

from .environment import _flag, custom_ops_enabled
from .hy4_ihc_launch import _device_arch
from .moe.ct_w4a8_prepare import HcuCTHTPrepareAndFinalize, _check_quant
from .quantization.hy4_ct_policy import is_hy4

MIN_TOKENS = 2048
logger = init_logger(f'vllm.{__name__}')


def _requested():
    return custom_ops_enabled() and _flag('HCU_VLLM_HY4_HT_INT8_DISPATCH', default=False)


def _supported(cfg, moe, prepare):
    if cfg is None:
        return False
    model, p = cfg.model_config, cfg.parallel_config
    actual = moe.moe_parallel_config
    return (is_hy4(model.hf_config) and not model.is_multimodal_model
        and model.dtype == torch.bfloat16 and model.enforce_eager
        and cfg.use_v2_model_runner and cfg.compilation_config.mode == 0
        and not cfg.scheduler_config.async_scheduling
        and cfg.lora_config is None
        and (p.tensor_parallel_size, p.pipeline_parallel_size,
             p.prefill_context_parallel_size, p.decode_context_parallel_size,
             p.data_parallel_size) == (1, 1, 16, 1, 1)
        and p.enable_expert_parallel and not p.enable_eplb and not p.enable_elastic_ep
        and not p.enable_dbo and p.ubatch_size in (0, 1)
        and (actual.tp_size, actual.dp_size, actual.pcp_size, actual.ep_size) == (1, 1, 16, 16)
        and actual.use_ep and actual.all2all_backend == 'deepep_high_throughput'
        and moe.hidden_dim == 6144 and moe.in_dtype == torch.bfloat16
        and moe.experts_per_token == 8 and not moe.has_bias
        and not moe.is_lora_enabled and not moe.apply_router_weight_on_input
        and prepare.num_dispatchers() == 16 and prepare.dp_size == 1
        and _device_arch() == 'gfx938')


def enabled(cfg, moe, prepare):
    return _requested() and _supported(cfg, moe, prepare)


class HY4HTInt8PrepareAndFinalize(HcuCTHTPrepareAndFinalize):
    def prepare_async(self, a1, topk_weights, topk_ids, num_experts, expert_map,
                      apply_router_weight_on_input, quant_config, defer_input_quant=False):
        if (not _requested() or a1.ndim != 2 or a1.shape[0] < MIN_TOKENS
                or a1.shape[1] != 6144 or a1.dtype != torch.bfloat16
                or not a1.is_contiguous() or apply_router_weight_on_input):
            return super().prepare_async(a1, topk_weights, topk_ids, num_experts,
                expert_map, apply_router_weight_on_input, quant_config,
                defer_input_quant=defer_input_quant)
        _check_quant(quant_config)
        if defer_input_quant:
            raise ValueError('HCU CT experts consume quantized inputs')
        q, scale = moe_kernel_quantize_input(a1, None, quant_dtype=torch.int8,
            per_act_token_quant=True, block_shape=None)
        if scale.ndim == 1:
            scale = scale.unsqueeze(-1)
        if (q.dtype != torch.int8 or scale.dtype != torch.float32
                or q.device != a1.device or scale.device != q.device
                or tuple(q.shape) != tuple(a1.shape)
                or tuple(scale.shape) != (q.shape[0], 1)):
            raise RuntimeError('HCU HT did not produce INT8 and per-token scales')
        # Native _receiver sees has_scales=True and retains the tuple. Do not
        # call the BF16 wrapper's receive closure and quantize a second time.
        # Errors propagate: retrying a failed collective can desynchronize ranks.
        return self._do_dispatch(tokens=q, token_scales=scale,
            rank_topk_ids=topk_ids, rank_topk_weights=topk_weights,
            num_experts=num_experts, a1_scale=None, quant_config=quant_config,
            defer_input_quant=True)


def select_prepare(native, cfg, moe):
    if type(native) is not HcuCTHTPrepareAndFinalize or not enabled(cfg, moe, native):
        return native
    selected = HY4HTInt8PrepareAndFinalize(native.buffer, native.num_dispatchers(),
                                         native.dp_size, native.rank_expert_offset)
    logger.info_once(f'HY4 HT INT8 dispatch selected; source tokens per rank >= {MIN_TOKENS}')
    return selected
