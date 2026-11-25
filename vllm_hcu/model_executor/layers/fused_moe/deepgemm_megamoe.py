# SPDX-License-Identifier: Apache-2.0
"""Standard deepgemm.mega adapter; backend/layout selection belongs to DeepGEMM."""
from __future__ import annotations

import math
import torch

from vllm.config import get_current_vllm_config
from vllm.distributed import get_ep_group, get_tp_group
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import FusedMoEMethodBase


def activation_clamp(moe, layer):
    """Resolve both explicit limits and the intrinsic SWIGLUSTEP activation."""
    limit = getattr(layer, "swiglu_limit", None)
    if limit is None:
        limit = getattr(moe, "swiglu_limit", None)
    if limit is None:
        activation = getattr(moe, "activation", "silu")
        activation = str(getattr(activation, "value", activation)).lower()
        if activation in ("swiglustep", "swiglu_step"):
            limit = 7.0
    return float("inf") if limit is None else float(limit)


def select_method(layer, quant_config, moe):
    """Select Channel-FP8 by its quantization contract, never by model name."""
    if quant_config is None or quant_config.get_name() not in (
        "compressed-tensors", "slimquant_compressed_tensors_marlin",
    ):
        return None
    from compressed_tensors.quantization import QuantizationStrategy
    from vllm.model_executor.layers.quantization.compressed_tensors.utils import should_ignore_layer
    if should_ignore_layer(layer.layer_name, ignore=quant_config.ignore,
                          fused_mapping=quant_config.packed_modules_mapping, use_fnmatch=True):
        raise ValueError("deepgemm.mega requires quantized expert projections")
    quant_config._add_fused_moe_to_target_scheme_map()
    schemes = [quant_config.get_scheme_dict(layer, layer.layer_name + f".0.{name}_proj")
               for name in ("gate", "up", "down")]
    if schemes[0] is None or any(value != schemes[0] for value in schemes[1:]):
        raise ValueError("deepgemm.mega requires matching gate/up/down quantization")
    weights, activations = schemes[0].get("weights"), schemes[0].get("input_activations")
    if not (quant_config._is_fp8_w8a8(weights, activations)
            and weights.strategy == QuantizationStrategy.CHANNEL
            and activations.strategy == QuantizationStrategy.TOKEN
            and activations.dynamic and weights.symmetric and activations.symmetric
            and not weights.dynamic):
        raise ValueError("deepgemm.mega requires symmetric Channel-FP8/dynamic-token FP8")
    config = get_current_vllm_config()
    parallel = config.parallel_config
    if not parallel.enable_expert_parallel or parallel.enable_eplb or parallel.enable_dbo:
        raise ValueError("deepgemm.mega requires EP and currently rejects EPLB/DBO")
    if moe.has_bias or moe.is_lora_enabled or moe.pcp_size > 1 or moe.is_sequence_parallel:
        raise ValueError("deepgemm.mega does not support bias, LoRA, PCP or sequence parallelism")
    if moe.in_dtype != torch.bfloat16 or moe.skip_final_all_reduce:
        raise ValueError("deepgemm.mega requires BF16 inputs and finalized routed output")
    if getattr(layer, "apply_router_weight_on_input", False):
        raise ValueError("deepgemm.mega applies routing weights after GEMM2")
    if layer.expert_map_manager.num_fused_shared_experts:
        raise ValueError("Shared experts must remain outside the routed-expert kernel")
    activation = str(getattr(moe.activation, "value", moe.activation)).lower()
    if activation not in ("silu", "swiglu", "swiglustep", "swiglu_step"):
        raise ValueError(f"deepgemm.mega does not support activation {activation}")
    return DeepGemmMegaMoEMethod(moe, weights)


def release_buffers():
    """Release standard MegaMoE buffers before distributed worker teardown."""
    try:
        ep, tp = get_ep_group(), get_tp_group()
    except AssertionError:
        return
    for group, name in ((tp, "_hcu_deepgemm_mega_gathers"),
                        (ep, "_hcu_deepgemm_mega_buffers")):
        cache = getattr(group, name, {})
        values = list(cache.values())
        cache.clear()
        for value in values:
            if name.endswith("gathers"):
                value.close()
            else:
                value.destroy()


class DeepGemmMegaMoEMethod(FusedMoEMethodBase):
    hcu_owns_dispatch = True
    hcu_output_is_reduced = True

    def __init__(self, moe, weight_quant):
        super().__init__(moe)
        self.weight_quant = weight_quant
        self.block_quant = False
        self.static_input_scales = False
        self.weight_block_size = None
        self._buffer = self._gather = None
        self._weights = None

    @property
    def topk_indices_dtype(self):
        return torch.int64

    def maybe_roundup_sizes(self, hidden_size, intermediate_size_per_partition,
                           act_dtype, moe_parallel_config):
        return hidden_size, intermediate_size_per_partition

    def create_weights(self, layer, num_experts, hidden_size,
                       intermediate_size_per_partition, params_dtype, **attrs):
        from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w8a8_fp8 import CompressedTensorsW8A8Fp8MoEMethod
        CompressedTensorsW8A8Fp8MoEMethod.create_weights(
            self, layer, num_experts, hidden_size, intermediate_size_per_partition,
            params_dtype, **attrs)

    def get_fused_moe_quant_config(self, layer):
        from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
        return FusedMoEQuantConfig.make(
            torch.float8_e4m3fn, w1_scale=layer.w13_weight_scale, w2_scale=layer.w2_weight_scale,
            per_act_token_quant=True,
            gemm1_clamp_limit=(None if math.isinf(activation_clamp(self.moe, layer))
                              else activation_clamp(self.moe, layer)))

    @torch.inference_mode()
    def process_weights_after_loading(self, layer):
        from deepgemm import mega
        ep, tp = get_ep_group(), get_tp_group()
        experts = int(self.moe.num_experts)
        hidden, intermediate = int(self.moe.hidden_dim), int(self.moe.intermediate_size_per_partition)
        if experts % ep.world_size:
            raise ValueError("deepgemm.mega requires evenly partitioned experts")
        expert_map = getattr(layer, "expert_map", None)
        if expert_map is not None:
            count = experts // ep.world_size
            expected = torch.full((experts,), -1, dtype=expert_map.dtype)
            first = ep.rank_in_group * count
            expected[first:first + count] = torch.arange(count, dtype=expert_map.dtype)
            if not torch.equal(expert_map[:experts].cpu(), expected):
                raise ValueError("deepgemm.mega requires contiguous physical expert IDs")
        if (layer.w13_weight.ndim, layer.w2_weight.ndim) != (3, 3):
            raise ValueError("MegaMoE weight preparation requires canonical [E,N,K] weights")
        if layer.w13_weight.dtype != torch.float8_e4m3fn or layer.w2_weight.dtype != torch.float8_e4m3fn:
            raise ValueError("deepgemm.mega requires E4M3FN expert weights")
        packed = mega.transform_weights_for_mega_moe(
            (layer.w13_weight, layer.w13_weight_scale.reshape(-1, 2 * intermediate).float().contiguous()),
            (layer.w2_weight, layer.w2_weight_scale.reshape(-1, hidden).float().contiguous()))
        # Drop canonical storage; keeping two complete expert copies exceeds HBM.
        layer.w13_weight.data, layer.w2_weight.data = packed[0][0], packed[1][0]
        self._weights = packed
        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        self._capacity = math.ceil(int(self.moe.max_num_tokens) / tp.world_size)
        cache = getattr(ep, "_hcu_deepgemm_mega_buffers", None)
        if cache is None:
            cache = ep._hcu_deepgemm_mega_buffers = {}
        key = (layer.w13_weight.device, experts, self.moe.experts_per_token,
               hidden, intermediate, self._capacity)
        if key not in cache:
            cache[key] = mega.get_symm_buffer_for_mega_moe(
                ep.cpu_group, experts, self._capacity, self.moe.experts_per_token,
                hidden, intermediate, use_fp8_dispatch=True, activation="swiglu",
                peer_memory_mode="ipc")
        self._buffer = cache[key]
        self._tp_size, self._tp_rank = tp.world_size, tp.rank_in_group
        if self._tp_size > 1:
            from vllm_hcu.distributed.device_communicators.ipc_all_gather import IpcAllGather
            gathers = getattr(tp, "_hcu_deepgemm_mega_gathers", None)
            if gathers is None:
                gathers = tp._hcu_deepgemm_mega_gathers = {}
            if key not in gathers:
                gathers[key] = IpcAllGather(tp.cpu_group, layer.w13_weight.device,
                                            self._capacity * hidden * 2)
            self._gather = gathers[key]

    def apply(self, layer, x, topk_weights, topk_ids,
              shared_experts=None, shared_experts_input=None):
        from deepgemm import mega
        if self._buffer is None:
            raise RuntimeError("MegaMoE weights and buffers must be initialized before capture")
        if x.ndim != 2 or x.dtype != torch.bfloat16:
            raise ValueError("deepgemm.mega requires BF16 [tokens,hidden] inputs")
        rows = x.shape[0]
        if topk_ids.shape != topk_weights.shape or topk_ids.shape != (rows, self.moe.experts_per_token):
            raise ValueError("Routing tensors must match [tokens,topk]")
        local_rows = math.ceil(rows / self._tp_size)
        if local_rows > self._capacity:
            raise ValueError("MoE batch exceeds its initialized source-token capacity")
        start = min(self._tp_rank * local_rows, rows)
        stop = min(start + local_rows, rows)
        valid = stop - start
        output = torch.empty((local_rows, x.shape[1]), dtype=x.dtype, device=x.device)
        # Decode publishes actual source counts; unused capacity is never read.
        # The original unified flow can inspect capacity rows, so clear its tail.
        if not self._buffer.decode:
            self._buffer.topk_idx[valid:].fill_(-1)
            self._buffer.topk_weights[valid:].zero_()
        mega.mega_moe_preprocess(x[start:stop], topk_ids[start:stop], topk_weights[start:stop],
                                 self._buffer, valid)
        mega.fp8_mega_moe(output[:valid], *self._weights, self._buffer,
                          activation_clamp=activation_clamp(self.moe, layer),
                          fast_math=True, tokens_all_ranks=-1)
        if valid < local_rows:
            output[valid:].zero_()
        if self._gather is not None:
            output = self._gather.gather([output], [local_rows] * self._tp_size)[0]
        return output[:rows]
