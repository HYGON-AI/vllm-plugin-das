"""Named INT8 ASM experts retaining Core CT loading and prepare/finalize."""

import torch

from vllm.model_executor.layers.fused_moe import modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import TopKWeightAndReduceNoOP
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kInt8DynamicTokenSym, kInt8StaticChannelSym,
)


def config_rejection(config):
    m = config
    if m.in_dtype not in (torch.bfloat16, torch.float16):
        return 'INT8 ASM requires BF16/FP16 inputs'
    if m.activation is not MoEActivation.SILU or not m.is_act_and_mul:
        return 'INT8 ASM currently supports gated SiLU only'
    if m.has_bias or m.is_lora_enabled or m.apply_router_weight_on_input:
        return 'INT8 ASM does not support bias, LoRA or router-weight-on-input'
    p = m.moe_parallel_config
    if p.use_ep or p.enable_eplb or p.dp_size != 1 or p.pcp_size != 1 or p.is_sequence_parallel:
        return 'INT8 ASM EP/EPLB/DP/PCP/sequence parallel is not validated'
    if m.aiter_fmoe_shared_expert_enabled:
        return 'INT8 ASM shared expert fusion is not validated'
    if (m.hidden_dim != m.hidden_dim_unpadded
            or m.intermediate_size_per_partition != m.intermediate_size_per_partition_unpadded
            or m.intermediate_pad not in (None, 0) or m.tp_shard_with_padding):
        return 'INT8 ASM padded weight storage is unsupported'
    if m.swiglu_limit not in (None, 0.0) or any(value is not None for value in (
        m.swiglu_alpha, m.swiglu_beta, m.activation_situ_beta, m.activation_situ_linear_beta,
    )):
        return 'INT8 ASM activation parameters are unsupported'
    return None


def _require_active():
    # Host check stays outside the owner lock; owner health never calls the host.
    from hcu_adapter.bridges.vllm import require_extension_healthy
    from hcu_vllm_das.int8_activation import CAPABILITY, require_published
    require_extension_healthy(CAPABILITY)
    require_published()


def convert_weights(*, w13, w2, layer=None, w13_scale=None):
    _require_active()
    for weight in (w13, w2):
        if (weight.dtype != torch.int8 or weight.ndim != 3 or not weight.is_contiguous()
                or getattr(weight, 'is_shuffled', False)
                or getattr(weight, 'weight_layout', 'raw') not in (None, 'raw')):
            raise ValueError('INT8 ASM requires canonical contiguous raw INT8 weights')
    return w13, w2


class HcuAiterInt8Experts(mk.FusedMoEExpertsModular):
    consumes_expert_mask = False

    def __init__(self, moe_config, quant_config):
        _require_active()
        reason = config_rejection(moe_config)
        if reason or not quant_config.use_int8_w8a8 or not quant_config.per_act_token_quant:
            raise ValueError(reason or 'INT8 ASM requires dynamic token INT8 W8A8')
        if any(getattr(quant_config, name) is not None for name in (
            'w1_bias', 'w2_bias', 'w1_zp', 'w2_zp', 'a1_scale', 'a2_scale',
        )):
            raise ValueError('INT8 ASM requires symmetric weights/dynamic activations without bias')
        super().__init__(moe_config, quant_config)

    @staticmethod
    def is_supported_config(cls, moe_config, weight_key, activation_key, activation_format):
        reason = config_rejection(moe_config)
        if reason:
            return False, reason
        return mk.FusedMoEExperts.is_supported_config(
            cls, moe_config, weight_key, activation_key, activation_format,
        )

    @staticmethod
    def _supports_current_device():
        _require_active()
        from hcu_adapter.bridges.vllm import detect_hcu
        detection = detect_hcu()
        return detection.detected and detection.architecture == 'gfx936'

    @staticmethod
    def _supports_no_act_and_mul():
        return False

    @staticmethod
    def _supports_activation(activation):
        return activation is MoEActivation.SILU

    @staticmethod
    def _supports_quant_scheme(weight_key, activation_key):
        return weight_key == kInt8StaticChannelSym and activation_key == kInt8DynamicTokenSym

    @staticmethod
    def _supports_parallel_config(p):
        return (not p.use_ep and not p.enable_eplb and p.dp_size == p.pcp_size == 1
                and not p.is_sequence_parallel)

    @staticmethod
    def activation_format():
        return mk.FusedMoEActivationFormat.Standard

    @property
    def expects_unquantized_inputs(self):
        return True

    def finalize_weight_and_reduce_impl(self):
        return TopKWeightAndReduceNoOP()

    def workspace_shapes(self, M, N, K, topk, global_num_experts, local_num_experts,
                         expert_tokens_meta, activation):
        return (0,), (0,), (M, K)

    def apply(self, output, hidden_states, w1, w2, topk_weights, topk_ids, activation,
              global_num_experts, expert_map, a1q_scale, a2_scale, workspace13,
              workspace2, expert_tokens_meta, apply_router_weight_on_input):
        _require_active()
        if (expert_tokens_meta is not None or apply_router_weight_on_input
                or a1q_scale is not None or a2_scale is not None or expert_map is not None
                or global_num_experts != self.moe_config.num_experts):
            raise ValueError('INT8 ASM received an unsupported routing/quantization contract')
        from .fused_moe_aiter import apply_aiter_int8_moe
        result = apply_aiter_int8_moe(
            hidden_states=hidden_states, w1=w1, w2=w2, topk_weights=topk_weights,
            topk_ids=topk_ids, moe_config=self.moe_config, quant_config=self.quant_config,
            activation=activation, expert_map=expert_map, output_dtype=output.dtype,
            asm_only=True,
        )
        output.copy_(result)
