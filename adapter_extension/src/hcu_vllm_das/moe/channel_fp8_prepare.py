# SPDX-License-Identifier: Apache-2.0
"""Validation and preparation for Channel-FP8 token activations."""
from __future__ import annotations

import inspect

import torch
from vllm.distributed import get_ep_group
from vllm.model_executor.layers.fused_moe import modular_kernel as mk
from vllm.model_executor.layers.fused_moe.all2all_utils import (
    get_ep_all2all_manager,
    maybe_make_prepare_finalize,
)
from vllm.model_executor.layers.fused_moe.prepare_finalize.deepep_ht import (
    DeepEPHTPrepareAndFinalize,
)
from vllm.model_executor.layers.fused_moe.prepare_finalize.deepep_ll import (
    DeepEPLLPrepareAndFinalize,
)
from vllm.model_executor.layers.fused_moe.utils import (
    moe_kernel_quantize_input,
    normalize_batched_scales_shape,
)
from vllm.platforms import current_platform
from vllm.v1.worker.ubatching import dbo_current_ubatch_id


class ChannelFP8PrepareError(RuntimeError):
    """The requested activation layout is outside the owned contract."""


def prepare_channel_activation(
    hidden_states: torch.Tensor,
    *,
    logical_k: int,
    padded_rows: int,
    clamp_limit: float | None = None,
    overlap: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if overlap:
        raise ChannelFP8PrepareError("Channel-FP8 activation overlap is unsupported")
    if not isinstance(hidden_states, torch.Tensor) or hidden_states.ndim != 2:
        raise ChannelFP8PrepareError("Channel-FP8 activation must be rank-2 [M,K]")
    if hidden_states.shape[1] != logical_k or logical_k <= 0:
        raise ChannelFP8PrepareError(
            f"Channel-FP8 activation K={hidden_states.shape[1]} does not match logical K={logical_k}"
        )
    rows = int(hidden_states.shape[0])
    if padded_rows < rows or padded_rows < 0:
        raise ChannelFP8PrepareError(
            f"Channel-FP8 padded rows {padded_rows} are smaller than request rows {rows}"
        )
    if not hidden_states.is_floating_point() or hidden_states.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ):
        raise ChannelFP8PrepareError("Channel-FP8 activation must be float16, bfloat16 or float32")
    values = hidden_states.float()
    if clamp_limit is not None:
        if not torch.isfinite(torch.tensor(clamp_limit)) or clamp_limit <= 0:
            raise ChannelFP8PrepareError("Channel-FP8 clamp_limit must be finite and positive")
        values = values.clamp(-float(clamp_limit), float(clamp_limit))
    if rows == 0:
        return (
            torch.empty((0, logical_k), dtype=torch.float8_e4m3fn, device=hidden_states.device),
            torch.empty((0, 1), dtype=torch.float32, device=hidden_states.device),
        )
    fp8_max = 448.0
    scale = values.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / fp8_max
    quantized = (values / scale).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
    return quantized.contiguous(), scale.to(torch.float32).contiguous()



def _check_quant_config(quant_config):
    if (quant_config.quant_dtype != torch.float8_e4m3fn
            or quant_config.block_shape is not None
            or not quant_config.per_act_token_quant
            or not quant_config.per_out_ch_quant
            or quant_config.a1_scale is not None
            or quant_config.is_scale_swizzled):
        raise ChannelFP8PrepareError(
            "Channel-FP8 requires unblocked dynamic token FP8 and channel scales"
        )


def _check_input(a1, topk_weights, topk_ids, num_experts):
    if a1.ndim != 2 or a1.dtype != torch.bfloat16:
        raise ChannelFP8PrepareError("DeepEP Channel-FP8 requires BF16 [M,K] input")
    if (topk_ids.ndim != 2 or topk_ids.shape[0] != a1.shape[0]
            or topk_weights.shape != topk_ids.shape
            # v0.33 core routers emit int32 topk_ids; accept both int32/int64.
            or topk_ids.dtype not in (torch.int32, torch.int64)
            or topk_weights.dtype != torch.float32
            or topk_weights.device != a1.device or topk_ids.device != a1.device
            or num_experts <= 0):
        raise ChannelFP8PrepareError("Channel-FP8 routing shape/dtype/device mismatch")


class HcuChannelFP8HTPrepareAndFinalize(DeepEPHTPrepareAndFinalize):
    """Preserve native HT transport and quantize tokens at the receiver.

    Native HT's non-block receiver quantizes per tensor. Deferring that step
    keeps the native event, expert-id restoration and metadata logic while
    selecting the owned per-token contract after dispatch.
    """

    def prepare_async(self, a1, topk_weights, topk_ids, num_experts, expert_map,
                      apply_router_weight_on_input, quant_config,
                      defer_input_quant=False):
        _check_quant_config(quant_config)
        _check_input(a1, topk_weights, topk_ids, num_experts)
        if defer_input_quant:
            raise ChannelFP8PrepareError("Channel-FP8 experts require quantized inputs")
        receiver = super().prepare_async(
            a1, topk_weights, topk_ids, num_experts, expert_map,
            apply_router_weight_on_input, quant_config, defer_input_quant=True,
        )

        def receive():
            values, _, metadata, ids, weights = receiver()
            if values.shape[0] == 0:
                q = values.to(torch.float8_e4m3fn)
                scale = torch.empty((0, 1), dtype=torch.float32, device=values.device)
            else:
                q, scale = moe_kernel_quantize_input(
                    values, None, quant_dtype=torch.float8_e4m3fn,
                    per_act_token_quant=True, block_shape=None,
                    is_scale_swizzled=False,
                )
            if (q.shape != values.shape or q.dtype != torch.float8_e4m3fn
                    or scale.dtype != torch.float32
                    or scale.shape != (values.shape[0], 1)
                    or not scale.is_contiguous()):
                raise ChannelFP8PrepareError("HT token quantizer returned invalid FP8/scale layout")
            return q, scale, metadata, ids, weights

        return receive


def _check_ll_buffer(buffer):
    # Validate the HCU Python wire ABI before allocating or dispatching.
    contracts = (
        ("low_latency_dispatch", (None,) * 5,
         dict(quant_type=2, quant_group_size=0, fp8_round_scale=False,
              async_finish=False, return_recv_hook=True),
         ("topk_weight", "quant_type", "quant_group_size", "fp8_round_scale")),
        ("low_latency_combine", (None,) * 4,
         dict(async_finish=False, zero_copy=False, return_recv_hook=True, out=None),
         ("topk_weights", "handle", "out")),
    )
    for name, args, kwargs, fields in contracts:
        function = getattr(buffer, name, None)
        try:
            signature = inspect.signature(function)
            if any(field not in signature.parameters for field in fields):
                raise TypeError("required HCU wire fields are absent")
            signature.bind(*args, **kwargs)
        except (TypeError, ValueError) as error:
            raise ChannelFP8PrepareError(
                f"HCU Channel-FP8 requires a compatible DeepEP {name} ABI: {error}"
            ) from error


class HcuChannelFP8LLPrepareAndFinalize(DeepEPLLPrepareAndFinalize):
    """Own HCU channel dispatch; inherit native handle slots and combine."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _check_ll_buffer(self.buffer)

    def prepare_async(self, a1, topk_weights, topk_ids, num_experts, expert_map,
                      apply_router_weight_on_input, quant_config,
                      defer_input_quant=False):
        _check_quant_config(quant_config)
        _check_input(a1, topk_weights, topk_ids, num_experts)
        if defer_input_quant:
            raise ChannelFP8PrepareError("Channel-FP8 LL cannot defer token quantization")
        if a1.shape[0] > self.max_tokens_per_rank:
            raise ChannelFP8PrepareError("Channel-FP8 LL request exceeds dispatch capacity")
        if a1.shape[1] not in self.SUPPORTED_HIDDEN_SIZES:
            raise ChannelFP8PrepareError("Channel-FP8 LL hidden size is unsupported by DeepEP")
        if apply_router_weight_on_input:
            if topk_ids.shape[1] != 1:
                raise ChannelFP8PrepareError("Channel-FP8 router-weight input requires topk=1")
            a1 = a1 * topk_weights.to(a1.dtype)
        ids = self._map_global_to_physical_ids(topk_ids)
        returned = self.buffer.low_latency_dispatch(
            a1, ids, topk_weights, self.max_tokens_per_rank, num_experts,
            quant_type=2, quant_group_size=0, fp8_round_scale=False,
            async_finish=False, return_recv_hook=True,
        )
        if not isinstance(returned, tuple) or len(returned) != 5:
            raise ChannelFP8PrepareError("HCU LL dispatch returned an invalid result tuple")
        expert_x, counts, handle, _, hook = returned
        self.handles[dbo_current_ubatch_id()] = handle
        if not callable(hook):
            raise ChannelFP8PrepareError("HCU LL dispatch did not provide a receive hook")

        def receive():
            if not isinstance(expert_x, tuple) or len(expert_x) != 2:
                raise ChannelFP8PrepareError("HCU LL dispatch must return (FP8, token scale)")
            values, scales = expert_x
            capacity = self.max_tokens_per_rank * self.num_dispatchers()
            if (values.ndim != 3 or values.dtype != torch.float8_e4m3fn
                    or values.shape[1:] != (capacity, a1.shape[1])
                    or values.device != a1.device or not values.is_contiguous()
                    or counts.dtype != torch.int32 or counts.shape != (values.shape[0],)
                    or counts.device != a1.device or not counts.is_contiguous()
                    or values.shape[0] * self.num_dispatchers() != num_experts):
                raise ChannelFP8PrepareError("HCU LL dispatch expert buffer/counts layout mismatch")
            if (scales.dtype != torch.float32 or scales.device != a1.device
                    or scales.shape not in (values.shape[:2], (*values.shape[:2], 1))
                    or not scales.is_contiguous()):
                raise ChannelFP8PrepareError("HCU LL dispatch requires contiguous FP32 token scales")
            scales = scales.view(values.shape[:2])
            metadata = mk.ExpertTokensMetadata(counts, None)
            return values, scales, metadata, None, None

        return hook, receive


def make_channel_prepare(moe, quant_config, routing_tables, config=None):
    """Use Core's manager/handle factory and reconstruct only the owned owner.

    Native constructors retain buffer lifetime and routing-table conversion;
    no process-wide factory, group or oracle is replaced.
    """
    _check_quant_config(quant_config)
    parallel = moe.moe_parallel_config
    if parallel.all2all_backend == "allgather_reducescatter":
        from hcu_vllm_das.dspark_agrs import make_channel_draft_prepare
        from vllm.config import get_current_vllm_config

        cfg = config if config is not None else get_current_vllm_config()
        return make_channel_draft_prepare(moe, cfg)
    if bool(parallel.use_deepep_ht_kernels) == bool(parallel.use_deepep_ll_kernels):
        raise ChannelFP8PrepareError("Channel-FP8 requires exactly one DeepEP HT/LL route")
    manager = get_ep_all2all_manager()
    if (manager.world_size != parallel.ep_size
            or manager.dp_world_size != parallel.dp_size
            or moe.num_local_experts * manager.world_size != moe.num_experts):
        raise ChannelFP8PrepareError("Channel-FP8 EP membership/expert capacity mismatch")
    native = maybe_make_prepare_finalize(
        moe, quant_config, routing_tables=routing_tables,
        allow_new_interface=True, all2all_manager=manager,
    )
    if parallel.use_deepep_ht_kernels:
        if not isinstance(native, DeepEPHTPrepareAndFinalize):
            raise ChannelFP8PrepareError("Core factory returned an incompatible HT owner")
        return HcuChannelFP8HTPrepareAndFinalize(
            native.buffer, native.num_dispatchers(), native.dp_size,
            native.rank_expert_offset,
        )
    if not isinstance(native, DeepEPLLPrepareAndFinalize):
        raise ChannelFP8PrepareError("Core factory returned an incompatible LL owner")
    return HcuChannelFP8LLPrepareAndFinalize(
        native.buffer, native.max_tokens_per_rank, native.num_dispatchers(),
        use_fp8_dispatch=True, global_to_physical=native.global_to_physical,
        physical_to_global=native.physical_to_global,
        local_expert_global_ids=native.local_expert_global_ids,
    )


__all__ = [
    "ChannelFP8PrepareError", "prepare_channel_activation",
    "HcuChannelFP8HTPrepareAndFinalize", "HcuChannelFP8LLPrepareAndFinalize",
    "make_channel_prepare",
]
