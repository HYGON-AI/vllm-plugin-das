# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""vLLM v0.25.1 target DeepEP prepare/finalize behavior plus HCU deltas.

The adapters install these functions only after the corresponding DeepEP
module has loaded.  Keeping all vLLM/DeepEP references behind the ``module``
argument avoids importing optional communication libraries eagerly.
"""

from __future__ import annotations

import inspect


def _has_hcu_low_latency_dispatch_abi(buffer) -> bool:
    dispatch = getattr(buffer, "low_latency_dispatch", None)
    if dispatch is None:
        return False
    try:
        parameters = inspect.signature(dispatch).parameters
    except (TypeError, ValueError):
        return False
    names = tuple(parameters)
    return (
        names[:5]
        == (
            "x",
            "topk_idx",
            "topk_weight",
            "num_max_dispatch_tokens_per_rank",
            "num_experts",
        )
        and {
            "quant_type",
            "quant_group_size",
            "fp8_round_scale",
            "async_finish",
            "return_recv_hook",
        }.issubset(parameters)
        and "use_fp8" not in parameters
    )


def ht_do_dispatch(
    module,
    self,
    tokens,
    token_scales,
    rank_topk_ids,
    rank_topk_weights,
    num_experts,
    a1_scale,
    quant_config,
    defer_input_quant,
):
    has_scales = token_scales is not None
    previous_event = module.dbo_get_previous_event(self.buffer.capture)
    module.dbo_yield_and_switch_from_compute_to_comm()
    (
        num_tokens_per_rank,
        num_tokens_per_rdma_rank,
        dispatch_expert_num_tokens,
        is_token_in_rank,
        event,
    ) = self.buffer.get_dispatch_layout(
        topk_idx=rank_topk_ids,
        num_experts=num_experts,
        previous_event=previous_event,
        async_finish=False,
        allocate_on_comm_stream=False,
    )
    token_data = (tokens, token_scales) if has_scales else tokens
    expert_alignment = (
        256
        if (
            (quant_config.use_int8_w8a8 or quant_config.use_fp8_w8a8)
            and quant_config.is_per_act_token
            and not quant_config.is_block_quantized
        )
        else 1
    )
    (
        token_data,
        expert_topk_ids,
        expert_topk_weights,
        expert_num_tokens_per_expert_list,
        handle,
        event,
    ) = self.buffer.dispatch(
        x=token_data,
        handle=None,
        num_tokens_per_rank=num_tokens_per_rank,
        num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
        is_token_in_rank=is_token_in_rank,
        num_tokens_per_expert=dispatch_expert_num_tokens,
        topk_idx=rank_topk_ids,
        topk_weights=rank_topk_weights,
        expert_alignment=expert_alignment,
        config=self._get_dispatch_config(),
        previous_event=previous_event,
        async_finish=self.async_prepare and not module.dbo_enabled(),
        allocate_on_comm_stream=False,
    )
    self.handles[module.dbo_current_ubatch_id()] = handle
    module.dbo_switch_to_compute_sync()
    return lambda: self._receiver(
        event,
        has_scales,
        token_data,
        expert_topk_ids,
        num_experts,
        expert_num_tokens_per_expert_list,
        expert_topk_weights,
        a1_scale,
        quant_config,
        defer_input_quant=defer_input_quant,
    )


def ht_receiver(
    module,
    self,
    event,
    has_scales,
    token_data,
    expert_topk_ids,
    num_experts,
    expert_num_tokens_per_expert_list,
    expert_topk_weights,
    a1_scale,
    quant_config,
    defer_input_quant,
):
    if event.event is not None:
        event.current_stream_wait()
    if has_scales:
        expert_x, expert_x_scale = token_data
    else:
        expert_x, expert_x_scale = token_data, None
    if expert_topk_ids is None:
        raise RuntimeError("DeepEP HT dispatch did not return expert_topk_ids")
    expert_topk_ids = module.torch.where(
        expert_topk_ids == -1,
        num_experts - 1 if self.rank_expert_offset == 0 else 0,
        expert_topk_ids + self.rank_expert_offset,
    )
    expert_tokens_meta = module.mk.ExpertTokensMetadata.make_from_list(
        expert_num_tokens_per_expert_list,
        device=expert_x.device,
    )
    if (
        not quant_config.is_block_quantized
        and not quant_config.is_per_act_token
        and not defer_input_quant
    ):
        expert_x_scale = None
        if expert_x.numel() != 0:
            expert_x, expert_x_scale = module.moe_kernel_quantize_input(
                expert_x,
                a1_scale,
                quant_dtype=quant_config.quant_dtype,
                per_act_token_quant=False,
                block_shape=quant_config.block_shape,
                is_scale_swizzled=quant_config.is_scale_swizzled,
            )
    return (
        expert_x,
        expert_x_scale,
        expert_tokens_meta,
        expert_topk_ids,
        expert_topk_weights,
    )


def ht_prepare_async(
    module,
    self,
    a1,
    topk_weights,
    topk_ids,
    num_experts,
    expert_map,
    apply_router_weight_on_input,
    quant_config,
    defer_input_quant=False,
):
    if apply_router_weight_on_input:
        if topk_ids.size(1) != 1:
            raise ValueError("DeepEP HT apply_router_weight_on_input requires topk=1")
        a1 = a1 * topk_weights.to(a1.dtype)
    if (
        (quant_config.is_block_quantized or quant_config.is_per_act_token)
        and not defer_input_quant
    ):
        a1q, a1q_scale = module.moe_kernel_quantize_input(
            a1,
            quant_config.a1_scale,
            quant_dtype=quant_config.quant_dtype,
            per_act_token_quant=quant_config.per_act_token_quant,
            block_shape=quant_config.block_shape,
        )
        if a1q_scale is not None and a1q_scale.numel() == 1:
            a1q_scale = a1q_scale.view(1, 1)
        a1_post_scale = None
    else:
        a1q = a1
        a1q_scale = None
        a1_post_scale = (
            quant_config.a1_gscale
            if quant_config.quant_dtype == "nvfp4"
            else quant_config.a1_scale
        )
    return self._do_dispatch(
        tokens=a1q,
        token_scales=a1q_scale,
        rank_topk_ids=topk_ids,
        rank_topk_weights=topk_weights,
        num_experts=num_experts,
        a1_scale=a1_post_scale,
        quant_config=quant_config,
        defer_input_quant=defer_input_quant,
    )


def ll_init(
    original,
    self,
    buffer,
    max_tokens_per_rank,
    num_dispatchers,
    use_fp8_dispatch=False,
    global_to_physical=None,
    physical_to_global=None,
    local_expert_global_ids=None,
    use_int8_dispatch=False,
):
    if use_fp8_dispatch and use_int8_dispatch:
        raise ValueError("DeepEP LL dispatch cannot enable FP8 and INT8 simultaneously")
    original(
        self,
        buffer,
        max_tokens_per_rank,
        num_dispatchers,
        use_fp8_dispatch,
        global_to_physical,
        physical_to_global,
        local_expert_global_ids,
    )
    self.use_int8_dispatch = bool(use_int8_dispatch)
    self._hcu_low_latency_dispatch_abi = _has_hcu_low_latency_dispatch_abi(buffer)


def ll_do_quant(module, self, x, a1_dtype, quant_config, expert_num_tokens=None):
    if self.use_fp8_dispatch:
        block_k = (
            quant_config.block_shape[1]
            if quant_config.block_shape is not None
            else None
        )
        if block_k == module.DEEPEP_QUANT_BLOCK_SIZE or (
            block_k is None and quant_config.per_act_token_quant
        ):
            if not isinstance(x, tuple) or len(x) != 2:
                raise RuntimeError("DeepEP LL FP8 dispatch did not return data and scales")
            x, x_scales = x
            if block_k is None:
                x_scales = module.normalize_batched_scales_shape(x_scales, x.size(0))
            return x, x_scales
        x_fp8, x_scales = x
        x = module.dequant_fp8(x_fp8, x_scales).to(dtype=a1_dtype)
    elif self.use_int8_dispatch:
        if not isinstance(x, tuple) or len(x) != 2:
            raise RuntimeError("DeepEP LL INT8 dispatch did not return data and scales")
        return x

    q_dtype = quant_config.quant_dtype
    if q_dtype == "nvfp4" and module.envs.VLLM_DEEPEPLL_NVFP4_DISPATCH:
        if not isinstance(x, tuple):
            raise RuntimeError("DeepEP LL NVFP4 dispatch did not return data and scales")
        x_scales = x[1]
        x = x[0].permute(2, 0, 1)
        num_experts, _, _ = x.shape
    else:
        if q_dtype == "nvfp4":
            q_dtype = None
        if not isinstance(x, module.torch.Tensor):
            raise RuntimeError("DeepEP LL unquantized dispatch returned a non-tensor")
        num_experts, _, hidden_dim = x.size()
        if (
            q_dtype == module.torch.int8
            and quant_config.per_act_token_quant
            and expert_num_tokens is not None
        ):
            from vllm.model_executor.layers.fused_moe import utils as moe_utils

            try:
                x, x_scales = moe_utils._int8_quantize(
                    x,
                    quant_config.a1_scale,
                    True,
                    quant_config.block_shape,
                    expert_num_tokens=expert_num_tokens,
                )
            except TypeError as exc:
                raise RuntimeError(
                    "HCU DeepEP INT8 requires the atomic MoE utils patch with "
                    "expert_num_tokens support"
                ) from exc
        else:
            x = x.view((-1, hidden_dim))
            x, x_scales = module.moe_kernel_quantize_input(
                x,
                quant_config.a1_scale,
                q_dtype,
                quant_config.per_act_token_quant,
                quant_config.block_shape,
            )
            x = x.view((num_experts, -1, hidden_dim))
    if q_dtype is not None and q_dtype != "nvfp4":
        if x_scales is None:
            raise RuntimeError("DeepEP LL quantization did not return scales")
        x_scales = module.normalize_batched_scales_shape(x_scales, num_experts)
    return x, x_scales


def ll_receiver(
    module,
    self,
    expert_x,
    expert_num_tokens,
    a1_scale,
    a1_dtype,
    quant_config,
):
    expert_x, expert_x_scale = self._do_quant(
        expert_x,
        a1_dtype,
        quant_config,
        expert_num_tokens,
    )
    expert_tokens_meta = module.mk.ExpertTokensMetadata(
        expert_num_tokens=expert_num_tokens,
        expert_num_tokens_cpu=None,
    )
    return expert_x, expert_x_scale, expert_tokens_meta, None, None


def ll_prepare_async(
    module,
    original,
    self,
    a1,
    topk_weights,
    topk_ids,
    num_experts,
    expert_map,
    apply_router_weight_on_input,
    quant_config,
    defer_input_quant=False,
):
    from vllm_hcu.platforms import envs as henvs

    hcu_dispatch_abi = getattr(self, "_hcu_low_latency_dispatch_abi", None)
    if hcu_dispatch_abi is None:
        hcu_dispatch_abi = _has_hcu_low_latency_dispatch_abi(
            getattr(self, "buffer", None)
        )
        self._hcu_low_latency_dispatch_abi = hcu_dispatch_abi
    use_hcu_api = bool(
        hcu_dispatch_abi
        or (
            henvs.VLLM_HCU_USE_CUSTOM_OPS
            and henvs.VLLM_HCU_DPSK_V4_DEEPEP_LL_USE_HCU_DISPATCH_API
        )
    )
    if not self.use_int8_dispatch and not use_hcu_api:
        return original(
            self,
            a1,
            topk_weights,
            topk_ids,
            num_experts,
            expert_map,
            apply_router_weight_on_input,
            quant_config,
            defer_input_quant,
        )
    if defer_input_quant:
        raise NotImplementedError("HCU DeepEP LL dispatch does not support defer_input_quant")
    hidden_size = a1.size(1)
    if hidden_size not in self.SUPPORTED_HIDDEN_SIZES:
        raise ValueError(f"DeepEP LL unsupported hidden size {hidden_size}")
    if self.use_fp8_dispatch and hidden_size % 128 != 0:
        raise ValueError("DeepEP LL FP8 dispatch requires hidden size divisible by 128")

    use_nvfp4 = bool(
        quant_config.quant_dtype == "nvfp4"
        and module.envs.VLLM_DEEPEPLL_NVFP4_DISPATCH
    )
    qc_scale = quant_config.a1_gscale if use_nvfp4 else quant_config.a1_scale
    has_per_token_scales = (
        qc_scale.numel() != 1
        if qc_scale is not None
        else (
            quant_config.a2_scale.numel() != 1
            if quant_config.a2_scale is not None
            else False
        )
    )
    if not use_nvfp4 and has_per_token_scales:
        raise ValueError("DeepEP LL cannot dispatch externally supplied per-token scales")
    if apply_router_weight_on_input:
        if topk_ids.size(1) != 1:
            raise ValueError("DeepEP LL apply_router_weight_on_input requires topk=1")
        a1 = a1 * topk_weights.to(a1.dtype)

    dispatch_topk_ids = self._map_global_to_physical_ids(topk_ids)
    quant_type = (
        3
        if self.use_ue8m0_dispatch
        else 1
        if self.use_int8_dispatch
        else 2
        if self.use_fp8_dispatch
        else 0
    )
    quant_group_size = 0
    if self.use_fp8_dispatch:
        block_k = quant_config.block_shape[1] if quant_config.block_shape else None
        quant_group_size = (
            0
            if block_k is None and quant_config.per_act_token_quant
            else module.DEEPEP_QUANT_BLOCK_SIZE
        )
    try:
        if use_hcu_api:
            result = self.buffer.low_latency_dispatch(
                a1,
                dispatch_topk_ids,
                topk_weights,
                self.max_tokens_per_rank,
                num_experts,
                quant_type=quant_type,
                quant_group_size=quant_group_size,
                fp8_round_scale=self.use_ue8m0_dispatch,
                async_finish=False,
                return_recv_hook=True,
            )
        else:
            # Compatibility with the HCU DeepEP v0.15-style API used by the
            # legacy INT8 path.  This branch is entered only for explicit INT8.
            result = self.buffer.low_latency_dispatch(
                a1,
                dispatch_topk_ids,
                topk_weights,
                self.max_tokens_per_rank,
                num_experts,
                quant_type=1,
                fp8_round_scale=False,
                async_finish=False,
                return_recv_hook=True,
            )
    except (TypeError, AttributeError) as exc:
        raise RuntimeError(
            "HCU DeepEP LL dispatch was enabled, but the installed DeepEP "
            "buffer does not expose the required HCU dispatch API"
        ) from exc
    expert_x, expert_num_tokens, handle, _, hook = result
    self.handles[module.dbo_current_ubatch_id()] = handle
    return hook, lambda: self._receiver(
        expert_x,
        expert_num_tokens,
        quant_config.a1_scale,
        a1.dtype,
        quant_config,
    )


__all__ = [
    "ht_do_dispatch",
    "ht_prepare_async",
    "ht_receiver",
    "ll_do_quant",
    "ll_init",
    "ll_prepare_async",
    "ll_receiver",
]
