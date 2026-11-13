# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Opt-in token partition for replicated TP inputs to the Step5 LL transport.

Attention and the TP-sharded shared expert keep their ordinary full-token
inputs. Only routed MoE tokens are split, then gathered after LL combine.
No model weights, routing decisions, or speculative sampling are changed.
"""
from __future__ import annotations

import torch


class Step5LLTPTokenBuffer:
    def __init__(self, buffer, max_tokens_per_rank: int, hidden_size: int):
        from vllm.distributed import get_tp_group

        self._buffer = buffer
        self._tp = get_tp_group()
        self._size = self._tp.world_size
        self._rank = self._tp.rank_in_group
        if self._size != 4:
            raise RuntimeError("Step5 LL token partition requires TP4")
        # Allocate before the first asynchronous transport. One buffer is
        # sufficient for this DBO-disabled, non-overlapping expert schedule.
        self._local_output = torch.empty(
            (max_tokens_per_rank, hidden_size),
            dtype=torch.bfloat16,
            device=torch.cuda.current_device(),
        )

    def __getattr__(self, name):
        return getattr(self._buffer, name)

    def low_latency_dispatch(
        self,
        x,
        topk_idx,
        topk_weight,
        num_max_dispatch_tokens_per_rank,
        num_experts,
        quant_type=1,
        quant_group_size=0,
        fp8_round_scale=False,
        async_finish=False,
        return_recv_hook=False,
        x_scales=None,
        ll_route_hint=None,
    ):
        if not isinstance(x, torch.Tensor) or x.dtype != torch.bfloat16:
            raise RuntimeError("Step5 LL token partition expects BF16 input")
        n = x.shape[0]
        # Uneven eager tails retain the original transport. Within a TP
        # group, every rank sees the same token count and takes this branch.
        partition = n > 0 and n % self._size == 0
        if partition:
            width = n // self._size
            start = self._rank * width
            x = x[start : start + width]
            topk_idx = topk_idx[start : start + width]
            topk_weight = topk_weight[start : start + width]
            if x_scales is not None and x_scales.numel() > 1:
                x_scales = x_scales[start : start + width]
        result = self._buffer.low_latency_dispatch(
            x,
            topk_idx,
            topk_weight,
            num_max_dispatch_tokens_per_rank,
            num_experts,
            quant_type=quant_type,
            quant_group_size=quant_group_size,
            fp8_round_scale=fp8_round_scale,
            async_finish=async_finish,
            return_recv_hook=return_recv_hook,
            x_scales=x_scales,
            ll_route_hint=ll_route_hint,
        )
        received, counts, native_handle, event, hook = result
        handle = (native_handle, n if partition else None)
        return received, counts, handle, event, hook

    def low_latency_combine(self, x, topk_idx, topk_weights, handle, **kwargs):
        native_handle, n = handle
        if n is None:
            return self._buffer.low_latency_combine(
                x, topk_idx, topk_weights, native_handle, **kwargs
            )
        width = n // self._size
        start = self._rank * width
        output = kwargs.pop("out", None)
        if output is None or output.shape != (n, x.shape[-1]):
            raise RuntimeError("Step5 LL token partition requires caller-owned output")
        local = self._local_output[:width]
        _, event, recv_hook = self._buffer.low_latency_combine(
            x,
            topk_idx[start : start + width],
            topk_weights[start : start + width],
            native_handle,
            out=local,
            **kwargs,
        )

        def gather():
            if recv_hook is not None:
                recv_hook()
            gathered = self._tp.all_gather(local, dim=0)
            output.copy_(gathered)

        if kwargs.get("return_recv_hook", False):
            return output, event, gather
        gather()
        return output, event, None

    def get_next_low_latency_combine_buffer(self, handle):
        native_handle, _ = handle
        return self._buffer.get_next_low_latency_combine_buffer(native_handle)
