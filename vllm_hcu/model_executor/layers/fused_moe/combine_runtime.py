# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Optional HCU EP combine ownership for naive DP/EP finalize."""

_combine_hook = None


def set_finalize_combine_hook(hook):
    global _combine_hook
    if hook is not None and not callable(hook):
        raise TypeError("The MoE combine hook must be callable or None")
    _combine_hook = hook


def finalize(
    self,
    output,
    fused_expert_output,
    topk_weights,
    topk_ids,
    apply_router_weight_on_input,
    weight_and_reduce_impl,
):
    from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
        TopKWeightAndReduceContiguous,
        TopKWeightAndReduceDelegate,
    )
    from vllm.distributed.parallel_state import get_ep_group

    if isinstance(weight_and_reduce_impl, TopKWeightAndReduceDelegate):
        weight_and_reduce_impl = TopKWeightAndReduceContiguous()
    out = weight_and_reduce_impl.apply(
        None, fused_expert_output, topk_weights, topk_ids, apply_router_weight_on_input
    )
    hook = _combine_hook
    if hook is not None and not self.is_sequence_parallel:
        combined = hook(out)
        if combined is not None:
            if combined.shape != output.shape or combined.dtype != output.dtype:
                raise ValueError("The HCU combine hook returned an incompatible tensor")
            return combined
    output.copy_(
        get_ep_group().combine(out, is_sequence_parallel=self.is_sequence_parallel)
    )
    return None
