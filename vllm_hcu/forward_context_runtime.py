# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""HCU-owned special forward-context path for Lightly-CP and DeepEP LL."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any


def attach_hcu_context_fields(
    context: object,
    *,
    scatter_indexes_tensor: object | None,
    gather_indexes_tensor: object | None,
    enable_lightly_cp: bool,
    enable_lightly_cplb: bool,
    deepep_auto_use_low_latency: bool = False,
) -> object:
    """Keep the official dataclass identity and add process-local HCU state."""

    context.scatter_indexes_tensor = scatter_indexes_tensor
    context.gather_indexes_tensor = gather_indexes_tensor
    context.enable_lightly_cp = bool(enable_lightly_cp)
    context.enable_lightly_cplb = bool(enable_lightly_cplb)
    context.deepep_auto_use_low_latency = bool(
        deepep_auto_use_low_latency
    )
    return context


def choose_deepep_auto_low_latency(
    vllm_config: object,
    num_tokens: int | None,
    num_tokens_across_dp: object | None,
    batch_descriptor: object | None,
) -> bool:
    """Choose LL for uniform decode and HT for prefill/mixed batches."""

    from vllm_hcu.patch.config import get_hcu_config

    if not get_hcu_config(vllm_config).deepep_auto:
        return False
    if batch_descriptor is not None:
        return bool(getattr(batch_descriptor, "uniform", False))

    # Draft/profiling forwards do not always carry BatchDescriptor. Preserve
    # the bounded-token fallback at this boundary, using the target scheduler
    # and speculative configuration rather than runner globals.
    if num_tokens_across_dp is None:
        max_num_tokens = int(num_tokens or 0)
    else:
        maximum = getattr(num_tokens_across_dp, "max", None)
        if not callable(maximum):
            raise TypeError("num_tokens_across_dp must expose max()")
        value = maximum()
        item = getattr(value, "item", None)
        max_num_tokens = int(item() if callable(item) else value)
    scheduler_config = getattr(vllm_config, "scheduler_config", None)
    max_num_seqs = int(getattr(scheduler_config, "max_num_seqs", 0))
    speculative_config = getattr(vllm_config, "speculative_config", None)
    num_speculative_tokens = int(
        getattr(speculative_config, "num_speculative_tokens", 0) or 0
    )
    return max_num_tokens <= max_num_seqs * (1 + num_speculative_tokens)


@contextmanager
def set_forward_context(
    module: object,
    attn_metadata: Any,
    vllm_config: object,
    num_tokens: int | None = None,
    num_tokens_across_dp: object | None = None,
    cudagraph_runtime_mode: object | None = None,
    batch_descriptor: object | None = None,
    ubatch_slices: object | None = None,
    slot_mapping: object | None = None,
    skip_compiled: bool = False,
    is_padding: object | None = None,
    *,
    scatter_indexes_tensor: object | None = None,
    gather_indexes_tensor: object | None = None,
    enable_lightly_cp: bool = False,
    enable_lightly_cplb: bool = False,
    deepep_auto_use_low_latency: bool = False,
):
    """Mirror v0.25.1's context manager while skipping invalid DeepEP-LL DP sync."""

    if cudagraph_runtime_mode is None:
        cudagraph_runtime_mode = module.CUDAGraphMode.NONE
    need_to_track_batchsize = module.track_batchsize and attn_metadata is not None
    if need_to_track_batchsize:
        module.forward_start_time = time.perf_counter()

    dp_metadata = None
    parallel_config = vllm_config.parallel_config
    low_latency = (
        parallel_config.all2all_backend == "deepep_low_latency"
        and not getattr(parallel_config, "_vllm_hcu_deepep_auto", False)
    )
    if (
        not low_latency
        and (
            parallel_config.data_parallel_size > 1
            or parallel_config.use_sequence_parallel_moe
        )
        and parallel_config.is_moe_model is not False
        and (attn_metadata is not None or num_tokens is not None)
    ):
        if (
            num_tokens_across_dp is None
            and parallel_config.data_parallel_size > 1
        ):
            assert ubatch_slices is None
            assert num_tokens is not None
            _, num_tokens_across_dp, _ = module.coordinate_batch_across_dp(
                num_tokens_unpadded=num_tokens,
                parallel_config=parallel_config,
                allow_microbatching=False,
            )
            assert num_tokens_across_dp is not None
        elif num_tokens_across_dp is None:
            assert num_tokens is not None
            num_tokens_across_dp = module.torch.tensor(
                [num_tokens], dtype=module.torch.int32
            )
        dp_metadata = module.DPMetadata.make(
            parallel_config, num_tokens or 0, num_tokens_across_dp
        )

    if (
        cudagraph_runtime_mode != module.CUDAGraphMode.NONE
        and num_tokens is not None
    ):
        batch_descriptor = batch_descriptor or module.BatchDescriptor(
            num_tokens=num_tokens
        )

    additional_kwargs = module.current_platform.set_additional_forward_context(
        attn_metadata=attn_metadata,
        vllm_config=vllm_config,
        dp_metadata=dp_metadata,
        num_tokens=num_tokens,
        num_tokens_across_dp=num_tokens_across_dp,
        cudagraph_runtime_mode=cudagraph_runtime_mode,
        batch_descriptor=batch_descriptor,
        ubatch_slices=ubatch_slices,
    )

    forward_context = module.create_forward_context(
        attn_metadata,
        vllm_config,
        dp_metadata,
        cudagraph_runtime_mode,
        batch_descriptor,
        ubatch_slices,
        slot_mapping,
        additional_kwargs,
        skip_compiled,
        is_padding,
        scatter_indexes_tensor=scatter_indexes_tensor,
        gather_indexes_tensor=gather_indexes_tensor,
        enable_lightly_cp=enable_lightly_cp,
        enable_lightly_cplb=enable_lightly_cplb,
        deepep_auto_use_low_latency=deepep_auto_use_low_latency,
    )

    try:
        with module.override_forward_context(forward_context):
            yield
    finally:
        if need_to_track_batchsize:
            batchsize = num_tokens
            synchronize = module.current_platform.synchronize
            if synchronize is not None:
                synchronize()
            now = time.perf_counter()
            module.batchsize_forward_time[batchsize].append(
                (now - module.forward_start_time) * 1000
            )
            if now - module.last_logging_time > module.batchsize_logging_interval:
                module.last_logging_time = now
                forward_stats = []
                for batch_size, times in module.batchsize_forward_time.items():
                    if len(times) <= 1:
                        continue
                    median = module.torch.quantile(
                        module.torch.tensor(times), q=0.5
                    ).item()
                    forward_stats.append((batch_size, len(times), round(median, 2)))
                forward_stats.sort(key=lambda item: item[1], reverse=True)
                if forward_stats:
                    module.logger.info(
                        "Batchsize forward time stats "
                        "(batchsize, count, median_time(ms)): %s",
                        forward_stats,
                    )


__all__ = [
    "attach_hcu_context_fields",
    "choose_deepep_auto_low_latency",
    "set_forward_context",
]
