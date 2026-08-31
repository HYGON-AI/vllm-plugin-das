# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""HCU-owned special forward-context path for Lightly-CP and DeepEP LL."""

from __future__ import annotations

import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any


_DEEPEP_AUTO_REQUEST_PHASE: ContextVar[list[object | None] | None] = ContextVar(
    "vllm_hcu_deepep_auto_request_phase", default=None
)


@contextmanager
def deepep_auto_request_phase_scope():
    """Keep Model Runner V2 phase evidence local to one invocation."""

    token = _DEEPEP_AUTO_REQUEST_PHASE.set([None])
    try:
        yield
    finally:
        _DEEPEP_AUTO_REQUEST_PHASE.reset(token)


def set_deepep_auto_request_phase(is_prefilling: object) -> None:
    phase_holder = _DEEPEP_AUTO_REQUEST_PHASE.get()
    if phase_holder is not None:
        phase_holder[0] = is_prefilling


def get_deepep_auto_request_phase() -> object | None:
    phase_holder = _DEEPEP_AUTO_REQUEST_PHASE.get()
    return None if phase_holder is None else phase_holder[0]


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
    attn_metadata: object | None = None,
    is_prefilling: object | None = None,
) -> bool:
    """Choose LL for uniform decode and HT for prefill/mixed batches."""

    from vllm_hcu.patch.config import get_hcu_config

    if not get_hcu_config(vllm_config).deepep_auto:
        return False
    from vllm_hcu.model_executor.layers.fused_moe.prepare_finalize.deepep_auto import (
        dspark_mooncake_pd_use_low_latency,
    )

    fixed_use_low_latency = dspark_mooncake_pd_use_low_latency(vllm_config)
    if fixed_use_low_latency is not None:
        return fixed_use_low_latency
    # Backend-specific attention metadata does not consistently retain
    # CommonAttentionMetadata.is_prefilling.  Require the runner's explicit
    # per-request phase vector instead of inferring phase from query lengths.
    if is_prefilling is None:
        is_prefilling = get_deepep_auto_request_phase()
    explicit_decode = _is_explicitly_non_prefilling(is_prefilling)
    local_decode = explicit_decode and (
        bool(
            batch_descriptor is not None
            and getattr(batch_descriptor, "uniform", False)
        )
        or _attention_metadata_is_pure_spec_decode(vllm_config, attn_metadata)
    )

    parallel_config = getattr(vllm_config, "parallel_config", None)
    data_parallel_size = int(
        getattr(parallel_config, "data_parallel_size", 1)
    )
    if data_parallel_size > 1:
        # Token counts cannot distinguish a short prefill from a short decode.
        # Synchronize phase evidence so every EP participant enters the same
        # DeepEP collective; ranks with no local tokens are neutral.
        return _synchronize_deepep_auto_phase(
            vllm_config,
            local_active=int(num_tokens or 0) > 0,
            local_decode=local_decode,
        )

    # Missing descriptor/metadata is not proof of decode. HT is safe for draft
    # and profiling forwards, while a token-count guess can send short prefills
    # through the masked LL path.
    return local_decode


def _synchronize_deepep_auto_phase(
    vllm_config: object,
    *,
    local_active: bool,
    local_decode: bool,
) -> bool:
    """Return true only when every active DP rank reports pure decode."""

    import torch
    import torch.distributed as dist
    from vllm.distributed.parallel_state import get_dp_group

    parallel_config = getattr(vllm_config, "parallel_config", None)
    dp_group = get_dp_group()
    device = dp_group.device
    process_group = dp_group.device_group
    if bool(
        getattr(parallel_config, "disable_nccl_for_dp_synchronization", False)
    ):
        device = "cpu"
        process_group = dp_group.cpu_group

    evidence = torch.tensor(
        [int(local_active), int(local_active and local_decode)],
        dtype=torch.int32,
        device=device,
    )
    dist.all_reduce(evidence, group=process_group)
    active_ranks, decode_ranks = (int(value) for value in evidence.cpu().tolist())
    return active_ranks > 0 and decode_ranks == active_ranks


def _attention_metadata_is_pure_spec_decode(
    vllm_config: object,
    attn_metadata: object | None,
) -> bool:
    speculative_config = getattr(vllm_config, "speculative_config", None)
    max_decode_query = 1 + int(
        getattr(speculative_config, "num_speculative_tokens", 0) or 0
    )
    if max_decode_query <= 1 or attn_metadata is None:
        return False
    pending = [attn_metadata]
    seen: set[int] = set()
    found_attention_metadata = False
    while pending:
        metadata = pending.pop()
        identity = id(metadata)
        if identity in seen:
            continue
        seen.add(identity)
        if isinstance(metadata, dict):
            pending.extend(metadata.values())
            continue
        if isinstance(metadata, (list, tuple)):
            pending.extend(metadata)
            continue
        max_query_len = getattr(metadata, "max_query_len", None)
        max_seq_len = getattr(metadata, "max_seq_len", None)
        if max_query_len is None or max_seq_len is None:
            continue
        found_attention_metadata = True
        max_query_len = int(max_query_len)
        max_seq_len = int(max_seq_len)
        if not (
            0 < max_query_len <= max_decode_query
            and max_seq_len > max_query_len
        ):
            return False
    return found_attention_metadata


def _is_explicitly_non_prefilling(value: object | None) -> bool:
    """Return true only for non-empty phase evidence containing only false."""

    if value is None:
        return False
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            value = tolist()
        except (TypeError, ValueError):
            return False
    if isinstance(value, (list, tuple)):
        return bool(value) and all(
            _is_explicitly_non_prefilling(item) for item in value
        )
    return isinstance(value, bool) and not value


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
