# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""HCU-owned helpers for worker framework adapters."""

from __future__ import annotations

import functools
import importlib
import inspect
from contextlib import contextmanager


_SPLIT_GROUP_COMPAT_MARKER = "_vllm_hcu_backend_compat"


def install_split_group_backend_compat(distributed: object) -> bool:
    """Adapt vLLM's split_group call to Torch versions without ``backend``.

    Torch 2.11 removed the keyword while vLLM 0.25.1 still supplies it.  The
    vLLM split-group path creates a device-bound parent process group with
    both Gloo and NCCL backends, so children can inherit those backends when
    the obsolete selector is omitted.
    """

    split_group = getattr(distributed, "split_group")
    if getattr(split_group, _SPLIT_GROUP_COMPAT_MARKER, False):
        return False
    try:
        parameters = inspect.signature(split_group).parameters
    except (TypeError, ValueError):
        return False
    if "backend" in parameters:
        return False

    @functools.wraps(split_group)
    def split_group_compat(*args, backend=None, **kwargs):
        del backend
        return split_group(*args, **kwargs)

    setattr(split_group_compat, _SPLIT_GROUP_COMPAT_MARKER, True)
    setattr(distributed, "split_group", split_group_compat)
    return True


@contextmanager
def suppress_pp_v2_warmup_sample_broadcast(model_runner: object):
    """Skip synthetic sampled-token broadcasts during PP+spec warmup.

    MRV2 warmup drives each PP rank with synthetic scheduler state.  With
    speculative decoding, that state can make a non-last rank enqueue a
    sampled-token receive when the last rank skips the matching broadcast.
    The unmatched side-stream collective survives warmup and stalls the first
    real request.  No rank consumes warmup samples, so suppress both sides and
    retain the terminal device sync as a check that all real work completed.
    """

    pp_handler = getattr(model_runner, "pp_handler")
    receive = getattr(pp_handler, "receive")
    broadcast = getattr(pp_handler, "broadcast")
    suppress_attr = "_vllm_hcu_suppress_pp_spec_draft_sync"
    missing = object()
    previous_suppress = getattr(model_runner, suppress_attr, missing)
    setattr(model_runner, suppress_attr, True)
    setattr(pp_handler, "receive", lambda *args, **kwargs: False)
    setattr(pp_handler, "broadcast", lambda *args, **kwargs: None)
    try:
        yield
    finally:
        setattr(pp_handler, "receive", receive)
        setattr(pp_handler, "broadcast", broadcast)
        if previous_suppress is missing:
            delattr(model_runner, suppress_attr)
        else:
            setattr(model_runner, suppress_attr, previous_suppress)


def share_eagle_topk_buffer(target_model: object, eagle_model: object) -> object:
    target_language_model = (
        target_model.get_language_model()
        if hasattr(target_model, "get_language_model")
        else target_model
    )
    target_inner = getattr(target_language_model, "model", None)
    draft_inner = getattr(eagle_model, "model", None)
    if (
        target_inner is None
        or draft_inner is None
        or not hasattr(target_inner, "topk_indices_buffer")
    ):
        return eagle_model
    target_buffer = target_inner.topk_indices_buffer
    if target_buffer is None:
        return eagle_model
    set_topk_indices_buffer = getattr(
        eagle_model, "set_topk_indices_buffer", None
    )
    if callable(set_topk_indices_buffer):
        set_topk_indices_buffer(target_buffer)
    for _, child in draft_inner.named_modules():
        if hasattr(child, "topk_indices_buffer"):
            child.topk_indices_buffer = target_buffer
    return eagle_model


def deep_gemm_has_sms_api(module: object) -> bool:
    if not module.has_deep_gemm():
        return False
    try:
        deep_gemm = importlib.import_module("deep_gemm")
    except (ImportError, RuntimeError):
        return False
    return callable(getattr(deep_gemm, "set_num_sms", None)) and callable(
        getattr(deep_gemm, "get_num_sms", None)
    )


def create_sm_control_context_without_compute(module: object, vllm_config: object):
    comm_sms = int(module.envs.VLLM_DBO_COMM_SMS)
    set_comm_sms = lambda sms: None
    if vllm_config.parallel_config.enable_expert_parallel:
        ep_group = module.get_ep_group()
        device_communicator = ep_group.device_communicator
        all2all_manager = (
            None
            if device_communicator is None
            else device_communicator.all2all_manager
        )
        if all2all_manager is not None:
            max_sms_used = all2all_manager.max_sms_used()
            if max_sms_used is not None:
                comm_sms = min(comm_sms, max_sms_used)
        if comm_sms > 0 and all2all_manager is not None:
            set_comm_sms = lambda sms: all2all_manager.set_num_sms(sms)
    return module.SMControlContextManager(
        comm_sms=comm_sms,
        set_comm_sms=set_comm_sms,
        set_compute_sms=lambda sms: None,
    )


def maybe_create_ubatch_slices(
    module: object,
    should_ubatch: bool,
    num_scheduled_tokens: object,
    num_tokens_padded: int,
    num_reqs_padded: int,
    num_ubatches: int,
    split_point: list[int] | int | None = None,
):
    if not should_ubatch:
        return None, None
    num_tokens_padded = int(num_tokens_padded)
    num_reqs_padded = int(num_reqs_padded)
    num_ubatches = int(num_ubatches)
    if num_ubatches <= 0:
        raise ValueError("num_ubatches must be positive")
    if split_point is None:
        split_point = num_tokens_padded // num_ubatches
    if isinstance(split_point, list):
        token_split_points = [int(point) for point in split_point]
        if len(token_split_points) != num_ubatches - 1:
            raise ValueError("split_point list must contain num_ubatches - 1 entries")
    else:
        split = int(split_point)
        token_split_points = [split * index for index in range(1, num_ubatches)]
    if token_split_points != sorted(token_split_points):
        raise ValueError("ubatch split points must be monotonically increasing")
    if any(point < 0 or point > num_tokens_padded for point in token_split_points):
        raise ValueError("ubatch split points must be within the padded token range")

    cu_num_tokens = module.np.zeros(
        len(num_scheduled_tokens) + 1, dtype=module.np.int32
    )
    module.np.cumsum(
        num_scheduled_tokens, dtype=module.np.int32, out=cu_num_tokens[1:]
    )
    ubatch_slices = []
    start_token = 0
    for end_token in token_split_points + [int(cu_num_tokens[-1])]:
        end_token = int(end_token)
        token_slice = slice(start_token, end_token)
        req_start = int(
            module.np.searchsorted(cu_num_tokens, start_token, side="right") - 1
        )
        req_stop = int(
            module.np.searchsorted(cu_num_tokens, end_token, side="left")
        )
        ubatch_slices.append(
            module.UBatchSlice(slice(req_start, req_stop), token_slice)
        )
        start_token = end_token

    padded = module._pad_out_ubatch_slices(
        ubatch_slices, num_tokens_padded, num_reqs_padded
    )
    assert sum(item.num_tokens for item in padded) == num_tokens_padded
    return ubatch_slices, padded


__all__ = [
    "create_sm_control_context_without_compute",
    "deep_gemm_has_sms_api",
    "install_split_group_backend_compat",
    "maybe_create_ubatch_slices",
    "share_eagle_topk_buffer",
    "suppress_pp_v2_warmup_sample_broadcast",
]
