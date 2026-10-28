# SPDX-License-Identifier: Apache-2.0
"""Fail-fast checks for distributed EngineArgs drift.

Every vLLM rank parses its own CLI. A small difference in scheduler or cache
options can survive process-group initialization and only fail later when PP
intermediate buffers are copied. Compare the shape-affecting settings before
model construction and weight loading.
"""
from __future__ import annotations

from typing import Any


def engine_config_signature(vllm_config: Any) -> dict[str, Any]:
    """Return the rank-independent settings that affect tensor contracts."""
    model = getattr(vllm_config, "model_config", None)
    parallel = getattr(vllm_config, "parallel_config", None)
    scheduler = getattr(vllm_config, "scheduler_config", None)
    cache = getattr(vllm_config, "cache_config", None)

    def value(obj: Any, name: str, default: Any = None) -> Any:
        item = getattr(obj, name, default)
        if isinstance(item, (str, int, float, bool, type(None))):
            return item
        return str(item)

    return {
        "model": value(model, "model"),
        "revision": value(model, "revision"),
        "dtype": value(model, "dtype"),
        "quantization": value(model, "quantization"),
        "max_model_len": value(model, "max_model_len"),
        "tensor_parallel_size": value(parallel, "tensor_parallel_size"),
        "pipeline_parallel_size": value(parallel, "pipeline_parallel_size"),
        "data_parallel_size": value(parallel, "data_parallel_size"),
        "world_size": value(parallel, "world_size"),
        "distributed_executor_backend": value(
            parallel, "distributed_executor_backend"
        ),
        "max_num_seqs": value(scheduler, "max_num_seqs"),
        "max_num_batched_tokens": value(scheduler, "max_num_batched_tokens"),
        "max_num_scheduled_tokens": value(scheduler, "max_num_scheduled_tokens"),
        "cache_dtype": value(cache, "cache_dtype"),
        "block_size": value(cache, "block_size"),
        "enable_prefix_caching": value(cache, "enable_prefix_caching"),
        "use_v2_model_runner": value(vllm_config, "use_v2_model_runner"),
    }


def validate_distributed_engine_config(vllm_config: Any, dist: Any) -> None:
    """Broadcast rank 0's signature and fail before model loading on drift."""
    if not dist.is_available() or not dist.is_initialized():
        return

    local = engine_config_signature(vllm_config)
    reference = [local if dist.get_rank() == 0 else None]
    dist.broadcast_object_list(reference, src=0)
    expected = reference[0]
    if local == expected:
        return

    differing = sorted(
        name
        for name in set(local) | set(expected)
        if local.get(name) != expected.get(name)
    )
    details = ", ".join(
        f"{name}={local.get(name)!r} (rank0={expected.get(name)!r})"
        for name in differing
    )
    raise RuntimeError(
        "Distributed EngineConfig mismatch before model loading: " + details
    )


__all__ = ["engine_config_signature", "validate_distributed_engine_config"]
