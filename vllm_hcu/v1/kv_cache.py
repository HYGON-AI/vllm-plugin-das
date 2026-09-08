"""HCU KV-cache allocation adapters for Model Runner V2."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from typing import Any

import torch

from vllm.v1.worker.gpu.attn_utils import get_shared_kv_cache_layers
from vllm.v1.worker.utils import allocate_kv_cache, bind_kv_cache


_HCU_FLASH_ATTN_MODULE = "vllm_hcu.v1.attention.backends.flash_attn"


def _is_hcu_flash_attn_backend(backend: type[Any]) -> bool:
    return backend.__module__ == _HCU_FLASH_ATTN_MODULE


def _uses_hcu_flash_attn(attn_groups: Sequence[Sequence[Any]]) -> bool:
    return any(
        _is_hcu_flash_attn_backend(group.backend)
        for groups in attn_groups
        for group in groups
    )


def _reshape_hcu_flash_cache(
    cache: torch.Tensor,
    block_size: int,
    num_kv_heads: int,
    head_size: int,
    stride_order: tuple[int, ...] = (0, 1, 2, 3, 4),
) -> torch.Tensor:
    """Reinterpret an official fused view as the HCU varlen cache ABI."""
    expected_shape = (num_kv_heads, block_size, 2 * head_size)
    if cache.ndim != 4 or tuple(cache.shape[1:]) != expected_shape:
        raise ValueError(
            "Unexpected official FlashAttention KV-cache shape: "
            f"got {tuple(cache.shape)}, expected [B, {expected_shape}]"
        )
    logical_shape = (
        cache.shape[0],
        2,
        block_size,
        num_kv_heads,
        head_size,
    )
    if sorted(stride_order) != list(range(len(logical_shape))):
        raise ValueError(f"Invalid HCU KV-cache stride order: {stride_order}")
    physical_shape = tuple(logical_shape[index] for index in stride_order)
    physical_stride = [0] * len(physical_shape)
    running_stride = 1
    for index in range(len(physical_shape) - 1, -1, -1):
        physical_stride[index] = running_stride
        running_stride *= physical_shape[index]
    logical_stride = tuple(
        physical_stride[stride_order.index(index)]
        for index in range(len(stride_order))
    )
    page_elements = 2 * block_size * num_kv_heads * head_size
    if cache.stride(0) < page_elements:
        raise ValueError(
            "Official KV-cache block stride is smaller than one HCU page: "
            f"stride={cache.stride(0)}, page={page_elements}"
        )
    return torch.as_strided(
        cache,
        size=logical_shape,
        stride=(cache.stride(0), *logical_stride[1:]),
        storage_offset=cache.storage_offset(),
    )


def _reshape_hcu_flash_caches(
    caches: dict[str, Any],
    attn_groups: list[list[Any]],
    kernel_block_sizes: list[int],
    shared_layers: dict[str, str],
) -> None:
    for groups in attn_groups:
        for group in groups:
            if not _is_hcu_flash_attn_backend(group.backend):
                continue
            block_size = kernel_block_sizes[group.kv_cache_group_id]
            spec = group.kv_cache_spec
            stride_order = group.backend.get_kv_cache_stride_order()
            for layer_name in group.layer_names:
                if layer_name in shared_layers:
                    continue
                caches[layer_name] = _reshape_hcu_flash_cache(
                    caches[layer_name],
                    block_size,
                    spec.num_kv_heads,
                    spec.head_size,
                    stride_order,
                )


def _init_hcu_kv_cache(
    runner: Any,
    runner_kv_caches: list[torch.Tensor | list[torch.Tensor]],
    forward_context: dict[str, Any],
    kv_cache_config: Any,
    device: torch.device,
    kernel_block_sizes: list[int],
    vllm_config: Any,
    kv_cache_allocation_context: Any = None,
) -> dict[str, Any]:
    shared_layers = get_shared_kv_cache_layers(vllm_config)
    allocation_context = kv_cache_allocation_context or nullcontext()
    with allocation_context:
        caches = allocate_kv_cache(
            kv_cache_config,
            device,
            vllm_config.cache_config.get_resolved_kv_cache_layout(),
            kernel_block_sizes,
        )
    _reshape_hcu_flash_caches(
        caches,
        runner.attn_groups,
        kernel_block_sizes,
        shared_layers,
    )
    for layer_name, target_layer_name in shared_layers.items():
        caches[layer_name] = caches[target_layer_name]

    num_attn_module = (
        2
        if vllm_config.model_config.hf_config.model_type
        in ("longcat_flash", "longcat_flash_ngram")
        else 1
    )
    bind_kv_cache(
        caches,
        forward_context,
        runner_kv_caches,
        num_attn_module,
        kv_cache_groups=kv_cache_config.kv_cache_groups,
    )
    return caches


@contextmanager
def use_hcu_flash_kv_cache_allocator(runner: Any):
    """Narrowly replace the official allocator during this runner's init."""
    from vllm.v1.worker.gpu import model_runner

    official_init_kv_cache = model_runner.init_kv_cache

    def init_kv_cache(*args: Any, **kwargs: Any):
        if not _uses_hcu_flash_attn(runner.attn_groups):
            return official_init_kv_cache(*args, **kwargs)
        return _init_hcu_kv_cache(runner, *args, **kwargs)

    model_runner.init_kv_cache = init_kv_cache
    try:
        yield
    finally:
        model_runner.init_kv_cache = official_init_kv_cache
