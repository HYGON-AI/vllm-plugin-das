"""HCU KV-cache allocation adapters for Model Runner V2."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from typing import Any, Sequence

import torch

from vllm.v1.worker.gpu.attn_utils import get_shared_kv_cache_layers
from vllm.v1.worker.utils import bind_kv_cache


_HCU_FLASH_ATTN_MODULE = "vllm_hcu.v1.attention.backends.flash_attn"

try:
    from vllm.v1.worker.utils import allocate_kv_cache
except ImportError:
    # v0.28 moved the allocator into ``gpu.attn_utils`` and kept the
    # public ``bind_kv_cache`` helper in ``worker.utils``.  Keep the HCU
    # Flash adapter importable on both target ABI variants.
    from vllm.v1.worker.gpu.attn_utils import _allocate_kv_cache

    def allocate_kv_cache(
        kv_cache_config: Any,
        device: torch.device,
        *_ignored: Any,
    ) -> dict[str, torch.Tensor]:
        return _allocate_kv_cache(kv_cache_config, device)


def _bind_kv_cache_compat(
    caches: dict[str, Any],
    forward_context: dict[str, Any],
    runner_kv_caches: list[torch.Tensor | list[torch.Tensor]],
    num_attn_module: int,
    kv_cache_groups: Any,
) -> None:
    """Bind caches across the v0.25 and v0.28 worker utility ABIs."""
    try:
        bind_kv_cache(
            caches,
            forward_context,
            runner_kv_caches,
            num_attn_module,
            kv_cache_groups=kv_cache_groups,
        )
    except TypeError as exc:
        if "kv_cache_groups" not in str(exc):
            raise
        bind_kv_cache(
            caches,
            forward_context,
            runner_kv_caches,
            num_attn_module,
        )


def _is_hcu_flash_attn_backend(backend: type[Any]) -> bool:
    return backend.__module__ == _HCU_FLASH_ATTN_MODULE


def _uses_hcu_flash_attn(attn_groups: Sequence[Sequence[Any]]) -> bool:
    return any(
        _is_hcu_flash_attn_backend(group.backend)
        for groups in attn_groups
        for group in groups
    )


def _has_packed_kv_descriptors(
    official_init_kv_cache: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> bool:
    """Detect HCU packed descriptors before choosing the modern adapter.

    DSA MLA models can use an HCU MLA backend rather than the ordinary HCU
    FlashAttention backend.  They still require the custom packed reshape;
    checking only ``_uses_hcu_flash_attn`` would silently send those models
    through vLLM's flat allocator and lose every per-layer offset.
    """
    import inspect

    try:
        bound = inspect.signature(official_init_kv_cache).bind(*args, **kwargs)
    except (TypeError, ValueError):
        return False
    config = bound.arguments.get("kv_cache_config")
    for descriptor in getattr(config, "kv_cache_tensors", ()) or ():
        if int(getattr(descriptor, "block_stride", 0) or 0) > 0:
            return True
    return False


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
            stride_order = group.backend.get_kv_cache_stride_order()
            for layer_name in group.layer_names:
                if layer_name in shared_layers:
                    continue
                spec = group.kv_cache_spec
                per_layer_specs = getattr(spec, "kv_cache_specs", None)
                if isinstance(per_layer_specs, dict):
                    spec = per_layer_specs[layer_name]
                caches[layer_name] = _reshape_hcu_flash_cache(
                    caches[layer_name],
                    block_size,
                    spec.num_kv_heads,
                    spec.head_size,
                    stride_order,
                )


def _cache_tensor_layers(cache_tensor: Any) -> list[str]:
    layers = getattr(cache_tensor, "shared_by", None)
    if layers is None:
        layers = getattr(cache_tensor, "layers", None)
    if not layers:
        raise ValueError("KV cache tensor must name at least one layer")
    return list(layers)


def _layer_spec(group: Any, layer_name: str) -> Any:
    spec = group.kv_cache_spec
    per_layer_specs = getattr(spec, "kv_cache_specs", None)
    if isinstance(per_layer_specs, dict):
        spec = per_layer_specs[layer_name]
    return spec


def _allocate_modern_raw_kv_cache(
    kv_cache_config: Any,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Allocate modern descriptors with the legacy shared backing semantics.

    v0.28's private allocator allocates one buffer per descriptor. HCU's
    packed layer-split descriptors intentionally describe multiple views into
    one backing allocation, so descriptors with the same explicit backing
    identity share it. Plain descriptors remain independent even when their
    sizes match.
    """
    from vllm_hcu.v1.dsa_cache_layer_split import packed_kv_backing_id

    packed_descriptors = [
        tensor
        for tensor in kv_cache_config.kv_cache_tensors
        if int(getattr(tensor, "block_stride", 0) or 0) > 0
    ]
    packed_ids = [packed_kv_backing_id(tensor) for tensor in packed_descriptors]
    missing_ids = [value for value in packed_ids if value is None]
    if missing_ids and (
        len(packed_descriptors) > 1
        or any(value is not None for value in packed_ids)
    ):
        raise ValueError(
            "Packed KV descriptors must carry explicit backing_id metadata "
            "when more than one packed descriptor is present"
        )

    raw_by_layer: dict[str, torch.Tensor] = {}
    descriptor_by_layer: dict[str, Any] = {}
    backing_by_layout: dict[tuple[Any, ...], torch.Tensor] = {}
    packed_layouts: dict[Any, tuple[int, int]] = {}
    for cache_tensor in kv_cache_config.kv_cache_tensors:
        size = int(cache_tensor.size)
        block_stride = int(getattr(cache_tensor, "block_stride", 0) or 0)
        is_packed = block_stride > 0
        if is_packed:
            backing_id = packed_kv_backing_id(cache_tensor)
            if backing_id is None:
                # A single descriptor is self-describing and needs no alias.
                key: tuple[Any, ...] = ("single-packed", size, block_stride)
            else:
                try:
                    hash(backing_id)
                except TypeError as exc:
                    raise ValueError(
                        "Packed KV backing_id must be hashable; refusing "
                        "ambiguous allocation"
                    ) from exc
                previous_layout = packed_layouts.get(backing_id)
                current_layout = (size, block_stride)
                if previous_layout is not None and previous_layout != current_layout:
                    raise ValueError(
                        "Packed KV descriptors sharing backing_id disagree on "
                        "size or block_stride"
                    )
                packed_layouts[backing_id] = current_layout
                key = ("packed", backing_id, size, block_stride)
            backing = backing_by_layout.get(key)
            if backing is None:
                backing = torch.zeros(size, dtype=torch.int8, device=device)
                backing_by_layout[key] = backing
        else:
            backing = torch.zeros(size, dtype=torch.int8, device=device)
        for layer_name in _cache_tensor_layers(cache_tensor):
            raw_by_layer[layer_name] = backing
            descriptor_by_layer[layer_name] = cache_tensor
    return raw_by_layer, descriptor_by_layer


def _reshape_modern_packed_cache(
    raw: torch.Tensor,
    descriptor: Any,
    spec: Any,
    backend: Any,
    num_blocks: int,
    cache_dtype: str,
) -> torch.Tensor:
    """Create a backend cache view from a packed byte descriptor."""
    storage_block_size = int(getattr(spec, "storage_block_size", spec.block_size))
    kernel_block_size = int(getattr(spec, "block_size", storage_block_size))
    if kernel_block_size <= 0 or storage_block_size % kernel_block_size:
        raise ValueError(
            "KV storage block size must be divisible by the kernel block size"
        )
    block_stride = int(getattr(descriptor, "block_stride", 0) or 0)
    if block_stride <= 0:
        raise ValueError("Packed KV descriptor must provide a positive block stride")
    if raw.numel() % block_stride:
        raise ValueError(
            "Packed KV backing size must be divisible by its block stride"
        )
    kernel_num_blocks = raw.numel() // block_stride
    if kernel_num_blocks != int(num_blocks):
        raise ValueError(
            "Packed KV descriptor and config disagree on block count: "
            f"descriptor={kernel_num_blocks} config={num_blocks}"
        )
    shape = backend.get_kv_cache_shape(
        kernel_num_blocks,
        storage_block_size,
        spec.num_kv_heads,
        spec.head_size,
        cache_dtype_str=cache_dtype,
    )
    stride_order = tuple(backend.get_kv_cache_stride_order())
    if len(stride_order) != len(shape):
        raise ValueError("HCU KV-cache stride order does not match its shape")
    physical_shape = tuple(shape[index] for index in stride_order)
    dtype = spec.dtype
    dtype_size = torch.empty((), dtype=dtype).element_size()
    offset = int(getattr(descriptor, "offset", 0) or 0)
    if block_stride % dtype_size or offset % dtype_size:
        raise ValueError("Packed KV offsets and strides must align to cache dtype")
    # Measure one cache page only. Logical dim 0 is the block count; folding
    # it into page_bytes overstates the view and rejects valid packed
    # layer-split descriptors. The as_strided layout below is unchanged.
    block_dim = stride_order.index(0)
    page_bytes = dtype_size
    for index, dimension in enumerate(physical_shape):
        if index == block_dim:
            continue
        page_bytes *= int(dimension)
    if page_bytes > block_stride:
        raise ValueError(
            "Packed KV backend view exceeds its per-block descriptor stride: "
            f"view={page_bytes} stride={block_stride}"
        )
    if (
        offset < 0
        or offset + (kernel_num_blocks - 1) * block_stride + page_bytes
        > raw.numel()
    ):
        raise ValueError(
            "Packed KV descriptor offset/view exceeds its backing allocation"
        )
    physical_stride = list(torch.empty(physical_shape, dtype=dtype).stride())
    physical_stride[block_dim] = block_stride // dtype_size
    return torch.as_strided(
        raw.view(dtype),
        size=physical_shape,
        stride=tuple(physical_stride),
        storage_offset=offset // dtype_size,
    ).permute(*[stride_order.index(i) for i in range(len(stride_order))])


def _init_modern_hcu_kv_cache(
    runner: Any,
    runner_kv_caches: list[Any],
    forward_context: dict[str, Any],
    kv_cache_config: Any,
    attn_backends: dict[str, Any],
    device: torch.device,
    cache_dtype: str,
) -> dict[str, Any]:
    """Modern v0.28 ``init_kv_cache`` adapter with packed-view support."""
    from vllm.v1.worker.gpu.attn_utils import _reshape_kv_cache

    shared_layers = get_shared_kv_cache_layers(runner.vllm_config)
    raw, descriptors = _allocate_modern_raw_kv_cache(kv_cache_config, device)
    packed_layers = {
        layer_name
        for layer_name, descriptor in descriptors.items()
        if int(getattr(descriptor, "block_stride", 0) or 0) > 0
    }
    kv_caches: dict[str, Any] = {}
    if packed_layers:
        from copy import deepcopy

        plain_tensors = [
            tensor
            for tensor in kv_cache_config.kv_cache_tensors
            if int(getattr(tensor, "block_stride", 0) or 0) <= 0
        ]
        if plain_tensors:
            plain_config = deepcopy(kv_cache_config)
            plain_config.kv_cache_tensors = plain_tensors
            plain_groups = []
            for group in plain_config.kv_cache_groups:
                group.layer_names = [
                    name for name in group.layer_names if name not in packed_layers
                ]
                if group.layer_names:
                    plain_groups.append(group)
            plain_config.kv_cache_groups = plain_groups
            kv_caches.update(
                _reshape_kv_cache(
                    plain_config,
                    {
                        name: raw[name]
                        for name in descriptors
                        if name not in packed_layers
                    },
                    attn_backends,
                    cache_dtype,
                )
            )
        reshaped_packed_layers: set[str] = set()
        for groups in runner.attn_groups:
            for group in groups:
                for layer_name in group.layer_names:
                    if layer_name not in packed_layers:
                        continue
                    spec = _layer_spec(group, layer_name)
                    kv_caches[layer_name] = _reshape_modern_packed_cache(
                        raw[layer_name],
                        descriptors[layer_name],
                        spec,
                        group.backend,
                        kv_cache_config.num_blocks,
                        cache_dtype,
                    )
                    reshaped_packed_layers.add(layer_name)
        if reshaped_packed_layers != packed_layers:
            missing = sorted(packed_layers - reshaped_packed_layers)
            raise ValueError(
                "Packed KV descriptors could not be reshaped by their attention "
                f"backends: {missing}"
            )
    else:
        kv_caches = _reshape_kv_cache(
            kv_cache_config,
            raw,
            attn_backends,
            cache_dtype,
        )
    for layer_name, target_layer_name in shared_layers.items():
        kv_caches[layer_name] = kv_caches[target_layer_name]

    num_attn_module = (
        2
        if runner.vllm_config.model_config.hf_config.model_type
        in ("longcat_flash", "longcat_flash_ngram")
        else 1
    )
    _bind_kv_cache_compat(
        kv_caches,
        forward_context,
        runner_kv_caches,
        num_attn_module,
        kv_cache_config.kv_cache_groups,
    )
    from vllm_hcu.v1.dsa_cache_layer_split import register_dsa_cache_tensors

    register_dsa_cache_tensors(kv_caches)
    return kv_caches


def _init_legacy_hcu_kv_cache(
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
    _bind_kv_cache_compat(
        caches,
        forward_context,
        runner_kv_caches,
        num_attn_module,
        kv_cache_config.kv_cache_groups,
    )
    from vllm_hcu.v1.dsa_cache_layer_split import register_dsa_cache_tensors

    register_dsa_cache_tensors(caches)
    return caches


def _init_hcu_kv_cache(
    runner: Any,
    official_init_kv_cache: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Dispatch the legacy and modern ``init_kv_cache`` call contracts."""
    import inspect

    bound = inspect.signature(official_init_kv_cache).bind(*args, **kwargs)
    values = bound.arguments
    if "attn_backends" in values:
        return _init_modern_hcu_kv_cache(
            runner,
            values["runner_kv_caches"],
            values["forward_context"],
            values["kv_cache_config"],
            values["attn_backends"],
            values["device"],
            values["cache_dtype"],
        )
    return _init_legacy_hcu_kv_cache(runner, *args, **kwargs)


@contextmanager
def use_hcu_flash_kv_cache_allocator(runner: Any):
    """Narrowly replace the official allocator during this runner's init."""
    from vllm.v1.worker.gpu import model_runner

    official_init_kv_cache = model_runner.init_kv_cache

    def init_kv_cache(*args: Any, **kwargs: Any):
        attn_groups = getattr(runner, "attn_groups", None)
        if (
            attn_groups is not None
            and not _uses_hcu_flash_attn(attn_groups)
            and not _has_packed_kv_descriptors(
                official_init_kv_cache,
                args,
                kwargs,
            )
        ):
            return official_init_kv_cache(*args, **kwargs)
        return _init_hcu_kv_cache(
            runner,
            official_init_kv_cache,
            args,
            kwargs,
        )

    model_runner.init_kv_cache = init_kv_cache
    try:
        yield
    finally:
        model_runner.init_kv_cache = official_init_kv_cache
