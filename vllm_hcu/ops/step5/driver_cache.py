# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple

CacheType = Dict[Tuple[Any, ...], Callable[..., None]]
GridMeta = Tuple[int, ...]
GridLaunch = Tuple[int, int, int]
MetaArgs = Tuple[Any, ...]
RuntimeArgs = Tuple[Any, ...]

_SIGNATURE_EXT = None
_SIGNATURE_EXT_READY = False
_SIGNATURE_EXT_LOCK = threading.Lock()
_TRITON_RUNTIME = None
_TRITON_RUNTIME_READY = False
_TRITON_RUNTIME_LOCK = threading.Lock()
_TRITON_SPECIALIZER = None
_TRITON_SPECIALIZER_READY = False
_TRITON_SPECIALIZER_LOCK = threading.Lock()
_TENSOR_BASE = None
_SIZE_BASE = None
_TORCH_TYPES_READY = False
_TORCH_TYPES_LOCK = threading.Lock()

_HASH_SEED = 1469598103934665603
_HASH_MAGIC = 0x9E3779B97F4A7C15
_HASH_MASK = (1 << 64) - 1
_STRICT_RUNTIME_SIGNATURE = (
    os.getenv("OPTIMUS_TRITON_DRIVER_STRICT_SIGNATURE", "0") == "1"
)


def _hash_combine(seed: int, val: int) -> int:
    seed &= _HASH_MASK
    val &= _HASH_MASK
    combined = seed ^ (val + _HASH_MAGIC + ((seed << 6) & _HASH_MASK) + (seed >> 2))
    return combined & _HASH_MASK


def _hash_tensor_signature(value) -> int:
    seed = _HASH_SEED
    dtype_bytes = str(value.dtype).encode("utf-8")
    for b in dtype_bytes:
        seed = _hash_combine(seed, b)
    for dim in value.shape:
        seed = _hash_combine(seed, int(dim))
    for stride in value.stride():
        seed = _hash_combine(seed, int(stride))
    return seed


def _default_grid_launch(grid_meta: GridMeta) -> GridLaunch:
    return (
        grid_meta[0],
        grid_meta[1] if len(grid_meta) > 1 else 1,
        grid_meta[2] if len(grid_meta) > 2 else 1,
    )


def _get_torch_types():
    global _TENSOR_BASE, _SIZE_BASE, _TORCH_TYPES_READY
    if _TORCH_TYPES_READY:
        return _TENSOR_BASE, _SIZE_BASE
    with _TORCH_TYPES_LOCK:
        if not _TORCH_TYPES_READY:
            try:
                import torch
            except (
                Exception
            ):  # pragma: no cover - torch should exist but guard just in case
                _TENSOR_BASE = ()
                _SIZE_BASE = ()
            else:
                _TENSOR_BASE = (torch.Tensor,)
                _SIZE_BASE = (torch.Size,)
            _TORCH_TYPES_READY = True
    return _TENSOR_BASE, _SIZE_BASE


def _load_signature_ext():
    # The reviewed base does not ship the optional signature extension source
    # or binary. Keep the Python signature fallback without depending on the
    # old model package or trying to compile a missing source file.
    try:
        from . import _signature as signature_ext
    except Exception:
        return None
    return signature_ext


def _get_signature_ext():
    global _SIGNATURE_EXT, _SIGNATURE_EXT_READY
    if _SIGNATURE_EXT_READY:
        return _SIGNATURE_EXT
    with _SIGNATURE_EXT_LOCK:
        if not _SIGNATURE_EXT_READY:
            _SIGNATURE_EXT = _load_signature_ext()
            _SIGNATURE_EXT_READY = True
    return _SIGNATURE_EXT


def _get_triton_runtime():
    global _TRITON_RUNTIME, _TRITON_RUNTIME_READY
    if _TRITON_RUNTIME_READY:
        return _TRITON_RUNTIME
    with _TRITON_RUNTIME_LOCK:
        if not _TRITON_RUNTIME_READY:
            from triton import knobs as triton_knobs
            from triton.runtime.driver import driver as triton_driver

            _TRITON_RUNTIME = (triton_knobs.runtime, triton_driver)
            _TRITON_RUNTIME_READY = True
    return _TRITON_RUNTIME


def _get_triton_specializer():
    global _TRITON_SPECIALIZER, _TRITON_SPECIALIZER_READY
    if _TRITON_SPECIALIZER_READY:
        return _TRITON_SPECIALIZER
    with _TRITON_SPECIALIZER_LOCK:
        if not _TRITON_SPECIALIZER_READY:
            from triton.compiler import make_backend
            from triton.runtime import jit as triton_jit

            _, triton_driver = _get_triton_runtime()
            backend = make_backend(triton_driver.active.get_current_target())

            native_specialize = getattr(triton_jit, "native_specialize_impl", None)
            if native_specialize is not None:

                def specialize(value):
                    return native_specialize(backend, value, False, True, True)
            else:
                create_specialize = triton_jit.create_specialize_impl
                specialize_impl = create_specialize(backend.get_arg_specialization)

                def specialize(value):
                    return specialize_impl(value, False, True, True)

            _TRITON_SPECIALIZER = specialize
            _TRITON_SPECIALIZER_READY = True
    return _TRITON_SPECIALIZER


def _value_signature(value: Any) -> Any:
    tensor_base, size_base = _get_torch_types()
    if isinstance(value, tensor_base):
        return ("tensor_hash", _hash_tensor_signature(value))
    if isinstance(value, size_base):
        return ("size", tuple(value))

    if isinstance(value, (int, float, bool, str, type(None))):
        return value
    if isinstance(value, tuple):
        return ("tuple", tuple(_value_signature(v) for v in value))
    if isinstance(value, list):
        return ("list", tuple(_value_signature(v) for v in value))
    if isinstance(value, dict):
        return (
            "dict",
            tuple(sorted((k, _value_signature(v)) for k, v in value.items())),
        )
    return (type(value).__name__, repr(value))


def _runtime_layout_signature(value: Any) -> Any:
    tensor_base, size_base = _get_torch_types()
    if isinstance(value, tensor_base):
        return (int(value.ndim), tuple(int(s) for s in value.stride()))
    if isinstance(value, size_base):
        return ("size", len(value))
    if isinstance(value, tuple):
        return (
            "tuple",
            tuple(_runtime_layout_signature(item) for item in value),
        )
    return None


def _fast_signature(
    grid_meta: GridMeta,
    meta_args: MetaArgs,
    runtime_args: RuntimeArgs,
) -> Tuple[Any, ...] | None:
    signature_ext = _get_signature_ext()
    if signature_ext is None:
        return None
    try:
        return tuple(signature_ext.build_signature(grid_meta, meta_args, runtime_args))
    except Exception:
        return None


def _default_signature(
    grid_meta: GridMeta,
    meta_args: MetaArgs,
    runtime_args: RuntimeArgs,
) -> Tuple[Any, ...]:
    if _STRICT_RUNTIME_SIGNATURE:
        fast = _fast_signature(grid_meta, meta_args, runtime_args)
        if fast is not None:
            return fast
        runtime_sig = tuple(_value_signature(arg) for arg in runtime_args)
    else:
        specialize = _get_triton_specializer()
        runtime_sig = (
            ("triton_specialization", specialize(runtime_args)),
            (
                "runtime_layouts",
                tuple(_runtime_layout_signature(arg) for arg in runtime_args),
            ),
        )
    meta_sig = tuple(_value_signature(arg) for arg in meta_args)
    return runtime_sig + (("meta",),) + meta_sig


@dataclass
class KernelLaunchSpec:
    kernel_id: str
    runtime_args: RuntimeArgs
    grid_fn: Callable[[], GridMeta]
    kernel_fn: Callable[[GridMeta], Any]
    autotuner: Any | None = None
    meta_fn: Callable[[GridMeta], MetaArgs] | None = None
    autotuned_meta_fn: Callable[..., MetaArgs] | None = None
    signature_fn: (
        Callable[[GridMeta, MetaArgs, RuntimeArgs], Tuple[Any, ...]] | None
    ) = None
    grid_launch_fn: Callable[[GridMeta], GridLaunch] = _default_grid_launch
    enforce_driver_launch_on_register: bool = False  # enforce driver launch on register step, which will launch the kernel twice


def _build_driver_call(
    kernel,
    meta_args: MetaArgs,
    best_config_meta_args: MetaArgs,
) -> Callable[..., None]:
    best_config_meta_args = (
        tuple(best_config_meta_args) if best_config_meta_args is not None else ()
    )
    runtime_knobs_obj, triton_driver = _get_triton_runtime()
    active_driver_ref = triton_driver.active
    get_current_device = active_driver_ref.get_current_device
    get_current_stream = active_driver_ref.get_current_stream

    def _driver_call(grid_meta: GridMeta, grid_launch: GridLaunch, *runtime_args):
        nonlocal active_driver_ref, get_current_device, get_current_stream
        kernel_args = runtime_args + meta_args + best_config_meta_args
        driver_active = triton_driver.active
        if driver_active is not active_driver_ref:
            active_driver_ref = driver_active
            get_current_device = driver_active.get_current_device
            get_current_stream = driver_active.get_current_stream
        device = get_current_device()
        stream = get_current_stream(device)
        launch_enter_hook = runtime_knobs_obj.launch_enter_hook
        launch_exit_hook = runtime_knobs_obj.launch_exit_hook
        enter_enabled = bool(getattr(launch_enter_hook, "calls", ()))
        exit_enabled = bool(getattr(launch_exit_hook, "calls", ()))
        launch_metadata = None
        if enter_enabled or exit_enabled:
            launch_metadata = kernel.launch_metadata(grid_meta, stream, *kernel_args)
        kernel.run(
            grid_launch[0],
            grid_launch[1],
            grid_launch[2],
            stream,
            kernel.function,
            kernel.packed_metadata,
            launch_metadata,
            launch_enter_hook if enter_enabled else None,
            launch_exit_hook if exit_enabled else None,
            *kernel_args,
        )

    return _driver_call


def _get_cached_driver_call(
    cache: CacheType,
    cache_lock: threading.Lock,
    cache_key: Tuple[Any, ...],
) -> Callable[..., None] | None:
    with cache_lock:
        return cache.get(cache_key)


def _register_driver_call(
    cache: CacheType,
    cache_lock: threading.Lock,
    cache_key: Tuple[Any, ...],
    kernel,
    meta_args: MetaArgs,
    best_config_meta_args: MetaArgs,
) -> Callable[..., None] | None:
    if kernel is None:
        return None
    driver_call = _build_driver_call(kernel, meta_args, best_config_meta_args)
    with cache_lock:
        cache.setdefault(cache_key, driver_call)
    return driver_call


def run_or_register_driver(
    cache: CacheType,
    cache_lock: threading.Lock,
    spec: KernelLaunchSpec,
) -> None:
    """Execute cached driver or build+register a new one for the provided spec."""

    grid_meta = spec.grid_fn()
    grid_launch = spec.grid_launch_fn(grid_meta)
    kernel_obj = None
    autotune_args_cache = None
    meta_args_cache = None

    def build_kernel():
        nonlocal kernel_obj
        if kernel_obj is None:
            kernel_obj = spec.kernel_fn(grid_meta)
        return kernel_obj

    def get_meta_args():
        nonlocal meta_args_cache
        if meta_args_cache is not None:
            return meta_args_cache
        if spec.meta_fn is None:
            meta_args_cache = ()
            return meta_args_cache
        try:
            meta_args_cache = tuple(spec.meta_fn(grid_meta))
        except RuntimeError:
            build_kernel()
            meta_args_cache = tuple(spec.meta_fn(grid_meta))
        return meta_args_cache

    def get_driver_meta_args(best_config: Any | None):
        if best_config is not None and hasattr(best_config, "kwargs"):
            driver_meta_args_cache = tuple(best_config.kwargs.values())
            return driver_meta_args_cache

    def get_autotune_args():
        nonlocal autotune_args_cache
        if autotune_args_cache is not None:
            return autotune_args_cache
        if spec.autotuned_meta_fn is None:
            autotune_args_cache = ()
            return autotune_args_cache
        autotune_args_cache = tuple(spec.autotuned_meta_fn(grid_meta))
        return autotune_args_cache

    meta_args = get_meta_args()
    # autotune_args = get_autotune_args()
    signature_fn = spec.signature_fn or _default_signature
    signature = signature_fn(grid_meta, meta_args, spec.runtime_args)
    cache_key = (spec.kernel_id, signature)
    driver_call = _get_cached_driver_call(cache, cache_lock, cache_key)
    if driver_call is None:
        kernel = build_kernel()
        best_config = None
        if spec.autotuner is not None:
            if best_config is None:
                best_config = getattr(spec.autotuner, "best_config", None)
        best_config_meta_args = get_driver_meta_args(best_config)
        driver_call = _register_driver_call(
            cache,
            cache_lock,
            cache_key,
            kernel,
            meta_args,
            best_config_meta_args,
        )
        if driver_call is None:
            return
        if spec.enforce_driver_launch_on_register:
            driver_call(grid_meta, grid_launch, *spec.runtime_args)
        return
    driver_call(grid_meta, grid_launch, *spec.runtime_args)


class CachedDriverLauncher:
    """Helper that launches Triton kernels with cached driver calls."""

    def __init__(
        self,
        cache: CacheType,
        cache_lock: threading.Lock,
    ) -> None:
        self._cache = cache
        self._cache_lock = cache_lock

    def launch(self, spec: KernelLaunchSpec) -> None:
        run_or_register_driver(self._cache, self._cache_lock, spec)

    __call__ = launch


class DriverCacheRegistry:
    """Provide reusable driver launchers keyed by a cache namespace."""

    def __init__(self) -> None:
        self._launchers: Dict[str, CachedDriverLauncher] = {}
        self._caches: Dict[str, CacheType] = {}
        self._locks: Dict[str, threading.Lock] = {}
        self._registry_lock = threading.Lock()

    def get_launcher(self, name: str) -> CachedDriverLauncher:
        with self._registry_lock:
            launcher = self._launchers.get(name)
            if launcher is None:
                cache: CacheType = {}
                cache_lock = threading.Lock()
                launcher = CachedDriverLauncher(cache, cache_lock)
                self._caches[name] = cache
                self._locks[name] = cache_lock
                self._launchers[name] = launcher
        return launcher


_default_cache_registry = DriverCacheRegistry()


def get_driver_launcher(
    name: str, registry: DriverCacheRegistry | None = None
) -> CachedDriverLauncher:
    registry = registry or _default_cache_registry
    return registry.get_launcher(name)


__all__ = [
    "CachedDriverLauncher",
    "DriverCacheRegistry",
    "KernelLaunchSpec",
    "get_driver_launcher",
    "run_or_register_driver",
]
