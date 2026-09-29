# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Opt-in debug weight loading behavior owned by the HCU plugin."""

from __future__ import annotations

import importlib
import inspect
import json
import sys
from contextlib import contextmanager
from collections.abc import Generator
from contextvars import ContextVar
from pathlib import Path
from types import ModuleType

from vllm.logger import init_logger

logger = init_logger(__name__)

_WEIGHT_UTILS_MODULE = "vllm.model_executor.model_loader.weight_utils"
_DEFAULT_LOADER_MODULE = "vllm.model_executor.model_loader.default_loader"
_WEIGHT_DEBUG_SKIP_DISABLED: ContextVar[bool] = ContextVar(
    "vllm_hcu_weight_debug_skip_disabled", default=False
)
_WEIGHT_PREFIXES_TO_SKIP: ContextVar[tuple[str, ...]] = ContextVar(
    "vllm_hcu_weight_prefixes_to_skip", default=()
)
_SAFE_OPEN_MARKER = "_hcu_safe_open_prefix_filter_applied"


class _SafeOpenPrefixView:
    """Filter safetensors keys before the underlying file reads tensor bytes."""

    def __init__(self, handle, prefixes: tuple[str, ...]):
        self._handle = handle
        self._prefixes = prefixes

    def __enter__(self):
        self._handle = self._handle.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return self._handle.__exit__(exc_type, exc_value, traceback)

    def keys(self):
        keys = self._handle.keys()
        if not self._prefixes:
            return keys
        return [
            key for key in keys
            if not any(key.startswith(prefix) for prefix in self._prefixes)
        ]

    def __getattr__(self, name):
        return getattr(self._handle, name)


@contextmanager
def skip_safetensors_weight_prefixes(
    prefixes: tuple[str, ...] | list[str],
):
    """Skip known-unused model prefixes before safetensors materializes them."""
    current = _WEIGHT_PREFIXES_TO_SKIP.get()
    normalized = tuple(dict.fromkeys((*current, *(str(p) for p in prefixes))))
    token = _WEIGHT_PREFIXES_TO_SKIP.set(normalized)
    try:
        yield
    finally:
        _WEIGHT_PREFIXES_TO_SKIP.reset(token)


@contextmanager
def disable_weight_debug_skip():
    """Temporarily load every layer in nested draft-model construction."""
    token = _WEIGHT_DEBUG_SKIP_DISABLED.set(True)
    try:
        yield
    finally:
        _WEIGHT_DEBUG_SKIP_DISABLED.reset(token)


def _require_exact_module(module: object, expected_name: str) -> ModuleType:
    if not isinstance(module, ModuleType) or module.__name__ != expected_name:
        actual_name = getattr(module, "__name__", None)
        raise TypeError(
            f"expected module {expected_name!r}, got {actual_name!r}"
        )
    return module


def _resolve_loader_modules(
    weight_utils: ModuleType | None,
    default_loader: ModuleType | None,
) -> tuple[ModuleType, ModuleType]:
    """Resolve direct-call inputs without importing from a partial package.

    Runtime callbacks always pass both completed modules explicitly.  The
    zero-argument form remains a supported direct/idempotent API: importing
    default_loader first lets vLLM finish its normal base_loader -> reload ->
    weight_utils chain before either module is patched.
    """

    if (weight_utils is None) != (default_loader is None):
        raise TypeError("weight_utils and default_loader must be provided together")
    if default_loader is None:
        default_loader = importlib.import_module(_DEFAULT_LOADER_MODULE)
        loaded_weight_utils = sys.modules.get(_WEIGHT_UTILS_MODULE)
        if not isinstance(loaded_weight_utils, ModuleType):
            raise RuntimeError(
                f"{_DEFAULT_LOADER_MODULE} loaded without {_WEIGHT_UTILS_MODULE}"
            )
        weight_utils = loaded_weight_utils

    return (
        _require_exact_module(weight_utils, _WEIGHT_UTILS_MODULE),
        _require_exact_module(default_loader, _DEFAULT_LOADER_MODULE),
    )


def install_weight_debug_skip_compat(
    weight_utils: ModuleType | None = None,
    default_loader: ModuleType | None = None,
) -> None:
    """Install the opt-in HCU debug weight-skipping iterator.

    Supplying both modules is the import-callback path and performs no vLLM
    import.  Calling without arguments remains supported for explicit callers.
    """

    weight_utils, default_loader = _resolve_loader_modules(
        weight_utils,
        default_loader,
    )

    weight_iterator = getattr(weight_utils, "safetensors_weights_iterator", None)
    default_iterator = getattr(default_loader, "safetensors_weights_iterator", None)
    if not callable(weight_iterator) or not callable(default_iterator):
        raise RuntimeError("vLLM safetensors iterator contract is unavailable")

    if getattr(weight_utils, "_hcu_skip_weight_patch_applied", False):
        if weight_iterator is not default_iterator:
            raise RuntimeError(
                "weight-debug marker is set but loader iterator bindings diverged"
            )
        return

    if weight_iterator is not default_iterator:
        raise RuntimeError(
            "weight_utils and default_loader iterator bindings differ before patch"
        )

    original_iterator = weight_iterator
    iterator_accepts_smoke_limit = "smoke_layer_limit" in inspect.signature(
        original_iterator
    ).parameters
    skip_logged = False

    DEFAULT_SAFETENSORS_PREFETCH_NUM_THREADS = 8
    DEFAULT_SAFETENSORS_PREFETCH_BLOCK_SIZE = 16 * 1024 * 1024

    if not getattr(weight_utils, _SAFE_OPEN_MARKER, False):
        original_safe_open = getattr(weight_utils, "safe_open", None)
        if not callable(original_safe_open):
            raise RuntimeError("vLLM safetensors safe_open contract is unavailable")

        def safe_open_with_prefix_filter(*args, **kwargs):
            return _SafeOpenPrefixView(
                original_safe_open(*args, **kwargs),
                _WEIGHT_PREFIXES_TO_SKIP.get(),
            )

        weight_utils.safe_open = safe_open_with_prefix_filter
        setattr(weight_utils, _SAFE_OPEN_MARKER, True)

    def wrapped_safetensors_weights_iterator(
        hf_weights_files: list[str],
        use_tqdm_on_load: bool,
        safetensors_load_strategy: str = "lazy",
        local_expert_ids: set[int] | None = None,
        smoke_layer_limit: int | None = None,
        *,
        safetensors_prefetch_num_threads: int = (
            DEFAULT_SAFETENSORS_PREFETCH_NUM_THREADS
        ),
        safetensors_prefetch_block_size: int = DEFAULT_SAFETENSORS_PREFETCH_BLOCK_SIZE,
    ) -> Generator[tuple[str, object], None, None]:
        import vllm_hcu.platforms.envs as henvs

        if smoke_layer_limit is not None:
            # The source loader filters tensors after opening every shard. On
            # very large NFS checkpoints that makes a 12-layer smoke run scan
            # the complete model. Use the standard HF weight map to avoid
            # opening shards that contain only omitted decoder layers or
            # multimodal tensors. Fail closed if the checkpoint index cannot
            # be matched exactly; this optimization is debug-smoke only.
            files = list(hf_weights_files)
            if files:
                index_path = Path(files[0]).parent / "model.safetensors.index.json"
                if not index_path.is_file():
                    raise RuntimeError(
                        "VLLM_SKIP_WEIGHT smoke loading requires "
                        f"{index_path} to select relevant checkpoint shards"
                    )
                with index_path.open(encoding="utf-8") as index_file:
                    weight_map = json.load(index_file).get("weight_map")
                if not isinstance(weight_map, dict):
                    raise RuntimeError(
                        f"invalid safetensors weight map in {index_path}"
                    )
                available = {Path(path).name for path in files}
                mapped = {str(shard) for shard in weight_map.values()}
                if not available.issubset(mapped):
                    raise RuntimeError(
                        "safetensors index does not cover every requested "
                        "checkpoint shard; refusing partial smoke filtering"
                    )
                from vllm.model_executor.model_loader.weight_utils import (
                    should_skip_smoke_layer_weight,
                )

                prefixes = _WEIGHT_PREFIXES_TO_SKIP.get()
                needed_shards = {
                    str(shard)
                    for name, shard in weight_map.items()
                    if str(shard) in available
                    and not should_skip_smoke_layer_weight(
                        name, smoke_layer_limit
                    )
                    and not any(name.startswith(prefix) for prefix in prefixes)
                }
                filtered_files = [
                    path for path in files if Path(path).name in needed_shards
                ]
                if not filtered_files:
                    raise RuntimeError(
                        "safetensors smoke filtering selected no checkpoint "
                        "shards; refusing to start an empty model"
                    )
                logger.info(
                    "VLLM_SKIP_WEIGHT selected %d of %d target checkpoint "
                    "shards using %s",
                    len(filtered_files),
                    len(files),
                    index_path,
                )
                hf_weights_files = filtered_files

        skip_weight_debug_enabled = henvs.VLLM_HCU_USE_SKIP_WEIGHT_DEBUG
        iterator_kwargs = dict(
            hf_weights_files=hf_weights_files,
            use_tqdm_on_load=use_tqdm_on_load,
            safetensors_load_strategy=safetensors_load_strategy,
            local_expert_ids=local_expert_ids,
        )
        if iterator_accepts_smoke_limit:
            iterator_kwargs["smoke_layer_limit"] = smoke_layer_limit
        for name, param in original_iterator(**iterator_kwargs):
            if (
                skip_weight_debug_enabled
                and not _WEIGHT_DEBUG_SKIP_DISABLED.get()
                and "layers." in name
            ):
                try:
                    layer_id = int(name.split("layers.")[1].split(".")[0])
                    if layer_id >= 5:
                        nonlocal skip_logged
                        if not skip_logged:
                            logger.warning(
                                "VLLM_HCU_USE_SKIP_WEIGHT_DEBUG is enabled "
                                "skipping weights from layer %s starting with %s",
                                layer_id,
                                name,
                            )
                            skip_logged = True
                        continue
                except Exception:
                    pass
            yield name, param

    weight_utils.safetensors_weights_iterator = wrapped_safetensors_weights_iterator
    default_loader.safetensors_weights_iterator = wrapped_safetensors_weights_iterator
    if (
        weight_utils.safetensors_weights_iterator
        is not default_loader.safetensors_weights_iterator
    ):
        raise RuntimeError("weight-debug iterator bindings diverged after patch")
    # Publish the marker only after both import-time copies are coherent.
    weight_utils._hcu_skip_weight_patch_applied = True
