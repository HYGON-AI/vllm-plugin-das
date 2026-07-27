# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Opt-in debug weight loading behavior owned by the HCU plugin."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Generator
from types import ModuleType

from vllm.logger import init_logger

logger = init_logger(__name__)

_WEIGHT_UTILS_MODULE = "vllm.model_executor.model_loader.weight_utils"
_DEFAULT_LOADER_MODULE = "vllm.model_executor.model_loader.default_loader"


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
    skip_logged = False

    DEFAULT_SAFETENSORS_PREFETCH_NUM_THREADS = 8
    DEFAULT_SAFETENSORS_PREFETCH_BLOCK_SIZE = 16 * 1024 * 1024

    def wrapped_safetensors_weights_iterator(
        hf_weights_files: list[str],
        use_tqdm_on_load: bool,
        safetensors_load_strategy: str = "lazy",
        local_expert_ids: set[int] | None = None,
        *,
        safetensors_prefetch_num_threads: int = (
            DEFAULT_SAFETENSORS_PREFETCH_NUM_THREADS
        ),
        safetensors_prefetch_block_size: int = DEFAULT_SAFETENSORS_PREFETCH_BLOCK_SIZE,
    ) -> Generator[tuple[str, object], None, None]:
        import vllm_hcu.platforms.envs as henvs

        skip_weight_debug_enabled = henvs.VLLM_HCU_USE_SKIP_WEIGHT_DEBUG
        for name, param in original_iterator(
            hf_weights_files=hf_weights_files,
            use_tqdm_on_load=use_tqdm_on_load,
            safetensors_load_strategy=safetensors_load_strategy,
            local_expert_ids=local_expert_ids,
        ):
            if skip_weight_debug_enabled and "layers." in name:
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
