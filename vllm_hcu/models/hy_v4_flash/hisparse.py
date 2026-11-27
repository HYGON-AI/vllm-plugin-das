# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Explicit, fail-closed HY4 Flash HiSparse admission for the v0.33 lane.

This module is deliberately import-light. Ordinary Flash/DFly inference must
not load the adapter HiSparse extension or allocate any host/device buffers.
"""

_HISPARSE_OPS = (
    "hisparse_resolve_residency",
    "hisparse_invalidate_written_slots",
    "hisparse_gather_plan",
    "hisparse_gather_compact",
)


def validate_flash_hisparse(vllm_config) -> bool:
    """Validate the target-only, eager HiSparse lane before constructing layers.

    Return whether HiSparse was explicitly requested. The pinned Core has no
    shared-hot-cache owner ABI for DFly; enabling that combination would alias
    host KV without sharing its residency metadata, so it remains rejected.
    """
    transfer = vllm_config.kv_transfer_config
    config = getattr(vllm_config.attention_config, "hisparse_config", None)
    if transfer is None:
        if config is not None:
            raise ValueError("HYV4 Flash HiSparse requires HiSparseConnector")
        return False
    if transfer.kv_connector != "HiSparseConnector":
        raise ValueError("HYV4 Flash supports only the HiSparseConnector KV connector")
    if config is None:
        raise ValueError("HiSparseConnector must resolve attention.hisparse_config")
    if vllm_config.speculative_config is not None:
        raise NotImplementedError(
            "HYV4 Flash HiSparse + DFly is not supported by the pinned v0.33 "
            "shared-hot-cache ABI; use target-only HiSparse or DFly without HiSparse"
        )
    if not vllm_config.model_config.enforce_eager:
        raise NotImplementedError("HYV4 Flash HiSparse currently requires enforce_eager")
    # Loading is opt-in and HCU-specific. CUDA continues to use Core's native ops.
    from .runtime import is_hcu_platform

    if is_hcu_platform():
        try:
            from hcu_adapter.kernels.hisparse import ensure_hisparse_ops
        except ImportError as exc:
            raise RuntimeError(
                "HYV4 Flash HiSparse requires adapter-das with native HIP "
                "HiSparse operators; do not enable it with the older adapter"
            ) from exc
        ensure_hisparse_ops()
    import torch

    missing = [
        name for name in _HISPARSE_OPS
        if not hasattr(torch.ops._C_cache_ops, name)
        or not torch._C._dispatch_has_kernel_for_dispatch_key(
            f"_C_cache_ops::{name}", "CUDA"
        )
    ]
    if missing:
        raise RuntimeError("HYV4 Flash HiSparse missing native operators: " + ", ".join(missing))
    return True
