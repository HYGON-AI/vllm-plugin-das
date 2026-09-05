# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Use the HCU MLA cache-update operator for vLLM's sparse MLA base."""

from __future__ import annotations

import functools
from types import ModuleType

from ._common import (
    already_applied,
    load_exact_module,
    require_callable,
    require_class,
    require_exact_signature,
)

TARGET_MODULE = "vllm.model_executor.layers.attention.sparse_mla_attention"
PATCH_ID = "worker.op_opt.attention.sparse_mla_cache_update"
TARGETS = (f"{TARGET_MODULE}.SparseMLACommonImpl.do_kv_cache_update",)
_MARKER = "_vllm_hcu_sparse_mla_cache_update_applied"
_WRAPPER = "_vllm_hcu_sparse_mla_cache_update_wrapper"


def apply_to_module(module: ModuleType) -> bool:
    sparse_module = load_exact_module(TARGET_MODULE, module)
    sparse = require_class(
        sparse_module,
        "SparseMLACommonImpl",
        f"{TARGET_MODULE}.SparseMLACommonImpl",
    )
    wrapped = ((sparse, "do_kv_cache_update", TARGETS[0], _WRAPPER),)
    if already_applied(sparse_module, _MARKER, wrapped):
        return False

    cache_update = require_callable(sparse, "do_kv_cache_update", TARGETS[0])
    require_exact_signature(
        cache_update,
        TARGETS[0],
        positional=(
            "self",
            "kv_c_normed",
            "k_pe",
            "kv_cache",
            "slot_mapping",
            "kv_cache_dtype",
            "k_scale",
        ),
    )

    @functools.wraps(cache_update)
    def hcu_cache_update(
        self,
        kv_c_normed,
        k_pe,
        kv_cache,
        slot_mapping,
        kv_cache_dtype,
        k_scale,
    ):
        if kv_cache.numel() == 0:
            return None
        from vllm.platforms import current_platform

        if not current_platform.is_rocm():
            return cache_update(
                self,
                kv_c_normed,
                k_pe,
                kv_cache,
                slot_mapping,
                kv_cache_dtype,
                k_scale,
            )
        try:
            from vllm_hcu.v1.attention.backends.fa_utils import hcu_ops  # noqa: F401

            op = sparse_module.torch.ops.hcu_ops.concat_and_cache_mla
        except (ImportError, AttributeError) as exc:
            raise RuntimeError(
                "HCU MLA concat-and-cache operator is required but unavailable"
            ) from exc
        op(
            kv_c_normed,
            k_pe.squeeze(1),
            kv_cache,
            slot_mapping.flatten(),
            kv_cache_dtype,
            k_scale,
        )
        return None

    setattr(hcu_cache_update, _WRAPPER, True)
    setattr(sparse, "_vllm_hcu_original_do_kv_cache_update", cache_update)
    setattr(sparse, "do_kv_cache_update", hcu_cache_update)
    setattr(sparse_module, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
