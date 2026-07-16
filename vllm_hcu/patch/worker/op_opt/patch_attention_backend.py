# SPDX-License-Identifier: Apache-2.0
"""HCU metadata and MLA cache-update adapters for the official backend API."""

from __future__ import annotations

import functools
import inspect
from types import ModuleType

from ._common import (
    PatchCompatibilityError,
    already_applied,
    load_exact_module,
    require_callable,
    require_class,
    require_exact_signature,
)

TARGET_MODULE = "vllm.v1.attention.backend"
PATCH_ID = "worker.op_opt.attention.backend_hcu_metadata"
TARGETS = (
    f"{TARGET_MODULE}.SparseMLAAttentionImpl.do_kv_cache_update",
    f"{TARGET_MODULE}.CpCommonAttentionMetadata",
    f"{TARGET_MODULE}.CommonAttentionMetadata.__init__",
    f"{TARGET_MODULE}.CommonAttentionMetadata.unpadded",
    f"{TARGET_MODULE}.CommonAttentionMetadata.replace",
)
_MARKER = "_vllm_hcu_attention_backend_metadata_applied"
_WRAPPER = "_vllm_hcu_attention_backend_metadata_wrapper"
_HCU_FIELDS = (
    "num_kv_actual_tokens", "seq_indexes_list", "scatter_indexes_tensor",
    "gather_indexes_tensor", "cp_common_metadata",
)


def apply_to_module(module: ModuleType) -> bool:
    backend = load_exact_module(TARGET_MODULE, module)
    common = require_class(backend, "CommonAttentionMetadata", f"{TARGET_MODULE}.CommonAttentionMetadata")
    sparse = require_class(backend, "SparseMLAAttentionImpl", f"{TARGET_MODULE}.SparseMLAAttentionImpl")
    wrapped = (
        (sparse, "do_kv_cache_update", TARGETS[0], _WRAPPER),
        (common, "__init__", TARGETS[2], _WRAPPER),
        (common, "unpadded", TARGETS[3], _WRAPPER),
        (common, "replace", TARGETS[4], _WRAPPER),
    )
    if already_applied(backend, _MARKER, wrapped):
        return False
    if "CpCommonAttentionMetadata" in vars(backend):
        raise PatchCompatibilityError(f"required HCU-owned type {TARGETS[1]} already exists")
    cache_update = require_callable(sparse, "do_kv_cache_update", TARGETS[0])
    require_exact_signature(
        cache_update, TARGETS[0],
        positional=("self", "kv_c_normed", "k_pe", "kv_cache", "slot_mapping", "kv_cache_dtype", "k_scale"),
    )
    original_init = require_callable(common, "__init__", TARGETS[2])
    try:
        init_names = tuple(inspect.signature(original_init).parameters)
    except (TypeError, ValueError) as exc:
        raise PatchCompatibilityError(f"cannot inspect {TARGETS[2]}") from exc
    required = {"self", "query_start_loc", "seq_lens", "num_actual_tokens", "slot_mapping"}
    if not required.issubset(init_names):
        raise PatchCompatibilityError(f"required HCU target {TARGETS[2]} has incompatible fields")
    unpadded = require_callable(common, "unpadded", TARGETS[3])
    require_exact_signature(unpadded, TARGETS[3], positional=("self", "num_actual_tokens", "num_actual_reqs"))
    replace = require_callable(common, "replace", TARGETS[4])
    require_exact_signature(replace, TARGETS[4], positional=("self",), var_keyword="kwargs")

    from vllm_hcu.v1.attention.metadata import CpCommonAttentionMetadata

    @functools.wraps(cache_update)
    def hcu_cache_update(self, kv_c_normed, k_pe, kv_cache, slot_mapping,
                         kv_cache_dtype, k_scale):
        if kv_cache.numel() == 0:
            return None
        from vllm.platforms import current_platform

        if not current_platform.is_rocm():
            return cache_update(self, kv_c_normed, k_pe, kv_cache, slot_mapping,
                                kv_cache_dtype, k_scale)
        try:
            from vllm_hcu.v1.attention.backends.fa_utils import hcu_ops  # noqa: F401

            op = backend.torch.ops.hcu_ops.concat_and_cache_mla
        except (ImportError, AttributeError) as exc:
            raise RuntimeError("HCU MLA concat-and-cache operator is required but unavailable") from exc
        op(kv_c_normed, k_pe.squeeze(1), kv_cache, slot_mapping.flatten(),
           kv_cache_dtype, k_scale)
        return None

    @functools.wraps(original_init)
    def hcu_common_init(self, *args, **kwargs):
        extras = {name: kwargs.pop(name, None) for name in _HCU_FIELDS}
        original_init(self, *args, **kwargs)
        if extras["num_kv_actual_tokens"] is None:
            extras["num_kv_actual_tokens"] = self.num_actual_tokens
        for name, value in extras.items():
            setattr(self, name, value)

    @functools.wraps(unpadded)
    def hcu_unpadded(self, num_actual_tokens, num_actual_reqs):
        result = unpadded(self, num_actual_tokens, num_actual_reqs)
        result.num_kv_actual_tokens = num_actual_tokens
        return result

    @functools.wraps(replace)
    def hcu_replace(self, **kwargs):
        extras = {
            name: kwargs.pop(name, getattr(self, name, None)) for name in _HCU_FIELDS
        }
        result = replace(self, **kwargs)
        for name, value in extras.items():
            setattr(result, name, value)
        return result

    for function in (hcu_cache_update, hcu_common_init, hcu_unpadded, hcu_replace):
        setattr(function, _WRAPPER, True)
    setattr(sparse, "_vllm_hcu_original_do_kv_cache_update", cache_update)
    setattr(common, "_vllm_hcu_original_init", original_init)
    setattr(common, "_vllm_hcu_original_unpadded", unpadded)
    setattr(common, "_vllm_hcu_original_replace", replace)
    setattr(sparse, "do_kv_cache_update", hcu_cache_update)
    setattr(common, "__init__", hcu_common_init)
    setattr(common, "unpadded", hcu_unpadded)
    setattr(common, "replace", hcu_replace)
    setattr(backend, "CpCommonAttentionMetadata", CpCommonAttentionMetadata)
    setattr(backend, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
