# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""HCU metadata adapter for the official v0.28 attention backend API.

``SparseMLAAttentionImpl`` was removed from this module before v0.28.  Its old
cache-update wrapper therefore cannot be carried forward as a backend-wide
requirement.  Sparse MLA remains a separate, fail-closed upgrade contract;
this callback only owns the common metadata fields used by HCU backends.
"""

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
    wrapped = (
        (common, "__init__", TARGETS[1], _WRAPPER),
        (common, "unpadded", TARGETS[2], _WRAPPER),
        (common, "replace", TARGETS[3], _WRAPPER),
    )
    if already_applied(backend, _MARKER, wrapped):
        return False
    if "CpCommonAttentionMetadata" in vars(backend):
        raise PatchCompatibilityError(f"required HCU-owned type {TARGETS[0]} already exists")
    original_init = require_callable(common, "__init__", TARGETS[1])
    try:
        init_names = tuple(inspect.signature(original_init).parameters)
    except (TypeError, ValueError) as exc:
        raise PatchCompatibilityError(f"cannot inspect {TARGETS[1]}") from exc
    required = {"self", "query_start_loc", "seq_lens", "num_actual_tokens", "slot_mapping"}
    if not required.issubset(init_names):
        raise PatchCompatibilityError(f"required HCU target {TARGETS[1]} has incompatible fields")
    unpadded = require_callable(common, "unpadded", TARGETS[2])
    require_exact_signature(unpadded, TARGETS[2], positional=("self", "num_actual_tokens", "num_actual_reqs"))
    replace = require_callable(common, "replace", TARGETS[3])
    require_exact_signature(replace, TARGETS[3], positional=("self",), var_keyword="kwargs")

    from vllm_hcu.v1.attention.metadata import CpCommonAttentionMetadata

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

    for function in (hcu_common_init, hcu_unpadded, hcu_replace):
        setattr(function, _WRAPPER, True)
    setattr(common, "_vllm_hcu_original_init", original_init)
    setattr(common, "_vllm_hcu_original_unpadded", unpadded)
    setattr(common, "_vllm_hcu_original_replace", replace)
    setattr(common, "__init__", hcu_common_init)
    setattr(common, "unpadded", hcu_unpadded)
    setattr(common, "replace", hcu_replace)
    setattr(backend, "CpCommonAttentionMetadata", CpCommonAttentionMetadata)
    setattr(backend, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
