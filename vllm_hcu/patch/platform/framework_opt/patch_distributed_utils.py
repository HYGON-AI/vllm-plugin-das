# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Preserve the PP=1 fast path before parsing manual layer partitions."""

from __future__ import annotations

import functools
import inspect
from types import ModuleType

from ._common import (
    PatchCompatibilityError,
    already_applied,
    load_exact_module,
    require_callable,
)

TARGET_MODULE = "vllm.distributed.utils"
PATCH_ID = "platform.framework_opt.pp_single_rank_partition"
TARGETS = (f"{TARGET_MODULE}.get_pp_indices",)
_MARKER = "_vllm_hcu_pp_single_rank_partition_applied"
_WRAPPER = "_vllm_hcu_pp_single_rank_partition_wrapper"


def apply_to_module(module: ModuleType) -> bool:
    distributed_utils = load_exact_module(TARGET_MODULE, module)
    if already_applied(
        distributed_utils,
        _MARKER,
        ((distributed_utils, "get_pp_indices", _WRAPPER),),
    ):
        return False

    original = require_callable(
        distributed_utils, "get_pp_indices", TARGETS[0]
    )
    signature = inspect.signature(original)
    if tuple(signature.parameters) != (
        "num_hidden_layers",
        "pp_rank",
        "pp_size",
    ):
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGETS[0]} has incompatible "
            f"signature {signature}"
        )

    @functools.wraps(original)
    def hcu_get_pp_indices(num_hidden_layers: int, pp_rank: int, pp_size: int):
        if pp_size == 1:
            return 0, num_hidden_layers
        return original(num_hidden_layers, pp_rank, pp_size)

    setattr(hcu_get_pp_indices, _WRAPPER, True)
    setattr(distributed_utils, "_vllm_hcu_original_get_pp_indices", original)
    setattr(distributed_utils, "get_pp_indices", hcu_get_pp_indices)
    setattr(distributed_utils, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
