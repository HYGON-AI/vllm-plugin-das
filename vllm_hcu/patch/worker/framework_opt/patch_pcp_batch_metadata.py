# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Attach ``PCPBatchMetadata`` on rank-local InputBatch (#62384 on 0.29).

Pinned 0.29 ``InputBatch`` has no ``pcp`` field. Upstream #62384 adds
``PCPBatchMetadata`` so decoder SWA bounded replay can see each request's
global trailing window. This adapter defines the dataclass and sets
``local_batch.pcp`` after ``PCPManager.partition_batch`` (instance attribute;
no dataclass rewrite).
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from types import ModuleType
from typing import Any

import numpy as np
import torch

from ._common import load_exact_module, require_callable, require_class

TARGET_MODULE = "vllm.v1.worker.gpu.pcp_manager"
PATCH_ID = "worker.framework_opt.pcp.batch_metadata"
TARGETS = (
    f"{TARGET_MODULE}.PCPBatchMetadata",
    f"{TARGET_MODULE}.PCPManager.partition_batch",
)
_MARKER = "_vllm_hcu_pcp_batch_metadata_applied"
_WRAPPER = "_vllm_hcu_pcp_batch_metadata_wrapper"


@dataclass(frozen=True)
class PCPBatchMetadata:
    """Per-step metadata of a PCP rank-local batch (upstream #62384)."""

    global_num_tokens: int
    global_query_lens_np: np.ndarray
    global_seq_lens_cpu_upper_bound_np: np.ndarray
    dcp_local_seq_lens_cpu_upper_bound: torch.Tensor | None = None


def apply_to_module(module: ModuleType) -> bool:
    pm = load_exact_module(TARGET_MODULE, module)
    wrapped = ((pm, "PCPManager", f"{TARGET_MODULE}.PCPManager", _WRAPPER),)
    # already_applied expects (owner, name, target, wrapper) on callables;
    # use marker-only for this module-level patch.
    if getattr(pm, _MARKER, False):
        return False

    setattr(pm, "PCPBatchMetadata", PCPBatchMetadata)
    mgr = require_class(pm, "PCPManager", f"{TARGET_MODULE}.PCPManager")
    original = require_callable(
        mgr, "partition_batch", f"{TARGET_MODULE}.PCPManager.partition_batch"
    )

    @functools.wraps(original)
    def partition_batch(self, input_batch, *args: Any, **kwargs: Any):
        local = original(self, input_batch, *args, **kwargs)
        idx_np = local.idx_mapping_np[: local.num_reqs]
        local.pcp = PCPBatchMetadata(
            global_num_tokens=int(input_batch.num_tokens),
            global_query_lens_np=np.asarray(
                input_batch.num_scheduled_tokens[idx_np], dtype=np.int32
            ),
            global_seq_lens_cpu_upper_bound_np=np.asarray(
                input_batch.seq_lens_cpu_upper_bound.numpy()[idx_np],
                dtype=np.int32,
            ),
            dcp_local_seq_lens_cpu_upper_bound=(
                local.dcp_local_seq_lens_cpu_upper_bound
            ),
        )
        return local

    setattr(partition_batch, _WRAPPER, True)
    mgr.partition_batch = partition_batch
    setattr(pm, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = [
    "PATCH_ID",
    "TARGET_MODULE",
    "TARGETS",
    "PCPBatchMetadata",
    "apply",
    "apply_to_module",
]
