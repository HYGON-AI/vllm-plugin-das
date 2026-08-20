# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Size AMD DeepSeek-V4 ragged SWA buffers for non-causal DSpark blocks."""

from __future__ import annotations

import functools
from types import ModuleType

import torch

from ._common import load_exact_module, require_callable, require_class

TARGET_MODULE = "vllm.models.deepseek_v4.amd.rocm"
PATCH_ID = "worker.core_fix.deepseek_v4_amd.dspark_ragged_swa_capacity"
_MARKER = "_vllm_hcu_dspark_ragged_swa_capacity_applied"


def apply_to_module(module: ModuleType) -> bool:
    rocm = load_exact_module(TARGET_MODULE, module)
    builder_cls = require_class(
        rocm,
        "DeepseekV4ROCMAiterSparseSWAMetadataBuilder",
        f"{TARGET_MODULE}.DeepseekV4ROCMAiterSparseSWAMetadataBuilder",
    )
    if getattr(rocm, _MARKER, False):
        return False

    original_init = require_callable(
        builder_cls,
        "__init__",
        f"{TARGET_MODULE}.DeepseekV4ROCMAiterSparseSWAMetadataBuilder.__init__",
    )
    original_copy = require_callable(
        rocm,
        "_copy_ragged_to_graph_buffers",
        f"{TARGET_MODULE}._copy_ragged_to_graph_buffers",
    )

    @functools.wraps(original_init)
    def hcu_dspark_builder_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if self.is_dspark and self.noncausal_index_width > self.window_size:
            self.decode_swa_ragged_indices_buffer = torch.empty(
                self._max_tokens * self.noncausal_index_width,
                dtype=torch.int32,
                device=self.device,
            )

    @functools.wraps(original_copy)
    def hcu_copy_ragged_to_graph_buffers(
        ragged_indices,
        ragged_indptr,
        ragged_indices_buffer,
        ragged_indptr_buffer,
        num_rows,
        max_entries_per_row,
    ):
        nnz = ragged_indices.numel()
        baseline_capacity = max(num_rows * max_entries_per_row, 1)
        if nnz <= baseline_capacity:
            return original_copy(
                ragged_indices,
                ragged_indptr,
                ragged_indices_buffer,
                ragged_indptr_buffer,
                num_rows,
                max_entries_per_row,
            )

        indptr_out = ragged_indptr_buffer[: num_rows + 1]
        indptr_out.copy_(ragged_indptr, non_blocking=True)

        required = max(baseline_capacity, nnz)
        if required > ragged_indices_buffer.numel():
            raise RuntimeError(
                "DeepSeek-V4 ragged metadata exceeds persistent buffer: "
                f"required={required}, capacity={ragged_indices_buffer.numel()}"
            )
        ragged_out = ragged_indices_buffer[:required]
        if nnz > 0:
            ragged_out[:nnz].copy_(ragged_indices, non_blocking=True)
        return ragged_out, indptr_out

    setattr(builder_cls, "__init__", hcu_dspark_builder_init)
    setattr(rocm, "_copy_ragged_to_graph_buffers", hcu_copy_ragged_to_graph_buffers)
    setattr(rocm, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "apply", "apply_to_module"]
