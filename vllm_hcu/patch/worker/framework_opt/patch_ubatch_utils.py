# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""DBO slicing normalization and metadata preservation."""

from __future__ import annotations

import functools
from types import ModuleType

from ._common import already_applied, load_exact_module, require_callable, require_exact_signature

TARGET_MODULE = "vllm.v1.worker.ubatch_utils"
PATCH_ID = "worker.framework_opt.dbo.ubatch_metadata"
TARGETS = (
    f"{TARGET_MODULE}._pad_out_ubatch_slices",
    f"{TARGET_MODULE}.maybe_create_ubatch_slices",
    f"{TARGET_MODULE}._make_metadata_with_slice",
)
_MARKER = "_vllm_hcu_ubatch_metadata_applied"
_WRAPPER = "_vllm_hcu_ubatch_metadata_wrapper"


def apply_to_module(module: ModuleType) -> bool:
    ubatch = load_exact_module(TARGET_MODULE, module)
    wrapped = (
        (ubatch, "_pad_out_ubatch_slices", TARGETS[0], _WRAPPER),
        (ubatch, "maybe_create_ubatch_slices", TARGETS[1], _WRAPPER),
        (ubatch, "_make_metadata_with_slice", TARGETS[2], _WRAPPER),
    )
    if already_applied(ubatch, _MARKER, wrapped):
        return False
    original_pad = require_callable(ubatch, "_pad_out_ubatch_slices", TARGETS[0])
    require_exact_signature(
        original_pad,
        TARGETS[0],
        positional=("ubatch_slices", "num_total_tokens", "num_reqs_padded"),
    )
    original_create = require_callable(ubatch, "maybe_create_ubatch_slices", TARGETS[1])
    require_exact_signature(
        original_create,
        TARGETS[1],
        positional=(
            "should_ubatch",
            "num_scheduled_tokens",
            "num_tokens_padded",
            "num_reqs_padded",
            "num_ubatches",
            "split_point",
        ),
        defaults={"split_point": None},
    )
    original_metadata = require_callable(ubatch, "_make_metadata_with_slice", TARGETS[2])
    require_exact_signature(
        original_metadata,
        TARGETS[2],
        positional=("ubatch_slice", "attn_metadata"),
    )

    @functools.wraps(original_pad)
    def hcu_pad(ubatch_slices, num_total_tokens, num_reqs_padded):
        result = original_pad(
            ubatch_slices, int(num_total_tokens), int(num_reqs_padded)
        )
        last = result[-1]
        result[-1] = ubatch.UBatchSlice(
            slice(
                int(last.request_slice.start), int(last.request_slice.stop)
            ),
            slice(int(last.token_slice.start), int(last.token_slice.stop)),
        )
        return result

    @functools.wraps(original_create)
    def hcu_create(
        should_ubatch,
        num_scheduled_tokens,
        num_tokens_padded,
        num_reqs_padded,
        num_ubatches,
        split_point=None,
    ):
        from vllm_hcu.v1.worker_framework_runtime import maybe_create_ubatch_slices

        return maybe_create_ubatch_slices(
            ubatch,
            should_ubatch,
            num_scheduled_tokens,
            num_tokens_padded,
            num_reqs_padded,
            num_ubatches,
            split_point,
        )

    @functools.wraps(original_metadata)
    def hcu_metadata(ubatch_slice, attn_metadata):
        result = original_metadata(ubatch_slice, attn_metadata)
        token_slice = ubatch_slice.token_slice
        request_slice = ubatch_slice.request_slice
        positions = getattr(attn_metadata, "positions", None)
        is_prefilling = getattr(attn_metadata, "is_prefilling", None)
        result.positions = None if positions is None else positions[token_slice]
        result.is_prefilling = (
            None if is_prefilling is None else is_prefilling[request_slice]
        )
        return result

    for function in (hcu_pad, hcu_create, hcu_metadata):
        setattr(function, _WRAPPER, True)
    setattr(ubatch, "_vllm_hcu_original_pad_out_ubatch_slices", original_pad)
    setattr(ubatch, "_vllm_hcu_original_maybe_create_ubatch_slices", original_create)
    setattr(ubatch, "_vllm_hcu_original_make_metadata_with_slice", original_metadata)
    setattr(ubatch, "_pad_out_ubatch_slices", hcu_pad)
    setattr(ubatch, "maybe_create_ubatch_slices", hcu_create)
    setattr(ubatch, "_make_metadata_with_slice", hcu_metadata)
    setattr(ubatch, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
