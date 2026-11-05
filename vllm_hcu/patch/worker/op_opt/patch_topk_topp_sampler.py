# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Bind vLLM's ROCm AITER sampler to DAS AITER sampling wrappers."""

from __future__ import annotations

import functools
from types import ModuleType

from vllm.logger import init_logger

from ._common import (
    already_applied,
    load_exact_module,
    require_callable,
    require_class,
    require_exact_signature,
)

logger = init_logger(__name__)

TARGET_MODULE = "vllm.v1.sample.ops.topk_topp_sampler"
PATCH_ID = "worker.op_opt.sampler.aiter_sampling_ops"
TARGETS = (
    f"{TARGET_MODULE}.TopKTopPSampler._init_aiter_ops",
    f"{TARGET_MODULE}.TopKTopPSampler.aiter_sample",
)
_CLASS_MARKER = "_vllm_hcu_aiter_sampling_ops_applied"
_WRAPPER_MARKER = "_vllm_hcu_aiter_sampling_ops_wrapper"


def apply_to_module(module: ModuleType) -> bool:
    sampler_module = load_exact_module(TARGET_MODULE, module)
    sampler_class = require_class(
        sampler_module,
        "TopKTopPSampler",
        f"{TARGET_MODULE}.TopKTopPSampler",
    )
    if already_applied(
        sampler_class,
        _CLASS_MARKER,
        ((sampler_class, "_init_aiter_ops", TARGETS[0], _WRAPPER_MARKER),),
    ):
        return False

    original = require_callable(sampler_class, "_init_aiter_ops", TARGETS[0])
    require_exact_signature(original, TARGETS[0], positional=("self",))
    # The facade mirrors the ops consumed by this exact aiter_sample shape.
    aiter_sample = require_callable(sampler_class, "aiter_sample", TARGETS[1])
    require_exact_signature(
        aiter_sample,
        TARGETS[1],
        positional=("self", "logits", "k", "p", "generators"),
    )

    @functools.wraps(original)
    def hcu_init_aiter_ops(self) -> bool:
        if not original(self):
            return False
        from vllm_hcu.ops.aiter_sampling import resolve_aiter_sampling_ops

        ops = resolve_aiter_sampling_ops()
        if ops is None:
            self.aiter_ops = None
            self._aiter_ops_import_failed = True
            self.forward = self.forward_native
            logger.warning_once(
                "AITER sampling ops are neither registered under torch.ops.aiter "
                "nor exposed by a supported aiter.ops.sampling; falling back to "
                "PyTorch-native sampling."
            )
            return False
        self.aiter_ops = ops
        return True

    setattr(hcu_init_aiter_ops, _WRAPPER_MARKER, True)
    setattr(sampler_class, "_vllm_hcu_original_init_aiter_ops", original)
    setattr(sampler_class, "_init_aiter_ops", hcu_init_aiter_ops)
    setattr(sampler_class, _CLASS_MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = [
    "PATCH_ID",
    "TARGET_MODULE",
    "TARGETS",
    "apply",
    "apply_to_module",
]
