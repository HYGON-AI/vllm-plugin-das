# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Bind vLLM's ROCm AITER sampler to validated HCU AITER sampling backends.

``_init_aiter_ops`` resolves a backend from ``vllm_hcu.ops.aiter_sampling``
and ``aiter_sample`` is replaced outright, so correctness never depends on the
body of vLLM's ``aiter_sample`` (which ops it calls or how it splits
arguments).  Only the call contract with ``forward_hip`` is audited.
"""

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


def _fall_back_to_native(sampler: object, message: str) -> bool:
    sampler.aiter_ops = None
    sampler._aiter_ops_import_failed = True
    sampler.forward = sampler.forward_native
    logger.warning_once(message)
    return False


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
        (
            (sampler_class, "_init_aiter_ops", TARGETS[0], _WRAPPER_MARKER),
            (sampler_class, "aiter_sample", TARGETS[1], _WRAPPER_MARKER),
        ),
    ):
        return False

    original_init = require_callable(sampler_class, "_init_aiter_ops", TARGETS[0])
    require_exact_signature(original_init, TARGETS[0], positional=("self",))
    # forward_hip calls aiter_sample(logits, k, p, generators); the replacement
    # keeps that contract and owns the implementation.
    original_sample = require_callable(sampler_class, "aiter_sample", TARGETS[1])
    require_exact_signature(
        original_sample,
        TARGETS[1],
        positional=("self", "logits", "k", "p", "generators"),
    )

    @functools.wraps(original_init)
    def hcu_init_aiter_ops(self) -> bool:
        import vllm_hcu.platforms.envs as henvs

        if not henvs.VLLM_HCU_USE_AITER_SAMPLER:
            return _fall_back_to_native(
                self,
                "VLLM_HCU_USE_AITER_SAMPLER=0; using PyTorch-native sampling "
                "instead of the AITER sampler.",
            )
        if not original_init(self):
            return False
        from vllm_hcu.ops.aiter_sampling import resolve_aiter_sampling_ops

        ops = resolve_aiter_sampling_ops()
        if ops is None:
            return _fall_back_to_native(
                self,
                "AITER sampling ops are neither registered under torch.ops.aiter "
                "with a compatible schema nor exposed by a supported "
                "aiter.ops.sampling; falling back to PyTorch-native sampling.",
            )
        self.aiter_ops = ops
        return True

    @functools.wraps(original_sample)
    def hcu_aiter_sample(self, logits, k, p, generators):
        # forward_hip routes per-request generators to forward_native.
        del generators
        from vllm_hcu.ops.aiter_sampling import aiter_sample

        assert self.aiter_ops is not None
        return aiter_sample(self.aiter_ops, logits, k, p)

    setattr(hcu_init_aiter_ops, _WRAPPER_MARKER, True)
    setattr(hcu_aiter_sample, _WRAPPER_MARKER, True)
    setattr(sampler_class, "_init_aiter_ops", hcu_init_aiter_ops)
    setattr(sampler_class, "aiter_sample", hcu_aiter_sample)
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
