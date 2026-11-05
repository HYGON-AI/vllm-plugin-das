# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Adapt DAS AITER sampling wrappers to vLLM's ``torch.ops.aiter`` ABI.

vLLM's ROCm ``TopKTopPSampler.aiter_sample`` calls upstream AITER custom ops
registered as ``torch.ops.aiter.*`` with flashinfer-style split arguments,
``(maybe_tensor, scalar)``.  DAS AITER exposes the same kernels only as Python
wrappers in ``aiter.ops.sampling`` taking ``Tensor | scalar`` directly, and
does not provide ``top_k_renorm_probs``.  This adapter keeps vLLM's sampling
algorithm and only translates the call boundary.
"""

from __future__ import annotations

import inspect
from types import ModuleType

import torch

_REQUIRED_OPS = (
    "top_k_top_p_sampling_from_probs",
    "top_p_sampling_from_probs",
    "top_k_renorm_probs",
)
_DAS_SIGNATURES = {
    "top_k_top_p_sampling_from_probs": ("probs", "top_k", "top_p"),
    "top_p_sampling_from_probs": ("probs", "top_p"),
}


def _merge(maybe_tensor: torch.Tensor | None, scalar):
    return maybe_tensor if maybe_tensor is not None else scalar


def _has_torch_ops() -> bool:
    namespace = torch.ops.aiter
    return all(hasattr(namespace, name) for name in _REQUIRED_OPS)


def _has_das_wrappers(sampling: ModuleType) -> bool:
    for name, prefix in _DAS_SIGNATURES.items():
        function = getattr(sampling, name, None)
        if not callable(function):
            return False
        try:
            parameters = tuple(inspect.signature(function).parameters)
        except (TypeError, ValueError):
            return False
        if parameters[: len(prefix)] != prefix:
            return False
    return True


def top_k_renorm_probs(probs: torch.Tensor, top_k: torch.Tensor | int) -> torch.Tensor:
    """Keep probabilities ranked within top-k per row and renormalize.

    Ties at the k-th value are kept, matching the threshold semantics of the
    upstream kernel.
    """
    vocab_size = probs.shape[-1]
    if isinstance(top_k, torch.Tensor):
        k = top_k.to(device=probs.device, dtype=torch.long).clamp(1, vocab_size)
    else:
        k = torch.full(
            (probs.shape[0],),
            min(max(int(top_k), 1), vocab_size),
            device=probs.device,
            dtype=torch.long,
        )
    sorted_probs = probs.sort(dim=-1, descending=True).values
    threshold = sorted_probs.gather(-1, (k - 1).unsqueeze(-1))
    kept = probs.masked_fill(probs < threshold, 0.0)
    return kept / kept.sum(dim=-1, keepdim=True)


class DasAiterSamplingOps:
    """``torch.ops.aiter``-compatible facade over ``aiter.ops.sampling``."""

    def __init__(self, sampling: ModuleType):
        self._sampling = sampling

    def top_k_top_p_sampling_from_probs(
        self,
        probs,
        indices,
        maybe_top_k_arr,
        top_k_val,
        maybe_top_p_arr,
        top_p_val,
        deterministic=False,
    ):
        return self._sampling.top_k_top_p_sampling_from_probs(
            probs,
            _merge(maybe_top_k_arr, top_k_val),
            _merge(maybe_top_p_arr, top_p_val),
            indices=indices,
            deterministic=deterministic,
        )

    def top_p_sampling_from_probs(
        self,
        probs,
        indices,
        maybe_top_p_arr,
        top_p_val,
        deterministic=False,
    ):
        return self._sampling.top_p_sampling_from_probs(
            probs,
            _merge(maybe_top_p_arr, top_p_val),
            indices=indices,
            deterministic=deterministic,
        )

    def top_k_renorm_probs(self, probs, maybe_top_k_arr, top_k_val):
        return top_k_renorm_probs(probs, _merge(maybe_top_k_arr, top_k_val))


def resolve_aiter_sampling_ops() -> object | None:
    """Return ops usable by ``aiter_sample``, or ``None`` if unsupported.

    Upstream AITER registrations are preferred unchanged.  DAS AITER wrappers
    are adapted when their audited signatures are present.
    """
    if _has_torch_ops():
        return torch.ops.aiter
    try:
        import aiter.ops.sampling as sampling
    except ImportError:
        return None
    if not _has_das_wrappers(sampling):
        return None
    return DasAiterSamplingOps(sampling)


__all__ = [
    "DasAiterSamplingOps",
    "resolve_aiter_sampling_ops",
    "top_k_renorm_probs",
]
