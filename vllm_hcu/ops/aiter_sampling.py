# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""AITER sampling backends for vLLM's ROCm ``TopKTopPSampler``.

vLLM's ROCm ``aiter_sample`` calls upstream AITER custom ops registered as
``torch.ops.aiter.*`` with flashinfer-style split arguments,
``(maybe_tensor, scalar)``.  DAS AITER exposes the same kernels only as Python
wrappers in ``aiter.ops.sampling`` taking ``Tensor | scalar`` directly, and
does not provide ``top_k_renorm_probs``.

The HCU sampler patch replaces ``aiter_sample`` with :func:`aiter_sample`,
which talks to one of the two backends below through a single merged-argument
interface.  Each backend is accepted only after its call boundary has been
validated against the exact arguments it will receive, so an incompatible
AITER build is rejected at resolve time (falling back to native sampling)
instead of raising inside the worker's sampling step.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from types import ModuleType

import torch


@dataclass(frozen=True, slots=True)
class _OpSpec:
    name: str
    # Leading positional ``torch.ops.aiter`` schema argument types.
    torch_types: tuple[str | tuple[str, ...], ...]
    # Leading DAS wrapper parameter names; ``None`` means no DAS wrapper and
    # the op is implemented in PyTorch (see :func:`top_k_renorm_probs`).
    das_prefix: tuple[str, ...] | None
    sampling: bool = True


# ``str(argument.type)`` spelling of ``Tensor?`` in a FunctionSchema.
_OPT = "Optional[Tensor]"

# Single source of truth for op names, upstream schemas and DAS signatures.
_OP_SPECS = (
    _OpSpec(
        "top_k_top_p_sampling_from_probs",
        ("Tensor", ("Tensor", _OPT), _OPT, "int", _OPT, "float"),
        ("probs", "top_k", "top_p"),
    ),
    _OpSpec(
        "top_p_sampling_from_probs",
        ("Tensor", ("Tensor", _OPT), _OPT, "float"),
        ("probs", "top_p"),
    ),
    _OpSpec(
        "top_k_renorm_probs",
        ("Tensor", _OPT, "int"),
        None,
        sampling=False,
    ),
)
# Keyword arguments every DAS sampling call passes.
_SAMPLING_KEYWORDS = ("indices", "deterministic")


def _split(value: torch.Tensor | float | int) -> tuple:
    """Same split as vLLM's ``_to_tensor_scalar_tuple``."""
    return (value, 0) if isinstance(value, torch.Tensor) else (None, value)


def _torch_op_compatible(namespace: object, spec: _OpSpec) -> bool:
    packet = getattr(namespace, spec.name, None)
    overload = getattr(packet, "default", None)
    schema = getattr(overload, "_schema", None)
    if schema is None:
        return False
    arguments = schema.arguments
    count = len(spec.torch_types)
    if len(arguments) < count:
        return False
    if not all(
        str(argument.type) in ((expected,) if isinstance(expected, str) else expected)
        for argument, expected in zip(arguments[:count], spec.torch_types)
    ):
        return False
    # Sampling ops receive ``deterministic`` by keyword; every other trailing
    # argument must have a default because it is never passed.
    if spec.sampling and "deterministic" not in {a.name for a in arguments}:
        return False
    return all(
        argument.has_default_value() or argument.name == "deterministic"
        for argument in arguments[count:]
    )


def _has_torch_ops() -> bool:
    namespace = torch.ops.aiter
    return all(_torch_op_compatible(namespace, spec) for spec in _OP_SPECS)


def _das_wrapper_compatible(sampling: ModuleType, spec: _OpSpec) -> bool:
    function = getattr(sampling, spec.name, None)
    if not callable(function):
        return False
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return False
    if tuple(signature.parameters)[: len(spec.das_prefix)] != spec.das_prefix:
        return False
    # Prove the exact call shape used by DasAiterSamplingOps binds, so a
    # wrapper without ``indices``/``deterministic`` is rejected here rather
    # than raising TypeError during sampling.
    try:
        signature.bind(
            *([None] * len(spec.das_prefix)),
            **dict.fromkeys(_SAMPLING_KEYWORDS),
        )
    except TypeError:
        return False
    return True


def _has_das_wrappers(sampling: ModuleType) -> bool:
    return all(
        _das_wrapper_compatible(sampling, spec)
        for spec in _OP_SPECS
        if spec.das_prefix is not None
    )


def top_k_renorm_probs(probs: torch.Tensor, top_k: torch.Tensor | int) -> torch.Tensor:
    """Keep probabilities ranked within top-k per row and renormalize.

    Follows vLLM's top-k convention: ``k <= 0`` or ``k >= vocab_size`` means
    top-k is disabled for that row.  Ties at the k-th value are kept, matching
    the threshold semantics of the upstream kernel.
    """
    vocab_size = probs.shape[-1]
    if not isinstance(top_k, torch.Tensor):
        k = int(top_k)
        if k <= 0 or k >= vocab_size:
            return probs
        threshold = probs.topk(k, dim=-1).values[:, -1:]
    else:
        k = top_k.to(device=probs.device, dtype=torch.long)
        disabled = (k <= 0) | (k >= vocab_size)
        # Disabled rows gather index 0 and then get a zero threshold.
        k = k.masked_fill(disabled, 1)
        # One host sync for the batch maximum, the same cost as vLLM's
        # apply_top_k_only; topk(max_k) avoids an O(V log V) full sort.
        max_k = int(k.max())
        threshold = probs.topk(max_k, dim=-1).values.gather(-1, (k - 1).unsqueeze(-1))
        threshold = threshold.masked_fill(disabled.unsqueeze(-1), 0.0)
    kept = probs.masked_fill(probs < threshold, 0.0)
    return kept / kept.sum(dim=-1, keepdim=True)


class TorchAiterSamplingOps:
    """Upstream AITER backend: split arguments for ``torch.ops.aiter``."""

    def __init__(self, namespace: object):
        self._ops = namespace

    def top_k_top_p_sampling_from_probs(self, probs, top_k, top_p):
        return self._ops.top_k_top_p_sampling_from_probs(
            probs, None, *_split(top_k), *_split(top_p), deterministic=True
        )

    def top_p_sampling_from_probs(self, probs, top_p):
        return self._ops.top_p_sampling_from_probs(
            probs, None, *_split(top_p), deterministic=True
        )

    def top_k_renorm_probs(self, probs, top_k):
        return self._ops.top_k_renorm_probs(probs, *_split(top_k))


class DasAiterSamplingOps:
    """DAS AITER backend: ``aiter.ops.sampling`` Python wrappers."""

    def __init__(self, sampling: ModuleType):
        self._sampling = sampling

    def top_k_top_p_sampling_from_probs(self, probs, top_k, top_p):
        return self._sampling.top_k_top_p_sampling_from_probs(
            probs, top_k, top_p, indices=None, deterministic=True
        )

    def top_p_sampling_from_probs(self, probs, top_p):
        return self._sampling.top_p_sampling_from_probs(
            probs, top_p, indices=None, deterministic=True
        )

    def top_k_renorm_probs(self, probs, top_k):
        return top_k_renorm_probs(probs, top_k)


AiterSamplingOps = TorchAiterSamplingOps | DasAiterSamplingOps


def resolve_aiter_sampling_ops() -> AiterSamplingOps | None:
    """Return a validated sampling backend, or ``None`` if unsupported.

    Schema-compatible upstream AITER registrations are preferred.  DAS AITER
    wrappers are used when their audited signatures are present.
    """
    if _has_torch_ops():
        return TorchAiterSamplingOps(torch.ops.aiter)
    try:
        import aiter.ops.sampling as sampling
    except ImportError:
        return None
    if not _has_das_wrappers(sampling):
        return None
    return DasAiterSamplingOps(sampling)


def aiter_sample(
    ops: AiterSamplingOps,
    logits: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor | None,
) -> torch.Tensor:
    """Sample from logits; same algorithm as vLLM's ``aiter_sample``."""
    if k is None and p is None:
        raise RuntimeError("aiter_sample was called with no active top-k or top-p.")
    if k is not None:
        # vLLM disables top-k with k == vocab_size; treat k <= 0 sentinels the
        # same.  The DAS joint top-k/top-p kernel never terminates on k <= 0.
        k = k.masked_fill(k <= 0, logits.shape[-1])
    probs = logits.softmax(dim=-1, dtype=torch.float32).contiguous()
    if k is not None and p is not None:
        next_token_ids = ops.top_k_top_p_sampling_from_probs(probs, k, p)
    elif p is not None:
        next_token_ids = ops.top_p_sampling_from_probs(probs, p)
    else:
        renorm_probs = ops.top_k_renorm_probs(probs, k)
        next_token_ids = torch.multinomial(renorm_probs, num_samples=1)
    return next_token_ids.view(-1)


__all__ = [
    "AiterSamplingOps",
    "DasAiterSamplingOps",
    "TorchAiterSamplingOps",
    "aiter_sample",
    "resolve_aiter_sampling_ops",
    "top_k_renorm_probs",
]
