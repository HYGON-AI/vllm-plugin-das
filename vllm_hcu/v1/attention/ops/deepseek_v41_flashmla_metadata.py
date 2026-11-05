# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""CPU-only ownership of V4.1's per-build, per-layer native decode plans.

An empty native scheduler does not allocate GPU buffers. The native call owns
initialization. Never share that initialized state between source layers or
successive forwards, even when shapes and compression ratios match.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from vllm_hcu.platforms import envs as henvs

SCHEDULERS_ATTR = "_vllm_hcu_flashmla_decode_schedulers"


def decode_context_supported(config) -> bool:
    parallel = config.parallel_config
    return (
        parallel.decode_context_parallel_size == 1
        and parallel.prefill_context_parallel_size == 1
        and decode_query_limit(config) > 0
    )


def decode_query_limit(config) -> int:
    spec = config.speculative_config
    if spec is None:
        return 1
    count = getattr(spec, "num_speculative_tokens", None)
    if (
        not henvs.dsv41_flashmla_dspark_enabled()
        or getattr(spec, "method", None) != "dspark"
        or type(count) is not int or count < 1
    ):
        return 0
    # Match AttentionMetadataBuilder's spec-as-decode threshold. Actual rows
    # still use explicit per-token indices; no causal mask is synthesized here.
    return 1 + (2 if getattr(spec, "parallel_drafting", False) else 1) * count


@dataclass(frozen=True)
class DecodeBuildPolicy:
    query_limit: int
    allow_graph: bool
    speculative: bool


_BUILD_POLICIES: ContextVar[dict | None] = ContextVar(
    "dsv41_decode_build_policies", default=None,
)


@contextmanager
def decode_metadata_build_scope():
    """Resolve decode policy once per config and attention-metadata pass."""
    token = _BUILD_POLICIES.set({})
    try:
        yield
    finally:
        _BUILD_POLICIES.reset(token)


def _resolve_decode_build_policy(config) -> DecodeBuildPolicy | None:
    if not henvs.dsv41_flashmla_decode_enabled():
        return None
    parallel = config.parallel_config
    if (
        parallel.decode_context_parallel_size != 1
        or parallel.prefill_context_parallel_size != 1
    ):
        return None
    query_limit = decode_query_limit(config)
    if query_limit < 1:
        return None
    return DecodeBuildPolicy(
        query_limit=query_limit,
        allow_graph=henvs.dsv41_flashmla_decode_graph_enabled(),
        speculative=config.speculative_config is not None,
    )


def decode_build_policy(config) -> DecodeBuildPolicy | None:
    policies = _BUILD_POLICIES.get()
    if policies is None:
        return _resolve_decode_build_policy(config)
    key = id(config)
    if key not in policies:
        policies[key] = (config, _resolve_decode_build_policy(config))
    return policies[key][1]


class DecodeScheduler:
    """Lease one fresh native plan per invocation, owned by step metadata.

    The builder supplies the first plan. Repeated eager execution of the same
    metadata (e.g. warmup) gets a fresh plan rather than reusing lengths from the
    preceding call. There is no process-global or builder-persistent plan cache.
    Capture also consumes a fresh plan, so native planner launches are recorded
    in the graph rather than freezing warmup lengths. Replay does not execute
    Python: graph-owned buffers retain the captured planner outputs. The runner
    owns input address stability and updates the persistent metadata buffers.
    """

    def __init__(
        self, factory: Callable, *, allow_graph: bool = False,
        query_limit: int = 1, speculative: bool = False,
    ):
        self._factory = factory
        self._pending = factory()[0]
        self.allow_graph = allow_graph
        self.query_limit = query_limit
        self.speculative = speculative

    def take(self):
        plan = self._pending
        if plan is None:
            plan = self._factory()[0]
        # Consume before launch: a failed native initialization must not leave
        # a partially initialized plan available to a subsequent invocation.
        self._pending = None
        return plan
