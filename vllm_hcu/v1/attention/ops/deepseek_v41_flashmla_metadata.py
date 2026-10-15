# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""CPU-only ownership of V4.1's per-build, per-layer native decode plans.

An empty native scheduler does not allocate GPU buffers. The native call owns
initialization. Never share that initialized state between source layers or
successive forwards, even when shapes and compression ratios match.
"""

from __future__ import annotations

from collections.abc import Callable

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
