# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Encoder-side APC half of DeepSeek-V4.1 SWA bounded replay (#56227) on 0.29.

Pinned vLLM 0.29 lacks ``prefix_replay_tokens``, ``Request.replay_start``,
``NewRequestData.replay_start``, and the scheduler rewind / past-hit allocate
path. This adapter ports those pieces without replacing the whole scheduler:

* ``KVCacheSpec.prefix_replay_tokens`` (default 0) + UniformType ``max`` forward
* ``Request.replay_start = 0`` on init
* ``NewRequestData.replay_start`` + ``from_request`` pass-through
* ``Scheduler``: ``prefix_replay_tokens`` init, ``_mark_prefix_replay``, remote-KV
  rewind, and a source-surgery of ``schedule`` for drop-short-hit + mark/rewind
  + ``num_tokens_past_hit`` allocate
"""

from __future__ import annotations

import functools
import inspect
import textwrap
from types import ModuleType
from typing import Any

from ._common import (
    PatchCompatibilityError,
    load_exact_module,
    require_callable,
    require_class,
)

KV_TARGET_MODULE = "vllm.v1.kv_cache_interface"
REQUEST_TARGET_MODULE = "vllm.v1.request"
OUTPUT_TARGET_MODULE = "vllm.v1.core.sched.output"
SCHEDULER_TARGET_MODULE = "vllm.v1.core.sched.scheduler"

KV_PATCH_ID = "platform.framework_opt.swa_bounded_replay.kv_spec"
REQUEST_PATCH_ID = "platform.framework_opt.swa_bounded_replay.request"
OUTPUT_PATCH_ID = "platform.framework_opt.swa_bounded_replay.new_request_data"
SCHEDULER_PATCH_ID = "platform.framework_opt.swa_bounded_replay.scheduler"

# Back-compat single-module entry used by some inventory helpers.
TARGET_MODULE = SCHEDULER_TARGET_MODULE
PATCH_ID = SCHEDULER_PATCH_ID

KV_TARGETS = (
    f"{KV_TARGET_MODULE}.KVCacheSpec.prefix_replay_tokens",
    f"{KV_TARGET_MODULE}.UniformTypeKVCacheSpecs.prefix_replay_tokens",
)
REQUEST_TARGETS = (f"{REQUEST_TARGET_MODULE}.Request.__init__",)
OUTPUT_TARGETS = (
    f"{OUTPUT_TARGET_MODULE}.NewRequestData.replay_start",
    f"{OUTPUT_TARGET_MODULE}.NewRequestData.from_request",
)
SCHEDULER_TARGETS = (
    f"{SCHEDULER_TARGET_MODULE}.Scheduler.__init__",
    f"{SCHEDULER_TARGET_MODULE}.Scheduler.schedule",
    f"{SCHEDULER_TARGET_MODULE}.Scheduler._mark_prefix_replay",
    f"{SCHEDULER_TARGET_MODULE}.Scheduler._update_waiting_for_remote_kv",
)
TARGETS = SCHEDULER_TARGETS

_KV_MARKER = "_vllm_hcu_swa_bounded_replay_kv_spec_applied"
_REQUEST_MARKER = "_vllm_hcu_swa_bounded_replay_request_applied"
_OUTPUT_MARKER = "_vllm_hcu_swa_bounded_replay_new_request_data_applied"
_SCHEDULER_MARKER = "_vllm_hcu_swa_bounded_replay_scheduler_applied"
_WRAPPER = "_vllm_hcu_swa_bounded_replay_apc_wrapper"

_SHORT_HIT_NEEDLE = (
    "assert num_computed_tokens <= request.num_tokens\n\n"
    "                # Skip request with pending mm encoding prefetches"
)
_SHORT_HIT_REPLACEMENT = (
    "assert num_computed_tokens <= request.num_tokens\n"
    "                if 0 < num_computed_tokens <= self.prefix_replay_tokens:\n"
    "                    # SWA bounded replay: a hit no longer than the\n"
    "                    # replayed window would be recomputed in full and\n"
    "                    # save nothing.\n"
    "                    new_computed_blocks = (\n"
    "                        self.kv_cache_manager.empty_kv_cache_blocks\n"
    "                    )\n"
    "                    num_new_local_computed_tokens = 0\n"
    "                    num_external_computed_tokens = 0\n"
    "                    connector_prefix_cache_hits = 0\n"
    "                    num_computed_tokens = 0\n"
    "                    load_kv_async = False\n"
    "\n"
    "                # Skip request with pending mm encoding prefetches"
)

_REPLAY_INIT_NEEDLE = (
    "            if load_kv_async:\n"
    "                # KVTransfer: loading remote KV, do not allocate for new work."
)
_REPLAY_INIT_REPLACEMENT = (
    "            # SWA bounded replay: recompute the tail of the hit without\n"
    "            # rewriting its cached KV. An async load replays once the KV\n"
    "            # has arrived (_update_waiting_for_remote_kv).\n"
    "            num_replay_tokens = 0\n"
    "\n"
    "            if load_kv_async:\n"
    "                # KVTransfer: loading remote KV, do not allocate for new work."
)

_MARK_NEEDLE = (
    "                request_token_budget = min(token_budget, input_budget - draft_slots)\n"
    "                # Number of tokens to be scheduled.\n"
    "                # We use `request.num_tokens` instead of\n"
    "                # `request.num_prompt_tokens` to consider the resumed\n"
    "                # requests, which have output tokens.\n"
    "                num_new_tokens = request.num_tokens - num_computed_tokens"
)
_MARK_REPLACEMENT = (
    "                request_token_budget = min(token_budget, input_budget - draft_slots)\n"
    "                if did_prefix_cache_lookup:\n"
    "                    # Fresh admission: local and/or sync external hit.\n"
    "                    num_replay_tokens = self._mark_prefix_replay(\n"
    "                        request, num_computed_tokens\n"
    "                    )\n"
    "                    num_computed_tokens -= num_replay_tokens\n"
    "                # Number of tokens to be scheduled.\n"
    "                # We use `request.num_tokens` instead of\n"
    "                # `request.num_prompt_tokens` to consider the resumed\n"
    "                # requests, which have output tokens.\n"
    "                num_new_tokens = request.num_tokens - num_computed_tokens"
)

_ALLOC_NEEDLE = (
    "            new_blocks = self.kv_cache_manager.allocate_slots(\n"
    "                request,\n"
    "                num_new_tokens,\n"
    "                num_new_computed_tokens=num_new_local_computed_tokens,"
)
_ALLOC_REPLACEMENT = (
    "            # Replayed tokens are already counted in the adopted hit; a\n"
    "            # chunk that ends inside the replayed range needs no new slots.\n"
    "            num_tokens_past_hit = max(num_new_tokens - num_replay_tokens, 0)\n"
    "            new_blocks = self.kv_cache_manager.allocate_slots(\n"
    "                request,\n"
    "                num_tokens_past_hit,\n"
    "                num_new_computed_tokens=num_new_local_computed_tokens,"
)


def _replace_once(source: str, needle: str, replacement: str, label: str) -> str:
    count = source.count(needle)
    if count != 1:
        raise PatchCompatibilityError(
            f"required HCU schedule surgery marker {label!r} matched {count} "
            "times; expected 1 (pinned 0.29 waiting-path fingerprint drifted)"
        )
    return source.replace(needle, replacement, 1)


def apply_kv_spec_to_module(module: ModuleType) -> bool:
    interface = load_exact_module(KV_TARGET_MODULE, module)
    kv_spec = require_class(interface, "KVCacheSpec", KV_TARGETS[0])
    uniform = require_class(interface, "UniformTypeKVCacheSpecs", KV_TARGETS[1])
    if getattr(interface, _KV_MARKER, False):
        return False

    if not hasattr(kv_spec, "prefix_replay_tokens"):

        @property
        def prefix_replay_tokens(self) -> int:
            """Tokens of a prefix hit to recompute for SWA bounded replay."""
            return 0

        kv_spec.prefix_replay_tokens = prefix_replay_tokens  # type: ignore[attr-defined]

    if "prefix_replay_tokens" not in vars(uniform):

        @property
        def uniform_prefix_replay_tokens(self) -> int:
            specs = self.kv_cache_specs.values()
            return max((spec.prefix_replay_tokens for spec in specs), default=0)

        uniform.prefix_replay_tokens = uniform_prefix_replay_tokens  # type: ignore[attr-defined]

    setattr(interface, _KV_MARKER, True)
    return True


def apply_request_to_module(module: ModuleType) -> bool:
    request_mod = load_exact_module(REQUEST_TARGET_MODULE, module)
    request_cls = require_class(request_mod, "Request", REQUEST_TARGETS[0])
    if getattr(request_mod, _REQUEST_MARKER, False):
        return False

    original_init = require_callable(request_cls, "__init__", REQUEST_TARGETS[0])

    @functools.wraps(original_init)
    def hcu_init(self, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        if not hasattr(self, "replay_start"):
            self.replay_start = 0

    setattr(hcu_init, _WRAPPER, True)
    request_cls.__init__ = hcu_init  # type: ignore[method-assign]
    setattr(request_mod, _REQUEST_MARKER, True)
    return True


def apply_new_request_data_to_module(module: ModuleType) -> bool:
    output_mod = load_exact_module(OUTPUT_TARGET_MODULE, module)
    new_req = require_class(output_mod, "NewRequestData", OUTPUT_TARGETS[0])
    if getattr(output_mod, _OUTPUT_MARKER, False):
        return False

    original_init = require_callable(new_req, "__init__", f"{OUTPUT_TARGET_MODULE}.NewRequestData.__init__")
    original_from = require_callable(new_req, "from_request", OUTPUT_TARGETS[1])

    @functools.wraps(original_init)
    def hcu_init(self, *args: Any, **kwargs: Any) -> None:
        replay_start = int(kwargs.pop("replay_start", 0) or 0)
        original_init(self, *args, **kwargs)
        self.replay_start = replay_start

    @classmethod
    @functools.wraps(original_from)
    def hcu_from_request(cls, request, block_ids, *args: Any, **kwargs: Any):
        data = original_from(request, block_ids, *args, **kwargs)
        data.replay_start = int(getattr(request, "replay_start", 0) or 0)
        return data

    setattr(hcu_init, _WRAPPER, True)
    setattr(hcu_from_request, _WRAPPER, True)
    new_req.__init__ = hcu_init  # type: ignore[method-assign]
    new_req.from_request = hcu_from_request  # type: ignore[method-assign]
    if not hasattr(new_req, "replay_start"):
        new_req.replay_start = 0
    setattr(output_mod, _OUTPUT_MARKER, True)
    return True


def _mark_prefix_replay(self, request, num_hit_tokens: int) -> int:
    """Record where a prefix hit's replay starts; return tokens to recompute."""
    if not self.prefix_replay_tokens:
        return 0
    num_replay_tokens = min(self.prefix_replay_tokens, num_hit_tokens)
    request.replay_start = num_hit_tokens - num_replay_tokens
    return num_replay_tokens


def _rebuild_schedule(original_schedule):
    try:
        source = textwrap.dedent(inspect.getsource(original_schedule))
    except (OSError, TypeError) as exc:
        raise PatchCompatibilityError(
            "required HCU patch could not read Scheduler.schedule source"
        ) from exc

    source = _replace_once(source, _SHORT_HIT_NEEDLE, _SHORT_HIT_REPLACEMENT, "short-hit")
    source = _replace_once(
        source, _REPLAY_INIT_NEEDLE, _REPLAY_INIT_REPLACEMENT, "num_replay_tokens init"
    )
    source = _replace_once(source, _MARK_NEEDLE, _MARK_REPLACEMENT, "mark+rewind")
    source = _replace_once(source, _ALLOC_NEEDLE, _ALLOC_REPLACEMENT, "past-hit allocate")

    namespace: dict[str, Any] = {}
    filename = (
        inspect.getsourcefile(original_schedule)
        or original_schedule.__code__.co_filename
    )
    exec(compile(source, filename, "exec"), original_schedule.__globals__, namespace)
    rebuilt = namespace.get("schedule")
    if not callable(rebuilt):
        raise PatchCompatibilityError(
            "could not rebuild Scheduler.schedule after SWA bounded-replay surgery"
        )
    functools.update_wrapper(rebuilt, original_schedule)
    setattr(rebuilt, _WRAPPER, True)
    return rebuilt


def apply_scheduler_to_module(module: ModuleType) -> bool:
    target = load_exact_module(SCHEDULER_TARGET_MODULE, module)
    scheduler = require_class(target, "Scheduler", SCHEDULER_TARGETS[0])
    if getattr(target, _SCHEDULER_MARKER, False):
        return False

    original_init = require_callable(scheduler, "__init__", SCHEDULER_TARGETS[0])
    original_schedule = require_callable(scheduler, "schedule", SCHEDULER_TARGETS[1])
    original_remote = require_callable(
        scheduler, "_update_waiting_for_remote_kv", SCHEDULER_TARGETS[3]
    )

    @functools.wraps(original_init)
    def hcu_init(
        self,
        vllm_config,
        kv_cache_config,
        structured_output_manager,
        block_size,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        original_init(
            self,
            vllm_config,
            kv_cache_config,
            structured_output_manager,
            block_size,
            *args,
            **kwargs,
        )
        replay_windows = {
            group.kv_cache_spec.prefix_replay_tokens
            for group in kv_cache_config.kv_cache_groups
            if group.kv_cache_spec.prefix_replay_tokens > 0
        }
        assert len(replay_windows) <= 1, (
            f"Prefix replay windows should agree: {sorted(replay_windows)}"
        )
        self.prefix_replay_tokens = replay_windows.pop() if replay_windows else 0

    def hcu_update_waiting_for_remote_kv(self, request) -> None:
        """KV Connector: update request state after async recv is finished."""
        assert self.connector is not None

        if request.request_id in self.failed_recving_kv_req_ids:
            if request.num_computed_tokens:
                self.kv_cache_manager.cache_blocks(
                    request, request.num_computed_tokens
                )
                if self.needs_kv_cache_zeroing:
                    self.kv_cache_manager.record_blocks_for_zeroing(
                        request.request_id, request.num_computed_tokens
                    )
            else:
                self.kv_cache_manager.free(request)
            self.failed_recving_kv_req_ids.remove(request.request_id)
        else:
            self.kv_cache_manager.cache_blocks(request, request.num_computed_tokens)

        # SWA bounded replay recomputes the tail of the hit, which covers the
        # last token; otherwise a full prompt hit re-computes that token so the
        # next one can be sampled.
        num_replay_tokens = self._mark_prefix_replay(
            request, request.num_computed_tokens
        )
        if num_replay_tokens > 0:
            request.num_computed_tokens -= num_replay_tokens
        elif request.num_computed_tokens == request.num_tokens:
            request.num_computed_tokens = request.num_tokens - 1
        self.finished_recving_kv_req_ids.remove(request.request_id)

    setattr(hcu_init, _WRAPPER, True)
    setattr(hcu_update_waiting_for_remote_kv, _WRAPPER, True)
    functools.update_wrapper(hcu_update_waiting_for_remote_kv, original_remote)

    scheduler.__init__ = hcu_init  # type: ignore[method-assign]
    scheduler.schedule = _rebuild_schedule(original_schedule)  # type: ignore[method-assign]
    scheduler._mark_prefix_replay = _mark_prefix_replay  # type: ignore[method-assign]
    scheduler._update_waiting_for_remote_kv = hcu_update_waiting_for_remote_kv  # type: ignore[method-assign]
    setattr(target, _SCHEDULER_MARKER, True)
    return True


def apply_to_module(module: ModuleType) -> bool:
    """Exact-import callback for the scheduler module (primary registration)."""
    return apply_scheduler_to_module(module)


def apply(module: ModuleType | None = None) -> bool:
    applied = apply_kv_spec_to_module(load_exact_module(KV_TARGET_MODULE, None))
    applied = (
        apply_request_to_module(load_exact_module(REQUEST_TARGET_MODULE, None))
        or applied
    )
    applied = (
        apply_new_request_data_to_module(load_exact_module(OUTPUT_TARGET_MODULE, None))
        or applied
    )
    applied = (
        apply_scheduler_to_module(load_exact_module(SCHEDULER_TARGET_MODULE, module))
        or applied
    )
    return applied


__all__ = [
    "PATCH_ID",
    "TARGET_MODULE",
    "TARGETS",
    "KV_PATCH_ID",
    "KV_TARGET_MODULE",
    "KV_TARGETS",
    "REQUEST_PATCH_ID",
    "REQUEST_TARGET_MODULE",
    "REQUEST_TARGETS",
    "OUTPUT_PATCH_ID",
    "OUTPUT_TARGET_MODULE",
    "OUTPUT_TARGETS",
    "SCHEDULER_PATCH_ID",
    "SCHEDULER_TARGET_MODULE",
    "SCHEDULER_TARGETS",
    "apply",
    "apply_to_module",
    "apply_kv_spec_to_module",
    "apply_request_to_module",
    "apply_new_request_data_to_module",
    "apply_scheduler_to_module",
]
