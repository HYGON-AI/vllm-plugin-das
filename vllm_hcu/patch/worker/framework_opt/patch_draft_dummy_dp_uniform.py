# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Reuse the agreed DP uniform token count in the MTP dummy propose."""

from __future__ import annotations

import functools
from types import ModuleType

from ._common import (
    already_applied,
    load_exact_module,
    require_callable,
    require_class,
    require_exact_signature,
)

TARGET_MODULE = "vllm.v1.worker.gpu.spec_decode.autoregressive.speculator"
PATCH_ID = "worker.framework_opt.spec_decode.draft_dummy_dp_uniform"
TARGETS = (f"{TARGET_MODULE}.AutoRegressiveSpeculator.propose",)
_MARKER = "_vllm_hcu_draft_dummy_dp_uniform_applied"
_WRAPPER = "_vllm_hcu_draft_dummy_dp_uniform_wrapper"

_POSITIONAL = (
    "self",
    "input_batch",
    "attn_metadata",
    "slot_mappings",
    "last_hidden_states",
    "aux_hidden_states",
    "num_sampled",
    "num_rejected",
    "last_sampled",
    "next_prefill_tokens",
    "temperature",
    "seeds",
    "dp_sync",
    "dummy_run",
    "skip_attn_for_dummy_run",
    "mm_inputs",
    "is_profile",
)
_DEFAULTS = {
    "dp_sync": None,
    "dummy_run": False,
    "skip_attn_for_dummy_run": False,
    "mm_inputs": None,
    "is_profile": False,
}
# Indices into ``args`` (``self`` excluded).
_DP_SYNC_IDX = _POSITIONAL.index("dp_sync") - 1
_DUMMY_RUN_IDX = _POSITIONAL.index("dummy_run") - 1


def _arg(args, kwargs, name, index):
    if name in kwargs:
        return kwargs[name]
    return args[index] if len(args) > index else _DEFAULTS[name]


def apply_to_module(module: ModuleType) -> bool:
    spec = load_exact_module(TARGET_MODULE, module)
    cls = require_class(
        spec, "AutoRegressiveSpeculator", f"{TARGET_MODULE}.AutoRegressiveSpeculator"
    )
    wrapped = ((cls, "propose", TARGETS[0], _WRAPPER),)
    if already_applied(spec, _MARKER, wrapped):
        return False
    original = require_callable(cls, "propose", TARGETS[0])
    require_exact_signature(
        original, TARGETS[0], positional=_POSITIONAL, defaults=_DEFAULTS
    )
    require_callable(
        spec, "get_uniform_decode_token_count", f"{TARGET_MODULE}.get_uniform_decode_token_count"
    )

    @functools.wraps(original)
    def hcu_propose(self, *args, **kwargs):
        dp_sync = _arg(args, kwargs, "dp_sync", _DP_SYNC_IDX)
        if dp_sync is None or not _arg(args, kwargs, "dummy_run", _DUMMY_RUN_IDX):
            return original(self, *args, **kwargs)
        # A dummy propose rebuilds its InputBatch from the cudagraph-padded
        # token count, so a locally recomputed uniform count can differ from the
        # one every DP rank agreed on. dispatch_cg_and_sync_dp already selects
        # the graph from dp_sync.uniform_token_count; only its consistency
        # assertion reads the local value, so hand it the agreed one.
        agreed = dp_sync.uniform_token_count
        local = spec.get_uniform_decode_token_count
        spec.get_uniform_decode_token_count = lambda *_args, **_kwargs: agreed
        try:
            return original(self, *args, **kwargs)
        finally:
            spec.get_uniform_decode_token_count = local

    setattr(hcu_propose, _WRAPPER, True)
    setattr(cls, "_vllm_hcu_original_propose", original)
    setattr(cls, "propose", hcu_propose)
    setattr(spec, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
