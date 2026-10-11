# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Reset the per-slot speculative acceptance count on every new request.

Fork 877e6996 ``model_states/mamba_hybrid.py`` resets
``num_accepted_tokens_gpu[req_index]`` unconditionally in ``add_request``
("must reset ... which could be stale").  The pinned v0.25.1 wheel performs the
same reset only inside ``if self._align_mode:``; with the default
``mamba_cache_mode="none"`` the branch never runs, so a recycled request slot
inherits the dead request's per-position acceptance counts and the GDN state
rollback rewinds by the wrong depths.  bs1 never reuses slots and keeps its
healthy acceptance; bs>=2 collapses (measured 2026-10-10: bs1 95.3% vs
bs8 15.8% with per-position deep falloff).

This adapter restores the fork-parity unconditional reset.  The wheel's
align-mode seeding inside ``_align_mode`` is left untouched.
"""

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

TARGET_MODULE = "vllm.v1.worker.gpu.model_states.mamba_hybrid"
PATCH_ID = "worker.framework_opt.model_states.mamba_acceptance_reset"
TARGETS = (f"{TARGET_MODULE}.MambaHybridModelState.add_request",)
_MARKER = "_vllm_hcu_mamba_acceptance_reset_applied"
_WRAPPER = "_vllm_hcu_mamba_acceptance_reset_wrapper"


def apply_to_module(module: ModuleType) -> bool:
    """Interpose ``MambaHybridModelState.add_request`` on the paired runtime."""
    state_mod = load_exact_module(TARGET_MODULE, module)
    cls = require_class(
        state_mod,
        "MambaHybridModelState",
        f"{TARGET_MODULE}.MambaHybridModelState",
    )
    wrapped = ((cls, "add_request", TARGETS[0], _WRAPPER),)
    if already_applied(state_mod, _MARKER, wrapped):
        return False

    original = require_callable(cls, "add_request", TARGETS[0])
    require_exact_signature(
        original,
        TARGETS[0],
        positional=("self", "req_index", "new_req_data"),
    )

    @functools.wraps(original)
    def hcu_add_request(self, req_index, new_req_data):
        result = original(self, req_index, new_req_data)
        # Fork 877e6996 parity: the wheel gates this reset on _align_mode,
        # which is inactive with the default mamba_cache_mode="none"; a
        # reused slot then inherits the dead request's acceptance counts and
        # the GDN rollback rewinds by wrong depths.
        self.num_accepted_tokens_gpu[req_index].fill_(1)
        return result

    setattr(hcu_add_request, _WRAPPER, True)
    cls.add_request = hcu_add_request
    setattr(state_mod, _MARKER, True)
    return True
