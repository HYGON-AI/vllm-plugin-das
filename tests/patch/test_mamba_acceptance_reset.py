# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Regression: the per-slot acceptance count must reset on add_request.

The v0.25.1 wheel performs the reset only under ``_align_mode``; with the
default ``mamba_cache_mode="none"`` a recycled slot inherits the dead
request's ``num_accepted_tokens_gpu`` entries and the GDN rollback rewinds by
wrong depths (fork 877e6996 resets unconditionally).
"""

from __future__ import annotations

import re
import sys
import types

import torch

from vllm_hcu.patch.worker.framework_opt.patch_mamba_acceptance_reset import (
    apply_to_module,
)


def _build_state_class():
    class MambaHybridModelState:
        _align_mode = False

        def __init__(self) -> None:
            self.num_accepted_tokens_gpu = torch.tensor([7, 3], dtype=torch.int64)
            self.reset_calls: list[int] = []

        def add_request(self, req_index: int, new_req_data) -> None:
            self.reset_calls.append(req_index)

    return MambaHybridModelState


def _build_module():
    module = types.ModuleType(
        "vllm.v1.worker.gpu.model_states.mamba_hybrid"
    )
    module.MambaHybridModelState = _build_state_class()
    return module


def test_patch_applies_and_marks_module():
    module = _build_module()
    assert apply_to_module(module) is True
    assert module._vllm_hcu_mamba_acceptance_reset_applied is True


def test_add_request_resets_slot_unconditionally():
    module = _build_module()
    apply_to_module(module)
    state = module.MambaHybridModelState()
    state.add_request(0, new_req_data=None)
    assert int(state.num_accepted_tokens_gpu[0]) == 1
    # other slots untouched
    assert int(state.num_accepted_tokens_gpu[1]) == 3

    state.add_request(1, new_req_data=None)
    assert int(state.num_accepted_tokens_gpu[1]) == 1


def test_reapply_is_idempotent():
    module = _build_module()
    assert apply_to_module(module) is True
    assert apply_to_module(module) is False
    state = module.MambaHybridModelState()
    state.add_request(1, new_req_data=None)
    assert int(state.num_accepted_tokens_gpu[1]) == 1
    assert state.reset_calls == [1]


def test_real_module_contract():
    """The deployed wheel must expose the exact wrap contract."""
    target = "vllm.v1.worker.gpu.model_states.mamba_hybrid"
    real = __import__(target, fromlist=["MambaHybridModelState"])
    cls = real.MambaHybridModelState
    import inspect

    sig = inspect.signature(cls.add_request)
    assert list(sig.parameters) == ["self", "req_index", "new_req_data"]
    # The reset target must EXIST on the deployed class: this patch rewinds
    # num_accepted_tokens_gpu per slot, so a wheel without the buffer makes
    # the whole patch a no-op (review 2026-10-13 P3: the ``or True`` made
    # this check vacuous).  Match an ATTRIBUTE USE/ASSIGNMENT, not bare
    # occurrences that could live in a comment or docstring (R3 #6).
    assert re.search(r"\.num_accepted_tokens_gpu\s*=", inspect.getsource(cls))
    # the wheel reset must be align-gated (the bug this patch fixes)
    src = inspect.getsource(cls.add_request)
    assert "_align_mode" in src
