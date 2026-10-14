# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Fix the wheel DSpark lm_head share for *ForConditionalGeneration targets.

``vllm.v1.worker.gpu.spec_decode.dspark.utils.load_dspark_model`` resolves the
target LM head with ``getattr(target_model, "lm_head", None)`` only.  A Kimi-K3
target is a ``*ForConditionalGeneration`` wrapper whose head lives on
``get_language_model()`` (mirroring fork 877e6996 ``dspark/utils.py``
``get_target_lm_head``), so the wheel share block silently skips and the K3
draft model stays without a head until ``compute_logits`` raises
"DSpark target LM head has not been attached" during ``profile_run``.

This adapter re-runs the wheel share with the fork-parity resolved head.  It
carries none of the role-separation semantics; the logic was extracted verbatim
from the previously bundled ``patch_dspark_role_separation`` adapter
(``_share_target_lm_head_fork_parity``) because the share is required for every
K3 DSpark start regardless of communicator topology.
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

TARGET_MODULE = "vllm.v1.worker.gpu.spec_decode.dspark.speculator"
PATCH_ID = "worker.framework_opt.spec_decode.dspark_target_lm_head_share"
TARGETS = (f"{TARGET_MODULE}.DSparkSpeculator.load_draft_model",)
_MARKER = "_vllm_hcu_dspark_target_lm_head_share_applied"
_LOAD_WRAPPER = "_vllm_hcu_dspark_target_lm_head_share_load_wrapper"


def _resolve_target_lm_head_fork_parity(target_model):
    """Fork parity for ``get_target_lm_head``: a *ForConditionalGeneration
    target keeps its lm_head on ``get_language_model()``, not on the wrapper.
    The wheel share block only probes ``target_model.lm_head`` and therefore
    silently skips the share for K3 targets, starving the guard's
    non-replicated-head premise (fork 877e6996 dspark/utils.py)."""

    language_model = (
        target_model.get_language_model()
        if hasattr(target_model, "get_language_model")
        else target_model
    )
    return getattr(language_model, "lm_head", None) or getattr(
        target_model, "lm_head", None
    )


def _share_target_lm_head_fork_parity(model, target_model) -> None:
    """Re-run the wheel lm_head share with the fork-resolved target head."""

    from vllm.v1.worker.gpu.spec_decode.eagle.utils import _should_share

    target_lm_head = _resolve_target_lm_head_fork_parity(target_model)
    if target_lm_head is None:
        return
    draft_lm_head = getattr(model, "lm_head", None)
    if not _should_share(model, "has_own_lm_head", draft_lm_head, target_lm_head):
        return
    if draft_lm_head is not None:
        del model.lm_head
    model.lm_head = target_lm_head


def apply_to_module(module: ModuleType) -> bool:
    """Interpose ``DSparkSpeculator.load_draft_model`` on the paired runtime."""
    spec = load_exact_module(TARGET_MODULE, module)
    cls = require_class(
        spec, "DSparkSpeculator", f"{TARGET_MODULE}.DSparkSpeculator"
    )
    wrapped = (
        (cls, "load_draft_model", TARGETS[0], _LOAD_WRAPPER),
    )
    if already_applied(spec, _MARKER, wrapped):
        return False

    original_load = require_callable(cls, "load_draft_model", TARGETS[0])
    require_exact_signature(
        original_load,
        TARGETS[0],
        positional=("self", "target_model", "target_attn_layer_names"),
    )

    @functools.wraps(original_load)
    def hcu_load_draft_model(self, target_model, target_attn_layer_names):
        model = original_load(self, target_model, target_attn_layer_names)
        # Wheel share miss: for a *ForConditionalGeneration target the wheel
        # resolves target_model.lm_head (absent on the wrapper) and skips the
        # sharing, leaving the K3 draft without a head.  Re-run the share with
        # the fork-resolved head.
        _share_target_lm_head_fork_parity(model, target_model)
        return model

    setattr(hcu_load_draft_model, _LOAD_WRAPPER, True)
    cls.load_draft_model = hcu_load_draft_model
    setattr(spec, _MARKER, True)
    return True
