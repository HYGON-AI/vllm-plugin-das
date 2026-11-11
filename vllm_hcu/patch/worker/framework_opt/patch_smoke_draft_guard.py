# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Load the DSpark draft with the smoke layer limit muted (upstream c4fc87056e).

Upstream wraps ``get_model`` inside ``load_dspark_model`` with
``disable_smoke_layer_limit()``.  The deployed core has no smoke layering, so
the switch lives in ``platform/core_fix/patch_smoke_layer_filter`` and is
entered here around every draft entry point by object identity, not by
rebinding a helper's module global (callers holding the old reference would
bypass that):

* ``DSparkSpeculator.load_draft_model`` — the outermost draft entry; wraps
  whatever is installed when this callback runs, including the HCU
  role-separation loader if it was applied first.
* ``load_dspark_model`` in ``dspark.utils`` and the name bound by
  ``from ... import load_dspark_model`` in ``dspark.speculator`` — still
  covered when a later patch replaces ``load_draft_model`` but keeps routing
  through ``load_dspark_model``.

The switch is reentrant, so stacked wrappers are harmless.  Register this
adapter after any other adapter that replaces these symbols.
"""

from __future__ import annotations

import functools
import importlib
import logging
from types import ModuleType

from vllm_hcu.patch.platform.core_fix.patch_smoke_layer_filter import (
    disable_smoke_layer_limit,
    smoke_weight_env_enabled,
)

from ._common import PatchCompatibilityError, load_exact_module

TARGET_MODULE = "vllm.v1.worker.gpu.spec_decode.dspark.speculator"
_UTILS_MODULE = "vllm.v1.worker.gpu.spec_decode.dspark.utils"
PATCH_ID = "worker.framework_opt.spec_decode.dspark_smoke_draft_guard"
TARGETS = (
    f"{TARGET_MODULE}.DSparkSpeculator.load_draft_model",
    f"{TARGET_MODULE}.load_dspark_model",
    f"{_UTILS_MODULE}.load_dspark_model",
)
_MARKER = "_vllm_hcu_smoke_draft_guard_applied"
_WRAPPER = "_vllm_hcu_smoke_draft_guard_wrapper"

_log = logging.getLogger("vllm_hcu.patch.kimi_k3_smoke_layer_filter")


def _guard(function):
    @functools.wraps(function)
    def smoke_safe(*args, **kwargs):
        with disable_smoke_layer_limit():
            return function(*args, **kwargs)

    # functools.wraps copies __dict__, so a previous owner's wrapper marker
    # (e.g. role separation's) stays visible to its own stale-marker check.
    setattr(smoke_safe, _WRAPPER, True)
    return smoke_safe


def _wrap_points(speculator: ModuleType):
    points = []
    cls = getattr(speculator, "DSparkSpeculator", None)
    if isinstance(cls, type) and callable(getattr(cls, "load_draft_model", None)):
        points.append((cls, "load_draft_model"))
    if callable(getattr(speculator, "load_dspark_model", None)):
        points.append((speculator, "load_dspark_model"))
    try:
        utils = importlib.import_module(_UTILS_MODULE)
    except ImportError:
        utils = None
    if utils is not None and callable(getattr(utils, "load_dspark_model", None)):
        points.append((utils, "load_dspark_model"))
    return points


def apply_to_module(module: ModuleType) -> bool:
    speculator = load_exact_module(TARGET_MODULE, module)
    points = _wrap_points(speculator)
    if getattr(speculator, _MARKER, False):
        for owner, name in points:
            if not getattr(getattr(owner, name), _WRAPPER, False):
                raise PatchCompatibilityError(
                    f"smoke draft guard on {name} was replaced after it was "
                    "applied; register patch_smoke_draft_guard last"
                )
        return False
    if not points:
        if smoke_weight_env_enabled():
            raise PatchCompatibilityError(
                "VLLM_SKIP_WEIGHT is set but no DSpark draft loader entry point "
                f"was found in {TARGET_MODULE}"
            )
        return False

    for owner, name in points:
        current = getattr(owner, name)
        if not getattr(current, _WRAPPER, False):
            setattr(owner, name, _guard(current))
    setattr(speculator, _MARKER, True)
    _log.info(
        "[k3-smoke-filter] DSpark draft loader guarded: %s",
        ", ".join(f"{getattr(o, '__name__', o)}.{n}" for o, n in points),
    )
    return True


__all__ = ["PATCH_ID", "TARGETS", "TARGET_MODULE", "apply_to_module"]
