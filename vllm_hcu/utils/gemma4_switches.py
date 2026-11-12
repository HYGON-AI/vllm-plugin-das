# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""The single switch for the Gemma4 BW1000 optimizations.

``VLLM_HCU_GEMMA_BW1000=1`` enables all of them together:

* the FA attention path for the full-attention (head 512) layers, including
  the paged FP8 gather that replaced the whole-pool FP8->BF16 conversion;
* the BW1000 attention launch tuning (global head512 tiling, SWA unit-scale
  FP8-read-as-BF16);
* the LM-head work: the rocBLAS algorithm table, the softcap lookup table,
  the 320/400 row buckets and the fused draft argmax.

They used to be two separate variables, ``VLLM_HCU_GEMMA4_USE_FA`` and
``VLLM_HCU_GEMMA4_BW1000_OPT``.  Nothing was ever measured with only one of
them set, and splitting them only made it possible to run half the stack by
accident, so they are now one.

The switch stays opt-in rather than defaulting to on.  The install paths
assert Gemma4-specific invariants -- vocabulary 262144, ``soft_cap`` 30.0, a
bf16 lm_head of a known shape, TP1 -- and load a rocBLAS table validated for
one torch/HIP/rocBLAS/arch combination.  This plugin also serves DeepSeek, GLM
and Kimi, where those assertions would fail at ``load_model``; the FP8
checkpoint of this very model has a quantized lm_head and would fail too.
Defaulting to on would therefore mean making every install path degrade
silently, which is the opposite of what those assertions are for: they are
what catches an "installed but doing nothing" state, which is exactly how the
320-only row bucket came to look installed while doing nothing in production.
"""

from __future__ import annotations

import os

from vllm.logger import init_logger

logger = init_logger(__name__)

SWITCH = "VLLM_HCU_GEMMA_BW1000"

# Retired spellings. A launch script still carrying one of these would
# otherwise serve an unoptimized engine while looking configured.
_RETIRED = ("VLLM_HCU_GEMMA4_USE_FA", "VLLM_HCU_GEMMA4_BW1000_OPT")

_TRUE = {"1", "true"}
_FALSE = {"", "0", "false"}


def gemma4_bw1000_enabled() -> bool:
    """Is the Gemma4 BW1000 optimization bundle switched on?"""
    raw = os.environ.get(SWITCH)
    if raw is None:
        return False
    value = raw.strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    # A typo must not read as "off": that would turn an intended optimization
    # into a no-op with nothing in the log to show it.
    raise ValueError(f"{SWITCH} must be 0/1 or false/true, got {raw!r}")


def _retired_in_environment() -> tuple[str, ...]:
    return tuple(name for name in _RETIRED if os.environ.get(name) is not None)


def warn_retired_switches() -> None:
    """Note any retired spelling, from a place that must not fail.

    Called during model registration, which runs for every model this plugin
    serves and from a ``vllm.general_plugins`` entry point whose invocation
    vLLM does not wrap in ``try``/``except`` -- raising here would abort
    startup for DeepSeek, GLM or Kimi over a variable that has nothing to do
    with the model being served.  The case worth failing on is narrower and
    is handled by ``check_retired_switches`` below.
    """
    retired = _retired_in_environment()
    if retired:
        logger.warning(
            "%s %s retired; the Gemma4 BW1000 bundle is now enabled by %s=1. "
            "The old name(s) are ignored.",
            ", ".join(retired),
            "is" if len(retired) == 1 else "are",
            SWITCH,
        )


def check_retired_switches(vllm_config) -> None:
    """Refuse to serve Gemma4 unoptimized while a retired switch says otherwise.

    Only one combination is actually dangerous: a retired name is set, the new
    switch is not, and the model being loaded is the one those names used to
    optimize.  The operator believes the optimization is on and it is off, and
    nothing in the log would say so.  Anything else -- a different model, or
    the new switch also being set -- makes the leftover variable harmless, so
    it stays a warning.
    """
    retired = _retired_in_environment()
    if not retired or gemma4_bw1000_enabled():
        return
    hf_config = getattr(getattr(vllm_config, "model_config", None), "hf_config", None)
    if getattr(hf_config, "model_type", None) != "gemma4":
        return
    raise ValueError(
        f"{', '.join(retired)} no longer enables anything, so this Gemma4 "
        f"engine would serve unoptimized. The whole BW1000 bundle is now one "
        f"switch: unset the old name(s) and set {SWITCH}=1."
    )
