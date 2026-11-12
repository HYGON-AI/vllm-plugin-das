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

SWITCH = "VLLM_HCU_GEMMA_BW1000"

# Read together with SWITCH so that a launch script still carrying one of them
# fails loudly instead of quietly serving an unoptimized engine.
_RETIRED = ("VLLM_HCU_GEMMA4_USE_FA", "VLLM_HCU_GEMMA4_BW1000_OPT")

_TRUE = {"1", "true"}
_FALSE = {"", "0", "false"}


def gemma4_bw1000_enabled() -> bool:
    """Is the Gemma4 BW1000 optimization bundle switched on?"""
    for retired in _RETIRED:
        if os.environ.get(retired) is not None:
            raise ValueError(
                f"{retired} has been replaced by {SWITCH}, which enables the "
                "whole Gemma4 BW1000 bundle at once. Unset the old variable "
                f"and set {SWITCH}=1."
            )
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
