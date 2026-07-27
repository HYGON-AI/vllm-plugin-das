# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

"""Source-level ownership guards for the HCU custom all-reduce extension."""

from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CUSTOM_AR_HEADER = REPO_ROOT / "vllm_hcu" / "csrc" / "custom_all_reduce.cuh"


def _class_destructor(source: str) -> str:
    marker = "~CustomAllreduce()"
    start = source.index(marker)
    body_start = source.index("{", start)
    depth = 0
    for offset, character in enumerate(source[body_start:], start=body_start):
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return source[body_start : offset + 1]
    raise AssertionError("CustomAllreduce destructor has an unbalanced body")


def test_optional_hdp_allocation_has_explicit_ownership_guard() -> None:
    source = CUSTOM_AR_HEADER.read_text(encoding="utf-8")

    assert re.search(
        r"uint32_t\s*\*\*\s*dev_curr_hdp_reg\s*=\s*nullptr\s*;", source
    )
    assert re.search(
        r"if\s*\(\s*!fully_connected\s*\)\s*\{\s*"
        r"CUDACHECK\(cudaMalloc\(\(void\*\*\)&dev_curr_hdp_reg,",
        source,
    )

    destructor = _class_destructor(source)
    assert re.search(
        r"if\s*\(\s*dev_curr_hdp_reg\s*!=\s*nullptr\s*\)\s*\{\s*"
        r"CUDACHECK\(cudaFree\(dev_curr_hdp_reg\)\);\s*\}",
        destructor,
    )
    assert destructor.count("cudaFree(dev_curr_hdp_reg)") == 1


def test_other_native_resources_keep_single_teardown_owner() -> None:
    source = CUSTOM_AR_HEADER.read_text(encoding="utf-8")
    destructor = _class_destructor(source)

    # The HDP allocation is conditional on topology. A surviving instance's
    # IPC entries have successful opens, and stopEvent is created for every
    # successfully-constructed CustomAllreduce instance.
    assert destructor.count("cudaIpcCloseMemHandle(ptr)") == 1
    assert destructor.count("cudaEventDestroy(stopEvent)") == 1
