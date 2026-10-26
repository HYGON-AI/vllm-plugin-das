# SPDX-License-Identifier: Apache-2.0
"""Read explicitly supplied tuning data; no machine-specific recipes ship here."""
from functools import lru_cache
import json
import os
from pathlib import Path
import re


@lru_cache(maxsize=8)
def _read(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("Step5 tuning data requires schema_version=1")
    if value.get("architecture") != "gfx938":
        raise ValueError("Step5 tuning data is qualified only for gfx938")
    return value


def tuning_data():
    path = os.environ.get("HCU_STEP5_TUNING_CONFIG", "")
    return _read(str(Path(path).resolve())) if path else {}


@lru_cache(maxsize=1)
def dense_recipe():
    dense = tuning_data().get("dense")
    if dense is None:
        return None
    if not isinstance(dense, dict) or not re.fullmatch(
        r"[0-9a-f]{64}", str(dense.get("hipblaslt_sha256", ""))
    ):
        raise ValueError("Step5 Dense tuning requires the qualified hipBLASLt SHA256")
    rows = dense.get("algorithms")
    if not isinstance(rows, dict) or not rows:
        raise ValueError("Step5 Dense tuning requires algorithm choices")
    choices = {}
    for shape, algorithm in rows.items():
        if not re.fullmatch(r"[1-9][0-9]*,[1-9][0-9]*,[1-9][0-9]*", shape):
            raise ValueError("Step5 Dense shape must be positive M,N,K")
        if type(algorithm) is not int or algorithm < 0:
            raise ValueError("Step5 Dense algorithm ID must be a nonnegative integer")
        choices[tuple(map(int, shape.split(",")))] = algorithm
    return choices, dense["hipblaslt_sha256"]


@lru_cache(maxsize=2)
def asm_recipe(local_experts):
    asm = tuning_data().get("asm_g1")
    if asm is None:
        return None
    if not isinstance(asm, dict) or asm.get("block_m") != 128:
        raise ValueError("Step5 ASM metadata requires a 128-row stage1 tile")
    solution = asm.get("solution")
    persist = asm.get("persist_by_local_experts", {}).get(str(local_experts))
    if any(type(x) is not int or x <= 0 for x in (solution, persist)):
        raise ValueError("Step5 ASM tuning requires positive solution and persist IDs")
    return solution, persist
