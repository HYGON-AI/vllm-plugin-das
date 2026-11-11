# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Gemma4 BW1000 LM-head GEMM tuning (TP1, BF16, full 262144 vocabulary).

Ported from the validated stage1 bundle; no new kernel:
  * load the fixed rocBLAS algorithm table (TunableOp, tuning disabled);
  * pad 128..319 hidden rows to the tuned 320-row GEMM.
"""

from __future__ import annotations

import csv
import hashlib
import time
from pathlib import Path

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

TABLE_PATH = Path(__file__).resolve().parent / "gemma4_head_results0.csv"
HIDDEN = 2816
VOCAB = 262144
SOFTCAP = 30.0
BUCKET = 320
BUCKET_ROWS = tuple(range(128, BUCKET))

_table_loaded = False
_bucket_installed = False


def load_head_table() -> dict | None:
    """Load and verify the fixed algorithm table in this worker process.

    Idempotent: a second call in the same process is a no-op.
    """
    global _table_loaded
    if _table_loaded:
        return None
    import torch.cuda.tunable as tunable

    raw = TABLE_PATH.read_bytes()
    if b"\r" in raw:
        raise RuntimeError("TunableOp CSV requires LF line endings")
    tunable.enable(True)
    tunable.tuning_enable(False)
    tunable.set_filename(str(TABLE_PATH), insert_device_ordinal=False)
    if not tunable.read_file(str(TABLE_PATH)):
        raise RuntimeError(
            "TunableOp validator rejected the Gemma4 head table "
            "(torch/HIP/rocBLAS/arch differ from the validated environment)"
        )
    expected = {
        tuple(r[:3])
        for r in csv.reader(raw.decode().splitlines())
        if r and r[0] != "Validator"
    }
    loaded = {tuple(r[:3]) for r in tunable.get_results()}
    if not expected.issubset(loaded):
        raise RuntimeError("Selected head algorithms missing from runtime database")
    report = dict(
        time=time.time(),
        verified=True,
        csv_sha=hashlib.sha256(raw).hexdigest(),
        filename=tunable.get_filename(),
        autotuning=False,
    )
    _table_loaded = True
    logger.info("Gemma4 BW1000 head table loaded: %s", report)
    return report


def install_head_bucket(runner, rows=BUCKET_ROWS) -> None:
    """Idempotent: a second call in the same process is a no-op."""
    global _bucket_installed
    if _bucket_installed:
        return
    model = runner.model
    while not hasattr(model, "lm_head"):
        model = model.language_model
    proc = model.logits_processor
    w = model.lm_head.weight
    assert w.shape == (VOCAB, HIDDEN) and w.dtype == torch.bfloat16
    assert proc.soft_cap == SOFTCAP and proc.scale == 1
    assert proc.head_dtype in (None, torch.bfloat16)
    assert not proc.logits_as_input
    assert runner.vllm_config.parallel_config.tensor_parallel_size == 1
    rows = frozenset(rows)
    assert rows and min(rows) >= 128 and max(rows) < BUCKET
    pads = {
        torch.cuda.current_stream().cuda_stream: torch.zeros(
            (BUCKET, HIDDEN), dtype=torch.bfloat16, device=w.device
        )
    }
    original = proc._get_logits

    def get_logits(hidden, lm_head, bias, skip_gather):
        n = hidden.shape[0]
        if (
            hidden.ndim != 2
            or hidden.dtype != torch.bfloat16
            or hidden.shape[1] != HIDDEN
            or n not in rows
        ):
            return original(hidden, lm_head, bias, skip_gather)
        stream = torch.cuda.current_stream().cuda_stream
        if stream not in pads:
            pads[stream] = torch.zeros(
                (BUCKET, HIDDEN), dtype=torch.bfloat16, device=hidden.device
            )
        pad = pads[stream]
        pad[:n].copy_(hidden)
        out = original(pad, lm_head, bias, skip_gather)
        return None if out is None else out[:n]

    proc._get_logits = get_logits
    proc._gemma4_head_pads = pads
    _bucket_installed = True
    logger.info(
        "Gemma4 BW1000 head bucket active: rows=%d..%d bucket=%d",
        min(rows), max(rows), BUCKET,
    )
