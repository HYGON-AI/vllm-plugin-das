# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Gemma4 BW1000 LM-head GEMM tuning (TP1, BF16, full 262144 vocabulary).

Ported from the validated stage1 bundle:
  * load the fixed rocBLAS algorithm table (TunableOp, tuning disabled);
  * pad 128..319 hidden rows to the tuned 320-row GEMM;
  * softcap the logits by table lookup instead of recomputing tanh;
  * fuse the 33..64-row greedy draft projection with its argmax.
"""

from __future__ import annotations

import csv
import hashlib
import time
from pathlib import Path

import torch

from vllm.logger import init_logger

from .gemma4_dense_head import dense_argmax
from .gemma4_softcap_lut import apply_lut, build_lut

logger = init_logger(__name__)

TABLE_PATH = Path(__file__).resolve().parent / "gemma4_head_results0.csv"
HIDDEN = 2816
VOCAB = 262144
SOFTCAP = 30.0
# Row counts are padded up to the nearest bucket so the head GEMM lands on a
# shape the rocBLAS table has an entry for. The two halves are one mechanism:
# a bucket without a table entry pads for nothing, and a table entry no row
# count reaches is dead weight.
#
# 320 alone left production uncovered. The 11-12 probe measured the live row
# count at 348..400 (num_seqs x MTP positions, 80 x 5 = 400 at the peak), i.e.
# above the only bucket, so those GEMMs ran on the default algorithm -- which
# measures 14.997 ms at 348 rows and 17.657 ms at 400 against 3.136 ms for the
# tuned 400 shape, about 6x off. Adding 400 covers them.
#
# Above 400 is still uncovered: at C80 the row count cannot exceed 80 x 5, but
# a higher concurrency would need its own bucket and its own tuned entry.
HEAD_BUCKETS = (320, 400)
BUCKET = HEAD_BUCKETS[-1]
BUCKET_ROWS = tuple(range(128, BUCKET + 1))
DRAFT_HIDDEN = 1024
# Widened from 33..64 after the 11-12 gate probe measured the live draft row
# count at 76..80 (C80, MTP4) -- the old ceiling sat just below production and
# gated the fusion out entirely. 256 is where the sweep stops, not where the
# gain does.
DENSE_ROWS = tuple(range(33, 257))
DENSE_CONFIG = {"BM": 32, "BN": 256, "BK": 64, "num_warps": 8, "num_stages": 1}
# BM is the only field that wants to track the row count. Measured against the
# stock matmul+argmax at 262144 vocab / 1024 hidden: BM=32 is best to 64 rows
# (1.52x at 33, 1.67x at 64), BM=64 from 65 to 96 (2.20x at 76, 2.27x at 96)
# and BM=128 above that (3.08x at 128, 3.19x at 160, 4.93x at 256). Keeping
# BM=32 everywhere would leave most of that on the table at the row counts
# production actually runs.
DENSE_BM_STEPS = ((64, 32), (96, 64))
DENSE_BM_ABOVE = 128
_BLOCK_ALIGN = 2048  # one lookup block, enough to trigger the compile

_table_loaded = False
_bucket_installed = False
_softcap_installed = False
_dense_draft_installed = False



def _head_bucket_for(rows: int) -> int:
    """Smallest bucket that fits this row count."""

    for bucket in HEAD_BUCKETS:
        if rows <= bucket:
            return bucket
    raise ValueError("no head bucket covers %d rows" % rows)


def _dense_config(rows: int, base: dict) -> dict:
    """Launch config for a given draft row count (see DENSE_BM_STEPS)."""

    for upper, bm in DENSE_BM_STEPS:
        if rows <= upper:
            return dict(base, BM=bm)
    return dict(base, BM=DENSE_BM_ABOVE)


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


def install_softcap_lut(runner) -> None:
    """Replace the logits softcap with a lookup over all bfloat16 patterns.

    Idempotent: a second call in the same process is a no-op. The table is
    built from the stock expression, so this is a bit-identical substitution
    and not an approximation -- but only for the soft_cap it was built with,
    and only when nothing scales the logits afterwards, both asserted here.
    """
    global _softcap_installed
    if _softcap_installed:
        return
    model = runner.model
    while not hasattr(model, "lm_head"):
        model = model.language_model
    proc = model.logits_processor
    assert proc.soft_cap == SOFTCAP and proc.scale == 1
    assert proc.head_dtype in (None, torch.bfloat16)
    assert not proc.logits_as_input
    assert runner.vllm_config.parallel_config.tensor_parallel_size == 1

    lut = build_lut(model.lm_head.weight.device, SOFTCAP)
    original = proc.forward

    def forward(lm_head, hidden_states, embedding_bias=None, skip_gather=False):
        if hidden_states.dtype != torch.bfloat16 or proc.logits_as_input:
            return original(lm_head, hidden_states, embedding_bias, skip_gather)
        logits = proc._get_logits(hidden_states, lm_head, embedding_bias, skip_gather)
        if logits is None:
            return logits
        if logits.dtype != torch.bfloat16 or not logits.is_contiguous():
            return original(lm_head, hidden_states, embedding_bias, skip_gather)
        return apply_lut(logits, lut)

    # Compile the lookup kernel now. A first compile inside CUDA graph capture
    # would be fatal, and whether logits processing falls inside the captured
    # region depends on the graph mode, so do not rely on it being outside.
    apply_lut(torch.zeros(_BLOCK_ALIGN, dtype=torch.bfloat16, device=lut.device), lut)
    torch.cuda.synchronize()

    proc.forward = forward
    proc._gemma4_softcap_lut = lut
    _softcap_installed = True
    logger.info(
        "Gemma4 BW1000 softcap lookup active: soft_cap=%s, 65536-entry BF16 table",
        SOFTCAP,
    )


def install_head_bucket(runner, rows=BUCKET_ROWS, buckets=HEAD_BUCKETS) -> None:
    """Pad the head GEMM's row count up to the nearest tuned shape.

    Idempotent: a second call in the same process is a no-op.

    Padding costs real rows of compute -- 348 rows padded to 400 is 15% more
    work -- so it only pays because the padded shape has a tuned rocBLAS entry
    and the true shape does not. That makes the table a hard prerequisite, not
    a nicety, so every bucket is checked against the loaded table here: a
    bucket whose shape has no entry would pad for nothing and silently cost
    throughput, which is exactly how the 320-only version came to look
    installed while doing nothing for production.
    """
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
    buckets = tuple(sorted(buckets))
    rows = frozenset(rows)
    assert rows and min(rows) >= 128 and max(rows) <= buckets[-1]

    # Every bucket must be a shape the loaded table actually covers, or the
    # padding is pure loss. load_head_table() runs before this, so the runtime
    # database is already populated.
    tuned = {r[1] for r in torch.cuda.tunable.get_results()}
    missing = [
        b for b in buckets
        if "tn_%d_%d_%d_ld_%d_%d_%d" % (VOCAB, b, HIDDEN, HIDDEN, HIDDEN, VOCAB)
        not in tuned
    ]
    assert not missing, (
        "head buckets %s have no tuned entry in the rocBLAS table; padding to "
        "them would add work for nothing" % missing
    )

    # One pad buffer per (stream, bucket). Keyed on the stream because CUDA
    # graph capture runs on its own.
    pads: dict = {}

    def pad_for(stream, bucket, device):
        buf = pads.get((stream, bucket))
        if buf is None:
            buf = torch.zeros((bucket, HIDDEN), dtype=torch.bfloat16, device=device)
            pads[(stream, bucket)] = buf
        return buf

    for b in buckets:
        pad_for(torch.cuda.current_stream().cuda_stream, b, w.device)

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
        bucket = _head_bucket_for(n)
        if n == bucket:
            # already a tuned shape; padding would copy for nothing
            return original(hidden, lm_head, bias, skip_gather)
        pad = pad_for(torch.cuda.current_stream().cuda_stream, bucket, hidden.device)
        pad[:n].copy_(hidden)
        out = original(pad, lm_head, bias, skip_gather)
        return None if out is None else out[:n]

    proc._get_logits = get_logits
    proc._gemma4_head_pads = pads
    _bucket_installed = True
    logger.info(
        "Gemma4 BW1000 head bucket active: rows=%d..%d buckets=%s "
        "(all present in the rocBLAS table)",
        min(rows), max(rows), list(buckets),
    )


def install_dense_draft(runner, config=None, rows=DENSE_ROWS) -> None:
    """Fuse the greedy draft projection with its argmax for 33..256 rows.

    Idempotent: a second call in the same process is a no-op. The stock step
    materialises the whole [rows, vocab] logits tensor only to reduce it, which
    at 80 rows over this vocabulary writes and reads back 40 MiB to produce 80
    indices.

    The range was 33..64 until the 11-12 gate probe measured the live draft row
    count at 76..80 under C80 + MTP4, i.e. just above the ceiling, so the
    fusion never ran in production. Re-swept to 256 rows it is worth 2.20x at
    76, 3.08x at 128 and 4.93x at 256 against the stock path, with the
    tie-aware check clean at every point.

    Unlike the softcap table this is not a bit-identical substitution: the
    GEMM's accumulation order differs from the library's, so a logit on a
    rounding boundary can fall either way and change which of several tied
    columns wins. The tie rule itself matches torch.argmax -- lowest index,
    NaN outranking everything -- and over 209 rows of random and
    small-magnitude input the indices agreed with the stock path exactly, with
    explicit ties, a suppressed tied winner and NaN rows all resolving the same
    way. A residual tie difference stays a legitimate greedy pick.
    """
    global _dense_draft_installed
    if _dense_draft_installed:
        return
    config = config or DENSE_CONFIG
    spec = getattr(runner, "speculator", None)
    model = runner.get_draft_model()
    if spec is None or model is None:
        return
    assert spec.draft_logits is None, "fused argmax returns indices, not logits"
    assert runner.vllm_config.parallel_config.tensor_parallel_size == 1
    w = model.lm_head.weight
    assert w.shape == (VOCAB, DRAFT_HIDDEN) and w.dtype == torch.bfloat16
    assert model.logits_processor.soft_cap is None
    assert model.logits_processor.scale == 1
    assert getattr(model, "masked_embedding", None) is None
    rows = frozenset(rows)
    assert rows and all(33 <= n <= 256 for n in rows)
    configs = {n: _dense_config(n, config) for n in rows}

    original = spec._greedy_sample_draft
    # Captured after load_weights, which is where gemma4_mtp fills it in; the
    # stock path suppresses the same ids inside compute_logits.
    suppress = getattr(model, "_suppress_idx", None)

    def greedy(hidden):
        if (
            hidden.dtype != torch.bfloat16
            or hidden.ndim != 2
            or hidden.shape[1] != DRAFT_HIDDEN
            or hidden.shape[0] not in rows
        ):
            return original(hidden)
        return dense_argmax(hidden, w, suppress, configs[hidden.shape[0]])

    # Compile before any CUDA graph capture: a first compile inside capture
    # would be fatal. M is a runtime argument, so one compile per launch config
    # serves every row count -- but Triton also specializes integer arguments
    # on divisibility by 16, so each config is compiled at both a divisible and
    # a non-divisible row count. That is a handful of compiles instead of one
    # per gated row: pre-compiling all 224 individually measured 211 s, which
    # lands inside load_model and would inflate the startup memory peak that
    # sets the KV budget.
    dummy = torch.zeros((max(rows), DRAFT_HIDDEN), dtype=torch.bfloat16, device=w.device)
    warmed = set()
    for n in sorted(rows):
        cfg = configs[n]
        key = (cfg["BM"], n % 16 == 0)
        if key in warmed:
            continue
        warmed.add(key)
        dense_argmax(dummy[:n], w, suppress, cfg)
    torch.cuda.synchronize()

    spec._greedy_sample_draft = greedy
    _dense_draft_installed = True
    logger.info(
        "Gemma4 BW1000 fused draft argmax active: rows=%d..%d vocab=%d "
        "BM schedule=%s (%d launch configs pre-compiled)",
        min(rows), max(rows), VOCAB,
        sorted({c["BM"] for c in configs.values()}), len(warmed),
    )
