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
BUCKET = 320
BUCKET_ROWS = tuple(range(128, BUCKET))
DRAFT_HIDDEN = 1024
DENSE_ROWS = tuple(range(33, 65))
DENSE_CONFIG = {"BM": 32, "BN": 256, "BK": 64, "num_warps": 8, "num_stages": 1}

_table_loaded = False
_bucket_installed = False
_softcap_installed = False
_dense_draft_installed = False


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

    proc.forward = forward
    proc._gemma4_softcap_lut = lut
    _softcap_installed = True
    logger.info(
        "Gemma4 BW1000 softcap lookup active: soft_cap=%s, 65536-entry BF16 table",
        SOFTCAP,
    )


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


def install_dense_draft(runner, config=None, rows=DENSE_ROWS) -> None:
    """Fuse the greedy draft projection with its argmax for 33..64 rows.

    Idempotent: a second call in the same process is a no-op. The stock step
    materialises the whole [rows, vocab] logits tensor only to reduce it, which
    at 64 rows over this vocabulary writes and reads back 32 MiB to produce 64
    indices.

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
    assert rows and all(32 <= n <= 64 for n in rows)

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
        return dense_argmax(hidden, w, suppress, config)

    # Compile every gated row count now: a first compile inside CUDA graph
    # capture would be fatal.
    dummy = torch.zeros((max(rows), DRAFT_HIDDEN), dtype=torch.bfloat16, device=w.device)
    for n in sorted(rows):
        dense_argmax(dummy[:n], w, suppress, config)
    torch.cuda.synchronize()

    spec._greedy_sample_draft = greedy
    _dense_draft_installed = True
    logger.info(
        "Gemma4 BW1000 fused draft argmax active: rows=%d..%d vocab=%d",
        min(rows), max(rows), VOCAB,
    )
