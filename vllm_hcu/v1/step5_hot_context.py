# SPDX-License-Identifier: Apache-2.0
"""Conservative CPU guard for a context-bounded Step5 decode graph.

Only CUDA graph replay is bounded. The model and KV-cache context limit stay
unchanged. Long batches vote for NONE through vLLM's existing DP mode sync.
"""

from __future__ import annotations

import os

import numpy as np

HOT_REGIONS = int(os.environ.get("HCU_STEP5_CONTEXT_GRAPH_REGIONS", "0"))
if HOT_REGIONS < 0:
    raise ValueError("HCU_STEP5_CONTEXT_GRAPH_REGIONS must be nonnegative")


def conservative_seq_upper_bound(runner, batch_state):
    """Bound the next query without waiting for the pending GPU sampler.

    The exact host mirror may lag one sampled batch. Every previous scheduled
    query token is an upper bound on the number it accepted. Taking the largest
    pending query is conservative even when request slots were reused.
    """
    if not getattr(runner, "_hsl_enabled", False):
        return None
    if batch_state is None or len(batch_state.idx_mapping_np) == 0:
        return 0
    idx = np.asarray(batch_state.idx_mapping_np, dtype=np.intp)
    computed = np.asarray(runner._hsl_exact[idx], dtype=np.int64)
    query = np.asarray(batch_state.num_scheduled_tokens, dtype=np.int64)
    if computed.shape != query.shape:
        raise ValueError("Step5 host mirror and scheduled queries disagree")
    pending = getattr(runner, "_hsl_pending", None)
    lag = int(np.max(pending[1], initial=0)) if pending is not None else 0
    return int(np.max(computed + query, initial=0)) + lag


def needs_context_eager(runner, batch_state, dummy_run, limit_regions=None):
    regions = HOT_REGIONS if limit_regions is None else limit_regions
    if not regions:
        return False
    runner._step5_hot_context_real_batch = not dummy_run
    if dummy_run:
        return False
    upper = conservative_seq_upper_bound(runner, batch_state)
    runner._step5_hot_context_upper = upper
    return upper is None or upper > regions * 8


def check_hot_graph_context(input_batch, cg_mode, real_batch, limit_regions=None):
    """Fail before replay if the exact, already-resolved host mirror disagrees."""
    regions = HOT_REGIONS if limit_regions is None else limit_regions
    if not regions or not real_batch or getattr(cg_mode, "name", None) != "FULL":
        return
    seq = getattr(input_batch, "seq_lens_cpu", None)
    if seq is None:
        raise RuntimeError("Cannot verify the Step5 hot-context FULL graph bound")
    values = seq.numpy() if hasattr(seq, "numpy") else np.asarray(seq)
    maximum = int(np.max(values, initial=0))
    if maximum > regions * 8:
        raise RuntimeError(f"Unsafe Step5 FULL graph context {maximum}>{regions * 8}")


def record_context_graph_mode(runner, batch_desc):
    if not HOT_REGIONS or not getattr(runner, "_step5_hot_context_real_batch", False):
        return
    mode = batch_desc.cg_mode.name
    if mode != getattr(runner, "_step5_hot_context_last_mode", None):
        runner._step5_hot_context_last_mode = mode
        print(
            f"[step5_context_graph] dp={runner.dp_rank} mode={mode} upper={getattr(runner, '_step5_hot_context_upper', None)} tokens={batch_desc.num_tokens}",
            flush=True,
        )
