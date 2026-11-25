"""Qualify the hoisted FP8 paged Attention and configure the fixed C4 bands."""
from collections import Counter
import hashlib
import json
from pathlib import Path

import torch
from hcu_vllm_das.parallel import minimax31_msa as msa
from .prefill_runtime_candidate import quant_query
from .prefill_attention_helpers import scoped_knobs, load, verify_outputs


PLANS = {2048: (4, 2, 16), 6144: (4, 4, 32)}
_active = None
_original = None
_candidate = None
_verify = False
_hits = Counter()
_checks = Counter()


def _eligible(q, selected, cache, meta):
    rows = q.shape[0]
    return (rows in PLANS and tuple(q.shape) == (rows, 64, 128) and q.dtype == torch.bfloat16
            and cache.layout == msa.LAYOUT and tuple(selected.shape) == (rows, 4, 16)
            and selected.dtype == torch.int32 and meta.k_end.numel() == rows
            and 131072 < meta.max_kv_len <= 147456 and len(meta.segments) == rows // 1024
            and all(length == 1024 and start >= 131072 for _, _, start, length in meta.segments))


def _attend(q, cache, selected, meta, *, q_scale, k_scale, v_scale, p_scale):
    if not _eligible(q, selected, cache, meta):
        return _original(q, cache, selected, meta, q_scale=q_scale,
                         k_scale=k_scale, v_scale=v_scale, p_scale=p_scale)
    rows = q.shape[0]
    for scale, label in zip((q_scale, k_scale, v_scale, p_scale), "QKVP"):
        msa._scalar(scale, q.device, label)
    native = _active.reference.native
    q8 = quant_query(q, q_scale)
    pages = selected.contiguous()

    def run(module, plan):
        with scoped_knobs(plan):
            return module.sparse_prefill_fwd(
                q8, cache.key, pages, 128 ** -.5, 128, None, None, cache.value,
                q_scale, k_scale, v_scale, p_scale, meta.k_end, meta.cu_q, meta.page_table,
            )

    output = run(_candidate, PLANS[rows])
    if _verify and _checks[rows] < 3:
        reference = run(_active.candidate, native.plans[rows])
        verify_outputs(reference, output, f"C4 runtime M={rows}")
        _checks[rows] += 1
    _hits[rows] += 1
    return output[0]


def configure(value, receipt_path):
    global _active, _original, _candidate, _verify
    if value == "status":
        return {"enabled": _active is not None, "verify": _verify, "plans": PLANS,
                "hits": dict(_hits), "verified": dict(_checks)}
    if value is not None and type(value) is not bool:
        raise ValueError("Use null for stock, true for verification, or false for timing")
    if value is None:
        if _active is not None:
            if msa._ADAPTER is not _active or msa._ADAPTER.attend is not _attend:
                raise ValueError("C4 Attention dispatch changed during the experiment")
            _active.attend = _original
            _original = None
            _active = None
    else:
        active = msa._ADAPTER
        if active is None or not hasattr(active, "candidate") or not hasattr(active, "reference"):
            raise RuntimeError("Enable the verified paged MSA path first")
        if _candidate is None:
            receipt = json.loads(Path(receipt_path).read_text())
            artifact = Path(receipt["extension"])
            if hashlib.sha256(artifact.read_bytes()).hexdigest() != receipt["sha256"]:
                raise ValueError("Hoisted Attention library differs from the validated build")
            _candidate = load(artifact, "c4_hoisted_attention")
        if _active is None:
            _active = active
            _original = active.attend
            active.attend = _attend
        elif _active is not active:
            raise RuntimeError("C4 Attention dispatch changed")
    _verify = value is True
    _hits.clear()
    _checks.clear()
    return configure("status", receipt_path)
