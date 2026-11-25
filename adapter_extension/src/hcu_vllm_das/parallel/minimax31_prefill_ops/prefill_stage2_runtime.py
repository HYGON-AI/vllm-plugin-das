"""Opt in to fused Stage2 for the long, prefix-cached PCP Prefill rows."""
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sysconfig

import torch


PREFIX_TOKENS = 131072
MAX_KV_TOKENS = 147456
PREFILL_ROWS = (2048, 4096, 6144, 8192)
_selection = None
_deferred_stage1 = None
_previous_tail_mode = None
_loaded = {}
_hits = Counter()
_verified = Counter()
_deferred = Counter()
_deferred_verified = Counter()
_deferred_bounds = Counter()
_poisoned_entries = {}


def _reference(native, score, meta, prefix, local, valid_pages, mark_local_pages):
    if local:
        mark_local_pages(score, meta.k_end, local)
    return native.msa128_stage2_topk_select(score, 16, valid_pages, prefix, 0, None, None)


def _supports(score, meta, valid_pages):
    # The planner's segment starts are CPU values, so the gate adds no device transfer.
    return (
        score.ndim == 3 and score.size(0) == 4 and score.size(2) in PREFILL_ROWS
        and 16 < valid_pages <= score.size(1) <= MAX_KV_TOKENS // 128
        and score.dtype == torch.float32 and score.is_cuda and score.is_contiguous()
        and meta.k_end.dtype == torch.int32 and meta.k_end.ndim == 1
        and meta.k_end.numel() == score.size(2) and meta.k_end.device == score.device
        and meta.k_end.is_contiguous()
        and PREFIX_TOKENS < meta.max_kv_len <= MAX_KV_TOKENS
        and meta.segments and all(start >= PREFIX_TOKENS for _, _, start, _ in meta.segments)
    )


def select_scored_pages(native, score, meta, prefix, local, mark_local_pages, *, selection=None):
    valid_pages = (meta.max_kv_len + 127) // 128
    if selection is None:
        selection = _selection
    if selection is None or not _supports(score, meta, valid_pages):
        return _reference(native, score, meta, prefix, local, valid_pages, mark_local_pages)

    module, tile_queries, check, _ = selection
    query_rows = score.size(2)
    if tile_queries < 0 and query_rows not in (2048, 6144):
        return _reference(native, score, meta, prefix, local, valid_pages, mark_local_pages)
    if tile_queries == -2:
        tile_queries = -1
    output = module.select_prefill_pages(score, meta.k_end, valid_pages, prefix, local, tile_queries)
    if check:
        # Compare actual runtime scores. The reference may mark the score in place.
        # It runs second on the same stream, after the fused kernel has read the scores.
        expected = _reference(native, score, meta, prefix, local, valid_pages, mark_local_pages)
        if not torch.equal(output, expected):
            raise ValueError(f"Fused MSA changed selected pages for {query_rows} local queries")
        _verified[query_rows] += 1
    _hits[query_rows] += 1
    return output


def _paired_geometry(index_q, meta):
    rows = index_q.size(0)
    segments = meta.segments
    return (
        rows in (2048, 6144) and len(segments) == rows // 1024
        and index_q.dtype == torch.float8_e4m3fn
        and PREFIX_TOKENS < meta.max_kv_len <= MAX_KV_TOKENS
        and all(length == 1024 and start >= PREFIX_TOKENS
                for _, _, start, length in segments)
        and max(start + length for _, _, start, length in segments) == meta.max_kv_len
        and os.environ.get("MSA_STAGE1_FP8_KERNEL") == "v9"
    )


def score_and_select_pages(native, index_q, index_cache, meta, prefix, local,
                           q_scale, k_scale, mark_local_pages):
    args = (index_q, index_cache.unsqueeze(1), meta.k_end, True, -1,
            meta.q_lens_cpu, meta.kv_lens_cpu, None, meta.packed_kv_pages,
            meta.max_kv_len, q_scale, k_scale, True, False, False)
    paired = _deferred_stage1
    if paired is None or not _paired_geometry(index_q, meta):
        score = native.msa128_stage1_page_max(*args)
        return select_scored_pages(native, score, meta, prefix, local, mark_local_pages)

    module, _, verify, _ = _selection
    score = paired.msa128_stage1_page_max(*args)
    valid_pages = (meta.max_kv_len + 127) // 128
    if not _supports(score, meta, valid_pages):
        raise RuntimeError("Deferred Stage1 produced a score outside the masked-reader contract")
    selected = module.select_prefill_pages(score, meta.k_end, valid_pages, prefix, local, -2)
    rows = index_q.size(0)
    if verify:
        if rows not in _poisoned_entries:
            pages = torch.arange(score.size(1), device=score.device, dtype=torch.int32)
            hidden = pages[:, None] * 128 >= meta.k_end[None, :]
            poisoned = int(torch.count_nonzero(torch.isnan(score) & hidden.unsqueeze(0)))
            if poisoned == 0:
                raise RuntimeError("Paired Stage1 did not write the expected diagnostic tail sentinel")
            _poisoned_entries[rows] = poisoned
        reference_score = native.msa128_stage1_page_max(*args)
        expected = _reference(native, reference_score, meta, prefix, local, valid_pages, mark_local_pages)
        if not torch.equal(selected, expected):
            raise RuntimeError(f"Paired Stage1/Stage2 changed the selected pages for {rows} rows")
        _deferred_verified[rows] += 1
    _deferred[rows] += 1
    _deferred_bounds[meta.max_kv_len] += 1
    return selected


def _load_stage1():
    receipt = json.loads(Path(os.environ["MINIMAX_PREFILL_STAGE1_TAIL_RECEIPT"]).read_text())
    binary = Path(receipt["extension"])
    digest = hashlib.sha256(binary.read_bytes()).hexdigest()
    if digest != receipt["sha256"]:
        raise ValueError("Deferred Stage1 module digest differs from the build receipt")
    if digest not in _loaded:
        spec = importlib.util.spec_from_file_location("prefill_deferred_stage1._native", binary)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _loaded[digest] = module
    return _loaded[digest]


def _load_extension(receipt_path):
    receipt = json.loads(Path(receipt_path).read_text())
    runtime = (str(torch.__version__), torch.version.hip, sysconfig.get_config_var("SOABI"))
    if tuple(receipt.get(k) for k in ("torch", "hip", "python_abi")) != runtime:
        raise ValueError("Stage2 build receipt is for a different Torch, HIP, or Python runtime")
    binary = Path(receipt["module"])
    digest = hashlib.sha256(binary.read_bytes()).hexdigest()
    if digest != receipt["module_sha256"]:
        raise ValueError("Stage2 module digest differs from the build receipt")
    if digest not in _loaded:
        spec = importlib.util.spec_from_file_location("prefill_stage2_fused", binary)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _loaded[digest] = module
    module = _loaded[digest]
    if not callable(getattr(module, "select_prefill_pages", None)):
        raise ValueError("Stage2 extension is missing select_prefill_pages")
    return module, digest


def load_prefill_plan(receipt_path, *, verify=False):
    """Bind the conservative sorted-page variant for all qualified row bands."""
    module, digest = _load_extension(receipt_path)
    return module, 0, verify, digest


def configure(value):
    global _selection, _deferred_stage1, _previous_tail_mode
    if value == "status":
        result = {"enabled": _selection is not None, "hits": dict(_hits), "verified": dict(_verified),
                  "deferred_hits": dict(_deferred), "deferred_verified": dict(_deferred_verified),
                  "deferred_bounds": dict(_deferred_bounds),
                  "poisoned_entries": dict(_poisoned_entries)}
        if _selection is not None:
            _, tile_queries, check, digest = _selection
            result.update(queries_per_cta=tile_queries, verify=check, module_sha256=digest)
        return result
    selection = None
    paired = None
    if value is not None:
        if not isinstance(value, dict):
            raise ValueError("expected an explicit Stage2 plan")
        tile_queries = value.get("queries_per_cta")
        check = value.get("verify", False)
        if type(tile_queries) is not int or tile_queries not in (-2, -1, 0, 4, 8) or type(check) is not bool:
            raise ValueError("invalid Stage2 tile or verification mode")
        receipt_key = (
            "MINIMAX_PREFILL_STAGE2_PRUNED_RECEIPT"
            if tile_queries == -1 else "MINIMAX_PREFILL_STAGE2_RECEIPT"
        )
        if tile_queries == -2:
            receipt_key = "MINIMAX_PREFILL_STAGE2_TAIL_RECEIPT"
            paired = _load_stage1()
        module, digest = _load_extension(os.environ[receipt_key])
        selection = (module, tile_queries, check, digest)
    if paired is not None:
        if _deferred_stage1 is None:
            _previous_tail_mode = os.environ.get("MSA_C4_SCORE_TAIL")
        os.environ["MSA_C4_SCORE_TAIL"] = "poison" if check else "defer"
    elif _deferred_stage1 is not None:
        if _previous_tail_mode is None:
            os.environ.pop("MSA_C4_SCORE_TAIL", None)
        else:
            os.environ["MSA_C4_SCORE_TAIL"] = _previous_tail_mode
        _previous_tail_mode = None
    _selection = selection
    _deferred_stage1 = paired
    _hits.clear()
    _verified.clear()
    _deferred.clear()
    _deferred_verified.clear()
    _deferred_bounds.clear()
    _poisoned_entries.clear()
    return configure("status")
