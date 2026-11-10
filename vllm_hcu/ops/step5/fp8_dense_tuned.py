# SPDX-License-Identifier: Apache-2.0
"""Measured gfx938 Step5 Dense choices through AITER.

Call from the existing opaque vllm::step5_fp8_gemm custom op. Other shapes
return None, preserving the original LightOp path and its compilation fake.
"""
import hashlib
from pathlib import Path
from functools import lru_cache
import threading

import torch
from vllm.logger import init_logger
from vllm_hcu.platforms import envs as henvs

logger = init_logger("vllm.hcu.step5_dense")

from .tuning import dense_recipe

_recipe = dense_recipe()
_CHOICES, _LIBRARY_SHA256 = _recipe if _recipe is not None else ({}, None)
_BY_NK = {}
for _shape, _algorithm in _CHOICES.items():
    _BY_NK.setdefault(_shape[1:], []).append((_shape[0], _algorithm))
_UNSPLIT_PROFILE_ROWS = 8192
_lock = threading.RLock()
_warmed = set()
_unsupported_devices = set()


def _check_library(library):
    if hashlib.sha256(library.read_bytes()).hexdigest() != _LIBRARY_SHA256:
        raise RuntimeError(f"Unvalidated hipBLASLt build for Step5 Dense recipe: {library}")


@lru_cache(maxsize=None)
def _aiter_gemm(device):
    from aiter import hipb_mm
    from vllm_hcu.model_executor.layers.fused_moe.aiter_ops import (
        _ensure_hipb_mm_extension_initialized,
    )

    with torch.cuda.device(device):
        _ensure_hipb_mm_extension_initialized()
    # Check the loaded binary without depending on the Step5 hcu_ops operator.
    libraries = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6:
            path = Path(fields[5])
            if path.is_absolute() and path.name.startswith("libhipblaslt.so"):
                libraries.add(path.resolve())
    if len(libraries) != 1:
        raise RuntimeError(f"Expected one loaded hipBLASLt library, found {libraries}")
    _check_library(libraries.pop())

    def run(x, xs, w, ws, algorithm):
        return hipb_mm(
            x, w.t(), solution_index=algorithm, out_dtype=torch.bfloat16,
            scaleA=xs, scaleB=ws.t(), scaleType=1,
        )

    return run


def tuned_fp8_gemm(x, xs, w, ws):
    """Return tuned BF16 output or None for the unchanged LightOp fallback."""
    if not henvs.custom_ops_enabled():
        return None
    m, k = x.shape
    n = w.shape[0]
    relevant = _BY_NK.get((n, k))
    if relevant is None:
        return None
    algorithm = _CHOICES.get((m, n, k))
    # In particular M16/M64 can occur before the profile. They must retain
    # their ordinary fallback without attempting to initialize these plans.
    if algorithm is None and m < _UNSPLIT_PROFILE_ROWS:
        return None
    device = x.device.index
    warm_key = (device, n, k)
    if device in _unsupported_devices:
        return None
    if warm_key not in _warmed:
        with _lock:
            if warm_key not in _warmed:
                architecture = torch.cuda.get_device_properties(x.device).gcnArchName.split(":")[0]
                if architecture != "gfx938":
                    _unsupported_devices.add(device)
                    return None
                # This is a runtime context predicate, not the configuration
                # enable_dbo flag (the profile deliberately runs unsplit).
                from vllm.v1.worker.ubatching import dbo_enabled
                profile_rows = max(
                    _UNSPLIT_PROFILE_ROWS, max(rows for rows, _ in relevant)
                )
                if m < profile_rows or dbo_enabled() or torch.cuda.is_current_stream_capturing():
                    raise RuntimeError(
                        "Step5 tuned Dense plans require an unsplit startup "
                        f"profile with at least {profile_rows} rows "
                        f"before DBO/graphs; got M={m},N={n},K={k}"
                    )
                op = _aiter_gemm(device)
                for rows, selected in relevant:
                    op(x[:rows], xs[:rows], w, ws, selected)
                    logger.info(
                        "[step5_dense_tuned] prepared M=%d N=%d K=%d "
                        "algorithm=%d backend=aiter",
                        rows, n, k, selected,
                    )
                _warmed.add(warm_key)
    if algorithm is None:
        return None
    return _aiter_gemm(device)(x, xs, w, ws, algorithm)
