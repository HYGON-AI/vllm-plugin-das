# SPDX-License-Identifier: Apache-2.0
"""Measured gfx938 Step5 Dense choices through the normal hcu_ops extension.

No standalone library or runtime code injection is used. Algorithm choices
come from explicitly supplied external tuning data.
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


@lru_cache(maxsize=1)
def _native_gemm():
    # Lazy worker-time import. The normal wheel must contain this operator;
    # do not silently bind an older installed extension or compile at runtime.
    import vllm_hcu.hcu_ops  # noqa: F401
    try:
        op = torch.ops.hcu_ops.step5_fp8_dense_gemm
        library = Path(torch.ops.hcu_ops.step5_fp8_dense_backend_library()).resolve()
        expected = _LIBRARY_SHA256
        if hashlib.sha256(library.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Unvalidated hipBLASLt build for Step5 Dense recipe: {library}")
        return op
    except AttributeError as error:
        raise RuntimeError(
            "The loaded hcu_ops extension lacks step5_fp8_dense_gemm; "
            "build/install the matching normal vllm_hcu wheel"
        ) from error


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
    if algorithm is None and m != _UNSPLIT_PROFILE_ROWS:
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
                if m != _UNSPLIT_PROFILE_ROWS or dbo_enabled() or torch.cuda.is_current_stream_capturing():
                    raise RuntimeError(
                        "Step5 tuned Dense plans require the unsplit 8192-row "
                        f"startup profile before DBO/graphs; got M={m},N={n},K={k}"
                    )
                op = _native_gemm()
                for rows, selected in relevant:
                    op(x[:rows], xs[:rows], w, ws, selected, True)
                    logger.info(
                        "[step5_dense_tuned] prepared M=%d N=%d K=%d "
                        "algorithm=%d workspace=0 backend=hcu_ops",
                        rows, n, k, selected,
                    )
                _warmed.add(warm_key)
    if algorithm is None:
        return None
    # The bool is False on every serving/capture call. The native operator
    # additionally rejects creation while a stream is capturing.
    return _native_gemm()(x, xs, w, ws, algorithm, False)
