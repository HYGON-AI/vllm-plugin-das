# SPDX-License-Identifier: Apache-2.0
"""R94/A' stage-1: route MLA prefill FA varlen calls through the FP8
specialization of the das FlashAttention wheel, mirroring the sglang HCU
form (`flash_fp8_fwd_kernel_gfx938`, Q/K/V e4m3 + descale args).

Why (R92-R94): the gfx938 bf16 b16 varlen specialization
(`compute_attn_mha_1rowblock_gfx938`) carries the causal tail OOB fault
(EXEC-79; diag79 clamps the kernel side).  sglang HCU never enters that
specialization because its prefill feeds FP8 Q/K/V -- the FP8 template
is a different binary whose tail loop is not the faulting one (verified
on the sglang rank0 trace: 20 FlashAttnVarlenFunc calls, all fp8_e4m3,
zero bf16 flash_fwd).

Capture-safety contract (cudagraph-capture-safety checklist):
- FULL graphs never contain this path (uniform-decode batches dispatch to
  the FlashMLA MQA backend, not the MLA prefill backend); PIECEWISE runs
  attention Python in the eager gap between graph pieces, where
  ``is_current_stream_capturing()`` is False at both profile/capture and
  every replay step, so the branch is dtype-driven and stable across
  all three regimes (capture / piecewise replay / eager).  Should a FULL
  capture ever reach here, the guard bakes the STOCK bf16 call into the
  graph (capture-time gate, no replay-time re-evaluation) -- the safe
  direction, never baked-fp8-then-replayed-stock.
- The requires_v_padding flip around the fp8 call is a host-side
  instance-attribute swap on an eager-only path (the rewrite never runs
  inside a capture window) and is restored in a ``finally`` before any
  stock re-entry, so the stock method always observes its own original
  flag.
- No host-device syncs, no python-scalar setitem, no pinned-copy sites
  are introduced; per-call FP8 materialization allocates through the
  regular eager allocator only (this code never executes inside a
  capture window; see guard below).

Rollback: module knob ``_A1_DISABLE`` (code-level, per R94: no new env
var in the deliverable).  This branch carries no VLLM_HCU_R39_VMF_GUARD
implementation -- the guard family lived only on diagnostic branches
and never merged to the line this fork starts from (d9995c9).

Validation: CPU gates 10/10 (route, capture guard, dtype guard, disable
knob, refusal retry, idempotency, finite cast, class install, kwargs
passthrough, trunk target audit); GPU gate on nmz16 proves the dispatch
split -- stock call walks ``flash_fwd_kernel_16x64_prefetch`` (the
faulting family), rewritten call walks ``flash_fp8_fwd_kernel_gfx938
<192,128,128>``, max|diff|=0.125 / rel 3.4% (e4m3 quantization noise),
lse delta 0.148.
ABI fixes after engine profile runs: descale bucket shaped to the wheel
varlen ABI (batch_size, num_heads_k); v padding suppressed for the fp8
call (stock pads v 128->192 when requires_v_padding is set, but the
wheel fp8 path accepts headdim_v == 128 only -- startup fault "fa fp8
only support dim=128").
"""

from __future__ import annotations

import torch

from vllm_hcu.patch.worker.core_fix._common import load_exact_module

PATCH_ID = "worker.core_fix.mla_prefill_fp8_a1"
TARGET_MODULE = "vllm.v1.attention.backends.mla.prefill.flash_attn"
TARGET_CLASS = "FlashAttnPrefillBackend"
TARGET_SYMBOL = f"{TARGET_MODULE}.{TARGET_CLASS}._flash_attn_varlen_diff_headdims"
_TARGET_METHOD = "_flash_attn_varlen_diff_headdims"
_MARK = "_vllm_hcu_a1_fp8_patched"

# Code-level kill (R94: no new env var in the deliverable). Flipped only by
# editing this file; default ON.
_A1_DISABLE = False

# One-way capability latch: set False on the first wheel-side rejection of
# the FP8 ABI so later calls skip the rewrite without a per-call try.
# Settled during eager warmup steps, read-only inside graph-managed steps.
_A1_CAPABLE = True

_FP8 = torch.float8_e4m3fn
_SUPPORTED_QDTYPES = (torch.bfloat16, torch.float16)


def _descale_bucket(q: torch.Tensor, k: torch.Tensor,
                    kwargs: dict) -> torch.Tensor | None:
    """sglang contract: unit (ones) descale in the wheel's varlen ABI shape.

    Wheel pybind requires exactly ``(batch_size, num_heads_k)`` -- the
    "(1,) scalar-broadcast" seen in the standalone GPU gate does not hold
    for real engine profile batches (startup fault, 12:45).  Semantics
    are unchanged: all layer attention scales are 1.0 in the K3
    deployment (scale-free latent store, R94 SS2), so the bucket is ones
    either way; only the shape is load-bearing.

    Host ints are taken from tensor SHAPES only (no .item()/D2H sync).
    Returns None when the caller gave no cu_seqlens_q -- the rewrite then
    stays off for that call rather than guessing an ABI shape.
    """
    cu = kwargs.get("cu_seqlens_q")
    if cu is None:
        return None
    batch_size = int(cu.shape[0]) - 1
    num_heads_k = int(k.shape[1])
    return torch.ones(
        (batch_size, num_heads_k), dtype=torch.float32, device=q.device
    )


def _quant_to_fp8(x: torch.Tensor) -> torch.Tensor:
    """Direct bf16/half -> e4m3 cast with unit descale.

    Matches the K3 deployment facts: all layer attention scales are 1.0,
    so sglang's ``scaled_fp8_quant`` degenerates to the same cast.  The
    cast saturates at the e4m3 range (max 448); prefill hidden states sit
    far below it.  Measured cost on-GPU: rel 3.4% vs bf16 (gaussian
    probe), i.e. e4m3 quantization noise -- the same output-fidelity
    class sglang ships as its HCU prefill default.
    """
    return x.to(_FP8)


def _a1_rewrites_call(backend, q: torch.Tensor, k: torch.Tensor,
                      v: torch.Tensor) -> bool:
    """Decide whether the fp8 rewrite applies to this call.

    Scope gate uses the backend's own audited head dims (trunk
    ``MLAPrefillBackend.__init__`` stores qk_nope/qk_rope/v_head_dim), so
    the rewrite only fires for MLA-prefill shapes -- K3: qk=192, v=128 --
    and never leaks onto other models' backends that happen to share the
    class.
    """
    if _A1_DISABLE or not _A1_CAPABLE:
        return False
    if q.dtype not in _SUPPORTED_QDTYPES or k.dtype not in _SUPPORTED_QDTYPES:
        return False
    qk_dim = getattr(backend, "qk_nope_head_dim", None)
    rope_dim = getattr(backend, "qk_rope_head_dim", None)
    v_dim = getattr(backend, "v_head_dim", None)
    if qk_dim is None or rope_dim is None or v_dim is None:
        return False
    qk_total = int(qk_dim) + int(rope_dim)
    # K3 deployment geometry only (R94 scope): the wheel's FP8 varlen
    # path accepts headdim_v == 128 exclusively ("fa fp8 only support
    # dim=128"), and the causal-tail fix targets qk=192/v=128.  Any other
    # geometry stays stock rather than risking an unrecognized ABI fault
    # (review R94-C S2).
    if qk_total != 192 or int(v_dim) != 128:
        return False
    if k.shape[-1] != qk_total or q.shape[-1] != qk_total:
        return False
    if v.shape[-1] not in (int(v_dim), qk_total):
        return False
    if torch.cuda.is_current_stream_capturing():
        # Capture windows bake the stock bf16 call (see module docstring).
        return False
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        return False
    return True


def _wheel_dispatch_refused(exc: Exception) -> bool:
    """True only for recognizable wheel-side FP8 dispatch refusals.

    Everything else -- OOM, kernel faults, metadata errors -- must
    propagate: a silent bf16 retry would double the peak memory and mask
    the real fault (review R94-B3).
    """
    if isinstance(exc, TypeError):
        return True
    if isinstance(exc, RuntimeError):
        text = str(exc).lower()
        return (
            "not supported" in text
            or "unsupported" in text
            or "no kernel" in text
            or "invalid dtype" in text
        )
    return False


def apply_to_module(module=None) -> bool:
    """Class-level install; marker-backed idempotent (registry contract:
    a validated reapply must return False, never reinstall).

    Returns False (and logs once) on target drift; the caller treats
    that as "stay stock".
    """
    try:
        if module is None:
            module = load_exact_module(TARGET_MODULE, None)
        cls = getattr(module, TARGET_CLASS, None)
        if cls is None:
            _log_once(f"{PATCH_ID}: {TARGET_CLASS} missing, staying stock")
            return False
        if getattr(cls, _MARK, False):
            # Already installed and validated in this process.
            return False
        fn = cls.__dict__.get(_TARGET_METHOD)
        if fn is None:
            _log_once(f"{PATCH_ID}: target method drifted, staying stock")
            return False

        import functools

        @functools.wraps(fn)
        def _a1(self, q, k, v, *args, **kwargs):  # noqa: ANN001, ANN202
            if not args and _a1_rewrites_call(self, q, k, v):
                ds = _descale_bucket(q, k, kwargs)
                if ds is not None:
                    q8 = _quant_to_fp8(q)
                    k8 = _quant_to_fp8(k)
                    v8 = _quant_to_fp8(v)
                    # The stock method pads v (128) up to the qk width
                    # (192) when ``requires_v_padding`` is set; the wheel's
                    # FP8 varlen path only accepts headdim_v == 128 (fault
                    # "fa fp8 only support dim=128", 13:36).  Flip the
                    # padding flag off for the fp8 call only -- the two
                    # wheel paths accept unpadded diff headdims natively
                    # (mla_traits<192,128> / fp8 traits).  Host flag,
                    # eager-only code path (the rewrite never runs inside
                    # a capture window), restored before any stock use.
                    # Symmetric attribute swap (R94-C NIT): remember
                    # existence so a flag-less instance is restored to
                    # flag-less, not left with a False we authored.
                    _had_pad = hasattr(self, "requires_v_padding")
                    pad_flag = self.requires_v_padding if _had_pad else None
                    try:
                        fkwargs = dict(kwargs)
                        fkwargs["q_descale"] = ds
                        fkwargs["k_descale"] = ds
                        fkwargs["v_descale"] = ds
                        self.requires_v_padding = False
                        return fn(self, q8, k8, v8, **fkwargs)
                    except Exception as exc:
                        if not _wheel_dispatch_refused(exc):
                            raise
                        # Wheel never accepts the FP8 ABI here: latch off
                        # so later calls skip the rewrite without a try,
                        # and finish THIS call on the stock bf16 path.  The
                        # fp8 temporaries are released below BEFORE the
                        # stock retry runs (R94-C S1: they must not stay
                        # referenced through the retry).
                        global _A1_CAPABLE  # noqa: PLW0603
                        _A1_CAPABLE = False
                        _log_once(
                            f"{PATCH_ID}: wheel refused fp8 dispatch "
                            f"({type(exc).__name__}), latching stock path"
                        )
                    finally:
                        if _had_pad:
                            self.requires_v_padding = pad_flag
                        else:
                            del self.requires_v_padding
                    del q8, k8, v8, ds, fkwargs
            return fn(self, q, k, v, *args, **kwargs)

        _a1._a1_wrapped = True  # type: ignore[attr-defined]
        setattr(cls, _MARK, True)
        setattr(cls, _TARGET_METHOD, _a1)
        return True
    except Exception as exc:  # pragma: no cover - patch must never raise
        _log_once(f"{PATCH_ID}: install failed ({exc}), staying stock")
        return False


def apply(module=None) -> bool:
    """Registry callback entry; never raises, logs refusals once."""
    return apply_to_module(module)


_LOGGED = False


def _log_once(message: str) -> None:
    global _LOGGED  # noqa: PLW0603
    if _LOGGED:
        return
    _LOGGED = True
    import os

    print(f"[VLLM_HCU {PATCH_ID}] {message}", flush=True)
    if os.environ.get("VLLM_LOGGING_LEVEL", "").lower() == "debug":
        import traceback

        traceback.print_stack()


__all__ = [
    "PATCH_ID",
    "TARGET_CLASS",
    "TARGET_MODULE",
    "TARGET_SYMBOL",
    "apply",
    "apply_to_module",
]
