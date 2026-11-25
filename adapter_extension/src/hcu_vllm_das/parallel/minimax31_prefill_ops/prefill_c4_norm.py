"""Fuse the validated C4 input norm and reuse its INT8 data at the same QKV call."""
from collections import Counter
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps

import torch
from triton.language.extra import libdevice
from vllm import _custom_ops as ops
from vllm.triton_utils import tl, triton

from hcu_vllm_das.parallel import minimax31_msa as msa
from hcu_vllm_das.parallel.minimax31_pcp import MiniMax31PCPSparseAttention
from hcu_vllm_das.quantization.minimax31_w8a8 import MiniMax31HipBlasLtInt8Kernel
from vllm.models.minimax_m3.amd.model import MiniMAXGemmaRMSNorm, MiniMaxM3DecoderLayer


@dataclass
class InputProjection:
    norm: object
    projection: object
    prepared: tuple | None = None


_current = ContextVar("c4_input_projection", default=None)
_enabled = False
_verify = False
_hits = Counter()
_verified = Counter()
_mismatches = Counter()
_installed = False


@triton.jit
def _norm_int8(x, residual, weight, y, carry, q, scales, eps, HAS_RESIDUAL: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, 8192)
    mask = col < 6144
    offset = row * 6144 + col
    total = tl.load(x + offset, mask, 0).to(tl.float32)
    if HAS_RESIDUAL:
        total += tl.load(residual + offset, mask, 0).to(tl.float32)
        tl.store(carry + offset, total.to(tl.bfloat16), mask)
    var = tl.sum(total * total, 0) / 6144
    inv = 1.0 / tl.sqrt(var + eps)
    w = tl.load(weight + col, mask, 0).to(tl.float32)
    norm = (total * inv * (1.0 + w)).to(tl.bfloat16)
    tl.store(y + offset, norm, mask)
    value = norm.to(tl.float32)
    maximum = tl.max(tl.where(mask, tl.abs(value), 0.0), 0)
    scale = tl.div_rn(maximum, 127.0)
    scaled = tl.where(maximum > 0.0, value * tl.div_rn(127.0, maximum), 0.0)
    rounded = tl.minimum(127.0, tl.maximum(-128.0, libdevice.rint(scaled)))
    tl.store(q + offset, rounded.to(tl.int8), mask)
    tl.store(scales + row, scale)


def _real_band(hidden):
    host = msa._HOST
    if host is None or hidden.shape != (2048, 6144):
        return False
    lengths, offsets, _ = host
    return (tuple(lengths) == (1024, 1024) and len(offsets) == 2
            and all(131072 <= start <= 146432 for start in offsets))


def _input_contract(x, residual, weight):
    return (
        x.dtype == torch.bfloat16 and x.is_cuda and x.is_contiguous()
        and weight.shape == (6144,) and weight.dtype == torch.bfloat16
        and weight.is_contiguous() and weight.device == x.device
        and (residual is None or (residual.shape == x.shape and residual.dtype == x.dtype
                                 and residual.device == x.device and residual.is_contiguous()))
    )


def _same(a, b):
    return torch.equal(a.view(torch.uint8), b.view(torch.uint8))


def _install():
    global _installed
    if _installed:
        return
    from .prefill_runtime_candidate import _patch

    old_layer = MiniMaxM3DecoderLayer.forward
    old_norm = MiniMAXGemmaRMSNorm.forward
    kernel = MiniMax31HipBlasLtInt8Kernel
    old_gemm = kernel.apply_weights

    def consume(layer, activation, flat):
        step = _current.get()
        if step is not None and step.projection is layer and step.prepared is not None:
            saved, q, scale = step.prepared
            if saved is activation:
                step.prepared = None
                _hits["qkv"] += 1
                return q, scale, None
        return ops.scaled_int8_quant(flat, None, None, symmetric=True)

    gemm = _patch(old_gemm,
                  "q,qs,zp=ops.scaled_int8_quant(flat,None,None,symmetric=True)",
                  "q,qs,zp=consume(layer,x,flat)",
                  consume=consume)

    @wraps(old_layer)
    def layer(self, positions, hidden_states, residual):
        if not _enabled or not isinstance(self.self_attn, MiniMax31PCPSparseAttention):
            return old_layer(self, positions, hidden_states, residual)
        projection = self.self_attn.qkv_proj
        selected = getattr(getattr(projection, "scheme", None), "kernel", None)
        if type(selected) is not kernel or not _real_band(hidden_states):
            return old_layer(self, positions, hidden_states, residual)
        token = _current.set(InputProjection(self.input_layernorm, projection))
        try:
            return old_layer(self, positions, hidden_states, residual)
        finally:
            _current.reset(token)

    @wraps(old_norm)
    def norm(self, x, residual=None):
        step = _current.get()
        if (step is None or step.norm is not self or step.prepared is not None
                or not _input_contract(x, residual, self.weight)):
            return old_norm(self, x, residual)
        y = torch.empty_like(x)
        carry = torch.empty_like(residual) if residual is not None else None
        q = torch.empty_like(x, dtype=torch.int8)
        scale = torch.empty((2048, 1), dtype=torch.float32, device=x.device)
        _norm_int8[(2048,)](
            x, x if residual is None else residual, self.weight, y,
            x if carry is None else carry, q, scale, self.variance_epsilon,
            residual is not None, num_warps=16,
        )
        if _verify:
            expected = old_norm(self, x, residual)
            ref_y, ref_carry = (expected, None) if residual is None else expected
            ref_q, ref_scale, zp = ops.scaled_int8_quant(ref_y, None, None, symmetric=True)
            if (zp is not None or not _same(y, ref_y) or not _same(q, ref_q)
                    or not _same(scale, ref_scale)
                    or (carry is not None and not _same(carry, ref_carry))):
                _mismatches["qkv"] += 1
                return expected
            _verified["qkv"] += 1
        step.prepared = y, q, scale
        return y if residual is None else (y, carry)

    MiniMaxM3DecoderLayer.forward = layer
    MiniMAXGemmaRMSNorm.forward = norm
    kernel.apply_weights = gemm
    _installed = True


def configure(value):
    global _enabled, _verify
    if value == "status":
        return {"enabled": _enabled, "verify": _verify, "hits": dict(_hits),
                "verified": dict(_verified), "mismatches": dict(_mismatches)}
    if value is not None and type(value) is not bool:
        raise ValueError("Use null for stock, true for comparison, or false for timing")
    _install()
    _enabled = value is not None
    _verify = value is True
    for counts in (_hits, _verified, _mismatches):
        counts.clear()
    return configure("status")
