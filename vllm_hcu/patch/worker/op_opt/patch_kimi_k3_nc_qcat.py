# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Route the Kimi-K3 fp8 decode epilogue through capture-safe Triton kernels.

Why: the vendor fused op
``fused_kimi_k3_mla_decode_q_concat_kv_cache_fp8_insert`` (PDL, warp-per-
(token, head) grid) intermittently drops whole warp writes at large batch
under FULL-cudagraph replay.  Observed poison signature: NC-entry ``q_spec``
rows written as all-zero while upstream activations are healthy, ``lse``
frozen at a constant, draft output collapsing onto one junk token.

What: the fp8 decode branch is rerouted to two Triton kernels —
Q lane: interleaved (GPT-J) RoPE over the adjacent pair + concat + static
E4M3 quant by ``q_scale_inv``; cache lane: packed-pair RoPE + E4M3 quant
written straight into the paged cache slot.  Measured against the vendor op
at the production 48x12 shape: Q lane 331776/331776 bytes and cache lane
27648/27648 bytes identical (bit-exact), graph replay 18.3us vs 18.4us.

The installed vLLM tree is never rewritten: this adapter replaces the module
symbol at import time, re-binds every already-imported consumer (including
``nvidia/mla.py``'s ``from ... import`` name), and registers a post-import
callback so a late ``nvidia.mla`` import is re-bound when it completes.

Contract: only the fp8 / ``ds_mla=False`` / bf16 / ``kv_lora_rank == 512`` /
``rope == 64`` shape is routed.  Every other call — including malformed
scales, unpaired ``positions``/``cos_sin_cache`` and non-3D caches — is
delegated to the vendor op unchanged so its own validation semantics are
preserved.

Kill switch: ``VLLM_HCU_K3_QCAT_TRITON=0`` read at worker start keeps the
vendor op (the adapter does not install at all).
"""
from __future__ import annotations

import os
import sys
from types import ModuleType

from ._common import load_exact_module

TARGET_MODULE = "vllm.models.kimi_k3.nvidia.ops.fused_mla_key_concat_kv_cache"
PATCH_ID = "worker.op_opt.kimi_k3.nc_qcat"
TARGETS = (f"{TARGET_MODULE}.fused_mla_decode_q_concat_kv_cache_insert",)
_MARKER = "_vllm_hcu_kimi_nc_qcat_applied"
_ENV = "VLLM_HCU_K3_QCAT_TRITON"
_MLA_MODULE = "vllm.models.kimi_k3.nvidia.mla"
_MLA_ATTR = "fused_mla_decode_q_concat_kv_cache_insert"
_CONSUMER_PREFIX = "vllm.models.kimi_k3"

_LORA = 512
_ROPE = 64

# Set by apply_to_module: (original, patched).  Consumers imported later are
# re-bound by the registered post-import callback.
_INSTALLED: tuple | None = None
_DEBUG = os.environ.get("VLLM_HCU_K3_NCQ_DEBUG", "0") == "1"
_DBG_BUDGET = [24]


def _debug_dump(original, out, ql_nope, q_pe, kv_c_normed, k_pe, kv_cache,
                slot_mapping, q_scale_inv, cache_scale_inv, positions,
                cos_sin_cache) -> None:
    """Eager-only, budgeted parity dump (never runs inside graph capture)."""
    import json

    try:
        import torch
    except Exception:  # noqa: BLE001
        return
    if torch.cuda.is_current_stream_capturing() or _DBG_BUDGET[0] <= 0:
        return
    _DBG_BUDGET[0] -= 1
    try:
        rec = {
            "b": int(out.shape[0]),
            "q_scale_inv": float(q_scale_inv),
            "cache_scale_inv": float(cache_scale_inv),
            "q_in_absmax": float(ql_nope.float().abs().max()),
            "pe_in_absmax": float(q_pe.float().abs().max()),
            "kvc_absmax": float(kv_c_normed.float().abs().max()),
            "kpe_absmax": float(k_pe.float().abs().max()),
            "slot_min": int(slot_mapping.min()),
            "slot_max": int(slot_mapping.max()),
            "pos_min": int(positions.min()) if positions is not None else -1,
            "pos_max": int(positions.max()) if positions is not None else -1,
            "out_absmax": float(out.float().abs().max()),
            "out_zero_frac": float((out.float() == 0).float().mean()),
            "out_sat_frac": float((out.float().abs() >= 440).float().mean()),
        }
        # same inputs through the vendor op on a scratch cache copy
        scratch = kv_cache.clone()
        ref = original(
            ql_nope, q_pe, kv_c_normed, k_pe, scratch, slot_mapping,
            ds_mla=False, q_scale_inv=q_scale_inv,
            cache_scale_inv=cache_scale_inv, positions=positions,
            cos_sin_cache=cos_sin_cache,
        )
        bits = (ref.view(torch.uint8).int() - out.view(torch.uint8).int())
        rec["ref_absmax"] = float(ref.float().abs().max())
        rec["diff_bytes"] = int((bits != 0).sum())
        rec["diff_max_step"] = int(bits.abs().max())
        rec["bytes"] = int(bits.numel())
        with open("/tmp/nc_dbg.log", "a") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception as exc:  # noqa: BLE001
        try:
            with open("/tmp/nc_dbg.log", "a") as f:
                f.write(json.dumps({"dbg_err": repr(exc)}) + "\n")
        except Exception:  # noqa: BLE001
            pass


def _enabled() -> bool:
    # Opt-in: the vendor op was cleared of blame (the DSpark poison traced to
    # the NC fast path instead), so this capture-safe replacement stays off
    # unless an operator asks for it.
    return os.environ.get(_ENV, "0").strip().lower() in ("1", "true", "yes")


def _rebind_consumers(original, patched) -> int:
    fixed = 0
    for name, mod in list(sys.modules.items()):
        if mod is None or not name.startswith(_CONSUMER_PREFIX):
            continue
        if getattr(mod, _MLA_ATTR, None) is original:
            setattr(mod, _MLA_ATTR, patched)
            fixed += 1
    return fixed


def _on_mla_import(module: ModuleType) -> None:
    """Post-import hook: a consumer imported after us must see the route."""
    if _INSTALLED is None:
        return
    original, patched = _INSTALLED
    if getattr(module, _MLA_ATTR, None) is original:
        setattr(module, _MLA_ATTR, patched)


def _register_late_rebind() -> None:
    try:
        from vllm_hcu.patch.import_coordinator import register_post_import
    except Exception:  # standalone harness without the coordinator
        return
    try:
        register_post_import(
            PATCH_ID + ".mla_binding", _MLA_MODULE, _on_mla_import,
            targets=TARGETS,
        )
    except Exception:  # already registered (idempotent re-apply)
        pass


def _build(original):
    """Return the drop-in replacement built around the vendor ``original``."""
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def _qcat_kernel(
        ql_ptr, qpe_ptr, cs_ptr, pos_ptr, out_ptr, qsi_ptr,
        stride_qt, stride_qh, stride_pt, stride_ph, stride_ot, stride_oh,
        LORA: tl.constexpr, ROPE: tl.constexpr,
        APPLY_ROPE: tl.constexpr,
    ):
        pid_t = tl.program_id(0)
        pid_h = tl.program_id(1)
        offs = tl.arange(0, LORA)
        ql = tl.load(ql_ptr + pid_t * stride_qt + pid_h * stride_qh + offs)
        offs_r = tl.arange(0, ROPE)
        pe = tl.load(
            qpe_ptr + pid_t * stride_pt + pid_h * stride_ph + offs_r)
        if APPLY_ROPE:
            pos = tl.load(pos_ptr + pid_t)
            even = offs_r % 2 == 0
            cos = tl.load(
                cs_ptr + pos * ROPE + offs_r // 2).to(tl.float32)
            sin = tl.load(
                cs_ptr + pos * ROPE + ROPE // 2 + offs_r // 2).to(tl.float32)
            xf = pe.to(tl.float32)
            peer_off = tl.where(even, offs_r + 1, offs_r - 1)
            peer = tl.load(
                qpe_ptr + pid_t * stride_pt + pid_h * stride_ph + peer_off,
            ).to(tl.float32)
            rot_even = xf * cos - peer * sin
            rot_odd = xf * cos + peer * sin
            pe = tl.where(even, rot_even, rot_odd).to(pe.dtype)
        qsi = tl.load(qsi_ptr)
        o_lo = (ql.to(tl.float32) * qsi).to(out_ptr.dtype.element_ty)
        o_hi = (pe.to(tl.float32) * qsi).to(out_ptr.dtype.element_ty)
        tl.store(out_ptr + pid_t * stride_ot + pid_h * stride_oh + offs,
                 o_lo)
        tl.store(out_ptr + pid_t * stride_ot + pid_h * stride_oh + LORA
                 + offs_r, o_hi)

    @triton.jit
    def _cache_kernel(
        kvc_ptr, kpe_ptr, cs_ptr, pos_ptr, cache_ptr, slot_ptr, ksi_ptr,
        stride_kvc, stride_kpe, cache_bs, cache_cs, cache_es, num_slots,
        LORA: tl.constexpr, ROPE: tl.constexpr,
        APPLY_ROPE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        slot = tl.load(slot_ptr + pid)
        if slot < 0 or slot >= num_slots:
            return
        ksi = tl.load(ksi_ptr)
        # Block size is an explicit argument: never derived from stride
        # ratios, so page-padded / block-major / strided cache layouts keep
        # their physical addressing.
        base = cache_ptr + slot // cache_bs * cache_cs \
            + slot % cache_bs * cache_es
        offs = tl.arange(0, LORA)
        v = tl.load(kvc_ptr + pid * stride_kvc + offs)
        tl.store(base + offs, (v.to(tl.float32) * ksi).to(
            cache_ptr.dtype.element_ty))
        offs_r = tl.arange(0, ROPE)
        r = tl.load(kpe_ptr + pid * stride_kpe + offs_r)
        if APPLY_ROPE:
            pos = tl.load(pos_ptr + pid)
            even = offs_r % 2 == 0
            cn = tl.load(cs_ptr + pos * ROPE + offs_r // 2).to(tl.float32)
            sn = tl.load(
                cs_ptr + pos * ROPE + ROPE // 2 + offs_r // 2).to(
                    tl.float32)
            xf = r.to(tl.float32)
            po = tl.where(even, offs_r + 1, offs_r - 1)
            pr = tl.load(kpe_ptr + pid * stride_kpe + po).to(tl.float32)
            re = xf * cn - pr * sn
            ro = xf * cn + pr * sn
            r = tl.where(even, re, ro).to(r.dtype)
        tl.store(base + LORA + offs_r, (r.to(tl.float32) * ksi).to(
            cache_ptr.dtype.element_ty))

    def _supported(ql_nope, q_pe, kv_c_normed, k_pe, kv_cache, slot_mapping,
                   ds_mla, q_scale_inv, cache_scale_inv, positions,
                   cos_sin_cache) -> bool:
        if ds_mla or q_scale_inv is None or cache_scale_inv is None:
            return False
        # The kernels index every tensor with unit element offsets, so all
        # operands must be CUDA tensors whose innermost dim is contiguous.
        if ql_nope.device.type != "cuda":
            return False
        if ql_nope.dim() != 3 or kv_cache.dim() != 3:
            return False
        if ql_nope.dtype is not torch.bfloat16:
            return False
        if ql_nope.shape[-1] != _LORA or ql_nope.stride(-1) != 1:
            return False
        if q_pe.shape[-1] != _ROPE:
            return False
        if q_pe.dtype is not torch.bfloat16 \
                or q_pe.stride(-1) != 1:
            return False
        # Cache entry width and dtype must match the kernels' write pattern.
        if kv_cache.shape[-1] != _LORA + _ROPE:
            return False
        if kv_cache.dtype not in (torch.float8_e4m3fn, torch.uint8):
            return False
        if kv_cache.stride(-1) != 1:
            return False
        # KV inputs: widths, dtype, batch, and innermost contiguity (the
        # kernels index offs 0..511 / 0..63 with +-1 rope peers directly).
        b = ql_nope.shape[0]
        if kv_c_normed.dim() != 2 or kv_c_normed.shape[-1] != _LORA:
            return False
        if kv_c_normed.dtype is not torch.bfloat16 \
                or kv_c_normed.stride(-1) != 1:
            return False
        if k_pe.dim() not in (2, 3) or k_pe.shape[-1] != _ROPE:
            return False
        if k_pe.dtype is not torch.bfloat16 or k_pe.stride(-1) != 1:
            return False
        if kv_c_normed.shape[0] != b or k_pe.shape[0] != b:
            return False
        # Scales: scalar, float32, on the compute device.
        for scale in (q_scale_inv, cache_scale_inv):
            if scale.numel() != 1 or scale.dtype is not torch.float32:
                return False
            if scale.device != ql_nope.device:
                return False
        if slot_mapping.dim() != 1 or slot_mapping.numel() != b:
            return False
        if slot_mapping.dtype not in (torch.int64, torch.int32):
            return False
        if slot_mapping.stride(0) != 1:
            return False
        if slot_mapping.device != ql_nope.device:
            return False
        # positions / cos_sin_cache are used as a pair by the vendor op; an
        # unpaired argument must reach it so its own error semantics hold.
        if (positions is None) != (cos_sin_cache is None):
            return False
        if positions is not None:
            if positions.dim() != 1 or positions.numel() != b:
                return False
            if positions.dtype not in (torch.int64, torch.int32):
                return False
            if positions.stride(0) != 1:
                return False
            if positions.device != ql_nope.device:
                return False
            if cos_sin_cache.dim() != 2 \
                    or cos_sin_cache.shape[-1] != _ROPE:
                return False
            if cos_sin_cache.dtype is not torch.float32:
                return False
            # the kernels index a packed row: cs + pos*ROPE + off
            if cos_sin_cache.stride(0) != _ROPE:
                return False
            if cos_sin_cache.stride(-1) != 1:
                return False
            if cos_sin_cache.device != ql_nope.device:
                return False
        for t in (q_pe, kv_c_normed, k_pe, kv_cache):
            if t.device != ql_nope.device:
                return False
        return True

    def _route(
        ql_nope, q_pe, kv_c_normed, k_pe, kv_cache, slot_mapping,
        q_scale_inv, cache_scale_inv, positions, cos_sin_cache,
    ):
        b, num_heads, kv_lora_rank = ql_nope.shape
        rope_dim = q_pe.shape[-1]
        has_rope = positions is not None and cos_sin_cache is not None
        dev = ql_nope.device
        mqa = torch.empty(
            (b, num_heads, kv_lora_rank + rope_dim),
            dtype=torch.float8_e4m3fn, device=dev,
        )
        if b == 0:
            return mqa
        if has_rope:
            pos_c, cs_c = positions, cos_sin_cache
        else:
            pos_c = torch.zeros(b, dtype=torch.int64, device=dev)
            cs_c = torch.zeros(rope_dim, dtype=torch.float32, device=dev)
        cache = (
            kv_cache
            if kv_cache.dtype == torch.float8_e4m3fn
            else kv_cache.view(torch.float8_e4m3fn)
        )
        k_pe2 = k_pe.reshape(b, -1)
        _qcat_kernel[(b, num_heads)](
            ql_nope, q_pe, cs_c, pos_c, mqa, q_scale_inv,
            ql_nope.stride(0), ql_nope.stride(1),
            q_pe.stride(0), q_pe.stride(1),
            mqa.stride(0), mqa.stride(1),
            LORA=kv_lora_rank, ROPE=rope_dim, APPLY_ROPE=has_rope,
            num_warps=4,
        )
        _cache_kernel[(b,)](
            kv_c_normed, k_pe2, cs_c, pos_c, cache, slot_mapping,
            cache_scale_inv,
            kv_c_normed.stride(0), k_pe2.stride(0),
            cache.shape[1], cache.stride(0), cache.stride(1),
            cache.shape[0] * cache.shape[1],
            LORA=kv_lora_rank, ROPE=rope_dim, APPLY_ROPE=has_rope,
            num_warps=4,
        )
        return mqa

    def fused_mla_decode_q_concat_kv_cache_insert(
        ql_nope, q_pe, kv_c_normed, k_pe, kv_cache, slot_mapping,
        *, ds_mla=False, q_scale_inv=None, cache_scale_inv=None,
        positions=None, cos_sin_cache=None,
    ):
        if not _supported(
            ql_nope, q_pe, kv_c_normed, k_pe, kv_cache, slot_mapping,
            ds_mla, q_scale_inv, cache_scale_inv, positions, cos_sin_cache,
        ):
            return original(
                ql_nope, q_pe, kv_c_normed, k_pe, kv_cache, slot_mapping,
                ds_mla=ds_mla, q_scale_inv=q_scale_inv,
                cache_scale_inv=cache_scale_inv, positions=positions,
                cos_sin_cache=cos_sin_cache,
            )
        out = _route(
            ql_nope, q_pe, kv_c_normed, k_pe, kv_cache, slot_mapping,
            q_scale_inv, cache_scale_inv, positions, cos_sin_cache,
        )
        if _DEBUG:
            _debug_dump(
                original, out, ql_nope, q_pe, kv_c_normed, k_pe, kv_cache,
                slot_mapping, q_scale_inv, cache_scale_inv, positions,
                cos_sin_cache,
            )
        return out

    return fused_mla_decode_q_concat_kv_cache_insert


def apply_to_module(module: ModuleType) -> bool:
    global _INSTALLED
    target = load_exact_module(TARGET_MODULE, module)
    if getattr(target, _MARKER, False):
        # Idempotent re-apply still repairs a consumer whose binding raced.
        if _INSTALLED is not None:
            _rebind_consumers(*_INSTALLED)
        return False
    if not _enabled():
        # Kill switch read at worker start: leave the vendor op in place.
        return False
    original = getattr(target, _MLA_ATTR)
    patched = _build(original)
    patched.__doc__ = getattr(original, "__doc__", None)
    patched.__name__ = getattr(original, "__name__", _MLA_ATTR)
    setattr(target, _MLA_ATTR, patched)
    _INSTALLED = (original, patched)
    _rebind_consumers(original, patched)
    _register_late_rebind()
    setattr(target, _MARKER, True)  # published last: consumers already fixed
    return True
