# SPDX-License-Identifier: Apache-2.0
"""Keep native fp8_ds_mla KV writes; avoid fused Q/KV scratch traffic.

The verified gfx938 binary leaves the first even rotation in each 16-element
lane group unfused. Its other even rotations and odd rotations use different
FMA operands. Preserve those boundaries; unknown native builds keep Core.
"""
from __future__ import annotations

from importlib import import_module
import logging
import os

import torch
from vllm.triton_utils import tl, triton

from hcu_adapter.operator_contracts import supports_operator_contract


_LOG = logging.getLogger('vllm.' + __name__)
_qualified_devices = set()
_declined_devices = set()


def _clear_preflight():
    _qualified_devices.clear()
    _declined_devices.clear()


if hasattr(os, 'register_at_fork'):
    os.register_at_fork(after_in_child=_clear_preflight)


def _has_qkv_contract():
    try:
        native = import_module('vllm._C_stable_libtorch')
        kv_insert = torch.ops._C.fused_deepseek_v4_kv_rope_insert
    except (ImportError, OSError, AttributeError):
        return False
    return (callable(kv_insert)
            and supports_operator_contract(native, 'dsv41_q_rope_and_padded_kv_insert'))


# Provider declarations can be checked without a device in the coordinator.
_NATIVE_BUILD_MATCHES = _has_qkv_contract()


def _native_ops():
    return (torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert,
            torch.ops._C.fused_deepseek_v4_kv_rope_insert)


def preflight_admitted():
    """Compile-cache factor for the result in the current worker process."""
    return bool(_qualified_devices)


def supported(q, kv, cache, slots, positions, cos_sin):
    tensors = (q, kv, cache, slots, positions, cos_sin)
    if (not all(isinstance(t, torch.Tensor) for t in tensors)
            or not (_NATIVE_BUILD_MATCHES or q.device in _qualified_devices)
            or q.ndim != 3 or q.shape[1:] != (64, 512)
            or not 0 < q.shape[0] <= 256
            or kv.shape != (q.shape[0], 512)
            or cache.ndim != 3 or cache.shape[0] <= 0
            or cache.shape[1] not in (32, 64) or cache.shape[2] != 584
            or cache.stride(2) != 1 or cache.stride(1) != 584
            or cache.stride(0) < cache.shape[1] * 584
            or slots.ndim != 1 or slots.shape[0] > q.shape[0]
            or positions.shape != (q.shape[0],)
            or cos_sin.ndim != 2 or cos_sin.shape[0] <= 0 or cos_sin.shape[1] != 64
            or q.dtype != torch.bfloat16 or kv.dtype != torch.bfloat16
            or cache.dtype != torch.uint8 or cos_sin.dtype != torch.float32
            or slots.dtype != torch.int64 or positions.dtype != torch.int64
            or not all(t.is_cuda and t.device == q.device for t in tensors)
            or not all(t.is_contiguous() for t in (q, kv, slots, positions, cos_sin))):
        return False
    return getattr(torch.cuda.get_device_properties(q.device), 'gcnArchName', '').split(':')[0] == 'gfx938'


@triton.jit
def _q_rope_native_512(Q, Positions, CosSin, Out):
    token = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1) * 4 + tl.arange(0, 4)[:, None]
    col = tl.arange(0, 512)[None, :]
    raw = tl.load(Q + token * 32768 + head * 512 + col).to(tl.float32)
    peer = tl.load(Q + token * 32768 + head * 512 + (col ^ 1), col >= 448, other=0).to(tl.float32)
    position = tl.load(Positions + token)
    index = (col - 448) // 2
    cosine = tl.load(CosSin + position * 64 + index, col >= 448, other=1.)
    sine = tl.load(CosSin + position * 64 + 32 + index, col >= 448, other=0.)
    even = col % 2 == 0
    ordinary = tl.where(even, tl.fma(raw, cosine, -peer * sine),
                        tl.fma(raw, cosine, peer * sine))
    unfused = raw * cosine + tl.where(even, -peer * sine, peer * sine)
    value = tl.where(col % 16 == 0, unfused, ordinary)
    value = tl.where(col >= 448, value, raw)
    tl.store(Out + token * 32768 + head * 512 + col, value)


@torch.inference_mode()
def qualify_qkv_split(device):
    """Compare Q RoPE and every KV cache byte with the installed fused op."""
    device = torch.device(device)
    if device.type != 'cuda':
        return False
    if device.index is None:
        device = torch.device('cuda', torch.cuda.current_device())
    if getattr(torch.cuda.get_device_properties(device), 'gcnArchName', '').split(':')[0] != 'gfx938':
        return False
    if _NATIVE_BUILD_MATCHES or device in _qualified_devices:
        return True
    if device in _declined_devices:
        return False
    with torch.cuda.device(device):
        if torch.cuda.is_current_stream_capturing():
            return False
        try:
            fused, _ = _native_ops()
            generator = torch.Generator(device=device).manual_seed(5924)
            angle = torch.randn((768, 32), device=device, dtype=torch.float32,
                                generator=generator)
            cos_sin = torch.cat((angle.cos(), angle.sin()), dim=1)
            for block_size in (32, 64):
                for rows in (1, 8, 48, 64, 192, 256):
                    q = torch.randn((rows, 64, 512), device=device, generator=generator).bfloat16()
                    kv = torch.randn((rows, 512), device=device, generator=generator).bfloat16()
                    positions = torch.arange(rows, device=device) * 3
                    slots = torch.arange(rows, device=device) * 2
                    slots[1::3] = -1
                    pages = max(4, (2 * rows + block_size - 1) // block_size)
                    for count, pitch in ((rows, 0), (rows // 2, 128)):
                        expected_backing = torch.full(
                            (pages, block_size * 584 + pitch), 17, dtype=torch.uint8, device=device)
                        actual_backing = expected_backing.clone()
                        expected_cache = expected_backing[:, :block_size * 584]
                        actual_cache = actual_backing[:, :block_size * 584].view(pages, block_size, 584)
                        expected_q = fused(q, kv, expected_cache, slots[:count], positions,
                                           cos_sin, 64, 1e-6, block_size, False, False, True, False)
                        actual_q = _split_qkv(q, kv, actual_cache, slots[:count], positions, cos_sin)
                        if (expected_q.shape != actual_q.shape or expected_q.dtype != actual_q.dtype
                                or not torch.equal(expected_q.view(torch.uint8), actual_q.view(torch.uint8))
                                or not torch.equal(expected_backing, actual_backing)):
                            _declined_devices.add(device)
                            _LOG.warning('HCU Q/KV split preflight differs from the installed fused op')
                            return False
        except (AttributeError, OSError, TypeError, ValueError, RuntimeError) as error:
            _declined_devices.add(device)
            _LOG.warning('HCU Q/KV split preflight unavailable: %s', error)
            return False
    _qualified_devices.add(device)
    _LOG.info('HCU Q/KV split admitted after native preflight')
    return True


def split_qkv(q, kv, cache, slots, positions, cos_sin):
    if not supported(q, kv, cache, slots, positions, cos_sin):
        raise ValueError('Q/KV split is outside the verified gfx938 operator contract')
    return _split_qkv(q, kv, cache, slots, positions, cos_sin)


def _split_qkv(q, kv, cache, slots, positions, cos_sin):
    out = torch.empty_like(q)
    _q_rope_native_512[(q.shape[0], 16)](
        q, positions, cos_sin, out, num_warps=4, enable_fp_fusion=False)
    torch.ops._C.fused_deepseek_v4_kv_rope_insert(
        kv, cache, slots, positions, cos_sin, cache.shape[1], None, False)
    return out
