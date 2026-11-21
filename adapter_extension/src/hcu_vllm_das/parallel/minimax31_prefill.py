# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""MiniMax prefill policy and stream-owned scratch; no private env or debug hooks."""

from ..adapter_api import adapter_module as _adapter_module
import os
import torch
hcu_config = _adapter_module('hcu_config')
registry = _adapter_module('registry')
op_disabled = _adapter_module('kill_switch').op_disabled

_POLICY = None
_ENABLED = None
_PID = os.getpid()
_PACK = {}
_ACTIVE = False


def _check_process():
    global _PID
    if os.getpid() != _PID:
        if _POLICY is not None or _PACK or _ACTIVE:
            raise RuntimeError("active MiniMax prefill state inherited across fork; use spawn")
        _PID = os.getpid()


def validate_policy():
    """Freeze at construction; reject drift once at each planner/profile boundary."""
    global _POLICY, _ENABLED
    _check_process()
    op = "vllm_minimax_prefill"
    if op not in hcu_config.BACKEND_CATALOG:
        raise RuntimeError("MiniMax prefill requires the paired Adapter policy catalog")
    requested = hcu_config.load()["ops"].get(op, "triton")
    if requested not in ("stock", "triton"):
        raise ValueError("unknown MiniMax prefill backend")
    decision = (requested, op_disabled(op))
    if _POLICY is None:
        backend = "stock" if decision[1] else requested
        registry.register(op=op, impl=backend,
                          target="hcu_vllm_das.parallel.minimax31_prefill",
                          mechanism="seam", source="hcu_vllm_das")
        _POLICY = decision
        _ENABLED = backend == "triton"
    elif decision != _POLICY:
        raise RuntimeError("MiniMax prefill policy changed; restart the engine")
    return _ENABLED


def enabled():
    """One composition, with only a PID comparison on initialized layer calls."""
    _check_process()
    if _POLICY is None:
        return validate_policy()
    return _ENABLED


def _pack_key(device):
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("PCP pack requires a CUDA/HIP device")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    stream = torch.cuda.current_stream(device).cuda_stream
    return device, torch.bfloat16, (1024, 128), stream


def reserve_pack_rows(rows, device):
    """Reserve the peak local payload during the existing vLLM memory probe."""
    _check_process()
    if type(rows) is not int or rows < 0:
        raise ValueError("invalid PCP pack profile row budget")
    key = _pack_key(device)
    old = _PACK.get(key)
    if old is None or old.shape[0] < rows:
        # The profile runs in inference_mode; persistent mutable scratch must
        # also be writable by a later no_grad/non-inference entry.
        with torch.inference_mode(False):
            _PACK[key] = torch.empty((rows, 1152), dtype=torch.bfloat16, device=key[0])


def pack_rows(parts):
    """Borrow profiled scratch until its same-stream synchronous gather consumes it.

    The caller must enqueue all source reads on this stream before the next pack.
    Do not retain the returned view or hand it to an async/foreign-stream gather.
    This function never grows persistent scratch after the KV memory budget.
    """
    _check_process()
    if len(parts) != 2 or any(p.ndim != 2 for p in parts):
        raise ValueError("expected rank-two PCP payload parts")
    x = parts[0]
    if (not x.is_cuda or x.dtype != torch.bfloat16 or
            tuple(p.shape[1] for p in parts) != (1024, 128) or
            any(p.device != x.device or p.dtype != x.dtype or p.shape[0] != x.shape[0]
                or p.requires_grad for p in parts)):
        raise ValueError("PCP pack requires same-device BF16 K/V/index inference tensors")
    old = _PACK.get(_pack_key(x.device))
    if old is None or old.shape[0] < x.shape[0]:
        raise RuntimeError("PCP pack workspace exceeds its profiled stream/row budget; restart the engine")
    out = old[:x.shape[0]]
    torch.cat(parts, dim=1, out=out)
    return out


def check_bound_scales(layer):
    """Return True only after a same-cache device guard was enqueued.

    New cache bindings return False so the original host validation runs. Strong
    cache references prevent allocator address reuse from inheriting a snapshot.
    The snapshot is private; neither producer tensors nor live cache are mutated.
    """
    _check_process()
    state = getattr(layer, "_hcu_prefill_scale_guard", None)
    if state is None or state[0].data_ptr() != layer.kv_cache.data_ptr():
        return False
    cache = layer.kv_cache
    geometry = (tuple(cache.shape), tuple(cache.stride()), cache.device, cache.dtype)
    if geometry != state[5]:
        raise RuntimeError("MiniMax scale guard cache geometry changed; restart the engine")
    stream = torch.cuda.current_stream(cache.device).cuda_stream
    if stream != state[4]:
        raise RuntimeError("MiniMax scale guard stream changed; construct a fresh engine")
    from .minimax31_prefill_triton import check_scales
    check_scales(layer._k_scale, layer._v_scale, state[1], state[2], state[3])
    return True


def bind_validated_scales(layer):
    global _ACTIVE
    _check_process()
    _ACTIVE = True
    stream = torch.cuda.current_stream(layer.kv_cache.device).cuda_stream
    layer._hcu_prefill_scale_guard = (layer.kv_cache, layer._k_scale.clone(),
                                    layer._v_scale.clone(),
                                    torch.empty((), dtype=torch.bool, device=layer.kv_cache.device), stream,
                                    (tuple(layer.kv_cache.shape), tuple(layer.kv_cache.stride()),
                                     layer.kv_cache.device, layer.kv_cache.dtype))
