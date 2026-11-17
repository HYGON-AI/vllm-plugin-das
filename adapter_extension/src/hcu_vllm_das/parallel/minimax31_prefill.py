# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""MiniMax prefill policy and stream-owned scratch; no private env or debug hooks."""
import os
import torch
from hcu_adapter import hcu_config, registry
from hcu_adapter.kill_switch import op_disabled

_FEATURES = frozenset(("local_pages", "q_fp8", "cache_store", "pack_buffer", "metadata", "scale_guard", "selector"))
_POLICY = None
_PID = os.getpid()
_PACK = {}
_ACTIVE = False


def _check_process():
    global _PID
    if os.getpid() != _PID:
        if _POLICY is not None or _PACK or _ACTIVE:
            raise RuntimeError("active MiniMax prefill state inherited across fork; use spawn")
        _PID = os.getpid()


def enabled(name):
    global _POLICY
    _check_process()
    if name not in _FEATURES:
        raise ValueError("unknown MiniMax prefill feature: " + str(name))
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
    elif decision != _POLICY:
        raise RuntimeError("MiniMax prefill policy changed; restart the engine")
    return requested == "triton" and not decision[1]


def pack_rows(parts):
    """One same-stream, inference-only pack workspace per device/geometry."""
    _check_process()
    if not parts or any(p.ndim != 2 for p in parts):
        raise ValueError("expected rank-two PCP payload parts")
    x = parts[0]
    if (not x.is_cuda or x.dtype != torch.bfloat16 or
            any(p.device != x.device or p.dtype != x.dtype or p.shape[0] != x.shape[0]
                or p.requires_grad for p in parts)):
        raise ValueError("PCP pack requires same-device BF16 inference tensors")
    stream = torch.cuda.current_stream(x.device).cuda_stream
    widths = tuple(p.shape[1] for p in parts)
    key = (x.device, x.dtype, widths, stream)
    old = _PACK.get(key)
    if old is None or old.shape[0] < x.shape[0]:
        cap = max(x.shape[0], old.shape[0] * 2 if old is not None else 1)
        old = torch.empty((cap, sum(widths)), dtype=x.dtype, device=x.device)
        _PACK[key] = old
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
    from triton_tmp_dir.minimax31_prefill_triton import check_scales
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
