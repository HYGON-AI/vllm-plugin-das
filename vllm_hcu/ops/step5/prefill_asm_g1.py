# SPDX-License-Identifier: Apache-2.0
"""Opt-in Step5 prefill ASM G1 with weights packed during normal loading.

Raw channel-FP8 weights remain authoritative for native fallback. Only the
expert-internal byte layout is cached; expert order, channel scales, activation,
router weighting, G2 and Directsum are unchanged. Cache allocations belong to
the model-loader memory profiler, before KV sizing and graph capture.
"""
from __future__ import annotations

from dataclasses import dataclass
import functools
import hashlib
import json
import os
import threading
import weakref

from .tuning import asm_recipe

import torch
import triton
import triton.language as tl

from vllm_hcu.model_executor.layers.fused_moe import ep_fused_moe as ep
from vllm_hcu.model_executor.layers.fused_moe.ep_fused_moe import _ep_swiglu_chunk

_ATTR = "_hcu_step5_prefill_asm_g1_cache"
_SKIP = "_hcu_step5_prefill_asm_g1_skip"
_BUFFER = "_hcu_step5_prefill_asm_g1_weight"
_LOCK = threading.RLock()
_CACHES = weakref.WeakValueDictionary()
_ONES = {}
_M, _N, _K, _TOPK, _BM = 65536, 3072, 4096, 8, 128
_LOCAL_EXPERTS = (22, 23)


def _persist_for_experts(local_experts: int) -> int:
    return asm_recipe(local_experts)[1]


@dataclass
class _PackedWeight:
    packed: torch.Tensor
    source_ref: weakref.ReferenceType
    source_ptr: int
    source_version: int | None
    device_index: int
    layer_name: str
    owner_ref: weakref.ReferenceType
    static_row: tuple[int, ...] | None
    used: bool = False


def _enabled() -> bool:
    return os.environ.get("HCU_STEP5_PREFILL_ASM_G1", "0") == "1"


@functools.lru_cache(maxsize=None)
def _is_gfx938(device_index: int) -> bool:
    return torch.cuda.get_device_properties(device_index).gcnArchName.split(":", 1)[0] == "gfx938"


def _version(weight: torch.Tensor) -> int | None:
    try:
        return int(weight._version)
    except RuntimeError:  # Inference tensors may have no version counter.
        return None


def _overlaps(left: torch.Tensor, right: torch.Tensor) -> bool:
    if not left.numel() or not right.numel():
        return False
    a, b = left.data_ptr(), right.data_ptr()
    return a < b + right.numel() * right.element_size() and b < a + left.numel() * left.element_size()


def _discard(layer) -> None:
    previous = getattr(layer, _ATTR, None)
    if previous is None:
        return
    _CACHES.pop((previous.device_index, previous.source_ptr), None)
    raw = previous.source_ref()
    if raw is not None and getattr(raw, _ATTR, None) is previous:
        delattr(raw, _ATTR)
    setattr(layer, _ATTR, None)
    setattr(layer, _BUFFER, None)


def _expert_layout_eligible(context, moe, layer, local_experts: int) -> bool:
    """E23 requires the immutable direct-load plan, never dynamic EPLB.

    The normal static loader binds its row before checkpoint callbacks. Thus
    weight postprocessing already sees the final replica slots; runtime commit
    only publishes routing tables and does not move or replace these weights.
    """
    if local_experts == 22:
        return (context.get("enable_eplb") is False
                and getattr(moe, "num_experts", None) == 352)
    if local_experts != 23:
        return False
    row = getattr(layer, "_vllm_hcu_static_eplb_row", None)
    return (
        context.get("enable_eplb") is True
        and bool(context.get("static_expert_map_path"))
        and not context.get("expert_map_record_path")
        and context.get("eplb_disable_rearrange") is True
        and context.get("static_dispatch_policy") == "token_hash"
        and context.get("num_redundant_experts") == 16
        and context.get("eplb_use_async") is False
        and context.get("enable_elastic_ep") is False
        and context.get("enable_ep_weight_filter") is False
        and getattr(moe, "num_experts", None) == 368
        and getattr(moe, "num_logical_experts", None) == 352
        and type(row) is tuple and len(row) == 368
        and all(type(expert) is int for expert in row)
        and set(row) == set(range(352))
    )


def prepare_prefill_asm_g1_cache(method, layer, owner_context) -> bool:
    """Pack the final postprocessed Parameter inside model_loader.load_model.

    The owner context is captured at quant-method construction, because its
    current VllmConfig context may have ended by weight postprocessing. Draft
    layers and dynamic EPLB are excluded. This function is idempotent and is
    called again after a normal reload; forward never allocates a weight cache.
    """
    if not _enabled():
        return False
    weight = getattr(layer, "w13_weight", None)
    if (not isinstance(weight, torch.Tensor) or weight.ndim != 3
            or weight.shape[0] not in _LOCAL_EXPERTS
            or tuple(weight.shape[1:]) != (_N, _K)):
        with _LOCK:
            _discard(layer)
        return False
    layer_name = str(getattr(layer, "layer_name", ""))
    context = owner_context or {}
    layer_name = layer_name or str(context.get("layer_name") or "")
    moe = getattr(method, "moe", None)
    quant = getattr(method, "moe_quant_config", None)
    local_experts = weight.shape[0]
    activation = getattr(getattr(layer, "activation", None), "value", getattr(layer, "activation", None))
    # Step5 selects SWIGLUSTEP from its per-layer limit but does not pass a
    # separate swiglu_limit into FusedMoEFactory. The native EP path defaults
    # a missing gemm1 limit to 7; preserve that exact effective-limit rule.
    limit = getattr(quant, "gemm1_clamp_limit", None)
    if limit is None:
        limit = getattr(moe, "swiglu_limit", None)
    if limit is None and activation == "swiglustep":
        limit = 7.0
    eligible = (
        context.get("is_step5") is True
        and not context.get("is_draft")
        and ".mtp_block." not in layer_name
        and context.get("tp_size") == 1 and context.get("dp_size") == 16
        and context.get("enable_ep") is True
        and _expert_layout_eligible(context, moe, layer, local_experts)
        and getattr(moe, "tp_size", None) == 1
        and getattr(moe, "num_local_experts", None) == local_experts
        and getattr(moe, "experts_per_token", None) == _TOPK
        and not getattr(moe, "is_lora_enabled", False)
        and not getattr(layer, "apply_router_weight_on_input", False)
        and activation == "swiglustep"
        and limit == 7.0
        and getattr(quant, "use_fp8_w8a8", False)
        and getattr(quant, "per_act_token_quant", False)
        and getattr(quant, "per_out_ch_quant", False)
        and getattr(quant, "block_shape", None) is None
        and weight.dtype == torch.float8_e4m3fn and weight.is_cuda
        and weight.is_contiguous() and _is_gfx938(weight.device.index)
    )
    if eligible:
        eligible = asm_recipe(local_experts) is not None
    if not eligible:
        with _LOCK:
            _discard(layer)
            setattr(weight, _SKIP, True)
        return False
    with _LOCK:
        version = _version(weight)
        # Always refresh on postprocessing: some weight loaders write through
        # .data/native code without advancing the Parameter's version counter.
        _discard(layer)
        from aiter.ops.shuffle import asm_shuffle_weight_b8

        before = int(torch.cuda.memory_allocated(weight.device))
        with torch.no_grad():
            # view is a byte reinterpretation, never a numeric FP8 conversion.
            packed = asm_shuffle_weight_b8(weight.view(torch.int8), stage=1).view(weight.dtype)
        if (packed.shape != weight.shape or packed.dtype != weight.dtype
                or not packed.is_contiguous() or packed.device != weight.device
                or _overlaps(packed, weight)):
            raise RuntimeError("Invalid Step5 ASM G1 post-load weight layout")
        # Establish readiness once at load time; no event wait or allocation
        # is added to the captured forward or to either DBO compute segment.
        torch.cuda.current_stream(weight.device).synchronize()
        static_row = (getattr(layer, "_vllm_hcu_static_eplb_row")
                      if local_experts == 23 else None)
        record = _PackedWeight(packed, weakref.ref(weight), weight.data_ptr(), version,
                               weight.device.index, layer_name, weakref.ref(layer), static_row)
        if _BUFFER not in layer._buffers:
            layer.register_buffer(_BUFFER, packed, persistent=False)
        else:
            setattr(layer, _BUFFER, packed)
        setattr(layer, _ATTR, record)
        setattr(weight, _ATTR, record)
        setattr(weight, _SKIP, False)
        _CACHES[(record.device_index, record.source_ptr)] = record
        records = tuple(_CACHES.values())
        print("[step5_prefill_asm_g1_cache] " + json.dumps(dict(
            layer=layer_name, bytes=packed.numel() * packed.element_size(),
            total_layers=len(records), total_bytes=sum(r.packed.numel() * r.packed.element_size() for r in records),
            allocation_delta=int(torch.cuda.memory_allocated(weight.device)) - before,
            source_ptr=record.source_ptr, packed_ptr=packed.data_ptr(),
            source_version=version, mtp_cached=False, solution=asm_recipe(local_experts)[0],
            persist=_persist_for_experts(local_experts), asm_block_m=_BM, shuffle=1,
            local_experts=local_experts, physical_experts=local_experts * 16,
            static_row_sha256=(hashlib.sha256(json.dumps(static_row).encode()).hexdigest()
                               if static_row is not None else None),
        ), sort_keys=True), flush=True)
    return True


def _lookup(weight: torch.Tensor) -> _PackedWeight | None:
    if getattr(weight, _SKIP, False):
        return None
    with _LOCK:
        record = getattr(weight, _ATTR, None)
        if record is None:
            record = _CACHES.get((weight.device.index, weight.data_ptr()))
        if record is None:
            raise RuntimeError("Step5 ASM G1 requires a cache prepared during weight loading")
        source = record.source_ref()
        version = None if source is None else _version(source)
        if (source is None or source.data_ptr() != record.source_ptr
                or source.shape != weight.shape or source.dtype != weight.dtype
                or source.device != weight.device
                or record.source_ptr != weight.data_ptr() or record.device_index != weight.device.index
                or record.packed.shape != weight.shape or record.packed.dtype != weight.dtype
                or (record.source_version is not None and version is not None
                    and record.source_version != version)):
            raise RuntimeError("Stale Step5 ASM G1 cache; rerun normal weight postprocessing after reload")
        if record.static_row is not None:
            owner = record.owner_ref()
            if (owner is None or getattr(owner, "_vllm_hcu_static_eplb_row", None)
                    is not record.static_row):
                raise RuntimeError("Stale Step5 E23 ASM G1 static replica placement")
        return record


@triton.jit
def _prepare_metadata(sid, eid, nv, safe, expanded,
                      CAP: tl.constexpr, BLOCK: tl.constexpr,
                      M: tl.constexpr, TOPK: tl.constexpr, ASM_BM: tl.constexpr):
    r = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    count = tl.minimum(tl.maximum(tl.load(nv), 0), CAP)
    encoded = tl.load(sid + r, mask=(r < CAP) & (r < count), other=M | (TOPK << 24))
    tl.store(safe + r, encoded, mask=r < CAP)
    expert = tl.load(eid + r * ASM_BM // 256,
                     mask=(r < CAP // ASM_BM) & (r * ASM_BM < count), other=0)
    tl.store(expanded + r, expert, mask=r < CAP // ASM_BM)


@triton.jit
def _route_quant_kernel(q, scale, cache1, weights, nv, sid,
                        ROWS: tl.constexpr, LIMIT: tl.constexpr,
                        BLOCK_M: tl.constexpr, CH: tl.constexpr,
                        M: tl.constexpr, TOPK: tl.constexpr, N: tl.constexpr):
    count = tl.minimum(tl.maximum(tl.load(nv), 0), ROWS)
    cols = tl.arange(0, CH)
    step = tl.num_programs(0) * BLOCK_M
    for start in range(tl.program_id(0) * BLOCK_M, count, step):
        rows = start + tl.arange(0, BLOCK_M)
        row_mask = rows < count
        encoded = tl.load(sid + rows, mask=row_mask, other=0)
        token, slot = encoded & 0xFFFFFF, encoded >> 24
        live = row_mask & (token < M) & (slot >= 0) & (slot < TOPK)
        flat = token.to(tl.int64) * TOPK + slot.to(tl.int64)
        offsets = flat[:, None] * N + cols[None, :]
        y0 = _ep_swiglu_chunk(cache1, offsets, live[:, None], LIMIT, N // 2)
        y1 = _ep_swiglu_chunk(cache1, offsets + CH, live[:, None], LIMIT, N // 2)
        y2 = _ep_swiglu_chunk(cache1, offsets + 2 * CH, live[:, None], LIMIT, N // 2)
        maximum = tl.maximum(tl.maximum(tl.max(tl.abs(y0), 1), tl.max(tl.abs(y1), 1)), tl.max(tl.abs(y2), 1))
        s = tl.maximum(maximum.to(tl.float32), 1.0e-10) / 448.0
        inverse = (1.0 / s)[:, None]
        out = rows[:, None].to(tl.int64) * (N // 2) + cols[None, :]
        tl.store(q + out, (y0 * inverse).to(q.dtype.element_ty), mask=row_mask[:, None], cache_modifier=".cs")
        tl.store(q + out + CH, (y1 * inverse).to(q.dtype.element_ty), mask=row_mask[:, None], cache_modifier=".cs")
        tl.store(q + out + 2 * CH, (y2 * inverse).to(q.dtype.element_ty), mask=row_mask[:, None], cache_modifier=".cs")
        w = tl.load(weights + rows, mask=row_mask, other=0.0)
        tl.store(scale + rows, s * w, mask=row_mask)


def try_step5_prefill_asm_g1(a, w1, w2, cache1, a_scale, w_scale,
                            sorted_ids, sorted_weights, expert_ids, nvalid,
                            *, num_tokens, top_k, sort_block_m, limit, workspace2):
    """Return sorted FP8 activation/scales, or None for the original path.

    One compute-stream segment covers ASM projection and activation. The stock
    ASM API receives the original exact-M input; this helper neither assumes
    extra backing rows nor copies input to add a tail. Exact-storage integration
    still requires the normal-source acceptance test, beyond the private probe.
    """
    if (not _enabled() or num_tokens != _M or top_k != _TOPK
            or sort_block_m != 256 or not isinstance(limit, (int, float))
            or float(limit) != 7.0):
        return None
    tensors = (a, w1, w2, cache1, a_scale, w_scale, sorted_ids, sorted_weights, expert_ids, nvalid)
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
        return None
    local_experts = w1.shape[0] if w1.ndim == 3 else 0
    if (a.shape != (_M, _K) or local_experts not in _LOCAL_EXPERTS
            or w1.shape != (local_experts, _N, _K)
            or w2.shape != (local_experts, _K, _N // 2)
            or a.dtype != torch.float8_e4m3fn or w1.dtype != a.dtype or w2.dtype != a.dtype
            or not a.is_cuda or not _is_gfx938(a.device.index)
            or cache1.ndim != 2 or cache1.shape[0] < _M * _TOPK
            or cache1.shape[1] != _N or cache1.dtype != torch.bfloat16):
        return None
    capacity = sorted_ids.numel()
    if (not capacity or capacity % 256 or a_scale.numel() != _M or w_scale.numel() != local_experts * _N
            or a_scale.dtype != torch.float32 or w_scale.dtype != torch.float32
            or sorted_ids.ndim != 1 or sorted_weights.ndim != 1 or expert_ids.ndim != 1
            or sorted_weights.numel() < capacity or expert_ids.numel() < capacity // 256
            or sorted_weights.dtype != torch.float32 or nvalid.numel() != 1
            or any(t.dtype != torch.int32 for t in (sorted_ids, expert_ids, nvalid))
            or any(t.device != a.device or not t.is_contiguous() for t in tensors)):
        return None
    record = _lookup(w1)
    if record is None:
        return None
    q = ep._carve(workspace2, (capacity, _N // 2), a.dtype, a.device)
    if any(_overlaps(cache1, t) for t in (a, a_scale, w1, w2, record.packed, w_scale, sorted_ids, sorted_weights, expert_ids, nvalid)):
        raise RuntimeError("Step5 ASM G1 output overlaps an active input")
    if any(_overlaps(q, t) for t in (cache1, a, a_scale, w1, w2, record.packed, w_scale, sorted_ids, sorted_weights, expert_ids, nvalid)):
        raise RuntimeError("Step5 ASM G1 activation output overlaps live projection data")
    safe = torch.empty_like(sorted_ids)
    expanded = torch.empty(capacity // _BM, dtype=torch.int32, device=a.device)
    _prepare_metadata[(triton.cdiv(capacity, 256),)](sorted_ids, expert_ids, nvalid, safe, expanded,
        CAP=capacity, BLOCK=256, M=_M, TOPK=_TOPK, ASM_BM=_BM, num_warps=4)
    key = (a.device.index, capacity)
    with _LOCK:
        ones = _ONES.get(key)
        if ones is None:
            ones = torch.ones(capacity, dtype=torch.float32, device=a.device)
            _ONES[key] = ones
    import aiter

    # The installed normal AITER API launches on the current HIP stream.
    aiter.asm_fmoe_a8(cache1, a, record.packed, w2, safe, ones, expanded, nvalid,
                     _TOPK, a_scale, w_scale, None, 4, asm_recipe(local_experts)[0], 1,
                     _persist_for_experts(local_experts), 1)
    scale = torch.empty((capacity, 1), dtype=torch.float32, device=a.device)
    block = ep.ACT_CFG["BLOCK_M"]
    grid = (max(1, min(triton.cdiv(capacity, block),
                       ep._num_cus(a.device.index) * ep.ACT_CFG["CTAS_PER_CU"])),)
    _route_quant_kernel[grid](q, scale, cache1, sorted_weights, nvalid, sorted_ids,
        ROWS=capacity, LIMIT=float(limit), BLOCK_M=block, CH=512, M=_M, TOPK=_TOPK, N=_N,
        num_warps=ep.ACT_CFG["num_warps"], num_stages=1)
    with _LOCK:
        if not record.used:
            record.used = True
            print("[step5_prefill_asm_g1] active: " + json.dumps(dict(
                layer=record.layer_name, M=_M, E=local_experts, topk=_TOPK, sort_bm=256,
                asm_bm=_BM, solution=asm_recipe(local_experts)[0],
                persist=_persist_for_experts(local_experts), shuffle=1,
                source_ptr=record.source_ptr, packed_ptr=record.packed.data_ptr(),
                weight_bytes=record.packed.numel(), input_storage_tail_added=False,
            ), sort_keys=True), flush=True)
    return q, scale
