"""Fuse sparse PCP cache writes and use FP8 transport only on larger cohorts."""
from collections import Counter
import inspect
import textwrap

import torch
import triton
import triton.language as tl

from hcu_vllm_das.parallel import minimax31_pcp as pcp
from .prefill_fp8_transport import gather_fp8
import pcp_write_rows

_original = None
_writer = None
_write_hits = Counter()
_write_checks = Counter()


@triton.jit(do_not_specialize=["N"])
def _write(rows, slots, side_slots, cache, side, ks, vs,
           N, RSTRIDE: tl.constexpr, IDX: tl.constexpr,
           INDEX_STRIDE: tl.constexpr, INDEX_BLOCK: tl.constexpr,
           INPUT_IS_FP8: tl.constexpr, TILE_ROWS: tl.constexpr):
    r = tl.program_id(0) * TILE_ROWS + tl.arange(0, TILE_ROWS)
    group = tl.program_id(1)
    d = tl.arange(0, 128)
    src = rows + r[:, None] * RSTRIDE + group * 128 + d[None, :]
    if IDX and group == 8:
        p = tl.load(side_slots + r, r < N, -1)
        valid = (r < N) & (p >= 0)
        value = tl.load(src, valid[:, None], 0)
        if not INPUT_IS_FP8:
            value = tl.clamp(value.to(tl.float32), -448., 448.)
            value = value.to(tl.float8e4nv, fp_downcast_rounding="rtne").to(tl.uint8, bitcast=True)
        dest = p // INDEX_BLOCK * INDEX_STRIDE + (p % INDEX_BLOCK) * 128
        tl.store(side + dest[:, None] + d[None, :], value, valid[:, None])
    else:
        p = tl.load(slots + r, r < N, -1)
        valid = (r < N) & (p >= 0)
        value = tl.load(src, valid[:, None], 0)
        if not INPUT_IS_FP8:
            scale = tl.load(ks) if group < 4 else tl.load(vs)
            value = (value.to(tl.float32) / scale).to(
                tl.float8e4nv, fp_downcast_rounding="rtne",
            ).to(tl.uint8, bitcast=True)
        dest = p * 1024 + (group % 4) * 256 + (group // 4) * 128
        tl.store(cache + dest[:, None] + d[None, :], value, valid[:, None])


def _write_bound(rows, slots, side_slots, cache, side, ks, vs):
    if (cache.ndim != 4 or tuple(cache.shape[1:]) != (128, 4, 128)
            or cache.stride() != (131072, 1024, 256, 1)
            or rows.ndim != 2 or rows.stride(1) != 1 or rows.size(1) not in (1024, 1152)
            or rows.dtype not in (torch.bfloat16, torch.float8_e4m3fn)
            or slots.numel() != rows.size(0) or not slots.is_contiguous()):
        raise ValueError("Unrecognized native Prefill cache layout")
    has_index = side is not None and rows.size(1) == 1152
    if has_index and (side.ndim != 3 or tuple(side.shape[1:]) != (128, 128)
                      or side.dtype != torch.float8_e4m3fn or side.stride()[1:] != (128, 1)
                      or not side_slots.is_contiguous() or side_slots.numel() != rows.size(0)):
        raise ValueError("Unrecognized native Prefill side cache layout")
    data = rows.view(torch.uint8) if rows.dtype == torch.float8_e4m3fn else rows
    raw = cache.view(torch.uint8)
    _write[(triton.cdiv(rows.size(0), 4), 9 if has_index else 8)](
        data, slots, side_slots if has_index else slots,
        raw, side.view(torch.uint8) if has_index else raw, ks, vs,
        rows.size(0), rows.stride(0), has_index, side.stride(0) if has_index else 0, 128,
        rows.dtype == torch.float8_e4m3fn, 4, num_warps=4,
    )


def cache_write(layer, rows, slots, binding):
    side = layer.indexer.index_cache
    live = side.kv_cache if not layer.skip_index_topk and side.kv_cache.numel() else None
    main_slots = slots[layer.layer_name]
    index_slots = slots[side.prefix] if live is not None else None
    main = binding.key_writer
    count = rows.shape[0]
    if (_writer is not None and count in (16384, 49152)
            and pcp_write_rows.supports(rows, main_slots, index_slots, main, live)):
        pcp_write_rows.write(rows, main_slots, index_slots, main, live, 16, 8, 8)
        if _writer and not _write_checks[count]:
            _check_write(rows, main_slots, index_slots, main, live)
            _write_checks[count] += 1
        _write_hits[count] += 1
    else:
        _write_bound(rows, main_slots, index_slots, main, live, layer._k_scale, layer._v_scale)


def _check_write(rows, main_slots, index_slots, main, index):
    encoded = rows.view(torch.uint8)
    kv_valid = main_slots >= 0
    kv_slots = main_slots[kv_valid]
    assert kv_slots.numel() > 0
    dense_kv = main.view(torch.uint8).as_strided((*main.shape[:-1], 256), main.stride())
    actual = dense_kv[kv_slots // 128, kv_slots % 128]
    expected = encoded[kv_valid, :1024].reshape(-1, 2, 4, 128).transpose(1, 2)
    if not torch.equal(actual, expected.reshape(-1, 4, 256)):
        raise RuntimeError("Grouped PCP writer changed packed K/V bytes")
    index_valid = index_slots >= 0
    index_slots = index_slots[index_valid]
    assert index_slots.numel() > 0
    actual_index = index.view(torch.uint8)[index_slots // 128, index_slots % 128]
    if not torch.equal(actual_index, encoded[index_valid, 1024:]):
        raise RuntimeError("Grouped PCP writer changed packed index bytes")


def configure_writer(value):
    global _writer
    if value == "status":
        return {"enabled": _writer is not None, "verify": _writer is True,
                "hits": dict(_write_hits), "checks": dict(_write_checks)}
    if value is not None and type(value) is not bool:
        raise ValueError("Use null for baseline, true for byte checks, or false for timing")
    _writer = value
    _write_hits.clear()
    _write_checks.clear()
    return configure_writer("status")


@triton.jit(do_not_specialize=["N"])
def _quant_index_query(source, output, N, BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(source + offset, offset < N, 0).to(tl.float32)
    limited = tl.where(value == value, tl.clamp(value, -448., 448.), value)
    encoded = limited.to(tl.float8e4nv, fp_downcast_rounding="rtne")
    tl.store(output + offset, encoded.to(tl.uint8, bitcast=True), offset < N)


def quant_index_query(source):
    if source.dtype != torch.bfloat16 or not source.is_contiguous():
        raise ValueError("Sparse PCP index query must be contiguous BF16")
    result = torch.empty_like(source, dtype=torch.uint8)
    if source.numel():
        _quant_index_query[(triton.cdiv(source.numel(), 512),)](
            source, result, source.numel(), 512, num_warps=4,
        )
    return result.view(torch.float8_e4m3fn)


def _qualify():
    from vllm.models.minimax_m3.amd.ops.sparse_pa import minimax_m3_insert_index_cache
    from vllm.v1.attention.ops.triton_reshape_and_cache_flash import triton_reshape_and_cache_flash

    device, fp8 = torch.device("cuda"), torch.float8_e4m3fn
    count = 253
    rows = torch.randn(
        (count, 1152), generator=torch.Generator(device=device).manual_seed(30),
        dtype=torch.bfloat16, device=device,
    ).mul_(128)
    slots = torch.arange(count, dtype=torch.int64, device=device)
    slots[::13] = -1
    index_slots = torch.arange(count, dtype=torch.int64, device=device).flip(0).contiguous()
    index_slots[::17] = -1
    sk, sv = (torch.tensor(value, device=device) for value in (1.375, 1.65))
    reference_index_q = rows.clamp(-448, 448).to(fp8)
    new_index_q = quant_index_query(rows)
    if not torch.equal(reference_index_q.view(torch.uint8), new_index_q.view(torch.uint8)):
        raise RuntimeError("Fused E4M3 index query changed quantized bytes")
    for encoded in (False, True):
        typ = fp8 if encoded else torch.bfloat16
        test = rows.to(typ)
        ref = torch.zeros((2, 128, 4, 256), dtype=torch.uint8, device=device)
        actual = torch.zeros_like(ref)
        iref = torch.zeros((2, 128, 128), dtype=torch.uint8, device=device)
        iactual = torch.zeros_like(iref)
        size = (count, 4, 128)
        triton_reshape_and_cache_flash(
            test[:, :512].view(size), test[:, 512:1024].view(size),
            ref[..., :128], ref[..., 128:], slots, "fp8", sk, sv,
        )
        index = test[:, 1024:]
        if not encoded:
            index = index.clamp(-448, 448)
        minimax_m3_insert_index_cache(index, iref.view(fp8), index_slots)
        _write_bound(test, slots, index_slots, actual[..., :128], iactual.view(fp8), sk, sv)
        if not torch.equal(ref, actual) or not torch.equal(iref, iactual):
            raise RuntimeError(f"Fused PCP writer changed cache bytes for {typ}")
    return 2


def configure(value):
    from .prefill_runtime_candidate import _original_sparse, _replace_sparse
    global _original

    if value is None:
        if _original is not None:
            pcp.MiniMax31PCPSparseAttention.forward = _original
            _original = None
        return {"enabled": False}
    if not isinstance(value, int) or value < 0:
        raise ValueError("FP8 transport threshold must be a nonnegative integer")
    cases = _qualify()
    if _original is None:
        _original = pcp.MiniMax31PCPSparseAttention.forward

    def hybrid(layer, qkv, slots, gather):
        if qkv.shape[0] >= value:
            return gather_fp8(layer, qkv, slots, gather)
        kv_end = layer.q_size + 2 * layer.kv_size
        rows = qkv[:, layer.q_size:kv_end]
        if not layer.skip_index_topk:
            rows = torch.cat((rows, qkv[:, kv_end + layer.index_q_size:]), dim=-1)
        return gather(rows, slots)

    # Use the pinned forward source. Replace only the writes for the native layout.
    code = inspect.getsource(_original_sparse)
    begin = code.index("            triton_reshape_and_cache_flash(")
    end = code.index("        if _EXPERIMENTAL_MSA:\n            from .minimax31_msa import run_native_attention", begin)
    block = code[begin:end]
    replacement = ("            if _EXPERIMENTAL_MSA and _EXPERIMENTAL_FP8_INDEX:\n"
                   "                cache_write(self, gathered, slots, binding)\n"
                   "            else:\n" +
                   "\n".join("    " + line if line else line for line in block.splitlines()) + "\n")
    code = code[:begin] + replacement + code[end:]
    old_quant = "index_query = index_query.clamp(-448.0, 448.0).to(torch.float8_e4m3fn)"
    if code.count(old_quant) != 1:
        raise ValueError("Unexpected sparse index query writer")
    code = code.replace(old_quant, "index_query = quant_index_query(index_query)")
    forward = _replace_sparse(hybrid, code=code, cache_write=cache_write,
                              quant_index_query=quant_index_query)
    pcp.MiniMax31PCPSparseAttention.forward = forward
    return {"enabled": True, "fp8_min_rows": value, "qualification_cases": cases}
