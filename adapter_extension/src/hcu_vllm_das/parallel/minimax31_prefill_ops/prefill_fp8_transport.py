"""Quantize sparse cache rows before the PCP gather, preserving cache bytes."""
import torch
import triton
import triton.language as tl


@triton.jit
def _pack_rows(qkv, packed, ks, vs, qkv_stride: tl.constexpr,
               q_width: tl.constexpr, kv_width: tl.constexpr,
               index_q_width: tl.constexpr, width: tl.constexpr, block: tl.constexpr):
    columns = tl.program_id(1) * block + tl.arange(0, block)
    row = tl.program_id(0)
    src_column = q_width + columns + tl.where(columns >= 2 * kv_width, index_q_width, 0)
    valid = columns < width
    real = tl.load(qkv + row * qkv_stride + src_column, valid, other=0).to(tl.float32)
    # The index cache has unit scale and the original writer explicitly clamps.
    scale = tl.where(columns < kv_width, tl.load(ks), tl.load(vs))
    clipped = tl.where(real == real, tl.clamp(real, -448.0, 448.0), real)
    result = tl.where(columns < 2 * kv_width, real / scale, clipped)
    encoded = result.to(tl.float8e4nv, fp_downcast_rounding="rtne")
    tl.store(packed + row * width + columns, encoded.to(tl.uint8, bitcast=True), valid)


def pack(qkv, q_width, kv_width, index_q_width, k_scale, v_scale):
    rows, total = qkv.shape
    width = total - q_width - index_q_width
    if qkv.dtype != torch.bfloat16 or qkv.stride(1) != 1 or width < 2 * kv_width:
        raise ValueError("Unsupported QKV layout for sparse FP8 cache transport")
    result = torch.empty((rows, width), device=qkv.device, dtype=torch.uint8)
    block = 256
    _pack_rows[(rows, triton.cdiv(width, block))](
        qkv, result, k_scale, v_scale, qkv.stride(0),
        q_width, kv_width, index_q_width, width, block, num_warps=4,
    )
    return result


def gather_fp8(layer, qkv, slots, gather_cache_rows):
    from vllm.platforms import current_platform

    if (layer.kv_cache_dtype != "fp8"
            or layer.kv_cache.dtype not in (torch.uint8, torch.float8_e4m3fn)
            or current_platform.fp8_dtype() != torch.float8_e4m3fn):
        raise ValueError("Sparse PCP FP8 transport requires E4M3FN main KV")
    index_q_width = 0 if layer.skip_index_topk else layer.index_q_size
    if layer.skip_index_topk:
        qkv = qkv[:, :layer.q_size + 2 * layer.kv_size]
    local = pack(qkv, layer.q_size, layer.kv_size, index_q_width,
                 layer._k_scale, layer._v_scale)
    return gather_cache_rows(local, slots).view(torch.float8_e4m3fn)


def validate_pack():
    from vllm.models.minimax_m3.amd.ops.sparse_pa import minimax_m3_insert_index_cache
    from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
        triton_reshape_and_cache_flash,
    )

    device = torch.device("cuda")
    dtype = torch.float8_e4m3fn
    rows, heads, dim, index_width = 257, 4, 128, 128
    q_width, index_q_width = 1024, 512
    kv_width = heads * dim
    qkv = torch.randn(
        (rows, q_width + 2 * kv_width + index_q_width + index_width),
        dtype=torch.float32, device=device,
        generator=torch.Generator(device=device).manual_seed(31),
    ).mul_(80).to(torch.bfloat16)
    index_start = q_width + 2 * kv_width + index_q_width
    qkv[0, index_start:index_start + 3] = qkv.new_tensor([-512, 512, -448])
    slots = torch.arange(rows, device=device, dtype=torch.int64)
    slots[23::79] = -1

    def write(data_k, data_v, data_index, sk, sv):
        keys = torch.zeros((3, 128, heads, dim), device=device, dtype=torch.uint8).view(dtype)
        values = torch.zeros_like(keys.view(torch.uint8)).view(dtype)
        idx = torch.zeros((3, 128, index_width), device=device, dtype=torch.uint8).view(dtype)
        triton_reshape_and_cache_flash(
            data_k.view(rows, heads, dim), data_v.view(rows, heads, dim),
            keys, values, slots, "fp8", sk, sv,
        )
        minimax_m3_insert_index_cache(data_index, idx, slots)
        return keys, values, idx

    for k_value, v_value in ((1.0, 1.0), (1.375, 1.65)):
        sk = torch.tensor(k_value, device=device, dtype=torch.float32)
        sv = torch.tensor(v_value, device=device, dtype=torch.float32)
        ref = write(qkv[:, q_width:q_width + kv_width],
                    qkv[:, q_width + kv_width:q_width + 2 * kv_width],
                    qkv[:, index_start:].clamp(-448, 448), sk, sv)
        encoded = pack(qkv, q_width, kv_width, index_q_width, sk, sv).view(dtype)
        candidate = write(encoded[:, :kv_width], encoded[:, kv_width:2 * kv_width],
                          encoded[:, 2 * kv_width:], sk, sv)
        for name, old, new in zip(("key", "value", "index"), ref, candidate):
            if not torch.equal(old.view(torch.uint8), new.view(torch.uint8)):
                mismatch = (old.view(torch.uint8) != new.view(torch.uint8)).sum().item()
                raise ValueError(f"{name} cache has {mismatch} changed bytes at {k_value}/{v_value}")
    return {"cache_byte_equality": True, "rows": rows, "scale_pairs": 2, "pad_slots": 3}
