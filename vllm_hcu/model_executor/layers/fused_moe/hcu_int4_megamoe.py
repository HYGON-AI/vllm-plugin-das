# SPDX-License-Identifier: Apache-2.0
# Ported from HYGON-AI/sglang-das PR #333.
"""Framework-local SlimQuant W4A8 support for the HCU MegaMoE INT8 engine.

Resident weights keep two signed nibbles per byte, in the raw checkpoint
layout or prepacked PACK5. Before GEMM they are expanded into a stream-owned
reusable PACK5 workspace. The raw path fuses permutation with unpacking. This is NOT a
native packed-INT4 GEMM: it trades an unpack launch/bandwidth for retaining
4-bit model storage and reusing the tuned HCU dispatch/GEMM/combine kernels.
"""

import inspect
import math

import torch
from vllm.triton_utils import tl, triton


def validate_megamoe_int8_runtime() -> None:
    """Fail early when the installed base wheel is too old for this adapter."""
    import megamoe

    required = (
        "int8_mega_moe", "mega_moe_pre_dispatch_int8", "SymmBuffer",
        "get_symm_buffer_for_mega_moe",
    )
    missing = [name for name in required if not callable(getattr(megamoe, name, None))]
    if missing:
        raise RuntimeError(
            "The installed MegaMoE wheel lacks the W8A8-I8 runtime required "
            "by vLLM HCU's W4A8 compatibility adapter: " + ", ".join(missing)
        )
    for name in ("SymmBuffer", "get_symm_buffer_for_mega_moe"):
        try:
            parameters = inspect.signature(getattr(megamoe, name)).parameters
        except (TypeError, ValueError):
            continue  # Some extension APIs do not expose a Python signature.
        if parameters and "quant_mode" not in parameters and not any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()
        ):
            raise RuntimeError(f"MegaMoE {name} does not support quant_mode='int8'")


@triton.jit
def _unpack_signed_int4(P, O, SIZE: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    b = tl.load(P + i // 2, i < SIZE, other=0).to(tl.int32)
    q = (b >> tl.where(i % 2 == 0, 4, 0)) & 15
    tl.store(O + i, ((q ^ 8) - 8).to(tl.int8), i < SIZE)


def unpack_signed_int4_out(packed, out, *, block=2048, num_warps=4):
    """Decode high-nibble-first two's-complement bytes, without FP quantization."""
    if packed.dtype not in (torch.int8, torch.uint8) or out.dtype != torch.int8:
        raise TypeError("packed must be byte INT4 and out must be signed INT8")
    if not packed.is_contiguous() or not out.is_contiguous():
        raise ValueError("packed and output must be contiguous")
    if packed.device != out.device or not packed.is_cuda:
        raise ValueError("packed and output must be on the same GPU")
    if out.numel() != 2 * packed.numel():
        raise ValueError("output must contain exactly two elements per packed byte")
    _unpack_signed_int4[(triton.cdiv(out.numel(), block),)](
        packed, out, out.numel(), block, num_warps=num_warps
    )
    return out


@triton.jit
def _unpack_raw_int4_pack5(
    P, O, SIZE: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    # Invert [E, KB, NB, N16a, K4, N16b, K16] back to raw [E, N, K].
    k16 = i % 16
    n16b = (i // 16) % 16
    k4 = (i // 256) % 4
    n16a = (i // 1024) % 16
    nb = (i // 16384) % (N // 256)
    kb = (i // (16384 * (N // 256))) % (K // 64)
    expert = i // (N * K)
    raw_byte = (
        (expert * N + nb * 256 + n16a * 16 + n16b) * (K // 2)
        + kb * 32 + k4 * 8 + k16 // 2
    )
    byte = tl.load(P + raw_byte, i < SIZE, other=0).to(tl.int32)
    q = (byte >> tl.where(k16 % 2 == 0, 4, 0)) & 15
    tl.store(O + i, ((q ^ 8) - 8).to(tl.int8), i < SIZE)


def unpack_raw_int4_pack5_out(packed, out, *, block=2048):
    """Decode checkpoint INT4 directly into PACK5 INT8 reusable scratch."""
    if packed.ndim != 3 or packed.dtype not in (torch.int8, torch.uint8):
        raise ValueError("raw INT4 must be byte [E,N,K/2]")
    e, n, kh = packed.shape
    if e <= 0 or n <= 0 or kh <= 0 or n % 256 or kh % 32:
        raise ValueError("normal PACK5 requires N%256=0 and K%64=0")
    if out.shape != (e, n * kh * 2) or out.dtype != torch.int8:
        raise ValueError("PACK5 output must be signed INT8 [E,N*K]")
    if not packed.is_contiguous() or not out.is_contiguous():
        raise ValueError("raw INT4 and output must be contiguous")
    if packed.device != out.device or not packed.is_cuda:
        raise ValueError("raw INT4 and output must be on the same GPU")
    _unpack_raw_int4_pack5[(triton.cdiv(out.numel(), block),)](
        packed, out, out.numel(), n, kh * 2, block, num_warps=4,
    )
    return out


def _pack(weight, scale, scale_multiplier, prepack=True):
    if weight.ndim != 3 or weight.dtype not in (torch.int8, torch.uint8):
        raise ValueError("checkpoint weight must be byte [E,N,K/2]")
    e, n, kh = weight.shape
    if e <= 0 or n <= 0 or kh <= 0 or n % 256 or kh % 32:
        raise ValueError("normal PACK5 requires N%256=0 and K%64=0")
    if scale.shape not in ((e, n), (e, n, 1)) or scale.device != weight.device:
        raise ValueError("per-channel scale must be [E,N] or [E,N,1] on weight device")
    # Same permutation as flatten_pack5_weight_asm_normal, with K16 compressed
    # to eight bytes. Adjacent K elements stay adjacent, so unpack commutes with
    # this permutation. No full INT8/FP weight copy is made at model load time.
    scales = scale.reshape(e, n).float().mul(scale_multiplier).contiguous()
    if not prepack:
        # Keep vLLM's loader-owned raw parameters. Fuse permutation with the
        # runtime unpack instead of retaining a second INT4 model copy.
        return {"raw_int4": (weight, scales)}
    packed = weight.reshape(e, n // 256, 16, 16, kh // 32, 4, 8)
    packed = packed.permute(0, 4, 1, 2, 5, 3, 6).contiguous().reshape(e, n * kh)
    return {"normal_int4": (packed, scales)}


def transform_int4_weights_for_mega_moe_normal(
    l1_weights, l2_weights, *, l1_scale, l2_scale,
    scale_multiplier=16.0, prepack=True,
):
    """Prepare raw SlimQuant checkpoint weights BEFORE any AITER/Marlin shuffle.

    Checkpoint byte = (even_K & 15)<<4 | (odd_K & 15). Nibbles are signed
    two's-complement, not offset-8. This model stores true channel scale / 16.
    Callers with true scales must explicitly use scale_multiplier=1.
    L1 rows are [all gate rows, all up rows], matching the existing INT8 K2.
    """
    if not math.isfinite(scale_multiplier) or scale_multiplier <= 0:
        raise ValueError("scale_multiplier must be finite and positive")
    if (l1_weights.ndim != 3 or l2_weights.ndim != 3
            or l1_weights.shape[0] != l2_weights.shape[0]
            or l1_weights.shape[1] != 4 * l2_weights.shape[2]
            or l2_weights.shape[1] != 2 * l1_weights.shape[2]):
        raise ValueError("expected L1=[E,2I,H/2], L2=[E,H,I/2]")
    return (_pack(l1_weights, l1_scale, scale_multiplier, prepack),
            _pack(l2_weights, l2_scale, scale_multiplier, prepack))


def int4_w4a8_mega_moe(y, l1_weights, l2_weights, sym_buffer, **kwargs):
    """W4A8 compatibility path; uses an INT8 activation SymmBuffer, eager only.

    Expanded storage is shared across layers on the same stream/buffer. A
    DSV4-Flash EP8 rank retains 384 MiB INT4 per layer and reuses 768 MiB total
    INT8 scratch, instead of expanding all layers persistently.
    """
    import megamoe

    if kwargs.get("graph", False) or torch.cuda.is_current_stream_capturing():
        raise NotImplementedError("W4A8 scratch expansion supports eager normal only")
    if kwargs.get("megamoe_backend", "normal") != "normal":
        raise NotImplementedError("W4A8 supports normal backend only")
    if getattr(sym_buffer, "quant_mode", None) != "int8":
        raise ValueError("W4A8 requires SymmBuffer quant_mode='int8'")
    layout = "raw_int4" if "raw_int4" in l1_weights else "normal_int4"
    p1, s1 = l1_weights[layout]
    p2, s2 = l2_weights[layout]
    stream = torch.cuda.current_stream(y.device)
    key = (y.device, stream.cuda_stream, layout, tuple(p1.shape), tuple(p2.shape))
    cache = getattr(sym_buffer, "_int4_unpack_workspaces", None)
    if cache is None:
        cache = sym_buffer._int4_unpack_workspaces = {}
    if key not in cache:
        cache[key] = tuple(torch.empty((p.shape[0], p.numel() // p.shape[0] * 2),
                                      dtype=torch.int8, device=p.device)
                           for p in (p1, p2))
    w1, w2 = cache[key]
    unpack = unpack_raw_int4_pack5_out if layout == "raw_int4" else unpack_signed_int4_out
    unpack(p1, w1)
    unpack(p2, w2)
    return megamoe.int8_mega_moe(
        y,
        {"normal": (w1, s1)},
        {"normal": (w2, s2)},
        sym_buffer,
        **kwargs,
    )
