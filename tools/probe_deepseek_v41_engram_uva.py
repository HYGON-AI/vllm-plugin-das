#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Probe DeepSeek V4.1 Engram Triton lookup through HCU UVA.

This intentionally uses a small synthetic table.  It compares the existing
vLLM Engram Triton kernel with GPU-resident storage and pinned-host storage
exposed through the HCU UVA bridge.
"""

from __future__ import annotations

import argparse
import json

import torch


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=32)
    return parser.parse_args()


def _lookup(
    kernel,
    weight: torch.Tensor,
    scales: torch.Tensor,
    ids: torch.Tensor,
    *,
    dim: int,
    heads: int,
    block_size: int,
) -> torch.Tensor:
    out = torch.empty(
        ids.shape[0], heads, dim, device=ids.device, dtype=torch.bfloat16
    )
    rows = ids.shape[0] * heads
    kernel[(1,)](
        weight,
        scales,
        ids,
        out,
        0,
        weight.shape[0],
        rows,
        ids.stride(0),
        ids.stride(1),
        HEAD_START=0,
        LOCAL_HEADS=heads,
        TOTAL_HEADS=heads,
        DIM=dim,
        QUANT_BLOCK=block_size,
        BLOCK_R=16,
        GRID=1,
    )
    return out


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("an HCU accelerator is required")
    if args.dim % args.block_size:
        raise ValueError("dim must be divisible by block-size")

    # Loading the stable extension registers the UVA bridge.  Importing the
    # HCU platform module mirrors the normal vLLM worker initialization path.
    import vllm._C_stable_libtorch  # noqa: F401
    import vllm_hcu.platforms.hcu  # noqa: F401
    from vllm.models.deepseek_v41.common.engram import _engram_lookup_kernel

    if not hasattr(torch.ops._C, "get_cuda_view_from_cpu_tensor"):
        raise RuntimeError("HCU UVA operator is unavailable after plugin initialization")

    torch.cuda.set_device(0)
    torch.manual_seed(1234)
    weight_host = torch.empty(
        args.rows,
        args.dim,
        dtype=torch.float8_e4m3fn,
        device="cpu",
    )
    # Integer-valued FP8 inputs are exactly representable and avoid accidental
    # reference differences unrelated to UVA addressing.
    weight_host.copy_(
        torch.randint(-8, 9, weight_host.shape, dtype=torch.int8).to(
            torch.float8_e4m3fn
        )
    )
    scales_host = torch.full(
        (args.rows, args.dim // args.block_size),
        127,
        dtype=torch.uint8,
        device="cpu",
    )
    # Large hipHostMalloc blocks are unreliable on HCU, so register anonymous
    # host memory instead (same mechanism the offload patch uses).
    from vllm_hcu.patch.worker.core_fix.patch_deepseek_v41_engram_offload import (
        _register_host_tensor,
    )

    class _Anchor:
        """Weakref-able owner keeping the registrations alive."""

    owner = _Anchor()
    _register_host_tensor(weight_host, owner)
    _register_host_tensor(scales_host, owner)
    ids = torch.randint(
        0,
        args.rows,
        (args.tokens, args.heads),
        dtype=torch.int64,
        device="cuda",
    )

    weight_device = weight_host.to("cuda")
    scales_device = scales_host.to("cuda")
    view = torch.ops._C.get_cuda_view_from_cpu_tensor
    weight_uva = view(weight_host)
    scales_uva = view(scales_host)

    want = _lookup(
        _engram_lookup_kernel,
        weight_device,
        scales_device,
        ids,
        dim=args.dim,
        heads=args.heads,
        block_size=args.block_size,
    )
    got = _lookup(
        _engram_lookup_kernel,
        weight_uva,
        scales_uva,
        ids,
        dim=args.dim,
        heads=args.heads,
        block_size=args.block_size,
    )
    torch.cuda.synchronize()
    if not torch.equal(got, want):
        delta = (got.float() - want.float()).abs().max().item()
        raise AssertionError(f"UVA lookup differs from HBM lookup: max_abs={delta}")

    weight_ptr = weight_uva.data_ptr()
    scale_ptr = scales_uva.data_ptr()
    for _ in range(args.iterations):
        got = _lookup(
            _engram_lookup_kernel,
            weight_uva,
            scales_uva,
            ids,
            dim=args.dim,
            heads=args.heads,
            block_size=args.block_size,
        )
    torch.cuda.synchronize()
    if not torch.equal(got, want):
        raise AssertionError("repeated UVA lookup became numerically unstable")
    if weight_uva.data_ptr() != weight_ptr or scales_uva.data_ptr() != scale_ptr:
        raise AssertionError("UVA device view address changed during repeated lookup")

    print(
        json.dumps(
            {
                "status": "PASS",
                "iterations": args.iterations,
                "device": torch.cuda.get_device_name(0),
                "weight_pinned": weight_host.is_pinned(),
                "scale_pinned": scales_host.is_pinned(),
                "weight_uva_device": str(weight_uva.device),
                "scale_uva_device": str(scales_uva.device),
                "shape": list(got.shape),
                "max_abs": 0.0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
