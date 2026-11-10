# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""CPU proof of the BoltOps EP activation contract used by the Step5 EP fast path.

Runs the real Triton kernel under Triton's NumPy interpreter and checks the
properties the Step5 EP path relies on: the device-side valid-row bound, the
router-weight fold into the per-row scale, the SWIGLUSTEP clamps, the FP8 scale
floor and the grid-stride coverage. No accelerator is required.

The module is loaded by file path on purpose: importing
``boltops.fused_moe.triton`` runs device queries in the package ``__init__``,
which a CPU-only host cannot answer.

Run directly (``python tests/runtime_patch/ep_activation_cpu_probe.py``) or via
``test_step5_ep_activation_ownership.py``.
"""
from __future__ import annotations

import importlib.util
import math
import os
import pathlib
import sys

os.environ["TRITON_INTERPRET"] = "1"

import torch

FP8 = torch.float8_e4m3fn
DTYPE_MAX = float(torch.finfo(FP8).max)
D = 1536
SENTINEL = 0x7F  # NaN payload byte, never produced by the kernel


def load_activation_module():
    import boltops

    path = pathlib.Path(boltops.__file__).parent / "fused_moe/triton/moe_activation.py"
    spec = importlib.util.spec_from_file_location("_step5_ep_activation_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # One CU x 64 CTAs would hide the grid-stride loop; 1 CTA forces it.
    module._ep_num_cus = lambda device: 1
    return module


def run(module, gate_up, weights, nvalid, rows, limit=7.0):
    q = torch.empty((rows, D), dtype=FP8)
    q.view(torch.uint8).fill_(SENTINEL)
    return module.ep_swiglustep_quant(gate_up, weights, nvalid, rows, limit=limit, q=q)


FAILURES = []


def check(name, condition, detail=""):
    print(f"{'PASS' if condition else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not condition:
        FAILURES.append(name)


def main():
    module = load_activation_module()
    torch.manual_seed(0)

    # A. nvalid == 0 writes nothing.
    x = torch.randn(4, 2 * D, dtype=torch.bfloat16)
    w = torch.rand(4)
    q, _ = run(module, x, w, torch.tensor([0], dtype=torch.int32), 4)
    check("nvalid=0 leaves q untouched", bool((q.view(torch.uint8) == SENTINEL).all()))

    # B. rows >= nvalid are never written.
    q, _ = run(module, x, w, torch.tensor([1], dtype=torch.int32), 4)
    check("rows >= nvalid untouched",
          bool((q[1:].view(torch.uint8) == SENTINEL).all())
          and bool((q[0].view(torch.uint8) != SENTINEL).any()))

    # C. The router weight lives in the scale, not in q.
    w1 = torch.full((4,), 1.0)
    w2 = torch.full((4,), 2.0)
    nv = torch.tensor([4], dtype=torch.int32)
    q1, s1 = run(module, x, w1, nv, 4)
    q2, s2 = run(module, x, w2, nv, 4)
    check("q independent of the router weight",
          torch.equal(q1.view(torch.uint8), q2.view(torch.uint8)))
    check("scale scales exactly with the router weight",
          torch.equal((s2 / s1), torch.full_like(s2, 2.0)))

    # D/E. Zero router weight and an all-zero row.
    _, s0 = run(module, x, torch.zeros(4), nv, 4)
    check("zero router weight gives a zero scale", bool((s0 == 0).all()))
    z = torch.zeros(4, 2 * D, dtype=torch.bfloat16)
    qz, sz = run(module, z, w1, nv, 4)
    check("all-zero row hits the scale floor",
          torch.equal(sz, torch.full((4, 1), 1.0e-10) / DTYPE_MAX),
          f"scale={sz[0].item():.3e}")
    check("all-zero row quantizes to zero", bool((qz.float() == 0).all()))

    # F. SWIGLUSTEP clamps: silu(gate) <= LIMIT and |up| <= LIMIT.
    big = torch.full((3, 2 * D), 1000.0, dtype=torch.bfloat16)
    big[:, D:] = -1000.0
    big[1, D:] = 1000.0
    qh, sh = run(module, big, w1[:3], torch.tensor([3], dtype=torch.int32), 3)
    expected = 49.0 / DTYPE_MAX
    check("clamped row scale is 49/448",
          torch.equal(sh, torch.full((3, 1), expected)))
    check("clamped positive row saturates fp8",
          bool((qh[1].float() == DTYPE_MAX).all()))
    check("clamped negative up row saturates to -fp8 max",
          bool((qh[0].float() == -DTYPE_MAX).all()))

    # H. SiLU curvature and the up multiplication: a plain product (identity
    # gate), a dropped up factor or a missing clamp all change the row scale.
    gates = torch.tensor([1.0, -1.0, 2.0], dtype=torch.bfloat16)
    ups = torch.tensor([1.0, 1.0, 3.0], dtype=torch.bfloat16)
    curved = torch.empty(3, 2 * D, dtype=torch.bfloat16)
    curved[:, :D] = gates[:, None]
    curved[:, D:] = ups[:, None]
    _, sc = run(module, curved, torch.ones(3), torch.tensor([3], dtype=torch.int32), 3)
    silu = lambda x: x / (1.0 + math.exp(-x))  # noqa: E731 - one-line reference
    expected = torch.tensor(
        [abs(silu(1.0)) * 1.0, abs(silu(-1.0)) * 1.0, abs(silu(2.0)) * 3.0],
        dtype=torch.float64) / DTYPE_MAX
    error = ((sc.double().view(-1) - expected).abs() / expected).max().item()
    # 1% tolerance: Triton's interpreter truncates bf16 casts where torch rounds,
    # which is up to one bf16 ulp (~0.8%) on a row max. Any structural mistake
    # (identity gate, dropped up factor, missing clamp) is off by >= 27%.
    check("row scale follows silu(gate) * up", error < 1.0e-2, f"max rel err {error:.2e}")

    # G. Grid stride: 300 rows with one CTA of BLOCK_M=2 needs several passes.
    rows = 300
    xg = torch.randn(rows, 2 * D, dtype=torch.bfloat16)
    qg, sg = run(module, xg, torch.rand(rows), torch.tensor([rows], dtype=torch.int32), rows)
    check("grid stride covers every valid row",
          bool((qg.view(torch.uint8) != SENTINEL).all()) and bool(torch.isfinite(sg).all()))

    print("ALL PASS" if not FAILURES else f"FAILED: {FAILURES}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
