# HY4 W4A8 Graph Implementation Plan

> For agentic workers: execute with superpowers:executing-plans.

**Goal:** Admit narrowly qualified FULL_AND_PIECEWISE while retaining W4A8 math.
**Architecture:** Shared existing Graph-bank predicate; Core opaque MoE boundary.
**Tech Stack:** Core033, HCU Plugin, Adapter integer Triton kernels.
**Spec:** ../specs/2026-11-25-hy4-w4a8-fullpiecewise-design.md

## Global Constraints

No Core patch or quantization change; retain explicit experimental opt-in.

## Review Focus

Unsupported topology, wrong Graph mode, eager capture misuse, dynamic routes,
and graphpool lifetime must fail closed or retain original ownership.

## Task1 Admission and guards

- [x] Write CPU regressions and run RED for missing qualified admission.
- [x] Add validate_w4a8_execution in quantization.py using fullgraph_runner.
- [x] Wire selector and apply capture checks without changing arithmetic.
- [x] Run CPU regressions and relevant existing tests:91passed,0skips.
- [x] Validate exact-reference method/Core and whole-model execution: Eager and
  FULL_AND_PIECEWISE each 161/164 on HumanEval 164/C64; different failed task sets.
- [ ] Qualify the separate explicit AITER candidate: installed method replay,
  whole-model loading, HumanEval C64 and matched throughput/TPOT.
- [ ] Review changes; commit/push only after coordinator approval.

## Explicit AITER candidate

Public AITER prepare/config decides layout; preserve CT loader/packed owners,
Core routing/shared/reduction and SwiGLU clipping. AITER INT8 rounding differs
from the CT oracle and is separately qualified. Padding IDs must be distinct
per token for the native sorter. Keep Draft until candidate gates pass.
