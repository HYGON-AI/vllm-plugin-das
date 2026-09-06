---
name: porting-vllm-hcu-operators
description: Use when comparing SGLang-DAS with vllm-plugin-das or adapting LightOp, AITER, Triton, MoE, attention, quantization, or routing operators to a pinned vLLM HCU release.
---

# Porting vLLM-HCU Operators

## Overview

Treat an operator as useful only after proving four things: the installed ABI
exists, vLLM has an exact ownership seam, numerical behavior matches an
independent reference, and the complete routed path wins on supported shapes.
A callable symbol alone proves none of these.

**REQUIRED REFERENCE:** Read
[references/validation-checklist.md](references/validation-checklist.md) before
changing code or accepting/rejecting a candidate.

## Required outcome

Maintain a disposition table for every SGLang LightOp/AITER import:
`accepted`, `already-covered`, `performance-rejected`,
`dependency-unavailable`, or `no-vllm-seam`. Record source SHAs, installed
package versions, exact import paths, shapes, and raw evidence. Re-audit the
live environment even when an older table exists.

For an accepted route:

- Keep implementation in plugin-owned modules. Patch only audited exact vLLM
  modules/symbols, validate signatures/types, use an idempotent marker, and
  patch bindings captured by real consumers—not only package exports.
- Require `VLLM_HCU_USE_CUSTOM_OPS` and one descriptive leaf switch. Default
  off for weight/layout mutation or uncertain coverage; default on only for a
  non-mutating route with a complete safe fallback.
- Import optional operators lazily from their public categorized modules. Do
  not import SGLang runtime code or private compatibility aliases.
- Reject unsupported dtype/device/shape/stride/topology before mutation. Once
  weights use a backend-specific layout, never send them to another backend.
- Fall back only when disabled, unavailable, or ineligible. After an eligible
  kernel is selected, ABI or execution failures are errors unless the path is
  demonstrably transactional and repository policy explicitly permits retry.

## Evidence gate

Write portable contract tests first and observe the expected failure. Cover
master off, leaf off, missing/non-callable dependency, eligible and ineligible
inputs, exact fallback, output contract, idempotence, and captured imports.

Then run live-HCU accuracy against an independent FP32/reference expression.
Declare tolerances before results; check shape, dtype, finiteness, mutation,
and production-derived edge shapes. Only then benchmark the complete route,
including conversion, packing, allocation, synchronization, and layout costs,
against the actual current backend. Use warmups, repeated medians, raw JSON,
and the requested gate (5% if none is specified). Narrow eligibility to stable
wins; otherwise remove the route and record `performance-rejected`.

Model validation follows operator validation. Use feature-off/on processes and
fresh log offsets to prove routing. For HumanEval-32 require exactly 32
predictions and 32 reviews with non-regressing Pass@1. Treat shared-host TPS as
observational unless the host is isolated. If model coverage is unavailable or
the user allows operator-only acceptance, report that limitation honestly;
never invent a model-level claim.

## Delivery

Register tests in HCU CI, run repository contract/coverage/boundary checks,
review the complete base-to-head diff, and keep the requested work in one PR.
Never place access tokens in commands, remotes, logs, reports, or commits.
