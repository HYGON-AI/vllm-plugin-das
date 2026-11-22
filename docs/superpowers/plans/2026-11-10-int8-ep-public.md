# Public INT8 EP Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans; implement inline,
> with one independent source/device-gate reviewer as required.

**Goal:** Correct canonical ASM expert routing, then qualify public EP2 through
Core's existing loader/PF/runner and real model HumanEval32.

**Architecture:** Plugin maps global routes to actual local expert slots for the
existing native sorter. Core retains public INT8 registry/CT/lifecycle and
communication; Adapter remains platform/health owner.

**Tech Stack:** Python3.12, Torch/DTK, existing AITER public native stages, Core0.33.

**Spec:** ../specs/2026-11-10-int8-ep-public-design.md

## Global Constraints

- No frozen environment/source/evidence edits, vendor CSV/private-cache edits,
  copied oracle/runner, architecture spoof or relaxed numerical thresholds.
- Whole-call official fallback keeps canonical map/raw weights; native errors
  propagate. Explicit backend only; auto unchanged.
- Strict BF16 relativeL2<=.005/maxabs<=.02/cos>=.9999 and actual Core unique top-k.
- Public EP refusal stays until corrected stage/device tests and admission review.

## Review Focus

- Valid local-index permutations retain exact routes instead of only ownership.
- Remote/-1/out-of-range padding cannot gather the last or first expert by wrap.
- Nomap/CK/vendor fallback semantics and native launch failure behavior stay intact.
- Graph uses changing device routes/maps with stable buffers and no host readback.
- Fresh installed public owner/actual CT loader/PF/runner/collective, not direct
  operator construction alone, must prove EP2 integration and model behavior.

## Task1: Preserve canonical map through ASM and remap sorter routes

Files: adapter_extension/src/hcu_vllm_das/moe/{fused_moe_aiter,int8_asm}.py;
tests/vllm033/{test_aiter_int8_provider,test_aiter_int8_asm}.py.

- [x] Write real tensor CPU tests for permutation/remote/padding/no-map routing;
  include original-int64 overflow/global-bound/extra-map entries and localE<topk
  prelaunch-fallback guards; update boundary test to require unchanged canonical
  map for ASM and retain CK-only binary ABI; run RED.
- [x] Implement safe global->local route preparation and local sorting domain;
  keep CK mask conversion at CK-only boundary; run focused existing suite GREEN.
- [x] Independently review source; install separate candidate artifact and run
  one-device native strict/reference/changed-Graph including permuted maps.
- [x] Record/commit actual scoped evidence; retain earlier R1/R2 failures.

Task1 evidence: upgrade-root evidence/v033-int8-ep-public-candidate-20261110/
cpu-routing-final.log (74PASS) and evidence/v033-int8-canonical-map-r5-20261110/
independent-terminal-canonical-map-review.json (148states/111Graph, worker0,
resources released). Scoped stage acceptance only; public EP is still closed.

## Task2: Qualified EP plans and public admission

- [ ] Write actual Core config/selector tests failing current EP refusal; test
  negative DP/PCP/SP/EPLB/shape/ABI and truthful local-miss plan fallback.
- [ ] Implement qualified supported topology/shape/plan compatibility with no
  vendor config rewrite. Expand native M/plan coverage by device tests.
- [ ] Test actual public factory, real checkpoint-style loader/reload and full
  official runner/collective/defaultGraph on2ranks in a separate installed env.

## Task3: Model accuracy, performance and publication

- [ ] Run real fitting MoE EP2 original HumanEval0..31 against control, keep raw
  and normalized score/template/token/dispatch/shutdown evidence separately.
- [ ] Run warranted warmed performance; update skill and full migration matrix.
- [ ] Fresh remote feature audit and independent final review; publish only
  missing tested delta via authorized fork and normally merge when real rules pass.
