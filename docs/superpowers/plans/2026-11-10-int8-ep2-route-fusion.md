# GPU EP route fusion implementation plan

> For agentic workers: use superpowers:executing-plans inline, independent launch/terminal/publication reviews.

**Goal:** Integrate the measured exact route primitive with current public INT8 EP2.
**Architecture:** One GPU-only canonical-map boundary kernel and its source fingerprint; Core/vendor stages unchanged.
**Tech Stack:** Actual DTK26.10/Torch2.11/Triton3.6; UV isolated package/runtime; realgfx936.
**Spec:** ../specs/2026-11-10-int8-ep2-route-fusion.md

## Global Constraints

Preserve latest2a80 documentation and frozen old sources/environments.
Only GPU/non-None map dispatch changes; no float math, admission/default/unknown fallback change.
Bind new module content before publication; no new exact-wheel runtime lock.
Original HE32/precision tolerances and required remote CI/normal approvals retained.

## Review Focus

New module file drift must invalidate cache identity before reuse.
No-map/int64 TP and CPU reference remain original Torch behavior.
Gfx938 TP stays admitted but gfx938 EP2 still refuses; no new GPU scope.
Signed64 padding/overflow, remote IDs and local permutations are exact.
Triton JIT must warm before capture, and new module enters all content factors.

## Tasks

- [x] New branch from current373head2a80 plus13profile delta;20runtime bytes equal P4.
- [x] Baseline166PASS;3contract checks originally missing VLLM_TARGET_ROOT then corrected3PASS (harness error, not behavioral RED).
- [x] Actual preparation-identity module-drift regression RED (DID NOT RAISE) then GREEN1.
- [x] Copy independently qualified primitive, select GPU+map only and bind source; source170 pass, critical lint/diff pass.
- [ ] New21module UVwheel/env; sourceCI/package/installed exact identities and independent code review.
- [ ] Real2rank full public matrix/native+fusion dispatch and strict reference/Graph/reload/resource audit.
- [ ] New unchanged HE32/warmed pair and independent quality/performance/lifecycle audit.
- [ ] Fresh remote support/MR checks before normal existing373 update and required CI/merge when allowed.
