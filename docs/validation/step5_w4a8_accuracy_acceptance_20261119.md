# Step5 W4A8 accuracy acceptance evidence — GSM8K and AA-LCR

**Evidence date:** 2026-11-12 through 2026-11-17.  
**Target baseline:** `HYGON-AI/vllm-plugin-das:step5-w4a8-pd@eb8c516`.  
**Artifact type:** validation evidence only; no changes to model, kernel, benchmark or verifier.  
**Overall evidence classification:** **PARTIAL / MIXED GATES**. This document does not claim all acceptance criteria passed.

## 1. Executive conclusion and acceptance matrix

| Validation | Available result | Acceptance interpretation |
| --- | --- | --- |
| Full GSM8K 1319, FP8 reference vs W4A8 | A=1270/1319; B=1265/1319; 5 fewer correct in W4A8 | **INCONCLUSIVE** under the pre-registered strict non-drop requirement |
| Full GSM8K, W4A8+MTP3 vs W4A8 no-speculation | B=1265/1319; D=1266/1319 | **PASS_NON_INFERIOR** under the task runbook rule; not identical |
| Full GSM8K, W4A8 IPC AG off vs on | B=C=1265/1319; 9 per-question flips in each direction | **PASS** under the runbook's equal-total gate; **not** per-question bitwise identity |
| AA-LCR High81 Rescue C3 | 243/243 terminal generations; official judge job completed, 243 judged; 105 `judge_correct` in handoff manifest; 16/16 independent checks PASS | **COMPLETED RUN / SCORE PROVENANCE NOT RECONCILED** |
| Terminal-Bench 2.1 | Not a completed 89×3 official-comparable score in the evidence reviewed | **PENDING**, independent future report |

**Important AA-LCR score caveat:** A separate historical status statement referred to a **diagnostically corrected 88.25%** against an **88.3% reference**. The currently available frozen handoff shows `judge_correct=105` for `judge_judged=243` (the direct quotient is ~43.21%). The corrected 88.25% is **not derivable from that raw pair**, and the correction formula, scope, and relevant result file were not independently inspected when authoring this report. Accordingly, **neither 88.25% nor a claimed official parity is certified by this PR**. They require an explicit reconciliation artifact and rerun of the independent verifier before being used as a headline.

This report is an **evidence archive for review**, not a production approval or official benchmark leaderboard submission.

## 2. Frozen GSM8K protocol

| Field | Observed protocol |
| --- | --- |
| Dataset | Full, frozen GSM8K test; 1319 questions |
| Harness | EvalScope 1.12.0; `accuracy/run_gsm8k_topk42.py` |
| Score | Pass@1, rule judge (`mean_acc`) |
| Sampling | Zero-shot, temperature 1.0, top_p 0.95, top_k 42, max_tokens 32768, seed 42, n=1 |
| Concurrency | 4 |
| Tokenizer / reasoning parser | Archived official tokenizer asset; `step3p5` |
| Model service | TP4/DP2/EP8, max batched tokens 32768, max model length 70000, max sequences 64, graph cap 64, GPU memory utilization 0.85 |
| MTP in A/B/C | Real MTP3, standard rejection sampling, greedy draft sampling, local argmax reduction; no synthetic acceptance |
| D | W4A8 with no speculation |
| W4A8 route | Triton/BoltOps MoE; AllGather/ReduceScatter |

These evaluation parameters are **not identical** to all historical official baselines: the historical reference used `top_k=-1` whereas this campaign used `top_k=42`. The FP8 arm also ran with its default FP8 MoE backend, which differs from the W4A8 Triton/BoltOps path. The 70K model context limit and other runtime constraints must be retained when interpreting this evidence.

## 3. GSM8K results

| Arm | Mode | Correct / 1319 | Accuracy | Request errors | Output-limit |
| --- | --- | ---: | ---: | ---: | ---: |
| A_FP8 | Channel-FP8, normal MTP3 | 1270 | 96.2851% | 0 | 0 |
| B_W4A8 | W4A8, normal MTP3, IPC AG off | 1265 | 95.9060% | 0 | 0 |
| C_W4A8_IPCAG | W4A8, normal MTP3, IPC AG on | 1265 | 95.9060% | 0 | 0 |
| D_W4A8_NOSPEC | W4A8, no speculation | 1266 | 95.9818% | 0 | 0 |

### Pre-registered gates

| Gate | Comparison | Observed difference | Reported 95% noise band | Verdict |
| --- | --- | --- | --- | --- |
| Gate 1 | B_W4A8 vs A_FP8 | −5 correct (−0.379 pp) | ±1.478 pp | **INCONCLUSIVE**; strict non-drop not met |
| Gate 2 | B_W4A8 vs D_W4A8_NOSPEC | −1 correct (−0.076 pp) | ±1.506 pp | **PASS_NON_INFERIOR**, under the defined task rule |
| Gate 3 | C_W4A8_IPCAG vs B_W4A8 | 0 net correct (0.000 pp) | ±1.512 pp | **PASS**, totals equal |

**Per-question paired counts:** A:B: both correct 1256, both wrong 40, only A 14, only B 9, McNemar p=0.4049. D:B: both correct 1249, both wrong 37, only D 17, only B 16, p=1.0000. B:C: both correct 1256, both wrong 45, only B 9, only C 9, p=1.0000.

**MTP acceptance (research metric, not task accuracy):** A=0.5672, B=0.5645, C=0.5665. The historical synthetic-derived 0.6 threshold is not an official real-MTP acceptance standard.

### Precise interpretation for IPC Split PR #472

The equal **1265/1319** total is a useful model-level regression check on the **W4A8 IPC AG off/on experiment**, but it does **not** prove all individual outputs are identical: the runbook measured nine answer flips in each direction. It is also not an eight-rank bitwise kernel check, and it must not be used to silently close PR #472's still-open 8-rank parity, Graph replay, Prefill non-regression or 1.8% performance promotion gates.

## 4. AA-LCR High81 Rescue C3 archived handoff

**Run ID:** `AA_LCR_W4A8_HIGH81_RESCUE_C3`.

The frozen handoff reported:

| Field | Result |
| --- | --- |
| Generation terminal count | 243 |
| Generation expected | 243 |
| Official judge status | COMPLETED |
| Judge records / judged | 243 / 243 |
| Raw `judge_correct` in handoff | 105 |
| Built-in verifier | PASS |
| Independent verifier | PASS (16 checks, 0 failures) |
| Manifest checked and frozen | YES |
| Original 400-question manifest unchanged | YES |

**Scope limitation:** The manifest proves completion and integrity of this rescued 243-generation subset. It does **not by itself** prove 400/400 coverage or an officially comparable corrected accuracy. A separately reported **88.25% diagnostic correction** must be tied to its correction algorithm, denominators, per-case ledger and independent verifier evidence before publishing as an accepted accuracy figure.

The model's service configuration in the handoff used W4A8, TP4/DP2/EP8, MTP3, Triton MoE, AllGather/ReduceScatter, graph FULL_AND_PIECEWISE and BF16/FP8 mixed runtime components. The exact service logs, checkpoint identity and manifest are authoritative; do not replace them with assumptions about an unrelated performance run.

## 5. Evidence provenance and reproduction pointers

The original results were produced in external evaluation environments and are **not copied into this PR**. Internal evidence owners should cross-check the original artifacts before signoff:

- GSM8K full A/B/C/D summary: `W4A8_ACCURACY_REPORT.md` (generated 2026-11-13), archived `accuracy_gate/RUNBOOK_GPU_ACCURACY.md`, `PROTOCOL_AUDIT.md`, `PER_QUESTION_COMPARISON.json`, `ACCURACY_VERIFICATION_RESULTS.json`, and each arm's `decode_accuracy/summary.json`.
- AA-LCR High81: run directory `AA_LCR_W4A8_HIGH81_RESCUE_C3`, frozen `HIGH81_TO_TB21_READY.json` and `HIGH81_RESULT_MANIFEST.sha256`; verify actual hashes against the frozen artifact set.
- Existing source/performance change in separate PR: [#472](https://github.com/HYGON-AI/vllm-plugin-das/pull/472).

The evidence set was **read from archived status and accuracy reports**, not from a fresh execution of GSM8K/AA-LCR, and no new hardware run is claimed.

## 6. Pending follow-up (not part of this PR)

1. Reconcile AA-LCR raw 105/243 judge count with the reported diagnostic 88.25% via the exact correction ledger and independent verifier; state whether reference protocol and denominator truly match.
2. If publication requires strict W4A8-vs-FP8 non-drop, resolve Gate 1 **INCONCLUSIVE** with a pre-registered follow-up.
3. Validate the code changes in PR #472 under the remaining device/Graph/Prefill gates independently of the GSM8K total-score comparison.
4. Run Terminal-Bench 2.1 only as a later, separately versioned evaluation; its current absence must not be misreported as a passed result.

**Proposed review scope:** Documentation accuracy, provenance, protocol fidelity and clear disclosure of outstanding validation. **No kernel or runtime changes.**
