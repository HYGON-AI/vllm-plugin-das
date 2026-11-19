# Step5 W4A8 accuracy acceptance evidence — GSM8K and AA-LCR

**Evidence date:** 2026-11-12 through 2026-11-17.  
**Target baseline:** `HYGON-AI/vllm-plugin-das:step5-w4a8-pd@eb8c516`.  
**Artifact type:** validation evidence only; no changes to model, kernel, benchmark or verifier.  
**Overall evidence classification:** **PARTIAL / MIXED GATES**. GSM8K Gate 1 remains INCONCLUSIVE. The AA-LCR diagnostic adjusted score is reconciled below but is **not** an official same-protocol score. This document does not claim all acceptance criteria passed.

## 1. Executive conclusion and acceptance matrix

| Validation | Available result | Acceptance interpretation |
| --- | --- | --- |
| Full GSM8K 1319, FP8 reference vs W4A8 | A=1270/1319; B=1265/1319; 5 fewer correct in W4A8 | **INCONCLUSIVE** under the pre-registered strict non-drop requirement |
| Full GSM8K, W4A8+MTP3 vs W4A8 no-speculation | B=1265/1319; D=1266/1319 | **PASS_NON_INFERIOR** under the task runbook rule; not identical |
| Full GSM8K, W4A8 IPC AG off vs on | B=C=1265/1319; 9 per-question flips in each direction | **PASS** under the runbook's equal-total gate; **not** per-question bitwise identity |
| AA-LCR High81 Rescue C3 | 243/243 terminal generations; official judge job completed, 243 judged; 105 `judge_correct` trials; 16/16 independent checks PASS; diagnostic adjusted score **88.25%** = (318 + 105/3) / 400 | **COMPLETED RUN / DIAGNOSTIC ADJUSTED SCORE RECONCILED; NOT OFFICIAL** |
| Terminal-Bench 2.1 | Not a completed 89×3 official-comparable score in the evidence reviewed | **PENDING**, independent future report |

**AA-LCR score reconciliation (2026-11-19, CPU-only):** The original 400-slot run is
318 correct + 41 non-empty wrong + 40 empty + 1 OOM (`qid 4 repeat 1`) = 400.
The rescue covers the 81 non-OOM incorrect slots (40 EMPTY + 41 WRONG) with
3 trials each = 243 trials; **105 trials were judged correct**
(105/243 = 43.2099% on the rescue subset). The diagnostic adjusted score is
**(318 + 105/3) / 400 = 353/400 = 88.25%**: each rescued slot receives the
fraction of its three trials that were correct. This is a
**SELECTIVE_RETEST_MIXED_PROTOCOL** diagnostic, not an official
same-protocol AA-LCR score. The reference is 88.3%; the rescue changed
`reasoning_effort` to `high` and `max_tokens` to 16384, and selectively
retested only the 81 failed slots. The 16/16 independent verifier checks
cover judge-contract identity, dataset/prompt hashes, result
coverage/uniqueness, raw-summary recomputation, and key-leak scanning; they
do not certify an official leaderboard score. Therefore **88.25% may be
published only as a diagnostic adjusted score, never as official parity**.

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
| Raw `judge_correct` in handoff | 105 trials |
| Diagnostic adjusted score | (318 + 105/3) / 400 = 88.25% |
| Built-in verifier | PASS |
| Independent verifier | PASS (16 checks, 0 failures) |
| Manifest checked and frozen | YES |
| Original 400-question manifest unchanged | YES |

**Reconciled scope:** The original 400-slot run is 318 correct + 41 non-empty
wrong + 40 empty + 1 OOM (`qid 4 repeat 1`) = 400. The original judge records
split exactly as **359 JUDGED + 40 EMPTY_CANDIDATE = 399**, plus **1
unjudged ERROR/OOM** slot = 400. The rescue covers the 81 non-OOM incorrect
slots with 3 trials each = 243 trials, of which 105 are correct. The 1 OOM
slot remains incorrect and stays in the denominator. The manifest verifies
400/400 original result files by SHA256 and the rescue judge verifies 243/243
judged tasks. The 88.25% is the diagnostic adjusted score defined above, not
an official same-protocol score.

**OOM provenance caveat:** The excluded slot `4:1` is classified as OOM by
the run summary (`OOM_FAILED=1`, `OOM_KEYS=[["4", 1]]`) and the attempt-5
driver log records `CUDA error: out of memory`; its final per-slot error is a
`URLError: Connection refused` retry. The slot remains incorrect in the
denominator. The run summary's `OOM_LINES_IN_LOG=0` does not match the driver
log and is treated as a summary-field caveat, not as a reason to re-add the
slot or change the score.

### 4.1 Reconciliation inputs (SHA256)

| Input | SHA256 |
| --- | --- |
| `HIGH81_SLOTS.json` | `5bbfa64081904b2c947b275220cdc465f2722d612db03550046f0556a032e98a` |
| `HIGH81_SLOT_RESULTS.csv` | `1006f6d5c15d792e380295b576bff66276028afe5bbfaff7af11a57bf56cab64` |
| `generations_243.jsonl` | `a88613f784afe7f03a148aa6a4beba4c9146e2d3144f67c8e774a7da16108836` |
| `judge/summary.json` | `cd65d646cec2cc80a6cddfa198a579f38a83fe89b44fe09876013e91e1188b8c` |
| `judge/independent_verifier.json` | `61b340ef26a723ab0d1715e7f26641d27c5e0ac101356f9f7b476981b97b6cb2` |
| `ORIGINAL_RESULT_MANIFEST.sha256` | `8b9e4fbc7af127313dd98aa8b215f02f7b0b3aa0371b8b4ee59670926913a83f` |

The CPU-only reconciliation script (`recompute_high81.py`) and its per-slot /
per-trial ledgers are kept in the internal delivery evidence archive
referenced by the PR body; they are intentionally not copied into this
repository. The script re-verifies all 400 original result hashes, the
359 JUDGED + 40 EMPTY_CANDIDATE + 1 ERROR/OOM split, the 81x3 rescue design,
and the 105/243 judge result.

The model's service configuration in the handoff used W4A8, TP4/DP2/EP8, MTP3, Triton MoE, AllGather/ReduceScatter, graph FULL_AND_PIECEWISE and BF16/FP8 mixed runtime components. The exact service logs, checkpoint identity and manifest are authoritative; do not replace them with assumptions about an unrelated performance run.

## 5. Evidence provenance and reproduction pointers

The original results were produced in external evaluation environments and are **not copied into this PR**. Internal evidence owners should cross-check the original artifacts before signoff:

- GSM8K full A/B/C/D summary: `W4A8_ACCURACY_REPORT.md` (generated 2026-11-13), archived `accuracy_gate/RUNBOOK_GPU_ACCURACY.md`, `PROTOCOL_AUDIT.md`, `PER_QUESTION_COMPARISON.json`, `ACCURACY_VERIFICATION_RESULTS.json`, and each arm's `decode_accuracy/summary.json`.
- AA-LCR High81: run directory `AA_LCR_W4A8_HIGH81_RESCUE_C3`, frozen `HIGH81_TO_TB21_READY.json` and `HIGH81_RESULT_MANIFEST.sha256`. The 2026-11-19 CPU reconciliation verifies 400/400 original manifest hashes, 318 original correct, 41+40+1 failure slots, 105/243 rescue correct trials, and `(318 + 105/3) / 400 = 88.25%`; per-slot and per-trial ledgers are in the adjacent `evidence/` directory.
- Existing source/performance change in separate PR: [#472](https://github.com/HYGON-AI/vllm-plugin-das/pull/472).

The evidence set was **read from archived status and accuracy reports**, not from a fresh execution of GSM8K/AA-LCR, and no new hardware run is claimed.

## 6. Follow-up status (not part of this PR)

1. **Resolved 2026-11-19 (CPU-only):** AA-LCR raw 105/243 is the trial-level
   rescue result; the diagnostic adjusted score is `(318 + 105/3) / 400 =
   88.25%`. The reference protocol and denominator do **not** match the
   official 88.3% protocol because this is a selective retest with
   `reasoning_effort=high` and `max_tokens=16384`. The remaining decision is
   reviewer acceptance of the diagnostic label, not arithmetic reconciliation.
2. If publication requires strict W4A8-vs-FP8 non-drop, resolve Gate 1 **INCONCLUSIVE** with a pre-registered follow-up.
3. Validate the code changes in PR #472 under the remaining device/Graph/Prefill gates independently of the GSM8K total-score comparison.
4. Run Terminal-Bench 2.1 only as a later, separately versioned evaluation; its current absence must not be misreported as a passed result.

**Proposed review scope:** Documentation accuracy, provenance, protocol fidelity and clear disclosure of outstanding validation. **No kernel or runtime changes.**
