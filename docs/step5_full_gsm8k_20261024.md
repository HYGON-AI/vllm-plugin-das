# Step5 full GSM8K comparison — 2026-10-24

In this full run, the three profiles differ by at most three questions (0.227 percentage points). No broad Prefill accuracy drop is observed relative to Decode. This single stochastic pass does not establish numerical equivalence or cross-run reproducibility; a same-profile comparison confirms that individual answers can change between online runs.

All three profiles generate fresh answers for the complete frozen GSM8K test
set: **1319 zero-shot examples at their original prompt lengths**, one repeat.
The previous 100-example answers are not reused. These accuracy runs use
published inference code `c2daed7ef0d34ea0313eda4d9e8058a0166766cd`; inference code and the normal native
extension are unchanged from the earlier source verification.

## Full-dataset results

| Profile | DP/TP/EP | Legacy evaluator | Exact numeric rescore | Request errors | Output limit reached |
| --- | --- | ---: | ---: | ---: | ---: |
| Decode | 4/4/16 | 1273/1319 (96.513%) | 1273/1319 (96.513%) | 0 | 0 |
| Prefill E22 | 16/1/16 | 1275/1319 (96.664%) | 1275/1319 (96.664%) | 0 | 0 |
| Prefill E23 | 16/1/16 | 1272/1319 (96.437%) | 1272/1319 (96.437%) | 0 | 0 |

The primary score uses the unchanged EvalScope-compatible legacy extractor
and numeric comparison. The additional score uses exact Decimal equality of
the **extracted** answer and target, without percentage scaling or relative
tolerance. Strict numeric rescoring does not repair extraction errors. Neither
column is replaced by manually edited scores.

## Common evaluation settings

- Standard MTP3; synthetic acceptance sampling is disabled for accuracy.
- Concurrency 64; temperature 1.0; top-p 0.95; top-k -1; seed 42; max output 32768.
- Same frozen 1319 prompts, targets and order; prompt hashes, token counts and
  source IDs are checked pairwise. Evaluator syntax trees are identical apart
  from their local API ports.
- FP8 E4M3 KV and prefix caching disabled on four nodes x four gfx938 GPUs.
- Prefill retains tokens 8192 / model length 70000, cached ASM G1, fused G2,
  pipeline 65536, DBO 4096 and Directsum 65536. E22 uses grid 256 / software queue
  cap 4; E23 uses grid 320 / cap 8 with 16 fixed replicas and the measured token-hash
  map. Actual hardware queue counts are not measured.
- Decode retains DP4/TP4/EP16, tokens 32768 / model length 1048576, Fabric,
  context-bound graph fallback, and the previous expert placement.

The performance runs remain separate and retain the requested synthetic MTP3
rates `[0.838992,0.598322,0.362686]`. Performance is not remeasured here.

The fixed seed alone does not establish batch-independent online
reproducibility. The same E22 source/profile and generation parameters scored
97/100 in the separate short run, versus 98/100 on the first 100 examples of
the fresh full run; only index 85 changed its extracted answer, from 48 to 44.
This confirms run variation on unchanged inference code. Its specific cause
has not been isolated using fixed batches, greedy repeats or logits comparison.
Dynamic scheduling and numerical-path differences are possible factors, not
proven diagnoses. See [vLLM reproducibility](https://docs.vllm.ai/en/latest/usage/reproducibility/).
No batch-invariant mode is assumed to be qualified for the HCU/Step5 setup.

## Paired results

Each row below compares the same 1319 examples. The JSON retains the actual
disagreement indices so differences can be inspected without substituting a
different benchmark subset.

| First profile | Second profile | Both correct | Only first correct | Only second correct | Second minus first (percentage points) |
| --- | --- | ---: | ---: | ---: | ---: |
| Prefill E22 | Prefill E23 | 1264 | 11 | 8 | -0.227 |
| Decode | Prefill E22 | 1264 | 9 | 11 | +0.152 |
| Decode | Prefill E23 | 1265 | 8 | 7 | -0.076 |

The parallel layouts, expert placement and kernel profiles differ, and
generation is stochastic. The results do not isolate one kernel or establish
formal quality equivalence.

## Separate long-input samples

Full GSM8K above uses the original short prompts. It does **not** mean all
1319 examples were padded to 64K. The earlier fresh checks on the same source
separately exercise high-token Prefill on every DP rank:

| Profile | About 24K input | About 64K input | Request errors |
| --- | ---: | ---: | ---: |
| Prefill E22 | 15/16 | 15/16 | 0 |
| Prefill E23 | 15/16 | 15/16 | 0 |

All four misses in these long-input samples are the same lemon-tree question (index 12): the
selected final answer is the 12-year break-even point, while the reference
requires the first positive-profit year, 13. All responses end normally.
Rank-tagged logs verify 88 ASM layers on each of 16 ranks, matched packed
weight pointers, and DBO/Directsum/pipeline execution.

## Scoring caveats and provenance

The earlier short samples scored Decode 98/100, E22 97/100 and E23 96/100.
The Decode versus E23 two-point gap consists of one extractor false negative
(index 7: 160 minutes also expressed as 2 hours 40 minutes) and one actual answer
disagreement (index 85: 44 versus 48). Index12 is marked wrong in both profiles
despite final prose explaining that profit starts in year 13, so that shared
extraction issue does not explain their score gap. These are prior-sample
observations, not corrections added to the fresh full-dataset scores.

The native library SHA256, source snapshot, per-node versions/image IDs,
dataset/evaluator hashes, actual finish reasons and raw prediction hashes
are recorded in the JSON. The fourth node retains its previously authorized
different image ID, with matching package versions. Configured limits remain
unchanged; actual 1M-token end-to-end inference is not tested.

All owned validation services are stopped and their evidence is archived
locally. Datasets, raw predictions, weights, native binaries and placement
assets are not published. The E22/E23 reproduction profiles remain opt-in.

See the [machine-readable results](../benchmarks/step5/results-full-gsm8k-image1021-20261024.json)
and [earlier source/Decode verification](step5_pd_verification_20261024.md).
