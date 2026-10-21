# AIOPT-1269 C1 — Frozen-base current-mainline reintegration

Status: C1 (Step5 MTP draft-layer MoE fixed-shape specialization) reintegrated
onto the frozen `v0.28.1-dev` head `56d23cd` and revalidated on nmz106.

This document reports the **frozen-base revalidation**. The historical
`+1.007%` / `-0.989%` / `+0.085%` result on the earlier lineage remains a
historical reference only; it is not an acceptance threshold for this round.

## Frozen base

```
FROZEN_UPSTREAM_BRANCH=upstream/v0.28.1-dev
FROZEN_UPSTREAM_HEAD=56d23cd55f191496e25bd67bb92755aec9736bc7
FROZEN_UPSTREAM_DATE=Tue Oct 20 21:53:18 2026 +0800
FROZEN_UPSTREAM_MESSAGE=perf(dsv4): optimize DeepEP MoE and static EPLB for PCP TTFT (#266)
```

## Candidate

```
BRANCH=aiopt-1269-c1-frozen-base
VALIDATED_CODE_HEAD=67b757d3b5617b964e90a2aed8481a8a766af5a1
A0_ARM=8b7fae2d9453ef3d2b71432f5a9136a0ffc99e79   (56d23cd + pin fix 509affd)
C1_ARM=67b757d3b5617b964e90a2aed8481a8a766af5a1   (8b7fae2 + C1 09d83cc)
PIN_FIX_COMMIT=509affd0df1a5efbe50f1c172c87755b01087dff
C1_SOURCE_COMMIT=09d83cc7fbf01d82269a6028e1c4341e6b3bda23
```

Pure upstream `56d23cd` cannot start the deployed Step5 runtime: the runtime
`_generate_drafts` source hashes to `dcabfd6f...`, which is absent from the
upstream reviewed set. The mandatory pin fix `509affd` is therefore applied to
**both** E2E arms; only the C1 diff differs between A0 and C1.

## Impact analysis (frozen base `56d23cd`)

Full ledger: `evidence/impact_ledger.md`.

- Upstream since the historical base: `dd706a3`, `8c827d9`, `4577bab`,
  `993e14b`, `56d23cd`; only `56d23cd` is new since the last audited head
  `993e14b`.
- `56d23cd` changes DeepSeek-V4 DeepEP/DeepGEMM/EPLB, modular-kernel workspace
  and shared-expert overlap. Those paths are gated by
  `deepep_auto`/`deepep_high_throughput` and DeepSeek-V4; the C1 workload uses
  Step5 bf16 `fused_experts` with AgRs all2all and EPLB disabled.
- No upstream change to `get_moe_configs`, `VLLM_TUNED_CONFIG_FOLDER`
  precedence, the Step5 MTP patch file, `vllm_hcu/__init__.py`, `setup.py`, or
  the C1-touched files.
- C1 is still required: frozen upstream ships no `E=44,N=1536` tuned config.

## Runtime guard

```
RUNTIME_IDENTITY=PASS
PLUGIN_CONTRACT=PASS
RUNTIME_CLASS=NMZ106_STEP5_EXACT
```

Fingerprint proof (`_generate_drafts`):

```
runtime source                 dcabfd6fe030337921106b7d409d26d44db3b152458c341b062c1c92c4a82e50
minus two trailing comments    88f8e6d38c898e68992d42e986a07bc7bd3277915224566399d00e93780abb78
AST / signature                identical
co_code / consts / names       identical
marshal bytes                  identical
disassembly                    169/169 identical
COMMENT_ONLY_DIFFERENCE=YES
SEMANTIC_EQUIVALENCE_PROVEN=YES
```

## Selector and fallback

```
A0 (frozen upstream + pin fix): VLLM_TUNED_CONFIG_FOLDER=None
                                get_moe_configs(44,1536,None)=None  -> default
C1 (candidate):                 VLLM_TUNED_CONFIG_FOLDER=<isolate>/.../moe_configs
                                get_moe_configs(44,1536,None)=19 keys
                                M=16..256 -> tuned tile BM16 BN64 BK128 w2 s1
FALLBACK: E=352 miss; dtype=fp8 miss; M=65536 -> vLLM default tile;
          user VLLM_TUNED_CONFIG_FOLDER is preserved and wins.
BW1100 / BW1101 alias smoke: PASS
```

## Local correctness

Frozen-base run (`evidence/local_bench_current_mainline.json`):

```
ROWS=12 (M=16/32/64/128 x seeds 0/1/2, HCU4/BW1101)
BIT_EXACT=YES (12/12)
MAX_ABS=0.0
ALL_FINITE=YES
INPUT_PRESERVED=YES
```

## Local performance

Same method as the historical microbench (CUDA-graph, 3 interleaved rounds x
20 replays, weight rotation):

| M | A0 µs | C1 µs | Δ median |
| ---: | ---: | ---: | ---: |
| 16 | 527.06 | 370.85 | −28.00% |
| 32 | 818.98 | 607.46 | −26.76% |
| 64 | 1053.38 | 840.08 | −22.86% |
| 128 | 1548.53 | 1005.28 | −34.12% |

```
LOCAL_PERF_VALID=YES
LOCAL_PERF_DIRECTION=POSITIVE
A0_FROZEN_MEDIAN=996.61 us   (median across the 12 rows)
C1_FROZEN_MEDIAN=743.37 us
RATIO_OF_MEDIANS_DELTA=-25.41%
MEDIAN_DELTA_ROWS=-27.19%    (12/12 negative)
```

The historical −25.79% is reference only.

## Paired E2E (frozen base)

3 interleaved clean-restart pairs on nmz106; TP4 × DP2 × EP8, MTP3, bs 32,
input 65536, output 4096, seed 42. A0 arm = `8b7fae2` (56d23cd + mandatory
pin fix); C1 arm = `67b757d` (+ C1 diff only).

| pair | A0 decode | C1 decode | delta |
| ---: | ---: | ---: | ---: |
| 1 | 1418.178 | 1418.768 | +0.042% |
| 2 | 1392.022 | 1392.009 | −0.001% |
| 3 | 1388.873 | 1399.747 | +0.783% |

```
E2E_VALID_PAIRS=3
PAIRED_WINS=2/3
A0_MEDIAN_DECODE=1392.022
C1_MEDIAN_DECODE=1399.747
MEDIAN_DECODE_DELTA_PERCENT=+0.555
MEAN_DECODE_DELTA_PERCENT=+0.273
ITL_DELTA_PERCENT=-0.575
PREFILL_DELTA_PERCENT=+0.200
MTP_ACCEPTANCE_DELTA=0.000000
REQUEST_FAILURES=0        (384 complete / 0 incomplete; 0 preemptions)
SERVER_5XX=0
OOM=NO
CRASH=NO                  (6 teardown-time EngineDeadError lines at SIGTERM,
                           after all formal cases; no in-run crash)
SELECTOR=A0 default 3/3 · C1 hit 3/3
STABILITY=PASS            (6/6 lives READY, 12/12 formal cases rc=0,
                           health 200, GPU_RELEASED=YES after every life)
```

The frozen extension rule (3/3 positive and stable) was **not** met:
PAIR2 is −0.001%, a statistical tie. The run therefore stopped at the
required 3 pairs and was **not** extended to 5. The 3-pair median, mean and
ITL are positive; the result is reported on its own merits. The historical
`+1.007%` was not used as an acceptance threshold.

## Historical C1 result — reference only

Earlier lineage (`567fd24 + 509affd + 09d83cc`, 5 pairs):

```
PAIRED_WINS=5/5
MEDIAN_DECODE_DELTA_PERCENT=+1.007
ITL_DELTA_PERCENT=-0.989
PREFILL_DELTA_PERCENT=+0.085
MTP_ACCEPTANCE_DELTA=0.000000
LOCAL: 12/12 bit-exact, max_abs=0.0, median -25.79%
```

These values are historical reference only and are not reused as frozen-base
acceptance evidence.

## Scope

```
merge-base(56d23cd, HEAD)=56d23cd
commits=3 (pin fix + C1 + this delivery doc)
files=8 (7 code files +571 lines, plus this doc)
OLD_PATCH_CHAIN_INCLUDED=NO
UNRELATED_COMMITS=NO
UNRELATED_DIFF=NO
SCOPE_CONTAMINATION=NO
```

## Delivery drift

Single read-only fetch after all validation:

```
FROZEN_UPSTREAM_HEAD=56d23cd55f191496e25bd67bb92755aec9736bc7
UPSTREAM_HEAD_AT_DELIVERY=56d23cd55f191496e25bd67bb92755aec9736bc7
UPSTREAM_MOVED_AFTER_FREEZE=NO
DELIVERY_DRIFT_IMPACT=NONE
INCREMENTAL_REVALIDATION_PERFORMED=NONE
```

## Evidence

Host path (not shipped in the repo):
`/public/home/liumengjie/vllm028-work/experiments/AIOPT-1269/C1_CURRENT_MAINLINE_56d23cd/`

- `evidence/impact_ledger.md`
- `evidence/runtime_preflight.txt`
- `evidence/A0_selector.txt`, `evidence/C1_selector.txt`,
  `evidence/C1_user_override.txt`, `evidence/selector_smoke.txt`
- `evidence/local_bench_current_mainline.json` / `.log`
- `evidence/e2e_frozen_base_summary.{json,txt}`
- `harness/runs/<RUN_ID>/` — 3 paired clean-restart lives

