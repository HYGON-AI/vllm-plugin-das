# Step5 prefill ASM G1

Historical measurement record. Current cleanup removes the retired Triton G1
experiment and keeps tuning data external; see [current source contract](step5_pd_verification_20261024.md#source-cleanup-after-the-recorded-measurements).

`HCU_STEP5_PREFILL_ASM_G1=1` opts into the existing AITER ASM stage1 API
for gfx938, DP16/TP1/EP16, E22 or fixed E23, M65536, topk8 and the original
BM256 expert sorting. The default remains disabled. Solution13001 uses a BM128
ASM tile, persistent256 for E22 or persistent320 for E23, and stage1 byte-shuffled weights. The normal
`hcu_ops` library, model weight files and E22 placement do not change.

The normal Channel-FP8 post-load hook packs each final raw W13 Parameter
after backend normalization, inside the model-loader memory profiler.
Original NK weights remain available for the original small-batch and
unsupported-shape paths. MTP/draft owners and dynamic EPLB are excluded. Per-layer
cache logs report source/packed pointers, allocation bytes and total cache
bytes. Reloading runs the post-load preparation again; stale or dead source
references cannot silently reuse a packed cache.

Each of the 88 main MoE layers adds 264 MiB of resident packed W13,
22.6875 GiB per rank in total. KV sizing must run after this allocation.
The complete-model run verified all 88 layers on every one of the 16 ranks:
each allocation delta was 276824064 bytes and the retained total was
24360517632 bytes per rank. Cache messages preceded model-loading completion
and KV sizing. Actual KV capacity was 416918–450880 tokens per rank,
above the 280000-token admission requirement for four 70000-token requests.
These are profiled capacities with the cache resident, rather than estimates
from the earlier runtime's available memory.

All ranks reported 89.85 GiB for model loading, compared with 68.29 GiB in
the preceding G2 run. HCU's loader profiler measures the difference between
two total-minus-free device-memory snapshots; it does not measure only live
Torch tensor bytes. The two runs' loading deltas cannot be combined as
68.29 + 22.6875. The exact cause of the 1.1275 GiB difference from that sum
has not been established. Admission instead verifies every cache allocation,
its lifetime before profiling, the resident-tensor lower bound and actual
KV capacity. KV profiling uses actual total consumed memory plus transient
peak headroom, including the persistent registered cache.

ASM G1 emits route-order BF16 projection rows. The activation kernel changes
only its input row address, retaining the native SwigluStep math, FP8
conversion, FP32 scale multiplication order and sorted FP8 output.
Triton G2, Directsum and the current same-stream DBO lifecycle stay in place.
No weight packing, extra input tail copy or independent native library is
created in forward.

Run `benchmarks/step5/check_prefill_asm_g1.py --run-gpu --fixture FIXTURE.npz
--fixture-meta FIXTURE.json --output RESULT.json` in the normal package
environment. It uses actual post-load cache preparation and flag dispatch,
exact-M AQ/AS logical storage, complete MoE output/guard checks and individual
weight-pair graph checks. It does not replace the model loader or constitute
a memory-sanitizer proof.

## Normal-source operator acceptance

The normal source delivery is `70af815dbffd32ba7cf497019e4df115c22f3eb2`,
running from `/opt/xuwq_step5_native_prefill_asm_g1_20261023`. The existing
normal `hcu_ops` binary remains SHA256
`50fbca2b1b957e5eb121d80847f044d51e3fd1c185ca4aacb7444d78db10c2d5`.

The installed checker passed 34 complete-MoE eager cases, eight individual
weight-pair graph cases and an eight-weight-ring graph check on the p50
routing fixture. It used the real normal cache API and dispatch, with no
private activation replacement. AQ and AS had exact-M logical storage,
without the extra backing rows used by earlier private probes. Maximum
relative L2 against the native complete-MoE result was 0.000260676; this is
a numerical operator check, not a complete-model quality result. Empty and
overflow routes, output guards, changed data, stale versions, dead-source
pointer reuse, draft exclusion and normal cache refresh were checked.

| Complete-MoE median | Native G1 + fused G2 | Cached ASM G1 + fused G2 |
| --- | ---: | ---: |
| CUDA events | 6.667310 ms | 6.251690 ms |
| Graph replay | 6.627890 ms | 6.208611 ms |

The eager ratio was 0.937663 and the graph ratio was 0.936740. Earlier
per-call workspace packing consumed most of the prepacked kernel gain;
the complete-model result below uses the resident post-load cache.

## Complete-model fixed-protocol performance, 2026-10-23

Four nodes with four gfx938 cards each used DP16/TP1/EP16, E22 placement,
64 concurrent requests, 65536 input tokens and one output token per request.
The fixed limits were `--max-num-batched-tokens 8192`,
`--max-model-len 70000`, `--max-num-seqs 64` and memory utilization 0.875.
KV used FP8 E4M3 and prefix caching was disabled. Synthetic MTP3 used exactly
`synthetic_acceptance_rates=[0.838992,0.598322,0.362686]`.

Dense tuning, native FA, cached ASM G1 and fused Triton G2 were enabled;
the earlier `HCU_STEP5_PREFILL_G1` specialization was disabled. Pipe,
DBO and Directsum thresholds were 65536, 4096 and 65536 respectively;
MoE chunk size was 65536, bounded workspace and uniform collectives were
enabled, and `NCCL_MAX_NCHANNELS=32`. E22 placement and model weights were
retained. All 16 ranks had the expected cache and actual ASM/G2 execution
markers, with no old-G1 execution marker.

Two complete warmups were excluded. Three consecutive formal rounds were
clean, each completing all 64 requests with zero preemptions, inference JIT
events or cache-file updates:

| Formal round | Prefill tokens/s | Prefill latency |
| --- | ---: | ---: |
| 3 | 42512.983558 | 98.659366 s |
| 4 | 42552.802042 | 98.567046 s |
| 5 | 42631.573851 | 98.384920 s |

Mean throughput was **42565.786483 tokens/s**, with sample standard
deviation **60.351977 tokens/s**. This is **0.80725%** above the preceding
G2-only mean of 42224.9259 tokens/s. The 45000 tokens/s target has not been
reached. The local complete-MoE improvement and complete-model improvement
are separate measurements.

## Quality status and evidence

The separate same-source standard-MTP quality run completed: GSM8K100
**96/100**, 24K-context GSM8K16 **15/16**, and 64K-context GSM8K16 **16/16**,
with zero request errors. All synthetic and quality process groups were
stopped after collection. No 98/100 adoption threshold was applied. The
preceding G2-only run scored 97/100, 15/16 and 15/16; these small stochastic
samples do not establish a causal quality change. Short GSM8K prompts
alone do not cover M65536.
The long-context tests direct one request to each DP rank; their full
8192-token chunks can enter the M65536 ASM path. Startup's once-per-layer
active messages alone do not establish execution within a particular QA run.

Run artifacts are retained under
`outputs/p5c1-4-regression-20261020/` outside this repository:

- `prefill-45k-asm-g1-integrated-full-moe-p50-v1-results.json`
- `prefill-45k-dp16-asm-g1-cached-summary.json`
- `prefill-45k-dp16-asm-g1-cached-all16-runtime-verification.json`
- `prefill-45k-dp16-asm-g1-cached-all16-asm-archive-readiness.json`
- `prefill-45k-dp16-asm-g1-cached-asm-cache-loading-admission.json`

The complete performance/quality receipt is `prefill-45k-dp16-asm-g1-cached-after-benchmark-result.json`.

## Fixed E23 follow-up, 2026-10-24

E23 admission requires a fixed direct-load map with 352 logical experts and
368 physical slots, 23 slots per rank, `token_hash` dispatch, 16 redundant
experts, and rearrangement/async/recording/elastic/filter paths disabled.
The normal loader binds the immutable per-layer row before loading final
replica weights. Cache lookup rejects replaced map rows and changed weights;
it never migrates experts during forward. E22 admission remains separate.

The E23-specific isolated search covered 30 solutions, BM16/32/64/128,
eight persistent grids and both shuffle layouts, across four route fixtures
and four representative ranks: 7,680 tuple samples. Solution13001/BM128/shuffle1
remained best. A separate 16-rank native AG/RS experiment selected grid320
for a complete-model trial; isolated grid64's small gain did not survive
communication. These operator fixtures preserve captured expert marginals
but use synthetic token associations and synthetic weights.

The normal-source grid320 trial used the same C64, 65536/1, 8192-token budget,
70000 context limit and exact synthetic MTP3 rates as above, with software
queue cap8. Three clean rounds measured 42656.609871, 42701.493007 and
42641.251358 tokens/s: mean **42666.451412**, sample standard deviation
**31.303450**. This is 0.5071% above the E23 Triton control (42451.191263).
The source delivery was `8c6ab08360f5a4b62cff4145a013c1edfeb79809`.

All 16 ranks retained and executed 88 caches before KV sizing. E23 adds
23.71875 GiB of packed weights per rank; actual minimum KV capacity was
364240 tokens. The E23 complete-model quality sample has **not** been run;
operator numerical checks and the E22 quality results do not replace it.
The E23 profile remains an explicit candidate, not an accuracy-equivalence claim.
The 45000 tokens/s target was not reached and further tuning was stopped.

Use the existing launcher with `STEP5_PREFILL_PROFILE=asm_g1_1021` for E22.
The E23 candidate uses `STEP5_PREFILL_PROFILE=e23_asm_1021`, an explicit
`STEP5_PREFILL_REDUNDANT_EXPERTS=16`, and a readable
`STEP5_PREFILL_STATIC_MAP`. Both require the external Prefill placement file.
The generic two-stage AITER tuning CSVs are not modified by this feature.

Additional external evidence:

- `prefill-45k-e23-asm-catalog-20261024-v1-summary.json`
- `prefill-45k-e23-asm-grid-native-pipe-20261024-v1-summary.json`
- `prefill-45k-dp16-e23-asm-g1-p320-cached-q8-v1-summary.json`

The merged publication checkout adds the branch's existing Decode integration.
The Prefill numbers above belong to the pinned measurement runtimes, rather
than a new full-model Prefill run of that consolidated checkout.
