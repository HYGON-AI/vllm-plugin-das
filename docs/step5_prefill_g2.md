# Step5 prefill G2 fusion

`HCU_STEP5_PREFILL_G2=1` enables a normal Python/Triton source path for the
fixed gfx938 E22, M65536, topk8, sort-BM256 channel-FP8 case. Default is off.
The normal `hcu_ops` library and the model weights do not change.

The existing Directsum metadata and scratch-overflow decision feed one G2
launch. It stores either to the Directsum destination or the original route
cache. The native final fix kernel retains the accumulation order. Unsupported
metadata/configurations retain the native two-launch path. Input/output offsets
use 64 bits, including the >2 GiB full-route overflow case.

Three isolated route levels, eight independent weight/input pairs, four
amplitudes, empty routes and forced overflow produced 102 complete-MoE bitexact
comparisons. Paired complete-MoE latency ratios were 0.94764/0.95510/0.96122
for p10/p50/p90; graph ratios were 0.94843/0.95519/0.96113.
These are operator measurements, not full-model throughput.

Run `benchmarks/step5/check_prefill_g2_fused.py --run-gpu --fixture FIXTURE.npz
--fixture-meta FIXTURE.json --output RESULT.json` in the normal package environment
to compare the integrated source, including device overflow and unsupported
metadata guards. Use the existing frozen routing fixture format.

With G1 specialization disabled, three clean full-model synthetic rounds gave
42,224.93 tokens/s (sample SD 8.25), 1.55% above the earlier 41,580.14 result.
DP16/TP1/EP16, batched-tokens8192, model-length70000, E22, native attention,
C64 x 65536 input tokens x 1 output token, and synthetic MTP3 rates
[0.838992,0.598322,0.362686] were retained. There were no preemptions or
in-measurement compilation/cache updates. This remains below 45,000 tokens/s.

Separate standard-MTP evaluation on the same delivered source scored GSM100
97/100, long24K16 15/16, and long64K16 15/16, with zero request errors. The
short GSM prompts total only 8,485 tokens (maximum 154 each); they cannot
exercise the M65536 specialization, so this sample alone does not establish
its quality effect. Long-context scores match the earlier 15/16 results;
evaluation-window dispatch counts were not collected. The independent
integrated complete-MoE check passed 34 bitexact comparisons and nine metadata
fallback guards. All performance and quality services were stopped afterward.

The separate G1 path is not required by G2 fusion. Prior stochastic short-GSM
scores of G1-enabled96/96 versus disabled98/98 need coverage-aware evaluation
before attributing a quality difference to that specialization.

## 2026-10-23 G2 tuning follow-up

The selected fused Triton G2 remains unchanged. These further measurements use
the frozen E22/M65536/topk8 p50 route fixture and are local operator results,
not full-model throughput.

The real ASM Mode5 catalog contains 21 recipes with both ordinary and shuffled
code objects. Two Ntile3584 recipes were excluded pending column-tail proof;
19 recipes x persist(0,128,256) x shuffle(0,1) gave 114 candidate tuples.
Both layout winners were solution23001/BM128/persist256.

| Measurement | Paired reference (ms) | Candidate (ms) | Candidate/reference |
| --- | ---: | ---: | ---: |
| ASM G2 + metadata, no shuffle | 2.41344 | 2.14416 | 0.88843 |
| ASM G2 + metadata, shuffled weights | 2.38176 | 2.09536 | 0.87975 |
| Complete MoE, ASM G2 through fixed cache3 | 6.66325 | 7.00813 | 1.05176 |
| Complete MoE, ASM G2 with full Directsum scratch | 6.65225 | 6.68945 | 1.00559 |
| Complete MoE, best graph-ranked additional fused Triton configuration | 6.65667 | 6.71461 | 1.00870 |

The first two rows compare preexisting route-layout ASM inputs against a native
sorted-to-route G2 reference; the reference gather is outside timing. They omit
G1, activation/quantization and final accumulation, so their 11.16%/12.02%
projection savings do not establish a complete-MoE benefit over the selected
fused G2/Directsum path. The independent destination bridge passed 9 cases and
36 graph checks with bitexact q/scales/metadata; this proves the bridge alone.

The fixed-cache3 complete pipeline must copy single-local-expert results from
cache3 to the real output, whereas native Directsum skips that copy. Its graph
latency was 6.95081 versus 6.60807 ms (ratio 1.05187). Full scratch removes that
copy but increases output/scratch capacity from 1 GiB to 4 GiB per M65536 chunk.
It still showed no speedup: graph 6.63497 versus 6.60381 ms (ratio 1.00472).
These variants passed 36/34 complete-MoE comparisons respectively; full scratch
also passed all 8 weight pairs as independent graphs and dynamic routing checks.
Maximum eager relative L2 was approximately 3.0e-5.

A separate 20-configuration fused Triton sweep passed 680 numerical and 160
graph comparisons, but none beat the existing integrated kernel. The best
graph-ranked candidate was BM256/BN256/BK64, warps8/stages2/GM1/persistent128:
graph 6.67585 versus 6.61569 ms (ratio 1.00909). The smallest eager ratio was 1.00831
with persistent256, also above 1. No new G2 configuration was adopted.

Evidence in `outputs/p5c1-4-regression-20261020`:

- `asm-g2-tune-guide-20261023-summary.json`
- `prefill-45k-asm-g2-destination-bridge-proof-v2-results.json`
- `asm-g2-directsum-fixed-cache3-20261023-summary.json`
- `asm-g2-fullscratch-20261023-summary.json`
- `fused-g2-tune-20261023-summary.json`
