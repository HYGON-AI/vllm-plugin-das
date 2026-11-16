# MiniMax31 aligned MoE full-model benchmark

The opt-in `HCU_MINIMAX_ALIGNED_MOE=1` selects the standalone aligned core inside the existing MiniMax31 DeepEP runner. Default routing is unchanged. The model keeps static136/17, shared overlap, FP8 index, eager execution and PCP8/EP8.

One aligned executor, communication buffer and scratch workspace are shared across the 57 serial MoE layers in each rank. Every call binds that layer’s existing HIPC weights under the single-inflight lock. No permanent weight repack or IPC layout change is introduced. `bind_weights` rejects geometry/device/stream changes. The shared branch and outer routed scaling retain the existing runner lifecycle.

## Validation

- Real GPU regression: rebind layer1 -> layer2 -> layer1; each result matches a fresh workspace bit-for-bit, scratch pointers stay fixed, changed geometry is rejected.
- Full60 loading: eight producer/consumer identities and mapped weight samples match; consumer pack count is zero. All 57 aligned layers are active per rank with one workspace.
- 1824 real shared-MLP serial/auxiliary-stream output checks passed.
- All 53 original cases and 1347 timed requests passed raw audits: prompt147456, output1, device prefix hit131072, frozen input SHA, rate/seed and native TTFT/P95/throughput formulas. Warmups and the separate diagnostic profile are excluded.

## Results

At mean TTFT <2000ms, the highest measured cached-input throughput is 200988.38 tok/s (c4_rr4.000_s42, mean 1520.07ms), up 12.11% from the prior measured matrix. It remains 15.65% below the historical SGLang peak.

| Case selected by original SGLang table | Prior ours mean TTFT(ms) | New ours mean TTFT(ms) | SGLang mean TTFT(ms) |
|---|---:|---:|---:|
| c1_rrinf_s42 | 980.12 | 862.40 | 745.81 |
| c2_rrinf_s42 | 1434.02 | 1275.67 | 1097.49 |
| c4_rr8.000_s42 | 2354.52 | 2099.79 | 1745.56 |
| c8_rr2.867_s42 | 2817.92 | 2376.36 | 1817.13 |
| c16_rr1.849_s42 | 3181.07 | 2184.74 | 1894.81 |
| c32_rr0.978_s42 | 2894.86 | 2329.30 | 1874.61 |
| c40_rr0.929_s42 | 2793.18 | 2116.63 | 1658.71 |
| c48_rr0.930_s42 | 2708.74 | 1996.57 | 1597.06 |
| c64_rr0.928_s42 | 2559.52 | 1965.63 | 1583.27 |

KV capacity is 493910 tokens versus 557343 in the prior run and 1202176 in the historical SGLang baseline. Full-batch-residency prechecks still fail for BS>=32; every actual request’s device cache hit is nevertheless verified. No fixed block override is used.

The original SGLang baseline is historical nmz18 data; this rerun is nmz20, so it is not a same-node SGLang A/B. The physical map is the same, but our selector remains splitmix64 versus SGLang locality_fair. Each case is one seeded run, with no confidence interval or new model-quality claim.

## Artifacts

Task directory: `/public/home/xiesl/inference/verify_torch213_image/task_dir/support_minmax_31_pcp8_ep8_perf/sglang_full_moe_alignment_20261115/`.

`optimized_results/` retains raw requests/native metrics; `runtime_evidence.json` independently reconciles all 53 cases; `readiness_proof.json` and `aligned_runtime_state.json` prove the loaded path and ownership; `traces/` is a separate four-request diagnostic. The updated two-sheet workbook preserves source reference data, all formulas, styles, table/conditional-format definitions and freeze panes.

The benchmark-owned API/engine and producer trees were stopped after fresh PID/birth-time verification. Final memory is approximately2MiB per GPU. SSH and artifacts are retained.

Baseline commit: `7c018ed`. Runtime changes are limited to the explicit runner branch and stream-ordered weight binding; the standalone module remains independently usable.
