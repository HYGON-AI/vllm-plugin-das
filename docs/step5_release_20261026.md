# Step5 P/D release on the updated v0.28.1 branch

This branch integrates the finalized Step5 runtime from
`857d114a56846b7a8f6831592bc735adc9923510` onto HYGON-AI's
`v0.28.1-dev` at `cd412a0d53b9b517fbc3dcdf96121dc3daa1096c`.
The original branch remains available. Its final tree was consolidated before
rebasing so reverted experiments are not replayed as intermediate states.

The shared MoE integration preserves upstream dispatch ownership and generic
packed-weight layouts. Step5 late shared-expert scheduling remains scoped to
Step5, while upstream automatic DeepEP/EPLB overlap is retained. Offline EPLB
callbacks register once after the final model-specific replacements.

## Frozen runtime profiles

| Profile | Parallel layout | Batch tokens | Model limit | Memory fraction | Main implementation |
| --- | --- | ---: | ---: | ---: | --- |
| Prefill | DP16/TP1/EP16, fixed E23 | 8192 | 70000 | 0.875 | DeepEP NORMAL MNVL and vendor DeepGEMM/WASP |
| Decode | DP4/TP4/EP16, E22 | 32768 | 1048576 | 0.85 | Native/AITER, DP Fabric, bounded workspace |

The direct launchers are
`benchmarks/step5/server_step5_prefill_dp16.sh` and
`benchmarks/step5/server_step5_decode_dp4tp4.sh`. Set `NODE_RANK` to 0 through 3
and configure the host IPs and external asset paths for the target cluster.
Both use normal installed wheels. The Decode default in `launch_multinode.sh`
also retains the previously validated bounded-workspace protection.

The associated external package supplies the MNVL-only DeepEP wheel, original
Prefill placement, E23 static replica map, Decode placement, Dense algorithm
selection and Native MoE table. Calibrated asset files and native binaries are
not added to this Git branch.

Performance runs use C64, 65,536 input tokens, and synthetic MTP3 acceptance
rates `[0.838992, 0.598322, 0.362686]`. Pure Prefill uses one output token;
Decode uses 4096. Accuracy runs use standard MTP3 and the frozen full GSM8K
evaluation protocol instead of synthetic acceptance.

## Client measurements

`benchmark_request_metrics.py` and `request_latency_stats.py` accompany the
delivered benchmark client. They export TTFT distributions, request-average
TPOT, the original common-window SSE-chunk ITL, end-to-end output throughput,
and raw request/chunk timestamps after the original timing has finished.
The unchanged base benchmark and metadata adapter are supplied with the
reproduction package.

Post-rebase acceptance is recorded with the exact tested wheel/source hashes
and the complete test reports in the accompanying release bundle.

## Acceptance results (2026-10-27)

The tested runtime commit is `eca1f8d7282449e10c0036964a01879d82c09c67`.
The following measurements use fresh containers, normally installed wheels,
and the frozen external assets from the prior complete delivery.

| Profile | Formal performance | Full GSM8K, standard MTP3 |
| --- | --- | --- |
| Prefill DP16/TP1/EP16, E23 | 45,439.7 input tokens/s, mean of 3 rounds | 1275/1319 (96.66%) |
| Decode DP4/TP4/EP16, E22 | 3,180.6 tokens/s in the common Decode window; ITL 56.608 ms | 1278/1319 (96.89%) |

Prefill formal rounds were 42,627.6, 46,821.9, and 46,869.6
input tokens/s. All three are retained in the mean, which is
-3.02% relative to the prior 46,854.9 mean. The first round
was lower; the final two were close to the prior reference. There were no
source or parameter changes between rounds.

Decode is one formal 4096-output-token round after an excluded warmup.
The prior reference was 3167.1 common-window tokens/s and 56.776 ms ITL.
The new run's request-average TPOT is 39.810 ms; end-to-end
output throughput, including Prefill, is 976.223 tokens/s.
All formal runs completed 64 requests with zero preemptions.
TTFT distributions and raw request/chunk timestamps are retained in the bundle.

Both quality runs use the same frozen 1319 zero-shot prompts, concurrency 64,
temperature 1.0, top_p 0.95, top_k -1, seed 42, and a 32768-token output limit.
Performance uses the specified synthetic MTP3 rates; quality uses standard MTP3.
Strict Decimal rescoring agrees with the original numeric grader. Both runs
have zero request errors and all 1319 responses finish with `stop`.
The prior same-protocol quality references are Prefill 1277/1319 and Decode
1275/1319. These are individual sampled runs, not a bitwise-equivalence claim.

The final focused compatibility regression has 483 passed and 3 GPU-only
skipped cases. GPU component checks cover six Dense shapes and 13 DSA cases;
the checked component sources/native binary match the accepted runtime.
The four installed packages match all 433 committed Python source files.

The rebase also restores upstream opt-in IPC initialization and cleanup,
keeps generic PCP/model paths outside Step5 late synchronization and DBO,
and retains upstream packed-weight handling. These changes are covered by
the compatibility regression.

Machine-specific placement and tuning files remain in the external bundle.
The companion `results-upstream-rebase-20261026.json` records the measurements;
it is not a runtime tuning configuration.
