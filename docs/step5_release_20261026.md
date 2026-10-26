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
