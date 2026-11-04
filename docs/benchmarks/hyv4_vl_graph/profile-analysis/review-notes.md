# Trace interpretation and limits

The BBuf llm-torch-profiler-analysis analyzer was run separately on all eight
rank files, preserving the original gzip traces and their SHA-256 hashes.
This was one mixed real-image request (same caption prompt and 32-token output
settings as the timing suite), bounded at three worker iterations. The window
contains image/prefill work and one decode forward, not three steady decode
steps. Timing claims come from separate unprofiled HTTP runs.

Rank 0 observations from this short profiled window:
- Prefill-labelled fused_moe_kernel: 147.47 ms, 28.3% of summed kernel time.
- Prefill-labelled sparse attention: 76.85 ms, 14.8%.
- Prefill-labelled NCCL all-reduce: 63.81 ms, 12.3%.
- Decode-labelled sparse attention: 16.26 ms, 26.5%.
- Decode-labelled fused MoE: 5.83 ms, 9.5%.
- Decode-labelled custom all-reduce: 5.07 ms, 8.3%.

These percentages are kernel-duration shares, not additive removable wall time.
The source warning confirms the installed runtime has no tuned Triton config
for E=256,N=256,device_name=BW1101,dtype=fp8_w8a8. MoE tile tuning is a possible
next experiment; no tuning is silently applied in this graph-only candidate.
Changing K tiles or reduction order requires its own numerical validation.

The generated fusion table's 'Confirmed' label identifies a known pattern,
not proof of HCU eligibility or that a compiler pass ran. This experiment
explicitly disables torch.compile. Sparse-attention, norm/quant and routing
patterns need concrete source/consumer checks before an implementation change.
No overlap opportunities cleared the analyzer's reporting threshold in this
single-trace mode. That does not prove no overlap exists.

Source attribution under replay frequently stops at hipGraphLaunch. No separate
eager mapping trace was transferred into this run, so deeper Python source
locations for those kernels remain unresolved. ROCTracer also reported duplicate
flow starts; retain that limitation when reading launch associations.

After the profiler automatically stopped, the service encountered HSA exception
0x1016 / KERNEL VMFault on the following decode step and returned HTTP 500.
The exported trace is a diagnostic window before that failure; it is not a
stability or accuracy pass. The source remained unchanged. A fresh service with
no profiler configured is used for full paired accuracy and timing controls.

All-rank summed kernel time ranges from 575.37 ms (rank 0) to 833.23 ms
(rank 2), with 14,608--14,688 kernel events per rank. This is visible rank
skew in the profiled window, not a claim about steady unprofiled imbalance.
Use all eight reports; do not generalize rank 0's shares to the whole run.
Profiler instrumentation/stop skew is an unresolved contributor.
