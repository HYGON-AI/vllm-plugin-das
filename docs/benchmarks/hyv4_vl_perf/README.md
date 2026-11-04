# HYV4VL performance validation artifacts

See [the report](../../hy4_vl_performance.md) for hardware, flags, limitations and
reproduction. No dataset images or model weights are included here.

- `baseline-*` and `optimized-*` timings: independent original (`bd1545a`) and
  optimized (`edeb432`) server runs. `*-metadata.json` records code/data hashes
  and cache settings. QA suites contain 64 requests per concurrency setting;
  the small fixed-length caption suites contain four requests each.
- `e2e-comparison.json`: changes computed from those timing summaries.
- `vit-comparison.json`: isolated TP1 vision forwards with real weights.
  `baseline_profile.txt` is the baseline TP1 profiler output.
- `tp8-vision-final.json`: isolated TP8 vision forwards, final per-image strategy.
  These vision benchmarks use native RMSNorm; they are separate from the server
  runtime's HCU RMSNorm path.
- `tp8-packed-numerical-probe.json`: **rejected** TP8 image-packing candidate.
  Its numerical differences motivated preserving per-image GEMM shapes/order;
  it is not the final implementation or a final performance result.
- `mrope-validation.json`: 72 native/fused exact-equality cases.
- `optimized-smoke-results.json`: seven constrained API checks plus one photo
  for manual review (`correct: null`).
- `baseline-accuracy-summary.json`: earlier normal-cache 765-item evaluation.
  `optimized-accuracy-summary.json`: optimized evaluation with processor/prefix
  caching disabled. `correctness-comparison.json` compares those two runs;
  their cache settings differ.
- `matched-baseline-accuracy-{summary,manifest}.json`,
  `baseline-control-metadata.json`: complete 765-item control restoring original
  forward methods on the same loaded weights, using the optimized run's cache
  settings and performance/warmup sequence. Its checkout revision denotes the
  loaded model code; the metadata explicitly identifies the original runtime
  forward methods. `baseline-control-*` timing files verify that restored mode.
- `matched-correctness-comparison.json` and `matched-predictions.jsonl`: the
  matched control versus optimized full-dataset comparison and all paired raw
  predictions, image hashes and correctness labels.
- `operator-audit.json` and `operator-audit-metadata.json`: sampled vision and
  live M-RoPE checks on the four historically changed items, on actual loaded
  TP8 workers. Exact sampled operators do not prove all dataset predictions
  must agree or establish the cause of run-level differences.
- `hyv4_perf_worker.py.txt`, `run_operator_audit.py.txt`,
  `run_baseline_control.py.txt`: diagnostic harness snapshots for the last two
  checks. These are temporary local development tools, not production plugin
  code or serving flags. The audit server was bound to localhost only.

The dataset/scorer revision and evaluator hash are recorded in accuracy
manifests. Use the evaluator in `tools/` to retain complete API responses for a
new run. Original full responses and logs from these runs remain under
`/root/hy4-vit-perf` on the test machine.
