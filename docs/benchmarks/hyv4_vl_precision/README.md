# HYV4-VL prefill precision artifacts

See [the investigation](../../hy4_vl_piecewise_precision.md) and
[public server/client commands](../../hy4_vl_serving.md).

The ablation runs the unguarded implementation at `ba44dc2` in a fresh,
profiler-free TP8 process. `stage4_worker.py` changes dispatcher permissions
or bypasses the PIECEWISE wrapper while preserving its padded descriptor.
`repro.py` repeats the three modes over the same 15 selected questions;
`snapshots.py` repeats the rank-0 sampled tensor captures for items 119/94.
Shape probes run on all eight ranks after the request-level ablations.
`install_source_guard.py` compiles the exact modified runner method into
that loaded process. The full accuracy pair and timing pair then execute
the source guard and a decode-only control using the same weights. A
separate fresh server loads the guard normally, without a worker extension,
and runs the public image smoke and the 15 selected questions.

The scripts are archival reproductions of the test host: repository, model,
dataset and output paths are absolute. Their test root is
`/root/hy4-perf-stage4`, the repository is `/tmp/vllm-plugin-das-pr`, and the
pinned data/scorer paths are documented in the serving guide. The diagnostic
server binds localhost and uses developer RPCs; ordinary serving commands
need neither the developer flag nor the worker extension. To repeat the
original ablation, use `ba44dc2`; apply `guard-source.diff` with `git apply --unidiff-zero` before installing
the source guard and running `formal.py`. Keep these steps sequential and
reset both multimodal caches when switching modes, as the scripts do.

`accuracy-comparison.json` and `paired-predictions.jsonl` retain the complete
765-item correctness and original-answer pair. `actual-dispatch-validation.json`
checks model-wrapper runtime modes on all ranks. Dispatcher-attempt counters
also contain the original padded proposal before the precision guard;
those proposals are not counted as actual replay. Python set ordering in
permission representations is normalized only when comparing per-rank
ablation counters; token counts and descriptor values are preserved.

The iHC/attention local shape probe tiles three captured effective rows to
M=22/43 and M=128 with the same effective prefix. The MLP/MoE shape probe uses
the complete captured 22-token valid input and actual padded tail. Neither
uses synthetic weights. Same-input probes are distinguished from the actual
layer-by-layer forward comparison. Snapshot JSON metrics cover rows 0,
middle and last; they do not claim a full-tensor error bound or logits parity.

The 40 raw `.pt` files (about 497 MiB) remain on the test host, not in Git;
`raw-tensor-manifest.json` records names, sizes and SHA-256 hashes. These
include eight sampled snapshot dictionaries and all-rank MLP input/output
captures. No dataset images, weights or profiler traces are added here.
Every committed artifact is covered by `SHA256SUMS`.
