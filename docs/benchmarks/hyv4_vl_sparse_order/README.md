# HYV4-VL sparse-order precision evidence

Tested inference source: `47bbdcda999e2ca8c6099da72423fec2f6adc7b9`.
See [analysis and results](../../hy4_vl_sparse_order_precision.md) and
[normal server/client commands](../../hy4_vl_serving.md).

- `cause/`: native/canonical/stable selector intervention, full-logit and all-rank selector comparisons, fixed-Q/K/V attention replay, executed diagnostic scripts and regression logs.
- `fresh/`: fresh committed-source verification, full 765-question predictions and scorer summary, actual wrapper dispatch counters, 32/128-token captions, modified long-prompt QA, matched performance requests, HTTP smoke and passing release gate.
- `environment.json`: recorded software/hardware environment used by these runs.
- `history-notes.md`: pinned infrastructure skill sources and investigation notes.
- `external-tensor-manifest.json`: paths, sizes and hashes of host-retained raw tensor traces; binary tensors, dataset images and model weights are excluded from Git.
- `log-normalization.json`: original log hashes and whitespace normalization applied only to archived text copies.

Scripts preserve the actual test host's absolute paths. Dataset revision and
scorer provenance are in the accuracy manifest and preceding precision report.
The diagnostic launcher `cause/serve.sh` uses development RPC and Stage4Worker;
its extension is archived in `../hyv4_vl_precision/scripts/stage4_worker.py`.
Caption payload helpers are in `../hyv4_vl_ready_validation/scripts/run.py`;
restore those helpers to the script paths or adapt imports before replaying.
Use the original pinned dataset/checkpoint and the exact recorded modes.

The root-cause selector ablation ran against baseline `7b5b973` before the
source fix. Replaying that ablation on the fixed source with canonicalization
still enabled would mask the native-order control. The temporary source-install
probe was eager-only; the `fresh/` run instead starts committed `47bbdcd`
normally, captures new graphs and records all-rank source flags. The native
long-QA control disables the order fix only in eager mode, then restores it.
The performance comparison enables the fix in both graph and eager modes.

`fresh/release_gate.py` checks saved evidence without loading the model. Full
trace comparisons additionally need the host-retained tensors. `SHA256SUMS`
covers every archived artifact except itself. Logs are force-added because the
repository ignores `*.log`. These measurements do not claim all questions are
correct, universal determinism, or an additional 50% gain over prior graphs.
