# Qwen4 optional dense-QSA checkpoint boundary (0.33)

Inspect the actual AMD decoder before carrying `VLLM_HCU_DISABLE_QSA` wrappers.
The 0.33 decoder already selects dense `Qwen3NextAttention` when the text config's
`indexer_n_heads` is `None`; current aliases include `indexed_attention` and
`qwen_sparse_attention`, rather than the old wrapper's `full_attention`.
Use public `ModelConfig`/CLI `hf_overrides` (nested `text_config` for a multimodal
root), verify the resulting typed text config and official config hash, and
retain the official `_qsa_layer_ids`/cache-scale mapping. A lower-level config
reader's replacement-dict failure does not establish a CLI override defect.

QSA checkpoints still contain disabled indexer weights. If a Plugin loader
adapter is needed, restrict it to explicit dense-mode language-model/draft
indexer paths and stream into the official loader; preserve QSA passthrough and
errors for unrelated/vision weights. Keep original constructors and multimodal
processor inheritance, and verify automatic MTP architecture selection. The
optional capability and remaining validation gates are documented in
`docs/qwen4_dense_checkpoint.md` in the Plugin repository. CPU constructor/loader
contracts do not establish device, mixed-batch, MTP or model accuracy acceptance.

Exercise complete `SpeculativeConfig` construction before claiming an override
reaches MTP. At the inspected 0.33 target, JSON/dict `hf_overrides` remain
target-only while callable overrides are composed into the reloaded draft.
Manually transforming a copy of the already overridden target misses this
boundary. For explicitly dense target and draft, use the official Python API
with an importable module-level callable, preserve Core's dummy/unmatched
config probe, and verify root/text configs, pickling and the actual draft class.
Keep this separate from the JSON CLI path and require real drafted/accepted
tokens before qualifying execution; see the repository's checkpoint document.

For native Qwen4 loader probes, keep the miniature configuration consistent
with the target's hybrid model and state manager. An all-attention fixture can
be rejected as having no Mamba layers before the loader is exercised. Inspect
the resolved GDN output gate too: the tested Transformers config defaults
`output_gate_type` to `None`, while the target GDN constructor requires an
explicit supported activation. Use the intended checkpoint's value for a
representative fixture; do not rewrite checkpoint configs or normalize production
semantics to make a synthetic test run. The local trained checkpoints use
`sigmoid`; a synthetic `silu` run only covers that synthetic configuration.

Bind native evidence to the actual worker model class, attention implementation,
cache configuration and changed-input Graph execution. A loader probe should
show a real parameter update through the official loader and restore its original
bytes before generation. Keep constructor failures and lifecycle diagnostics;
synthetic output parity does not establish trained-checkpoint, MTP or HumanEval
acceptance.

Check the actual platform/warmup guard before installing a missing CUDA
dependency on HIP. Installed CUDA CUTLASS can make an optional warmup
appear available even when it cannot execute on HCU. A separate HIP-only
dependency cohort must retain the original failure and reverify all retained
distributions; it cannot qualify the original package combination.

For a trained checkpoint that fails in PLE construction, inspect the allocation
arguments before classifying a HIP OOM as exhausted VRAM. The official pinned
host implementation uses `torch.empty(device="cpu", pin_memory=True)` and can
fail before the Plugin's checkpoint filter runs. Record the table dtype and
ETP-local shape, total resident host footprint, available host memory and worker
cleanup. Free VRAM alone does not establish pinned-host capacity or the driver
failure's cause. Reuse prior matching allocation diagnostics where applicable.

When the target exposes `--engram-config '{"cpu_offload":false}'`, its official
device-resident PLE is a separate supported configuration to validate. Keep
Graph, prefix, chunked prefill and the original accuracy/exit gates; verify the
actual worker table device and dtype. Preserve the failed default-offload run
and mark CPU offload unqualified even if device-resident inference passes.
Do not replace pinned tables with pageable tensors or transfer older-version
offload environment switches without checking the installed allocation/UVA path.


When observing a loaded-model profiling failure, record actual worker/draft
classes, runtime tensor shapes/layout and allocator counters, and sample
physical device residency throughout the run. Keep SMI card totals separate
from per-process attribution. A global Python profiler can change compilation
and allocation timing substantially; a successful observed GEMM does not fix a
prior unobserved model failure. Preserve original startup/exit deadlines and
NOT_RUN results. If less intrusive evidence is needed, first validate local
code-event monitoring against the installed Python/runtime: preserve existing
tools, original returns/exceptions and callable identity, and release the
observer on success, exception and observer failure. Keep such instrumentation
in diagnostic worker hooks, separate from production model/attention code.
