# Optional Qwen4 dense checkpoint loading (vLLM 0.33)

The official AMD Qwen4 decoder already constructs `Qwen3NextAttention` when
`indexer_n_heads` is `None`, including the `indexed_attention` and
`qwen_sparse_attention` layer types. Keep that constructor and its cache-scale
mapping. The old wrapper for `full_attention` does not match this interface.

When reusing a QSA checkpoint for this deliberately different dense mode,
unused language-model/draft `self_attn.indexer` weights must be discarded before
the official loader traverses the module tree. The optional
`qwen4-dense-qsa-checkpoint` capability provides only that boundary. It publishes
lazy subclasses through the public model registry, preserving architecture names
and automatic MTP selection. It inherits constructors, forwards, compilation
annotations and the official multimodal processor without replacement.

Select the extension explicitly, using the Adapter's existing configuration:

```bash
export HCU_ADAPTER_VLLM_REQUIRED_EXTENSIONS=das_hcu_qwen4_dense
# For a multimodal/root Qwen4 config, add to the normal vllm serve arguments:
# --hf-overrides '{"text_config":{"indexer_n_heads":null}}'
# For a standalone text config instead:
# --hf-overrides '{"indexer_n_heads":null}'
```

Apply nested overrides through the public `ModelConfig`/CLI path. Calling the
lower-level config reader with a nested replacement dict has different semantics
and is not a test of CLI overrides. A root-only override on a multimodal config
does not disable the text indexer. The official model-config hash changes when
the real text configuration changes; no private environment switch is added.

Installation/discovery alone does not publish these subclasses. Activation
requires HCU and the epoch-zero vLLM 0.33 release line. Foreign model owners are
preserved, partial publication is fatal, and later ownership drift is rejected
without registry repair. Process locks/failure handling use Adapter's shared
`ProcessLocalLifecycle`; a healthy fork must reactivate in the child, while an
interrupted publication cannot be resumed there.

With QSA still enabled, the exact input iterator reaches the official loader.
In dense mode only anchored language-model/draft indexer paths are filtered,
lazily; other unknown weights and visual indexers retain official errors.
The checkpoint and installed Core/FlashAttention files are never rewritten.

Validation entry point: `tests/vllm033/test_qwen4_dense_cpu.py`. Its official
constructors use explicit CPU leaf doubles, and its real loaders use synthetic
CPU module trees. It also tests actual public config parsing and installed entry
points. These checks do not establish full checkpoint construction, GPU kernels,
Graph replay, mixed decode/prefill, MTP acceptance, model accuracy or performance.
Those remain required device/end-to-end gates before this capability is Ready.
Disabling QSA changes the model's attention semantics; it is an explicit mode,
not a transparent accuracy-preserving optimization.

The installed CPU qualification for runtime `1fba2bc1` (Core `c440c089`,
Adapter/native `d9bb7fa5`) passes the complete portable selection: 1408 cases,
including the 15 new CPU scenarios, plus 11 fresh public-selector processes.
There are no failures/errors/skips; 46 HCU cases are deliberately deselected.
All 18 owned distribution RECORDs are unchanged before/after the run.
Real checkpoint-index inspection selects 39 indexer names and preserves all
152049 other names. Clean-environment probes also retain the official MTP
architecture mapping for multimodal and standalone text configurations.
Configuration probes use `tokenizer_mode="skip"`; they do not qualify actual
multimodal processing. The device/model gates above remain open.


## Native startup follow-up (2026-11-20)

The original 18-distribution environment failed during default router warmup:
installed CUDA CuTeDSL made an SM90-style warmup appear available on gfx938,
then its `quack` import failed. No loader probe or generation was reached in
that attempt. Earlier miniature fixtures also retain their pre-loader failures:
all-attention layers are incompatible with the hybrid state manager, and the
Transformers default output gate `None` is rejected by the target GDN.

A separate environment omits the four optional CUDA CUTLASS distributions and
their package path entry. Core, Adapter, Plugin and FlashAttention code/native
payloads stay identical. All 14 retained owned distributions are reverified;
this is a different dependency cohort and does not qualify the original one.
The miniature uses one dense attention layer, one GDN layer, and the local
trained checkpoint's `sigmoid` output gate; weights remain synthetic.

The separate HIP dependency cohort completed both synthetic arms with exact saved token/logprob parity.
Default Graph and prefix caching remain enabled. Graph capture is recorded
separately from replay-trace qualification. Full trained-checkpoint loading,
MTP, mixed-batch coverage, HumanEval and performance remain unqualified.
Both synthetic arms also report nanobind leaks and framework forced cleanup at
shutdown. Their lifecycle gates fail despite exit0 and no surviving owned process.
Keep Draft; earlier validation failures and model gates are unchanged.

## Trained-checkpoint startup follow-up (2026-11-20)

The HIP dependency cohort was also exercised with the local trained
`Qwen3.8-Flash-Next-FP8-Channelwise` checkpoint, whose architecture is Qwen4,
on four gfx938 devices. The 20 affected installed CPU checks pass. The first
full-checkpoint startup retained default CPU-offloaded PLE, dense attention,
default Graph, prefix caching and chunked prefill.

That startup fails in the inherited official PLE constructor at
`torch.empty(device="cpu", pin_memory=True)`, before checkpoint weight loading
or generation. The BF16 PLE allocation is `[80000384, 160]`,
25,600,122,880 bytes per rank and 102,400,491,520 bytes across four ranks.
The HIP OOM at this host allocation does not establish exhausted device memory
or a failure of the Plugin's indexer filter. The underlying driver cause has
not been established. HumanEval is **NOT_RUN** for this attempt.

All 14 retained distribution RECORDs and the checkpoint's full before/after
double-read seal pass. The service exits 1 with no owned survivors or controller
cleanup signals, while the framework's worker SIGTERM diagnostic remains in the
log. This is a failed startup, with no normal-serving lifecycle qualification.
Evidence: `evidence/qwen4-dense-full-checkpoint-20261120-r1/`.

The official `--engram-config '{"cpu_offload":false}'` selects a separate
device-resident PLE implementation. A result from that configuration cannot
qualify the failed default CPU-offload path. Neither configuration changes
Core, Adapter, FlashAttention or the checkpoint bytes.

The separate TP4 device-resident run completes all 131 weight shards and uses
56.02 GiB per rank. Actual worker inspection confirms the Plugin conditional
model class, `indexer_n_heads=None`, the checkpoint's `sigmoid` output gate,
and one GPU-resident BF16 PLE table of the shape above on each rank. Official
Triton FP8 MoE and default FULL/PIECEWISE Graph remain enabled. Compilation
emits allocator OOM warnings but subsequently completes; the warnings are
retained rather than classified as a fatal startup result.

Arithmetic passes 13/13, prefix probes pass 3/3 with 3136 cache hits, and
concurrency probes cover 1/4/8 requests. The four worker traces contain
3162/3163/3163/3163 GPU kernel events and 53 Graph launches each. This observes
Graph execution; it does not close the separate changed-input replay and
mixed decode/prefill matrix.

Original HumanEval0–31, temperature0/seed0/max2048/thinking-off, completes with
**raw 0/32, normalized 32/32**, 4815 completion tokens, and no truncation.
All 32 responses start with a Python Markdown fence; raw scoring retains its
syntax failures, and both score variants execute in the unchanged isolated
uid65534 evaluator. This candidate has no matched trained QSA baseline, so the
result does not establish QSA-equivalent semantics or a numerical regression
comparison.

The server exits0 within the original 15-second grace without controller
cleanup signals or survivors. Nevertheless, framework worker SIGTERM,
`EngineDeadError` and nanobind leak diagnostics keep the original lifecycle
gate **FAIL**. All 14 distribution RECORDs, source controls and 146 checkpoint
assets (237,133,280,448 bytes, full double-read before/after) verify unchanged.
Evidence: `evidence/qwen4-dense-full-checkpoint-20261120-r2-device/audit.json`.

**Keep Draft.** Default CPU offload, original raw-format/lifecycle findings,
MTP acceptance, the remaining replay/mixed-batch matrix and performance gates
remain open. This documentation/skill follow-up changes no production code.
