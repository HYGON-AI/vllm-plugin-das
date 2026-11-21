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

## MTP configuration boundary

At the inspected Core target, dictionary `hf_overrides` apply only to the
target. Actual `SpeculativeConfig` construction reloads the draft checkpoint:
the nested JSON override above gives target `indexer_n_heads=None` but draft
`indexer_n_heads=4` on the trained checkpoint. Copying the already overridden
target config into `hf_config_override` does not test this construction path.
The JSON CLI example alone therefore does not select dense MTP.

For explicitly dense target **and** draft, the official Python API composes a
callable override into both configurations. Put the callable at module scope
in an importable application module so spawned workers can deserialize it:

```python
def dense_hf_override(config):
    from vllm.transformers_utils.config import get_hf_text_config

    if config.model_type in {"qwen4_exp", "qwen4_exp_text", "qwen4_exp_mtp"}:
        get_hf_text_config(config).indexer_n_heads = None
    return config

# Pass through the public LLM/EngineArgs Python API:
# hf_overrides=dense_hf_override,
# speculative_config={"method": "mtp", "num_speculative_tokens": 3}
```

Core also probes this callable with a dummy `PretrainedConfig`; preserve
unmatched configurations unchanged. Keep the explicit extension selection.
This uses the official configuration composition, model constructors and MTP
implementation without changing checkpoint files or a shared config method.
The `speculative-config` CPU scenario exercises complete `ModelConfig` and
`SpeculativeConfig` construction for root and standalone text checkpoints,
including unchanged defaults, JSON target-only behavior, callable propagation
and pickling. Actual accepted tokens and model accuracy remain device gates.

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

## Trained MTP startup follow-up (2026-11-20)

A separate TP4/device-PLE run uses the explicit callable Python API above and
official MTP with three speculative tokens. The new installed CPU scenario
passes; the previous 20 affected CPU results belong to the unchanged production
payload. Target and draft checkpoint loading both complete their 131-shard
iteration, and workers report 56.7 GiB model memory each. The official
speculator identifies `HcuQwen4ExpMTP` and retains text-only draft inputs.

This run fails during target profiling/compilation before service health,
worker RPC inspection, Graph replay or generation. The first fatal stack
reaches the target HC projection's `vllm.rocm_unquantized_gemm`, whose compiled
input has K=10240 and weight shape `[336, 10240]`, and reports
`HIPBLAS_STATUS_INTERNAL_ERROR`. A later GDN prefill warmup reports
`hipErrorNoBinaryForGpu`; its timing does not establish that it caused the
earlier GEMM failure. Preserve the allocator warnings and both diagnostics;
the underlying memory/driver/kernel cause is not established.

A separate process on the same gfx938 device executes Torch linear and the
installed official GEMM at K=10240/N=336 and M=1/4/16/64/1024 against a nonzero
analytic BF16 reference. All ten comparisons and three changing-input Graph
replays at M=1024 pass exactly. This does not reproduce the loaded model's
memory state, establish its actual profiling M, or resolve the startup failure.

HumanEval and drafted/accepted counters are **NOT_RUN**, rather than zero-score
or MTP success. No trained output comparison with the non-MTP run is available.
The server exits1 with tracker0, no controller cleanup signals and no owned
survivors; framework worker SIGTERM remains. All 14 distribution RECORDs,
source controls and the complete 146-file checkpoint double-read before/after
seal verify unchanged. Evidence:
`evidence/qwen4-dense-full-checkpoint-20261120-r3-mtp/audit.json` and
`evidence/qwen4-dense-mtp-preflight-20261120/`.

Keep Draft. This follow-up adds the configuration regression and documents the
official API boundary; production loader, activation, Core and FlashAttention
bytes remain unchanged. MTP execution/accuracy and the previous gates are open.


## Observed profiling follow-up (2026-11-20)

A separate diagnostic run retains the same trained checkpoint, TP4, device PLE,
MTP3, max batch1024 and default Graph/prefix/chunked configuration. Its public
worker hook observes the installed profile and ROCm GEMM code with Python
profiling events; it does not replace those functions or alter tensors. The
four workers confirm the actual target/draft classes and dense overrides before
profiling. Each records 102 successful HC calls, with first input `[1024,10240]`
and weight `[336,10240]`, contiguous BF16 and 256-byte-aligned pointers.
All 675 observed GEMM calls per worker return normally, with zero observer
errors, allocator retries or allocator OOMs. Physical HCU memory peaks at
85,365/85,401/85,577/85,127 MiB on cards0–3. These are external card readings;
the SMI process query does not establish per-process attribution.

The global Python profiler changes compilation timing substantially: backbone
compile takes about271s. Target/draft prewarming and Graph capture finish, but
the API misses the original900s startup deadline. The controller retains that
failure and sends its original graceful interrupt; HumanEval, served probes and
drafted/accepted counters are **NOT_RUN**. Server exit is -2, tracker0, with no
controller-forced cleanup or survivors within the original15s grace; framework
worker SIGTERM and nanobind diagnostics remain. The 14 RECORDs, control sources
and all146 checkpoint files verify unchanged before/after.

This is diagnostic evidence, not a production fix, performance result or proof
that the earlier HIPBLAS failure is resolved. Keep the original failed run and
its unknown root cause. Evidence:
`evidence/qwen4-dense-full-checkpoint-20261120-r4-observed/audit.json`.


A follow-up uses Python3.12 local `sys.monitoring` events on the same installed
profile/GEMM code, preserving callable identity and original returns/exceptions.
Six observer CPU checks and installed event registration/cleanup pass. The same
trained configuration again completes target/draft weight loading, then fails
before health. Dynamo takes7.8–8.3s and backbone compilation about64s; these
instrumented timings are diagnostic, not performance qualification.

The first HC call on each rank has actual input `[1024,10240]` and weight
`[336,10240]`, contiguous BF16, offset0 and pointer alignment256. Rank0's call
raises `HIPBLAS_STATUS_INTERNAL_ERROR`; the other three HC calls return normally
before their profile scopes unwind. Each rank's preceding110 GEMMs return
normally, with no observer error. At the HC call, allocated memory is
61,171,682,816 bytes/rank and reserved memory is
92,547,317,760 / 92,274,688,000 / 92,274,688,000 / 92,652,175,360 bytes
for TP0/1/2/3. The observed allocator counters still report zero retries/OOMs,
while the complete server log retains769 allocator allocation-failure warning
lines. Do not equate those two measurements or discard either.

External card0–3 memory peaks at94,107/94,104/94,104/94,116 MiB. Those sampled
card totals do not establish the allocation responsible for the error, continuous
peak usage or a unique root cause. The later GDN no-binary warning from the
original failed run does **not** recur here.

HumanEval, served probes, accepted-token metrics and runtime Graph replay are
**NOT_RUN**. Server1/tracker0 exit within the original15s grace with no controller
forced cleanup or survivors. All14 RECORDs, control sources and146 checkpoint
assets pass full before/after verification. This reproduces the startup error
with measured loaded-memory/layout evidence; it does not fix it. Keep Draft and
retain the default CPU-offload, raw-format/lifecycle and wider model gates.
Evidence: `evidence/qwen4-dense-full-checkpoint-20261120-r5-local-events/audit.json`.

## Compiler-option diagnostic and device-memory change (2026-11-21)

The R6 diagnostic uses the public CLI compilation configuration with explicit
`combo_kernels=true, benchmark_combo_kernel=false`. Explicitly preserving the
first flag matters: this Core injects both defaults only when neither key is
present. Installed public CLI/EngineArgs parsing and actual four-worker RPC
configuration confirm the pair. Production payloads remain Core `c440c089`,
Adapter `d9bb7fa5`, Plugin `1fba2bc1`; no attention or native source was changed.

This allocation reports 147,440 MiB total memory per card (about144 GiB), whereas
the earlier failed allocation reported about96 GiB. R6 therefore does **not**
isolate the compiler option as the cause of the earlier HIPBLAS failure. It
retains TP4, device PLE, trained target/draft, MTP3, default Graph, prefix and
chunked prefill, the900-second health deadline and15-second exit grace.

All675 observed GEMMs, including102 HC calls per rank, return normally. There
are no observed allocator warning lines or HIPBLAS internal errors in this run.
The service becomes healthy, arithmetic passes13/13 and prefix probes3/3 with
3200 hit tokens. Four worker traces each contain59 Graph-launch events; these
alone do not qualify changed-input replay or the complete mixed-batch matrix.
MTP counters record3738 drafted and3645 accepted tokens (97.51%). HumanEval0–31
passes0/32 raw and32/32 after isolated normalization,4811 completion tokens,
without truncation. The raw code-fence syntax failures remain failures. Compared
with the earlier non-MTP run,29/32 contents are identical; both the hardware and
runtime settings differ, so this is not a matched MTP-only comparison.

Server0/tracker0 exit within15 seconds without controller-forced cleanup or
survivors. Framework worker SIGTERM, EngineDeadError and nanobind diagnostics
still fail the existing lifecycle gate. All14 RECORDs, control sources and the
146-file/237,133,280,448-byte checkpoint verify unchanged before/after. Keep
Draft: raw format, normal lifecycle, default compiler, default CPU PLE and the
wider accuracy/performance gates are still open. Evidence:
`evidence/qwen4-dense-full-checkpoint-20261120-r6-no-combo-bench/audit.json`.

A fresh default-compiler control is prepared for the same physical cards0–3,
with the same observer and serving settings, independent cold caches and port.
Other clients occupied those cards immediately after R6 teardown. The control
must satisfy the unchanged85%+4GiB memory admission before launch; unavailable
devices provide no default-configuration result. Adapter154 has since merged
into a newer main; its CI and payload are separate from this frozen runtime.
