# AITER CK GDN Prefill Backend Design

## Objective

Add a shape-gated AITER CK fused GDN prefill path without weakening the
existing HCU provider controls or vLLM execution modes. The new path must be
numerically compatible with the current Qwen GDN contract, work under
`FULL_AND_PIECEWISE` for supported graph shapes, and demonstrate real-model
correctness and end-to-end performance against the current plugin before it is
eligible to replace the current mixed prefill pipeline.

## Existing behavior

With custom operations enabled and `hcu-aiter` selected, prefill currently
uses the upstream Triton/FLA frontend and resolves operators independently:

- Triton/FLA: chunk-local cumsum, KKT, and triangular solve;
- BoltOps: standalone `recompute_w_u_fwd`;
- AITER HIPC: chunk delta-H and output-O.

`VLLM_HCU_USE_CUSTOM_OPS=0` is the existing master override. For GDN it
forces the established BoltOps-oriented fallback behavior. The new fused path
must respect that behavior and must not introduce another environment switch.

## Selected design

### Ownership and selection

The fused path is selected at the `ChunkGatedDeltaRule` boundary, where the
complete prefill contract is available. It does not pretend to provide a
standalone `recompute_w_u_fwd` implementation and does not alter the existing
per-operator fallbacks.

The candidate is considered only when all of the following are true:

- `VLLM_HCU_USE_CUSTOM_OPS` is enabled;
- the requested HCU GDN prefill provider is `hcu-aiter`;
- AITER exports ABI-compatible `ck_gdn_prefill` and workspace-size entry
  points;
- key and value dimensions are both 128;
- the local value-head/key-head ratio is exactly 1, 2, or 4;
- input dtypes, state dtype, layout, and metadata satisfy the audited CK
  contract.

Any failed capability or input check uses the current complete prefill path.
Explicit `triton` and `hcu-boltops` selections retain their current meaning.
Provider runtime errors after CK has been selected are propagated rather than
silently changing kernels.

### Tensor semantics

The adapter maps vLLM's `[1, tokens, heads, 128]` packed tensors to CK's
`[tokens, heads, 128]` view without copying. It preserves normalized q/k,
log-decay g, sigmoid-applied beta, scale, FP32 initial/final state, output
dtype, and `output_final_state` semantics. It copies into `core_attn_out` when
vLLM supplies a backing buffer.

The CK invocation uses `use_exp2=False`, `transpose_state=True`, and
`qk_l2norm=False`, matching the already-preprocessed Qwen GDN path verified by
the operator probe.

### Workspace lifecycle

Workspace size is derived from AITER's workspace-size entry point. Buffers are
cached by device and required capacity and reused across layers and calls.
Growth is allowed only outside graph capture. A call requiring allocation or
growth during capture falls back before capture begins or fails closed with a
clear compatibility error; it must not allocate inside a captured region.

The cache must not retain one maximum-size workspace per model layer. Its
ownership and concurrency rules must prevent two overlapping calls from
mutating the same live workspace.

### Graph and variable-length behavior

`FULL_AND_PIECEWISE` remains enabled throughout validation.

For a single packed sequence whose metadata represents `[0, tokens]`, the
adapter passes `cu_seqlens=None` to CK. This avoids the host/device validation
inside the current AITER wrapper and is graph-safe. Capture/replay must keep
addresses stable, react to changed input contents, and match eager CK.

Eager packed-varlen calls may use CK with device `cu_seqlens`. During graph
capture, packed-varlen calls use the current prefill path because AITER 0.1.7
attempts a CPU-to-HCU copy when `cu_seqlens` is present. No host read or stream
synchronization may be added to determine sequence contents during capture;
the decision uses metadata shape and framework-owned invariants.

### Logging

Log once per process when fused CK is selected and once for each stable
fallback reason. Logs distinguish the frontend/backend decision from actual
execution and include enough information to prove CK was reached during model
validation. They must not claim AITER lacks recompute code; its recompute
kernel is internal to the fused pipeline.

## Failure handling

- Missing AITER symbols or an unsupported audited signature: use the existing
  prefill path and log the capability reason.
- Unsupported dimension, ratio, dtype, or layout: use the existing path and
  log the input reason once.
- Packed-varlen graph capture: use the existing path with an explicit graph
  compatibility reason.
- Provider execution failure after selection: raise the provider error.
- Invalid or insufficient caller output/state storage: fail before launching.

Qwen3.5-27B remains on the current path because its local `Hv/Hqk` ratio is 3.

## Verification

### Automated tests

Contract tests cover master-switch behavior, explicit provider selection,
missing/signature-drifted AITER, every shape gate, output-buffer handling,
workspace reuse/growth, graph allocation prohibition, and fallback logging.

Native operator tests cover lengths 1, 16, 63, 64, 65, 127, 128, 256, 512,
1024, 2048, and 4096; head ratios 1, 2, and 4; zero and nonzero initial state;
fixed and packed-varlen inputs; final-state behavior; and finite results.
They compare output and final state with official FLA and the current mixed
pipeline and record absolute error and normalized RMSE.

Graph tests cover fixed single-sequence capture/replay with stable addresses
and changed inputs. Packed-varlen graph is a required fallback test until the
AITER wrapper becomes graph-safe.

### Required model validation

Use `/model/Qwen3.5-35B-A3B` in BF16 with TP4 on one eight-card node. Keep the
same model, tokenizer, request set, memory settings, attention/MoE providers,
and `FULL_AND_PIECEWISE` compilation mode for both builds. Qwen3.5-0.8B TP1
may be used as an additional smoke test but does not replace the 35B row.

The baseline is an isolated wheel built from the pre-integration commit. The
candidate is an isolated wheel built from the CK integration commit. This
avoids adding a benchmark-only runtime switch and compares the exact old and
new default `hcu-aiter` behavior.

Both builds must demonstrate:

- actual provider logs (mixed pipeline for baseline, CK for candidate);
- successful `FULL_AND_PIECEWISE` capture and replay;
- deterministic greedy output parity on a fixed prompt corpus;
- dynamic prompt lengths including chunk boundaries and long prefill;
- concurrent requests;
- three repeated shared-prefix requests with correct output and measurable
  prefix-cache hits;
- clean API/EngineCore/worker shutdown when the API PID is signaled directly.

### End-to-end performance

Measure the baseline and candidate on the same idle node with identical server
and client settings. Separate cold startup/compile time from steady state.
Use at least three independent repetitions and report distributions for:

- short prefill and decode;
- 1K, 4K, and the largest practical prefill for the model configuration;
- single-request latency and concurrent throughput;
- TTFT, TPOT/ITL, end-to-end latency, request throughput, token throughput,
  error rate, and peak HCU memory.

The fused path is eligible for the selected shapes only if correctness gates
pass, it produces a repeatable improvement of roughly 5% or more on the
prefill-critical workloads, and no required workload shows a repeatable
unexplained regression. Raw results are retained even when the performance
gate fails; a faster microbenchmark alone is insufficient.

## Deliverables

- Narrow fused-prefill adapter and workspace manager.
- CPU contract and native HCU operator tests.
- Isolated baseline and candidate wheels with source SHAs and checksums.
- Model validation logs and raw performance results.
- Updated provider documentation describing supported shapes, graph behavior,
  fallback conditions, and the remaining AITER varlen-graph limitation.
