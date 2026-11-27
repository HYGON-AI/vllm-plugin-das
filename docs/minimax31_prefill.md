# MiniMax PCP8 prefill fusion

This follow-up builds on the merged MiniMax PCP8/EP8 path. Adapter owns the
configuration/catalog/veto; Plugin owns the implementation. The admitted scope
remains eager TP1/PP1/DP1/PCP8/EP8/DCP1 text inference with the audited native MSA,
main FP8 KV and the existing BF16 or FP8 side cache. No model precision, weight
packing, placement hash or IPC validation is relaxed.

## Configuration

The paired Adapter defaults `ops.vllm_minimax_prefill` to `triton` within this
model path. Use `stock` in a fresh process for the unfused comparison:

```json
{"version":1,"ops":{"vllm_minimax_prefill":"stock"}}
```

Select that file through the existing `HCU_CONFIG` interface. The existing
`HCU_DISABLE_OPS=vllm_minimax_prefill` startup veto wins over configuration.
There is no new private environment variable or per-feature debug switch.
Policy is frozen/reported before model loading. Planner/profile entry checks
reject later changes before mutating inputs; per-layer `enabled()` reads only the
cached composition after a PID check. There are no per-feature selections.
Inherited active fork state is refused; use the existing upstream
`VLLM_WORKER_MULTIPROC_METHOD=spawn` startup setting. Both catalog choices remain candidates.

Use explicit stock/triton configuration for A/B work: the old private experimental
switch is no longer read, and an unset configuration now selects the optimized
composition. Normal safetensors loading does not require preload. For IPC, derive
an honest current identity and follow the existing publication contract; do not
rewrite old hashes to reuse a producer.

## Execution contracts

| Optimization | Preserved boundary |
|---|---|
| Local-page update | Same FP32 sentinel, Top16 ordering and zero-end masking; strides/page count are non-specialized runtime scalars. Only the local-page setting remains constexpr. |
| Q conversion | BF16 input, correctly rounded FP32 division, identical E4M3FN bytes; element count is a non-specialized runtime scalar, only tile size remains constexpr. |
| Fused cache store | Gather order K512/V512/index-K128, independent main/index physical slots, -1 padding, native token-major strides, unchanged descales and FP8-index clamp. |
| Pack workspace | BF16 K512/V512/index-K128 scratch is reserved by the existing local memory-profile entry. Inference never grows it; unprofiled streams or row budgets refuse. The borrowed view must be consumed by the synchronous, same-stream PCP gather before reuse; inputs remain untouched. |
| Metadata reuse | Reset every forward; retain only the three source tensors, not their metadata wrapper or unrelated tensors. Both stock and optimized keys bind physical identity/version; only the optimized key shares fresh wrappers. Inference tensors without versions retain tensor object identity. |
| Scale guard | First binding retains finite-positive host validation; subsequent same-cache checks compare device values with private snapshots on the bound stream. Geometry/stream changes refuse. Failed device assertions require process restart. |
| Static selector | Original immutable tables and logical-ID checks; identical uint64 hash. N/REAL/LAYER remain runtime scalars to avoid per-layer/row JIT variants. No first-forward kernel rebinding. |

The fused store is scoped to native MSA cache views. Other supported MiniMax
paths retain their existing composition. The original layer/stream, layout,
source ABI and same-generation IPC guards continue to apply.

## Verification

CPU contracts cover real configuration/veto/default selection, policy drift and
fork rejection, metadata storage/version/forward ownership and existing FP8 guards.
HCU checks compare exact FP8/cache bytes (both uint8 and FP8 backing), real strided
inputs and independent Python uint64 selector outputs for all 57 MoE layers,
0/1 boundaries and nine-replica tables. Selector compilation has one variant in
that check; the cache-store check also varies rows and page capacity without
creating a new variant. Local-page and Q checks also vary geometry/element count
with at most one variant per fixed setting. Main-cache finite overflow values
(up to BF16 65504 with several scales) are compared byte-for-byte with the official
`triton_reshape_and_cache_flash`, in addition to the independent small-value oracle.
The real `is_profile=True` runner boundary (rather than the later shared dummy
preparation) reserves scratch as normal mutable storage, even under inference
mode. Tests cover later non-inference writes and same-stream scratch consumption. Static frame admission precedes MSA metadata mutation,
so a rejected concurrent frame cannot overwrite the active owner. If MSA setup
throws after admission, the existing `execute_model` / `_dummy_run` failure guard
releases the host frame and poisons further model execution until restart; it does
not roll back to a seemingly healthy engine.
Destructive scale mutations run only in isolated probe processes;
an arbitrary crash is not a passing result: the expected assertion kernel and
message must be identified.

```bash
# CPU: hide devices and use the paired source/install paths.
HCU_ADAPTER_HOST=none HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= \
  python -m pytest -q tests/vllm033/test_minimax31_prefill_cpu.py
# HCU component checks; this is not a full-model accuracy test.
HCU_ADAPTER_HOST=none python -m pytest -q tests/vllm033/test_minimax31_prefill_gpu.py
# Run each destructive kind in a separate process; invalid kinds must refuse.
HCU_ADAPTER_HOST=none python tests/vllm033/minimax31_scale_guard_probe.py valid
```

Probe kinds are `valid`, `fill`, `zero`, `nan`, `native`, `event`. A valid probe
must exit 0. Invalid probes must identify the scale assertion and exit nonzero;
on the tested HCU runtime the assertion is reported as SIGABRT with the assertion
kernel/message in the driver dump. Never run destructive probes inside a model
server or count a missing dependency/unrelated VM fault as successful rejection.

## Input-workspace benchmark evidence

The supplied seven-feature candidate was measured on one eight-card BW1100 node
with about 96 GiB/card, full 60-layer W4A8/W8A8 PCP8/EP8, eager FP8 KV/index,
static136/17 and matching 493782-token KV capacity. Both arms completed 53 cases
and 1347 timed requests. Under mean TTFT <2 s, peak cached-input throughput was
202274.36 -> 225036.71 token/s (+11.25%); with a 1950 ms margin it was +7.49%.
The 2 s winner was close to its limit (1997.87 ms). Requests hit a 131072-token
prefix, add 16384 tokens and emit one token; new-input throughput is 1/9 of the
cached-input number. Earlier six-feature regressions and remaining statistical
regressions are retained in the external experiment archive.

These measurements precede this port onto the merged baseline and its ownership
hardening. They motivate the candidate, not a fresh throughput result for this
MR. Current component checks do not establish full-model quality, long-output
TPOT, or a same-node SGLang speedup. A later 144 GiB/card run used a different
node/warmup protocol and is not a memory-only A/B.

Archive identity: `benchmark_fixed_evidence_20261117.tar.gz`, SHA256
`bf05dd354f00543f6962d18db7c8d8e0a8ceb2529e38b527d97a5f9d5cb90f03`.
Full request/timing/source manifests remain in the independent experiment archive;
no personal account, internal host address or private launcher is required by the
public commands above.

## Review disposition and result identity

The paired Adapter prerequisite and merge order are recorded in
[minimax31_configuration.md](minimax31_configuration.md). The admitted-path default
remains `triton` under the requested deployment policy; both backends stay
`candidate`, and `stock` remains the explicit startup comparison. Component checks
and a default selection do not promote a backend to production.

The 2026-11-17 HumanEval thinking run used Adapter `1f643926` and Plugin `5e48369c`:
163/164 unique tasks completed, 156 passed, seven failed, and task 145 was stopped
at the user's request. There were 162 normal stops and one 16384-token truncation
(task 32); complete pass@1 is null. Failed IDs were 32/39/75/106/119/134/163.
This partial result predates the review fixes above and must not be relabelled as
current-head or complete-model acceptance. The stopped task is not resumed.

Scale mutation is a correctness failure while FP8 KV is bound. The optimized
same-stream device assertion can terminate the worker (SIGABRT on the measured
HCU runtime); restart the engine, never catch and continue with uncertain cache.
The explicit stock comparison retains its synchronous Python validation. Clean
forks before policy/device state can initialize independently, but inherited
active state is never reset to bypass ownership; use spawn workers.

## Installed-wheel boundary

The prefill implementation is packaged as
`hcu_vllm_das.parallel.minimax31_prefill_triton`. All five production import
sites use that package-relative module; it is included by the independent
Plugin project's existing setuptools namespace discovery. It does not require
the repository-root `triton_tmp_dir` directory on `PYTHONPATH`.

The wheel boundary test checks the exact module in the built wheel and its
RECORD-owned installed origin from an unrelated directory with isolated Python
paths and repository-root imports blocked. The portable source lane uses
import-only Torch/Triton stand-ins; a separate installed-wheel probe imports the
real runtime dependencies. Relocation preserves the kernel implementation bytes.

## DeepEP early-combine merge scope

Native Prefill early-combine is excluded from the MiniMax 3.1 integration.
The experiment in `d1ec968` added `prepare_intranode_combine` to the native
DeepEP extension and used it in an eight-rank component A/B with synthetic
expert compute. That experiment was based on reference source `5e424c0`.
The source matching the deployed DeepEP wheel (`g44d179`) is unavailable to
this integration, and the wheel lacks the required native entry point.
The experiment has no full-model benchmark result.

The merged C4 shared-expert down projection overlap uses the existing
combine interface in the deployed wheel.
