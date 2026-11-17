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
Policy is frozen/reported before model loading; later changes and inherited
active fork state are refused. Both catalog choices remain candidates.

Use explicit stock/triton configuration for A/B work: the old private experimental
switch is no longer read, and an unset configuration now selects the optimized
composition. Normal safetensors loading does not require preload. For IPC, derive
an honest current identity and follow the existing publication contract; do not
rewrite old hashes to reuse a producer.

## Execution contracts

| Optimization | Preserved boundary |
|---|---|
| Local-page update | Same FP32 sentinel, Top16 ordering and zero-end masking; write only selected local pages. |
| Q conversion | BF16 input, correctly rounded FP32 division, identical E4M3FN bytes. |
| Fused cache store | Gather order K512/V512/index-K128, independent main/index physical slots, -1 padding, native token-major strides, unchanged descales and FP8-index clamp. |
| Pack workspace | Same-stream BF16 inference storage, geometric growth/reuse, original inputs untouched. |
| Metadata reuse | Reset every forward; hold the original tensor references (even if wrapper fields change) and bind device/address/shape/stride/dtype/version. Inference tensors without versions retain object identity instead of merging fresh wrappers. |
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
creating a new variant. Static frame admission precedes MSA metadata mutation,
so a rejected concurrent frame cannot overwrite the active owner.
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
