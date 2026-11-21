# Runtime-SP dense MLP: retain the original TP weight layout

The previous candidate replicated both dense and shared MLP weights. The legacy
GLM4/HY3 ordinary dense MLP instead gathers token rows, computes with TP-sharded
gate/up and down weights, and reduce-scatters the partial outputs. Restore that
policy through the existing public Linear OOT classes. Shared experts retain
replicated weights and local tokens. Official MLP forward, activation, allocation,
checkpoint loaders, quantization methods and TP-status updates stay inherited.

This is a correction to an unfinished candidate, not model qualification.
Attention output projection still uses the original inverse head exchange and
replicated full GEMM. Its ordinary-TP numerical disagreement remains open.
No Core, Adapter or native FlashAttention source/binary is changed.

## Construction and execution

- Scoped ordinary `.mlp.gate_up_proj` keeps TP allocation and gathers masked
  local token rows before calling the official forward.
- Scoped ordinary `.mlp.down_proj` keeps TP allocation, sets
  `reduce_results=False` before construction, runs official partial GEMM and
  calls the official TP group's public `reduce_scatter(dim=0)`.
- Shared `.mlp.shared_experts` / `.mlp.shared_mlp` continue selecting
  `disable_tp=True` before allocation. Unrelated/ordinary construction keeps
  the official TP behavior.
- Dense TP divisibility remains required; nondivisible shared dimensions are
  tested separately. An explicitly conflicting dense replicated-allocation
  request is refused before weight creation.
- Clear local NaN padding before all-gather, global padding before down GEMM,
  and local padding after reduce-scatter. Device mask values may change at
  fixed addresses on Graph replay. Existing process/group ownership guards
  still run before execution.

At TP4, the dense projection parameter count equals ordinary TP and is one
quarter of the previous replicated candidate's dense projection count. This
is a weight-layout fact, not an end-to-end memory or throughput measurement.

## Validation on 2026-11-19

Recorded dependencies: Core `c440c089fe9621c5b03e1618cf86a13e95de8bf1`,
unpublished Adapter `004cce0a83055b2f67acdc1060258773c3275787`. The later
Adapter main advancement is not relabeled as the tested installation.

| Selection | Result | Boundary |
| --- | --- | --- |
| Source CPU | 212 passed | Actual official constructors/loaders; explicit CPU GEMM/collective doubles |
| Independent installed CPU | 212 passed | Same copied tests, installed Plugin selected before collection |
| Old candidate negative control | 2 failed as expected | Old replicated dense allocation accepts the conflicting request |
| Clean stdlib CI helpers | 98 passed | Initial shared-environment attempt's 3 metadata-isolation failures retained |
| Source/package contracts | 8 passed | Actual target Core source; not native/model qualification |
| Native dense MLP | 16 distinct rank/cases passed | GLM4/HY3 × BF16/FP16 × 4 ranks; two weight loads, active 5/7/3 and raw Graph |

Native dense eager/Graph versus ordinary TP has maximum error
`0.00048828125` at unchanged `atol=rtol=0.016`. All 96 Graph/eager comparisons
are exactly equal. Loaded TP weights match, padding stays finite/zero and
addresses stay fixed. Controller exit 0, owned-process tracker exit 0, no
cleanup signals/survivors, and all 18 private package RECORDs are unchanged.
The standalone tool is `tools/probe_runtime_sp_dense_mlp.py`.

The devices already held another service allocation. The controller required
at least 16 GiB free per selected device; this tiny probe bounds its own Torch
allocator to 5% and measured a 64.4 MiB Torch allocation peak per rank. That
figure excludes native runtime/context allocations. No other job was cleaned
up; these numerical results make no uncontended performance claim.

## Complete synthetic models remain unqualified

The original two-layer official GLM4/HY3 model diagnostic retains the same
synthetic weights, precision, backend, Graph policy and numerical assertions.
These are not pretrained checkpoints or HumanEval samples.

| Family | Eager vs ordinary TP max error | Graph vs TP active 5 / 7 | Graph vs SP eager |
| --- | --- | --- | --- |
| GLM4 | 0.0478515625 | 0.0335693359375 / 0.03125 | 0 / 0 |
| HY3 | 0.0390625 | 0.05859375 / 0.03125 | 0 / 0 |

All ranks agree on these values. The unchanged ordinary-TP gate fails 16/24
comparisons; all 16 Graph/SP-eager comparisons agree exactly. Active-7 cases
pass the original combined absolute/relative assertion. The maximum ordinary-TP
error is larger than in the previous replicated-dense candidate. Restoring the
legacy weight/communication policy does not resolve or waive that failed gate.

The first diagnostic exits 1 and preserves all rank results, but `mp.spawn`
also logs three SIGTERM attempts after numerical exceptions. Its controller's
empty `forced_cleanup` list does not establish signal-free shutdown. The prior
boundary diagnostic had the same logging pattern; retain that correction with
its original evidence. The follow-up probe keeps per-rank failed assertions,
lets ranks complete cleanup and reports numerical failure from the parent after
joining every rank. These diagnostic teardown observations do not qualify the
original 15-second normal model-service shutdown gate.

That follow-up completed with exit 1 for the same 16 numerical failures. All
four ranks' cases, comparisons and failure receipts exactly match the first
run. The follow-up log has no process-termination/SIGTERM/SIGKILL messages;
the controller sent no cleanup signals, tracker exited 0, no owned process
survived and all 18 RECORDs remained unchanged. This fixes diagnostic failure
collection, while preserving the failed model gate.

## Delivery and remaining gates

The paired Adapter selector remains unpublished. There is still no automatic
ModelRegistry publication. Keep PR #479 Draft pending original model numerical
agreement, default-runner compilation, native quantized weights/scales/reloads,
checkpoint/HumanEval32, target/draft/MTP, real PP/DP/PCP/DeepEP/stateless,
performance and normal model-service lifecycle. No DeepSeek V4.1 checkpoint
test is included. Previous validation remains bound to its original sources.

The current external delivery ledger records exact source/wheel/installation
hashes, native launch/cleanup receipts and CI heads separately. An old-head
successful CI run does not qualify this changed production payload.
