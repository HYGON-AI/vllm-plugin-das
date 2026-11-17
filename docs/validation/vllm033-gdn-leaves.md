# v0.33 GDN/FLA leaf migration candidate

The production `adapter_extension` previously provided no replacement for the
three legacy FLA leaves or two Qwen GDN decode functions. This candidate adds
an independent, explicitly requested `hcu-gdn-fla` extension. It keeps Core's
complete algorithm and its constructor's final backend dispatch.

## Ownership and temporary seam

Core v0.33 exposes whole-op/layer OOT registration but no FLA leaf registry.
The Plugin therefore replaces only three aliases in
`vllm.third_party.flash_linear_attention.ops.chunk` and two aliases in
`vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn`. Existing pipeline
functions and model methods resolve those same module globals. Canonical leaf
modules and unrelated consumers remain official.

This is a temporary consumer alias seam. Migrate to Core's public leaf-provider
registry when an interface covers the required metadata, state and output
contracts. Do not copy the complete pipeline or reactivate the legacy root.
The Plugin neither registers extra official backend names nor modifies the
official prefill selector, convolution, recurrent state allocator, weight
loader, RMSNormGated or fused post-convolution branches.

## Selection and contracts

Request capability `hcu-gdn-fla`. The effective policy is the conjunction of
`VLLM_HCU_USE_CUSTOM_OPS` (default OFF) and the two stage selections:

| Setting | Values | Default |
| --- | --- | --- |
| `VLLM_HCU_GDN_PREFILL_BACKEND` | `triton`, `hcu-boltops`, `hcu-aiter` | `triton` |
| `VLLM_HCU_GDN_DECODE_BACKEND` | `triton`, `hcu-boltops`, `hcu-aiter` | `triton` |

The official `gdn_prefill_backend` remains authoritative. A whole-op backend
such as `aiter_flydsl` can bypass the prefill aliases. Optional fused decode
branches can bypass the two decode aliases. Activation is not execution proof;
qualification must establish the actual route for the selected model.

Before publication, all five Core signatures and selected provider signatures
are checked, including parameter defaults and synchronous execution kind.
Missing packages or symbols follow AITER → BoltOPs → official, or
BoltOPs → official. Missing internal dependencies and incompatible installed
ABIs are errors. Native execution errors propagate after selection.

BoltOPs chunk-O has two audited exact signatures: the original entry and an
extended entry with trailing `full_chunks=None`. The latter's default retains
masked varlen execution and auto-detects only fixed aligned chunks. The Plugin
does not assert `full_chunks=True`; different defaults or additional parameters
still refuse before publication.

Adapters retain the official metadata, `[N,H,V,K]` state layout, initial/final
state, `save_new_value` and output backing. BoltOPs indexed decode receives explicit
`null_state_index=0`. Prefill chunk-H receives dense gathered per-sequence state,
where row 0 is real data and there are no state indices; it explicitly uses
`null_state_index=-1`. Treating that dense row as the cache NULL slot discarded
its initial state in the installed provider and failed native varlen final-state
parity (max absolute error about 0.012). AITER decode converts indices ≤0 to -1
without mutating the original index tensor. AITER chunk-H receives int64 offsets. AITER has no
standalone recompute entry, so that leaf uses BoltOPs or official.

Gated AITER chunk-O remains excluded because earlier device evidence showed
numerical divergence. AITER-incompatible inputs use the next prepared route.
The recorded exact BF16 chunk-H `(Hk,Hv,K,V)=(4,12,128,128)` retains official
execution on the BoltOPs route. These constraints do not qualify other shapes
or promote a provider default.

All prechecks finish before any alias changes. Partial publication, reentry,
or late policy/owner drift permanently refuses the process. A complete
inherited publication requires fresh child runtime activation; a fork during
preparation/publication refuses. Adapter owns runtime admission and the serving
health boundary, including checks before cached/Graph execution. Python alias
guards alone do not run inside an already captured device Graph.

Effective routes and adapter/provider source identity enter official normal
and AOT compile factors. This is a source identity, not an exact native wheel
pin or a replacement for the providers' native/JIT cache ownership.

## Validation and remaining gates

Source CPU qualification with real Core `98a0985` and Adapter `719e203`:
**39 passed / 0 skipped** across `test_gdn_leaf_providers.py` and
`test_package_boundary.py`. This includes real Core aliases and constructor
dispatch, strict provider call contracts, NULL and speculative metadata,
output backing/padding, native-error propagation, publication failure,
owner drift, reentry and actual fork tests. Provider doubles do not prove
native kernel numerics. Earlier failed test-fixture runs remain in the task
evidence rather than being relabeled as device failures.

`test_gdn_leaf_hcu.py` provides an opt-in complete official FLA dataflow test:
legal WY intermediates, real gates, head ratios 4/8 and 4/12, nonaligned varlen,
initial/final state, padded output backing and three changed-input Graph
replays. Run each provider selection in a fresh process with
`HCU_V033_TEST_GDN=1`. The explicit test-only `HCU_V033_TEST_GDN_ALLOCATOR_MIB` budget defaults
to 512 MiB and requires that budget plus 16 MiB actually free. Cold official
Triton autotuning itself allocates a 256 MiB benchmark cache; the earlier
64 MiB cap failed there before native prefill execution. Shapes and official
autotuning remain intact. Output absolute tolerance is tightened to 1e-4, with
independent decode state checks at 5e-4. This candidate is still being qualified.
`HCU_V033_TEST_GDN_PUBLIC_STARTUP=1` requires the real Adapter general-plugin
activation before pytest; the fixture does not activate the extension itself.
Its official reference always binds canonical Core leaves, even after public
startup has replaced consumer aliases. Packed decode and sigmoid MTP tests
additionally compare the official kernels and an independent CPU FP32 recurrence, with NULL state,
accepted-token metadata and fixed-address changed-input Graph replay.

On gfx938, the installed Core98/Adapter719/Plugin candidate completed all eight
component cases through real Adapter public startup for BoltOPs (cold Triton
autotuning, 288 MiB test budget) and AITER (verified genuine warm cache, 64 MiB).
Each selection includes 24 changed-input Graph replays. AITER chunk-H/decode
execution is required explicitly; recompute and gated chunk-O still execute
BoltOPs. Each process exits 0 but retains two known provider weakref cleanup
errors: this is numerical/Graph component evidence, not clean lifecycle closure.
An earlier device-6 run failed with `hipErrorNoBinaryForGpu` during comparison;
the device-7 synchronous/private-cache repeat passes, but its root cause is
still unattributed. The old 64 MiB cold-autotune and pre-fix numerical failures
remain recorded separately.

The earlier component cleanup warnings above remain historical evidence. A
later isolated source-provider installation (AITER `1cb6ed3`, BoltOPs
`2d71c75`) revalidated both public-startup selections: eight native cases and
24 changed-input Graph replays each, with zero weakref errors or tracebacks.
That result does not establish normal model-server teardown.

### Frozen Qwen3.5 INT8 model comparison, 2026-11-17

Three fresh gfx938 TP1 runs use the original `Qwen3.5-35B-A3B-W8A8` checkpoint,
Core `98a09850`, Adapter `20d18cc6` and installed Plugin `3836b8b3`. Before
target integration, all 64 production Python files at PR head `94c35088`
match that installation byte-for-byte. Source-provider package RECORDs and
the successful native prerequisites are checked before launch and after
evaluation. These source-built providers are candidates, not a qualified
published provider release.

All arms retain BF16 activations/CT INT8 weights, text-only mode, context 4096,
batch tokens 1024, max sequences 16, fixed 2 GiB KV, admission fraction 0.50,
official FULL_AND_PIECEWISE Graphs, prefix caching and chunked prefill. The
official GDN prefill selector is explicitly `triton` in all three arms. The
stock arm verifies policy `(triton, triton)`, no leaf publication and all five
actual consumer aliases bound to canonical Core functions.

HumanEval/0–31 uses identical saved request bytes across the three arms:
temperature 0, seed 0, max tokens 2048, concurrency 4 and thinking disabled.
Request/protocol SHA256 is
`c3f27a1e28bdb623aaa4a9bd7f38fff7ca287d73dd68d4714c7beea449ce19f3`.
This is the current fixed protocol; the historical handoff's original request
bytes were unavailable and are not claimed to have been recovered. Raw and
fence-normalized completions are scored separately in the isolated scorer.

| Leaf selection | Raw | Normalized | Raw failed task IDs | Completion tokens |
| --- | --- | --- | --- | --- |
| Official stock | 31/32 | 32/32 | 13 | 4966 |
| AITER | 30/32 | 32/32 | 24, 31 | 4773 |
| BoltOPs | 29/32 | 32/32 | 15, 24, 31 | 4793 |

Every completion ends with `stop`, with no truncation. Both candidates recover
task 13, but AITER loses previously passing tasks 24/31 and BoltOPs loses
15/24/31. These raw failures are fenced Python reported as `invalid syntax`;
stripped completions pass the tests. **The raw score decreases remain open:
normalized 32/32 does not authorize acceptance or merge.** No prompt, scorer,
backend or score threshold was changed to hide them.

Each arm passes 13/13 changing arithmetic/concurrency probes and 3/3 prefix
probes with 4224 additional cache-hit tokens. The effective manager page is
1056 tokens with four cache groups; the prefix prompt has 2271 tokens. Actual
worker snapshots report no observer errors or pending submissions. AITER
submits chunk-H/decode through AITER and chunk-O/recompute through BoltOPs;
BoltOPs submits all five leaves through BoltOPs. These are Python submission
and capture observations, not counts of GPU-only Graph replays.

Real GPU traces contain 1939/1939/1945 kernel events for stock/AITER/BoltOPs
and 44 `hipGraphLaunch` calls in each arm. The AITER trace includes 30 native
`FLA_NAMESPACE::chunk_gated_delta_rule_fwd_kernel` events. Decode Graph-node
attribution remains incomplete where replay traces do not expose individual
kernels; a provider-selection setting alone is not direct native attribution.

All 23 checkpoint assets (37,602,505,378 bytes) match independent pre/post
dual reads across all arms. Server, evaluator and tracker exit 0; recorded
owned identities disappear, ports release, and post-run package/asset checks
pass. Weakref errors are zero, but **nanobind and abort shutdown diagnostics
remain in every arm**, including stock. Process release does not establish
normal lifecycle closure.

The local evidence index is `qwen35-gdn-tp1-r7-closure.json`, binding original
requests/results, worker observations, traces, asset seals and package gates.
No performance, MTP, gfx936 or other topology qualification is claimed. The
later target integration includes Plugin `030c6849` changes outside GDN, and
Adapter main has advanced beyond `20d18cc6`; frozen scores do not qualify
those complete merged combinations. Keep Draft pending raw accuracy,
lifecycle, remaining feature/architecture/provider delivery and current-head
CI/review gates. DeepSeek V4.1 model testing remains user-deferred.
