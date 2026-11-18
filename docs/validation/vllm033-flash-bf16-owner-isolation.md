# Qwen3-8B BF16 code comparison and owner isolation

This is a scoped PR #410 validation record, not a passing release gate.
Core `c440c089fe9621c5b03e1618cf86a13e95de8bf1`, Adapter
`89fc929c14551985121dd8eabc8d9da8b639c0da` and Plugin runtime
`877a96ad81df020f7e708467c94a2c4bddb63bb0` are fixed on gfx938.
Adapter main later advanced to `ef63bbbd73a42812db9de1683e6720239c42fd68`;
these results do not qualify that new Adapter main.

## Original complete model cohorts

The official baseline has no Plugin module, distribution or extension entry
point. The candidate installs the exact current Plugin source/wheel payload.
All four shared packages have identical, fully verified RECORDs. Worker RPC
confirms 36 actual `TritonAttentionImpl` or `HcuFlashAttentionImpl` layers and
BF16 physical caches before/after evaluation. Backend-selected cache layouts
differ: official LBNHC, Plugin LBHNC. This compares complete backend and
registration choices; it does not isolate one kernel.

Both use `/models/Qwen3-8B`, BF16 weights/KV, TP1, block64, model length4096,
16 sequences, 1024 batched tokens, 1GiB KV, default Graph and enabled prefix
caching. HumanEval0–31 uses temperature0, seed0, max_tokens2048, thinking-off
and concurrency4. `return_token_ids=true` is the only request change from
r127. New request SHA256:
`48f39070274c24a9a65ddf7e03aa5b09196cacc56b2a708175c670062cfa9385`.

| Arm | Raw /32 | Normalized /32 | Normalized failures |
| --- | ---: | ---: | --- |
| Official BF16 |20|31|26|
| Plugin BF16 |20|30|18,26|
| Fresh complete official BF16 repeat |20|31|26|

All cohorts return32 normal stops without length truncation. Every response
contains nonempty generated and prompt ID arrays; rendered prompt IDs match.
The first pair differs on tasks1,12,18. The official repeat reproduces the
Plugin task1 ID sequence; tasks12 and18 still differ. This variability does
not waive the task18 semantic loss against both complete official runs.

Task18 first differs at emitted-ID index106, absolute221 after the115-token
prompt. The official answer uses regex positive lookahead for overlapping
substring matches; the Plugin answer uses ordinary `finditer` without it.
Both raw answers fail on formatting; after unchanged normalization, only
the official answer passes. Generated code runs only in the existing read-only
chroot, mount/net/PID namespace, uid65534 isolated evaluator.

Each arm passes arithmetic13/13, prefix3/3, actual hits+1280 and observed
Graph execution. Traces distinguish official `kernel_unified_attention` from
vendor varlen/paged attention. Complete package RECORDs and checkpoint seals
match before/after. Services exit0 naturally in9.93/10.23/9.68s under the
unchanged15s grace, without forced cleanup. Nanobind leak diagnostics remain;
normal lifecycle is not qualified.

Local evidence: `evidence/plugin410-official-bf16-code-pair-r132/` in the
upgrade workspace, including per-arm requests/responses, isolated results,
provenance, traces and `independent-model-review.json`. Original failures
and the earlier E5M2 task19 loss remain separate and retained.

## Diagnostic method and acceptance boundary

The paired installation also registers `HcuRMSNorm`; its reported forward is
still `RMSNorm.forward_native`. To test registration effects separately from
attention selection, compare no-Plugin/official-TRITON, Plugin/official-TRITON
and Plugin/FLASH using the same frozen history. Verify actual layer classes,
operator forwards, resolved layouts and kernel traces; an active extension
report alone is not execution evidence.

Use original response prompt IDs concatenated with generated IDs, without
re-tokenizing answer text. Verify current chat rendering matches saved prompt
IDs, bind source response hashes and probe both saved continuation histories.
Full teacher-forced prompt logprobs and warmed-prefix requests have different
geometry from the original concurrent free-running generation. A221-token
cached-prefix request reuses192 tokens and computes29 query tokens; only a
measured64n+1 input with64n reused tokens establishes a one-query call.

This diagnostic does not produce a new HumanEval score, recover the failed
cohort, establish a vendor-kernel defect or justify a speculative reader
replacement. FP8 model accuracy, pretrained sliding/softcap combinations,
normal lifecycle, applicable performance and exact-head CI/review/release
requirements remain independent gates. No Core runtime change is introduced.

## Completed emitted-ID diagnostic

The three new services use GPU6 and the same frozen installations. Each
replays both actual emitted histories, repeats each full prompt-logprob
request twice and executes21 cached-prefix probes. No answer is rescored.
Full returned prefill logprobs repeat exactly within every arm/history.

| Arm | Reference | Full-prefill top ID | Top-two gap | Cached branch emitted IDs |
| --- | --- | ---: | ---: | --- |
| official | baseline | 671 | 0.0000 | 671,671,671 |
| official | candidate | 671 | 0.0000 | 671,671,671 |
| plugin_triton | baseline | 671 | 0.2500 | 671,671,671 |
| plugin_triton | candidate | 671 | 0.2500 | 671,671,671 |
| plugin_flash | baseline | 671 | 0.0000 | 671,671,671 |
| plugin_flash | candidate | 671 | 0.0000 | 671,671,671 |

IDs671 and5383 decode to ` #` and ` pattern`. Prefill top IDs are sorted
by returned logprob, then increasing token ID for ties; cached branch IDs
are actual sampled outputs. Both histories use the same shared prefix at
absolute221; cached requests reuse192 tokens and compute29 query tokens.

| Pair | Reference | Returned full prefill logprobs exactly equal | Prefill top-ID differences | Max reference logprob delta | Cached-prefix emitted-ID differences |
| --- | --- | --- | ---: | ---: | ---: |
| official_vs_plugin_triton | baseline | False | 0 | 0.162644 | 0 |
| official_vs_plugin_triton | candidate | False | 0 | 0.138315 | 0 |
| official_vs_plugin_flash | baseline | False | 0 | 0.087206 | 0 |
| official_vs_plugin_flash | candidate | False | 0 | 0.087206 | 0 |
| plugin_triton_vs_plugin_flash | baseline | False | 0 | 0.162667 | 0 |
| plugin_triton_vs_plugin_flash | candidate | False | 0 | 0.136804 | 0 |

The Plugin/official-TRITON control itself changes returned probabilities
relative to the no-Plugin official service. Therefore the original loss
cannot be attributed solely to the Flash reader. Reported RMS forwards
remain native, but registration/compilation effects and the original
concurrent decode geometry still need isolation; this does not establish
a specific RMSNorm or native attention defect.

Every cached branch request in this diagnostic emits671 (` #`), including
Plugin/FLASH. It does not reproduce the original concurrent candidate
choice5383 (` pattern`); the task18 model failure is not recovered or
waived by these sequential probes.

Source/wheel/installed binding, all complete package RECORDs and checkpoint
seals remain verified. Actual traces identify the requested attention
provider and Graph execution in all three arms. The original15s grace
is unchanged; shutdown details are retained below, including native leaks.
Owned identities and all three service ports are absent after terminal
controller exit0. The original BF16 task18 and E5M2 task19 losses remain.
No production reader substitution, Core change or accuracy waiver is made.

Local evidence: `evidence/plugin410-bf16-owner-isolation-r133/`, including
`frozen-plan.json`, original response hashes, complete public distributions,
worker/operator observations, traces and `analysis.json`.

| Arm | Exit | Natural exit seconds | Frontend timeout | Forced cleanup | Remaining at grace |
| --- | ---: | ---: | --- | --- | --- |
| official | 0 | 9.829 | False | False | False |
| plugin_triton | 0 | 9.878 | False | False | False |
| plugin_flash | 0 | 9.629 | False | False | False |

Native leak messages are preserved separately from process exit status;
no arm is promoted to full lifecycle or model qualification.

## RMS registration and autotune controls

A follow-up on the same gfx938/Core/Adapter/Plugin runtime uses four fresh
official-TRITON services: no Plugin, unchanged Plugin, Plugin with only its
RMSNorm OOT entry removed by a diagnostic worker hook before model construction,
and a fresh unchanged-Plugin repeat. The hook keeps other registry entries
and compile-factor getters intact. It is not a production Plugin fix. Actual
constructed RMS classes and native forwards, attention layers, cache layout,
source origins and Graph traces are observed. All four keep the original
BF16/Graph/prefix serving configuration and exact saved-ID requests.

| Arm | History | Prefill top ID | Top-two gap | Cached branch emitted IDs |
| --- | --- | ---: | ---: | --- |
| official | baseline | 671 | 0.0000 | 671,671,671 |
| official | candidate | 671 | 0.0000 | 671,671,671 |
| plugin_triton | baseline | 671 | 0.2500 | 671,671,671 |
| plugin_triton | candidate | 671 | 0.2500 | 671,671,671 |
| plugin_no_rms | baseline | 671 | 0.0000 | 671,671,671 |
| plugin_no_rms | candidate | 671 | 0.0000 | 671,671,671 |
| plugin_repeat | baseline | 671 | 0.0000 | 671,671,671 |
| plugin_repeat | candidate | 671 | 0.0000 | 671,671,671 |

| Pair | History | Prefill top-ID differences | Max reference logprob delta | Cached emitted-ID differences |
| --- | --- | ---: | ---: | ---: |
| official_vs_plugin_triton | baseline | 0 | 0.162644 | 0 |
| official_vs_plugin_triton | candidate | 0 | 0.138315 | 0 |
| official_vs_plugin_no_rms | baseline | 0 | 0.000000 | 0 |
| official_vs_plugin_no_rms | candidate | 0 | 0.000000 | 0 |
| official_vs_plugin_repeat | baseline | 0 | 0.000000 | 0 |
| official_vs_plugin_repeat | candidate | 0 | 0.000000 | 0 |
| plugin_triton_vs_plugin_no_rms | baseline | 0 | 0.162644 | 0 |
| plugin_triton_vs_plugin_no_rms | candidate | 0 | 0.138315 | 0 |
| plugin_triton_vs_plugin_repeat | baseline | 0 | 0.162644 | 0 |
| plugin_triton_vs_plugin_repeat | candidate | 0 | 0.138315 | 0 |
| plugin_no_rms_vs_plugin_repeat | baseline | 0 | 0.000000 | 0 |
| plugin_no_rms_vs_plugin_repeat | candidate | 0 | 0.000000 | 0 |

Prefill rankings retain the original tie rule; emitted IDs are sampled
responses. Branch requests still reuse192 tokens and compute29 query
tokens. Full prefill repeats exactly within each arm/history. The three Plugin
arms have equal compile factors.

| Pair | Model graph AST equal | Generated kernel AST sets equal after removing comment-only lines | Different selected autotune configurations |
| --- | --- | --- | ---: |
| official_vs_plugin_triton | True | True | 2 |
| official_vs_plugin_no_rms | True | True | 0 |
| official_vs_plugin_repeat | True | True | 1 |
| plugin_triton_vs_plugin_no_rms | True | True | 2 |
| plugin_triton_vs_plugin_repeat | True | True | 3 |
| plugin_no_rms_vs_plugin_repeat | True | True | 1 |

The unchanged-Plugin repeat retains all145 `HcuRMSNorm` instances, yet
both complete returned prefill distributions and all cached-prefix top
logprobs match the official service exactly, for both saved histories.
The RMS-entry-removed arm matches them too. The first Plugin service
differs from its own fresh repeat despite the same class registration.
Several tuning choices differ in that repeat: it is not a single-tile
causal intervention. These observations do not establish an RMS class
defect or identify the cause of the original concurrent model failure.

The preceding official/Plugin-TRITON diagnostic already had matching
model graph ASTs and matching generated-source AST sets after removing
comment-only source paths. Its selected RMS reduction differed:
`XBLOCK=2,R0_BLOCK=2048` versus `XBLOCK=1,R0_BLOCK=4096`, both with
`num_warps=8,num_stages=1`. Different compile-cache keys and OOT class
names therefore do not by themselves establish different algorithms.

An independent gfx938 operator probe retains that exact official generated
Triton body and removes only its timing-autotuner decorator to request
the two recorded configurations on identical CPU-seeded BF16 tensors.
Across 15 cases, 32 BF16 elements differ; maximum tile-output absolute delta is 0.015625.
Both outputs are finite; independent FP64 mean/rsqrt reference errors
and input hashes are retained. These are synthetic inputs, not captured
model activations. This demonstrates reduction-choice rounding sensitivity,
not the cause of the original model loss, a numerical acceptance waiver,
or a requirement to pin tuning globally. No tolerance is changed.

| Arm | Exit | Natural exit seconds | Frontend timeout | Forced cleanup | Remaining at grace |
| --- | ---: | ---: | --- | --- | --- |
| official | 0 | 9.377 | False | False | False |
| plugin_triton | 0 | 9.729 | False | False | False |
| plugin_no_rms | 0 | 9.579 | False | False | False |
| plugin_repeat | 0 | 9.178 | False | False | False |

All package RECORDs and checkpoint seals remain unchanged. The original
15s grace and native leak diagnostics are preserved. No new HumanEval
score, original concurrent-cohort recovery, full lifecycle acceptance,
Core change or production registry modification is claimed. Original
BF16 task18 and E5M2 task19 losses remain; keep Draft.

Evidence: `evidence/plugin410-rms-registration-r134/`, including actual
registry snapshots, source-bound distributions, compiler artifacts,
selected configurations, synthetic tile probe, traces and `analysis.json`.

## Matched live tuning and one-configuration control

Four fresh gfx938 services retain the preceding Core/Adapter/Plugin runtime
and actual saved histories: official/TRITON, Plugin/TRITON, Plugin/TRITON
with one RMS reduction choice changed, and Plugin/FLASH. Each private cache
replays ten previously observed official tuning choices. The third arm only
changes `3e47a29ee0274aefe888dd7cbc7d7a4dbb737d72431586ead332762cac963a22.best_config`
from XBLOCK2/R0_BLOCK2048 to XBLOCK1/R0_BLOCK4096; warps8/stages1 remain.
All original cache validation is retained. The installed Torch cache API
accepts the recorded configurations and rejects mismatched config hashes
for the target choice in a separate CPU control. No production tuning policy or RMS registration
is changed.

Before and after historical requests, actual worker CachingAutotuner
instances must report cache hits and the selected active launcher for all
ten choices. Disk files alone are insufficient. The first attempt stopped
before requests because gc.get_objects omitted frozen instances. Its failure
is retained. The second attempt exposed released candidate configs=None
in the observer; its terminal failure is retained too. A CPU contract
checks the finalized/frozen state. The completed attempt reads modules held by
PyCodeCache/sys.modules, without unfreezing GC, reloading kernels or
substituting functions.

| Arm | History | Prefill top ID | Top-two gap | Cached branch emitted IDs |
| --- | --- | ---: | ---: | --- |
| official | baseline | 671 | 0.000000 | 671,671,671 |
| official | candidate | 671 | 0.000000 | 671,671,671 |
| plugin_triton | baseline | 671 | 0.000000 | 671,671,671 |
| plugin_triton | candidate | 671 | 0.000000 | 671,671,671 |
| plugin_flip | baseline | 671 | 0.250000 | 671,671,671 |
| plugin_flip | candidate | 671 | 0.250000 | 671,671,671 |
| plugin_flash | baseline | 671 | 0.000000 | 671,671,671 |
| plugin_flash | candidate | 671 | 0.000000 | 671,671,671 |

| Pair | History | Max reference logprob delta | Prefill top-ID differences | Cached emitted-ID differences |
| --- | --- | ---: | ---: | ---: |
| official_vs_plugin_triton | baseline | 0.000000 | 0 | 0 |
| official_vs_plugin_triton | candidate | 0.000000 | 0 | 0 |
| official_vs_plugin_flip | baseline | 0.162644 | 0 | 0 |
| official_vs_plugin_flip | candidate | 0.138315 | 0 | 0 |
| official_vs_plugin_flash | baseline | 0.049198 | 0 | 0 |
| official_vs_plugin_flash | candidate | 0.030302 | 0 | 0 |
| plugin_triton_vs_plugin_flip | baseline | 0.162644 | 0 | 0 |
| plugin_triton_vs_plugin_flip | candidate | 0.138315 | 0 | 0 |
| plugin_triton_vs_plugin_flash | baseline | 0.049198 | 0 | 0 |
| plugin_triton_vs_plugin_flash | candidate | 0.030302 | 0 | 0 |
| plugin_flip_vs_plugin_flash | baseline | 0.162670 | 0 | 0 |
| plugin_flip_vs_plugin_flash | candidate | 0.147028 | 0 | 0 |

Exact equality of the complete returned prefill distributions and cached
top-logprobs, respectively, is recorded below for both saved histories:

- `official_vs_plugin_triton`: `{"baseline": {"cached": true, "prefill": true}, "candidate": {"cached": true, "prefill": true}}`.
- `plugin_triton_vs_plugin_flip`: `{"baseline": {"cached": false, "prefill": false}, "candidate": {"cached": false, "prefill": false}}`.
- `official_vs_plugin_flash`: `{"baseline": {"cached": false, "prefill": false}, "candidate": {"cached": false, "prefill": false}}`.

With the same verified tuning choices, Plugin RMS registration and
official/TRITON produce exactly equal returned distributions in these probes.

Changing just the observed RMS tuning choice alters returned model
probabilities here. This identifies a cause within the sequential control;
it does not establish the cause of the original concurrent task failure.

Generated-source ASTs, graph ASTs, final configuration sets and active
launcher provenance are retained in the analysis. The Plugin/TRITON flip
has exactly one effective configuration difference; official/TRITON and
Plugin/TRITON or Plugin/FLASH have no effective configuration differences.
This isolates a tuning choice in these sequential history probes, not the
original concurrent free-running HumanEval failure. The branch still uses
cached192/query29. Full model recovery and a vendor-kernel defect are not
established, and no new HumanEval score or accuracy waiver is introduced.

| Arm | Exit | Natural exit seconds | Forced cleanup |
| --- | ---: | ---: | --- |
| official | 0 | 9.879 | False |
| plugin_triton | 0 | 10.078 | False |
| plugin_flip | 0 | 9.380 | False |
| plugin_flash | 0 | 8.878 | False |

The original15s grace, native leak diagnostics, original BF16 task18 loss
and E5M2 task19 loss remain. Package RECORDs/checkpoint seals are unchanged;
owned processes and ports are absent after terminal controller completion.
Keep Draft while the original accuracy/lifecycle and applicable release
gates remain open. No Core production MR is introduced.

Evidence: `evidence/plugin410-matched-tuning-r135/` (retained first attempt
and CPU cache controls), and `evidence/plugin410-matched-tuning-r135a3/`
(completed actual live-launcher controls, responses, traces and analysis).
