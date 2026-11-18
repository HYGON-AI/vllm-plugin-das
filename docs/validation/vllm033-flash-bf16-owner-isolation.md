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
