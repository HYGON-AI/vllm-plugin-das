# v0.33 HCU Flash attention softcap

The [BF16 code-pair and owner-isolation record](vllm033-flash-bf16-owner-isolation.md)
preserves the complete official/Plugin Qwen3-8B comparison and its unresolved
task18 loss. Component softcap/window passes do not replace that model gate.

The public HCU FLASH_ATTN implementation used to forward a positive
`logits_soft_cap` to the vendor function for every batch. The installed vendor
wrapper's direct paged path explicitly ignores softcap. A writer-to-reader
regression with softcap 0.5 also fails on short prefill: the original wheel
has a maximum absolute output difference of 1.5859375 from the independent
FP32 softcap reference, with unchanged 0.016 absolute/relative tolerances.

Positive softcap now uses Core's unchanged `unified_attention` implementation.
The adapter passes the existing paged K/V views in NHD order, cumulative query
lengths, sequence lengths, block table, causal flag, scale and ALiBi slopes.
Core owns the attention algorithm and softcap calculation. No Core source,
scheduler, metadata builder, cache allocation or custom kernel is copied.
Both explicit E5M2 and E4M3 paths use the existing query/output descale adapter,
including its rounding behavior; each keeps its own FP8 typed cache view.
The underlying attention call uses Core instead of vendor Flash. A zero/absent
softcap with a full window keeps the existing vendor route and optional
prefill gather policy. Enabling prefill gather cannot bypass softcap dispatch.

This is an explicit provider fallback for a correctness boundary, and is not
evidence that the vendor implements softcap. The Core path produces the
public AttentionImpl output only; diagnostic requests for LSE or attention
probabilities are refused. Remove this fallback only after the vendor passes
the same writer/reader, non-unit-scale and changed-input Graph tests for both
prefill and decode, plus applicable model validation.

## Causal sliding-window composition

Pretrained Gemma2 combines positive attention softcap with alternating full
and sliding layers. The previous provider advertised no sliding support and
rejected its constructor, so the full-attention synthetic fixture cannot
qualify this checkpoint. The Plugin now accepts positive integer causal
windows and routes every finite window to Core's unchanged `unified_attention`,
including zero softcap. It passes `(window - 1, 0)`, counting the current token
in the public window size. E4M3/E5M2 retain typed views and the same one-time
KV descales. Core retains the per-layer policy, metadata and cache allocation;
no checkpoint configuration or Core source is edited. Sink, RSWA, noncausal
and context-parallel combinations remain rejected.

The new native regression uses an independent FP32 mask/softcap reference,
native KV writes and public `forward`, head size256, both layouts, BF16/FP16/
E4M3/E5M2, non-unit FP8 scales, windows1/17/64/4096 and caps0/50. Ragged rows,
empty rows, changed page tables and lengths, invalid writer slots, output
padding and stable-address Graph replay are exercised. Results and pretrained
HE32 comparison remain separate gates; historical results below bind their
original runtime rather than this new composition.

On gfx936, the sliding composition passes128 CPU Flash contracts,8 source/
package contracts, and64 installed native cases with no skips and unchanged
0.016 absolute/relative numerical tolerances. All six installed package
RECORDs verify before/after. The wheel SHA256 is
`12e1858293aef1312609eb34eea7e8b4b83a97ccad474b9fa1a6c4bd5da966d1`;
Core0.33 source `98a09850d56b` and baseline Adapter `89fc929c1455` are unchanged.
The initial native attempt failed because the new object-based test fixture
omitted `alibi_slopes`; that failure is retained, the fixture now declares its
absent bias, and the complete64-case matrix was rerun. No production change
or tolerance relaxation was needed for that test-fixture failure.

A pretrained `google/gemma-2-2b-it` checkpoint has been sealed from the public
ModelScope mirror at revision `c400b2564e9db9e5a0bc6dfbe07a05da3cd09f52`:
all12 assets (5,250,588,019bytes) match remote SHA256 and an independent full
reread. Its original26layers,13full/13sliding4096 policy, head size256 and
attention cap50 are retained. Original HE0–31 paired validation is running
with official TRITON_ATTN and Plugin FLASH_ATTN at BF16 and E4M3 KV, default
Graph/prefix and the original15s shutdown grace. This is pending model/accuracy
evidence, not a passed gate. The local checkpoint weights reside in tmpfs;
only the bounded metadata/seal is durably archived. These running/pending
statements describe that historical cohort, not the current process state or
qualification of later runtime payloads. gfx836 remains unqualified.

## E4M3 composition follow-up (2026-11-18)

Replaying the original #410 payload over the merged E4M3 provider exposed a
dispatch defect: only E5M2 consumed KV descales before entering the Core
softcap helper, so an E4M3 public forward raised its unapplied-descale guard.
Four E4M3 CPU regressions fail on that composition while four E5M2 controls
pass. The corrected Plugin routes both formats through the existing shared
scale adapter, retaining their distinct typed cache views and the official
Core attention algorithm. No Core or Adapter production code changes.

The isolated installed candidate passes **93 CPU Flash contracts**, **8
source/package contracts** and **219 actual HCU numerical/Graph cases**, all
with zero skips. The latter includes all original 80 E4M3 cases, 32 new
positive-softcap E4M3 cases and the existing bounded-scratch test, plus the
BF16/FP16 and E5M2 suites. Original tolerances remain unchanged. Source,
wheel and installed Plugin payloads match; all six native-owner/package
RECORDs verify before/after. The immediate native terminal still observed a
KFD owner; three later independent checks found none. Eventual device release
does not qualify clean model lifecycle.

This run uses actual **gfx936**, Core **0.33.0** source `98a09850d56b`,
Torch 2.13 and the baseline Adapter from `89fc929c1455`, independently of
Draft Adapter #140. The Plugin wheel SHA256 is
`ab6bca09f3cf76fc60b0e705210df8e5909ece17718a18112038dc68d6ea84fa`.
Positive-softcap public LLM and fresh original HE32 regression gates for this
integrated payload are pending. Historical results below do not certify the
updated payload, gfx836 or full upgrade acceptance.

## Historical component validation

Environment: real gfx936 (not gfx836/gfx938), DTK/Torch 2.13 stack;
Core 0014f6fce6de03125c02aa1541344ecba54ef114 and Adapter 21cf8f7.
Original and candidate installed wheels retain identical dependencies.

- Original installed Plugin: the new numerical test fails as described above.
- Candidate installed Plugin: 106 HCU tests pass, covering BF16/FP16,
  HND/NHD, prefill/decode, E5M2, per-sequence non-unit descales and changed
  input/page table/scale CUDA Graph replay. Existing tolerances are retained.
- Portable Flash contracts: 81 pass, including actual public forward dispatch
  and output padding with prefill gather both enabled and disabled.
- Expanded non-DeepSeek portable contracts: 564 pass, 6 deselected, 14 warnings.
  Source and packaging contracts: 8 pass.

Run against an installed candidate wheel with the prepared HCU dependencies:

```bash
VLLM033_HCU_TEST=1 HCU_ADAPTER_HOST=vllm VLLM_PLUGINS=hcu_adapter \
  python -m pytest tests/vllm033/test_flash_attention_hcu.py \
  tests/vllm033/test_e5m2_scales_hcu.py -q
VLLM_TARGET_ROOT=/path/to/installed/site-packages HCU_ADAPTER_HOST=none \
  python -m pytest tests/vllm033/test_flash_attention.py -q
```

## Public LLM integration

`tools/probe_softcap_llm.py` builds a seeded serialized two-layer Gemma2 control
with full attention, softcap 0.5, BF16 weights, hidden size 256 and vocabulary
128. Q/K weights are amplified to make the softcap observable. This fixture
is synthetic, not pretrained model accuracy. The same checkpoint bytes are
used in three isolated installed-wheel environments: official Core+Adapter
TRITON_ATTN, original Plugin FLASH_ATTN, and corrected Plugin FLASH_ATTN.

Both baseline and candidate use block size 64, default FULL_AND_PIECEWISE
Graphs, prefix caching, three changed suffixes and two generation waves.
The second wave reuses at least 64 prefix tokens for every request. Parent
and real worker package origins, both effective softcap layers, all 128
logprobs per output token and greedy token IDs are recorded. Candidate and
official baseline agree on all 48 generated token IDs; maximum absolute
logprob difference across the recorded distributions is 0.0077385902.
This records a measured difference, not a new logprob acceptance tolerance.
All first-output-token distributions match exactly. The original Plugin
matches only 30/48 generated tokens; its first-output-token distributions
already differ by up to 1.353480816 at identical prompt histories. Comparisons
after greedy histories diverge are retained as diagnostics, not same-history
numerical comparisons.

Run each arm in a fresh process with its installed environment's Python.
For example, after building the fixture with `--build --checkpoint /tmp/cap-model`:

```bash
HIP_VISIBLE_DEVICES=0 HCU_ADAPTER_HOST=vllm VLLM_PLUGINS=hcu_adapter \
  VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
  VLLM_HCU_USE_CUSTOM_OPS=0 VLLM_HCU_USE_CUSTOM_RMS_NORM=0 \
  VLLM_CACHE_ROOT=/tmp/cap-candidate-cache \
  /path/to/candidate-env/bin/python -I tools/probe_softcap_llm.py \
  --checkpoint /tmp/cap-model --backend FLASH_ATTN --output-dir /tmp/cap-candidate
```

The explicit serialization setting permits the probe's own local worker
inspection callback. Baseline uses its environment and `--backend TRITON_ATTN`;
original Plugin uses its environment and `--backend FLASH_ATTN`. Use different
cache/output directories and retain warnings. An earlier local run imported
system packages and was discarded; another stopped at the RPC serialization
gate before generating outputs. Neither contributes passing evidence.

## Real checkpoint regression

Qwen3-8B BF16, gfx936 TP1, matched frozen original HumanEval/0-31 requests,
block64 FLASH_ATTN, default Graph/prefix: candidate and original Plugin produce
byte-identical raw completions and the same **20/32 raw, 30/32 normalized**
scores. All 32 finish normally, the prefix probe hits 3840 tokens, and two
four-request concurrent waves pass. Scoring runs in an isolated namespace/chroot
as an unprivileged user. No generated Python runs in the serving/host environment.

Qwen3 uses softcap zero. Its existing additional HumanEval/18 semantic failure
against official TRITON_ATTN (31/32 normalized) remains open and is not explained
or repaired by this change. This run bounds the zero-softcap regression claim;
it is not pretrained positive-softcap model acceptance. No supported pretrained
positive-softcap checkpoint was validated here. Gemma4's other unsupported
attention requirements are not enabled by this fallback.

## gfx938 BF16 / E5M2 pretrained KV comparison

On 2026-11-18, an independent installed-wheel run used Core `c440c089fe96`,
Adapter main `89fc929c1455` and Plugin runtime `7fa427b81f8f` on gfx938.
Both arms use the same Qwen3-8B checkpoint, FLASH_ATTN, TP1/BF16 weights,
block 64, default Graph/prefix, max length 4096, seqs 16, batched tokens 1024
and a fixed 1 GiB KV allocation. The baseline requests `auto` KV; the other
arm requests public `fp8_e5m2`. This measures the KV dtype change inside the
same Plugin; it is not official-versus-Plugin or positive-softcap acceptance.

The HumanEval 0–31 requests are byte-identical across the arms:
temperature 0, seed 0, max output 2048, thinking disabled and concurrency 4.
Generation, raw outputs and scoring are retained independently. Scoring uses
the existing unprivileged read-only chroot with mount/network/PID namespaces.

| KV dtype | Raw | Normalized | Normalized failed tasks | Completion tokens |
| --- | ---: | ---: | --- | ---: |
| BF16 (`auto`) | 20/32 | 30/32 | 18, 26 | 4218 |
| E5M2 | 21/32 | 30/32 | 19, 26 | 4064 |
| Fresh full BF16 repeat | 20/32 | 31/32 | 26 | 4218 |

All 32 responses stop normally in each arm. E5M2 loses task 19 and gains task 18
after normalization; equal totals do **not** establish no regression. Task 19
emits an unordered `Enum` as the sorting key, and the isolated checker reports
that `<` is unsupported between two `Number` values. This is a semantic
failure, not a Markdown-only normalization issue. No kernel cause is established.

Both arms pass 13 arithmetic requests at concurrency 1/4/8, three repeated-prefix
requests and the profiled request; prefix hits increase by 1280 tokens per arm.
Real worker inspection verifies all 36 Flash layers, BF16 versus byte E5M2
cache storage, installed package origins and native library paths. Traces
observe 40 Graph launches per arm. E5M2 traces include the byte writer,
row scales, BF16 Flash prefill and byte-cache paged decode; this does not
establish native FP8-compute throughput.
Plugin source/wheel/installed payloads, five full hashed package RECORDs and
the checkpoint's complete pre/post seal verify unchanged. Owned descendants
are absent after cleanup. The frontends exit 0 in 10.63s/9.23s under the original
15s grace without forced cleanup, but both logs retain nanobind native-leak
diagnostics: normal lifecycle remains unqualified.

The full BF16 baseline was then repeated in a fresh service with identical
request bytes, probe sequence and default settings. Four completions differ
(tasks 1, 18, 25, 31); task 18 passes after normalization, while task 19 passes
in both BF16 runs. This establishes baseline variation for task 18, without
reproducing or waiving the E5M2 task 19 loss or identifying a kernel cause.
The repeat also passes all probes, unchanged RECORD/checkpoint checks and
Graph inspection. It exits 0 in 9.38s without forced cleanup, retaining the
native-leak diagnostics.

Evidence: `evidence/plugin410-gfx938-public-llm-r127` in the upgrade workspace,
including original responses, isolated results, launch receipts, worker
provenance, traces, `independent-model-review.json` and
`independent-baseline-repeat-review.json`. This is scoped
pretrained E5M2 execution evidence with an unresolved pass-set loss. It neither
erases the older BF16 task 18 regression nor qualifies other checkpoints,
architectures, positive softcap or performance. The original selected CI and
component results remain bound to their tested runtime commit.

## Frozen-history reader diagnostic on gfx938

The subsequent r128 diagnostic keeps Core `c440c089`, Adapter main `89fc929c`
and the tested Plugin runtime. Both saved task19 continuations are retokenized
with an unchanged prompt boundary. Complete teacher-forced prefill repeats
are identical within each KV arm. The first BF16/E5M2 greedy difference is at
absolute token149 / continuation17: BF16 chooses220 (space), E5M2 chooses364
(space+quote), with top-two logprob gaps9.75/2.00. Full prefill reproduces it,
so a single-token decode-only explanation is insufficient. Each arm's18
aligned-prefix probes verify192/256/320 cached tokens for193/257/321-token
prompts. No new HumanEval cohort or score is claimed.

Local diagnostic Plugin `57182d8609202d748e5a14f8758a0fb0af441f12` (r129)
substitutes Core's unchanged `unified_attention` only for output-only E5M2
reads. It preserves the writer/cache/scales and the original statistics-return
provider. All46 unchanged E5M2 numerical/Graph cases pass with their original
tolerances. An independent physical installation then runs the same frozen
requests; trace shows `kernel_unified_attention`, the E5M2 writer and row
scales, plus actual Graph launches. At token149 it chooses220 with gap1.00,
matching BF16's choice; complete prefill repeats are again identical. Against
the vendor reader, each saved reference has3 greedy differences, and3/18
aligned-prefix choices differ. Against BF16, the two references have3/1
greedy differences; aligned-prefix choices agree18/18. This establishes reader
sensitivity. It does not establish a vendor defect or qualify the local
substitution as a production repair, performance improvement or HE32 recovery.
Next numerical comparison must use identical captured attention inputs and an
independent reference.

The other four package RECORDs are identical to r128; five full installed
RECORDs, source/wheel/installed binding and full checkpoint pre/post seals pass.
All other90 Plugin payload files are unchanged. The frontend exits0 in9.53s
under the original15s grace, no forced cleanup, owned descendants and port19994
released; native nanobind leak diagnostics persist. Original accuracy and
lifecycle gates remain open. Adapter main advanced separately to `ef63bbbd`
during this diagnostic; these results remain bound to installed `89fc929c`.
The reader experiment remains local and is absent from this MR's production
tree. Evidence: `evidence/plugin410-e5m2-frozen-history-r128` and
`evidence/plugin410-official-reader-r129` in the upgrade workspace.

## Component timing and limits

One gfx936, BF16, HND, 32 query/8 KV heads, head size128, cap0.5: three warmed
repetitions of 100 eager calls per shape give the following medians. GPU events
span the call loop, including launch gaps; these are not isolated kernel times
or serving latency measurements. Clocks/power are not locked.

Reproduce with `python -I tools/bench_softcap.py --output /tmp/cap-timing.json`
in the installed candidate environment with one idle HCU visible.

| Query/KV tokens | Core fallback (us/call) | Unchecked vendor (us/call) |
| --- | ---: | ---: |
| 1/128 | 206.11 | 83.92 |
| 1/2048 | 195.08 | 94.95 |
| 128/128 | 172.79 | 1002.45 |
| 512/512 | 288.02 | 1046.78 |

The unchecked vendor is numerically incorrect at these positive-softcap
shapes (maximum absolute differences 0.708/0.124/4.538/5.674 from the FP32
reference). Its timing does not establish a valid performance baseline.
Decode can cost more; no softcap speedup or serving performance pass is claimed.
Existing native finalizer warnings and forced worker shutdowns remain recorded.
These results do not certify all Flash accuracy, other hardware or the complete
upgrade. DeepSeek V4 remains deferred until the preceding adaptation is complete.


## 2026-11-18 pretrained Gemma2 BF16/E4M3 four-arm follow-up

This cohort tests runtime `877a96ad81df020f7e708467c94a2c4bddb63bb0` on actual
**gfx936**, official Core 0.33 `98a09850d56b` and unchanged baseline Adapter.
The installed Plugin's 91 production files match their published Git blobs.
Adapter/kernel payloads match the baseline (279 files); the build's
`exitcleanup1` version suffix does not imply Draft Adapter #140's exit hook.

The unchanged pretrained `google/gemma-2-2b-it` checkpoint is pinned to
ModelScope revision `c400b2564e9db9e5a0bc6dfbe07a05da3cd09f52`; its 12 assets
and full pre/post package RECORDs were verified. Attention head size 256, softcap 50,
13 full layers and 13 sliding 4096 layers remain intact. All four arms use TP1,
BF16 compute, context8192, block64, default FULL_AND_PIECEWISE Graph and
prefix caching, original15s shutdown grace, and the same original HE0–31
requests at temperature0/seed0/max_tokens2048. We retain failed format probes.

| Arm | Raw HE32 | Normalized HE32 | Completion tokens | Normal stops |
| --- | ---: | ---: | ---: | ---: |
| reference-bf16 | 0/32 | 23/32 | 4998 | 32 |
| flash-bf16 | 0/32 | 23/32 | 4986 | 32 |
| reference-e4m3 | 0/32 | 24/32 | 4969 | 32 |
| flash-e4m3 | 0/32 | 24/32 | 4979 | 32 |

Pairwise original passing-task sets and answer comparisons:

- reference-bf16 / flash-bf16: identical request payloads=True; raw/normalized answer texts identical 30/32 and 30/32; lost passes {"raw": [], "normalized": []}; gained passes {"raw": [], "normalized": []}.
- reference-e4m3 / flash-e4m3: identical request payloads=True; raw/normalized answer texts identical 28/32 and 28/32; lost passes {"raw": [], "normalized": []}; gained passes {"raw": [], "normalized": []}.
- reference-bf16 / reference-e4m3: identical request payloads=True; raw/normalized answer texts identical 25/32 and 25/32; lost passes {"raw": [], "normalized": []}; gained passes {"raw": [], "normalized": ["HumanEval/18"]}.
- flash-bf16 / flash-e4m3: identical request payloads=True; raw/normalized answer texts identical 27/32 and 27/32; lost passes {"raw": [], "normalized": []}; gained passes {"raw": [], "normalized": ["HumanEval/18"]}.

Raw scores retain the original response-format failures; normalization uses
the unchanged isolated evaluator. Every score verifies its input hashes,
namespace isolation, nobody UID/GID and absent host credentials/GPU nodes.
The first official BF16 arm generated all32 answers before scoring rejected
a temporary output symlink resolving outside its allowed root. That failed
run and its raw bytes were preserved. All27 existing evidence files were
copied and verified under a real permitted directory; only scoring resumed.
The other three arms use real output directories. No answers were regenerated
for this recovery and no scorer/path/normalization checks were relaxed.

Worker probes confirm26 selected attention implementations and the original
13 sliding/13 full cache groups before/after every arm. The three remaining
arms additionally inspect every implementation's softcap 50/head size 256/window bounds
by RPC; that additional RPC was not collected in the first official BF16 arm
because its original scoring wrapper failed. Native traces show official
`kernel_unified_attention` and native cache writes. Each arm retains 3840
prefix-cache hits and two concurrent waves.

**Qualification limits:** strict functional probes that request only an integer
remain failures when the model emits an equation. Inspect the original
`functional-probe-status.json` for each arm. All recorded services, workers,
ports and device handles release, but native leaks and forced worker cleanup
remain failures of the original lifecycle gate. These short HE32 inputs do
not cover contexts beyond the 4096-token sliding window or cache reclamation
at that boundary. Source/native tests are separate evidence. gfx936 results
do not qualify gfx836; independent earlier Qwen3 BF16 and gfx938 E5M2 losses
remain recorded. Keep Draft and `last_good=false`.

Runtime CI44305568352 is **success**:8 source,1204 portable,383 gfx936 native/Graph and1 gfx938 synthetic case, all zero failures/errors/skips. The64 local sliding cases overlap with the383 CI native cases. Full immutable-head checkout logs and artifact SHA256 digests were verified. gfx938 is synthetic operator coverage. This evidence belongs to the tested runtime SHA above; the following documentation commit has no new runtime payload.


## 2026-11-18 Gemma2 long-context four-arm comparison

This separate boundary probe uses the same unchanged pretrained Gemma2
checkpoint and runtime payload as the four-arm HE32 cohort above, on **gfx936**.
Core remains 0.33 (`98a09850d56b`); there is no new Core or Adapter change.
TP1, BF16 compute, context8192, block64, batched tokens1024, default
FULL_AND_PIECEWISE Graph, prefix caching, softcap50 and the original
13 full/13 sliding4096 layers are preserved. Worker RPC checks all26 actual
attention implementations, window bounds, head size256, softcap and cache dtype.
All six installed package RECORDs and the checkpoint seal match before/after
each arm; the published Flash runtime is byte-identical to the tested source.

Six fixed integer-token prompts have lengths4095,4096,4097,6144,6144,4352;
their requested keys are RED, BLUE, GREEN, GOLD, SILVER and BLACK. A subsequent
four-request concurrent wave repeats RED/GOLD/SILVER/BLACK. Identical request
payloads use temperature0, seed0, max_tokens16 and sampled/top5 logprobs.
The first-wave files retain their original `*-cold.json` names, but later
first-wave requests reuse common prefix blocks too. These are long sequence
positions and cache reuse probes, **not evidence of a completely uncached
6144-token prefill** or a performance benchmark.

| Arm | Exact key after whitespace | Normal stops | Completion tokens | Prefix hits | Actual Graph launches |
| --- | ---: | ---: | ---: | ---: | ---: |
| reference-bf16 | 10/10 | 10/10 | 40 | 42624 | 8 |
| flash-bf16 | 10/10 | 10/10 | 40 | 42624 | 8 |
| reference-e4m3 | 10/10 | 10/10 | 40 | 42624 | 8 |
| flash-e4m3 | 10/10 | 10/10 | 40 | 42624 | 8 |

Exact greedy token/text comparisons over the same ten requests:

- reference-bf16 / flash-bf16: equal token sequences 10/10; equal texts 10/10; maximum sampled-logprob difference along identical histories 0.015656114.
- reference-e4m3 / flash-e4m3: equal token sequences 10/10; equal texts 10/10; maximum sampled-logprob difference along identical histories 0.030717403.
- reference-bf16 / reference-e4m3: equal token sequences 10/10; equal texts 10/10; maximum sampled-logprob difference along identical histories 0.03996912.
- flash-bf16 / flash-e4m3: equal token sequences 10/10; equal texts 10/10; maximum sampled-logprob difference along identical histories 0.035611284.

The logprob differences are observations, not a new numerical tolerance.
Raw logprobs, integer prompt IDs, generated token representations, request/
response hashes and unchanged-source checks remain in the evidence. Native
profiles show actual `kernel_unified_attention` execution in all four arms.
The E4M3 reference traces Core's `reshape_and_cache_kernel_flash`; its retained
compiled IR has `f8E4M3FN` key/value cache pointers. The E4M3 Plugin traces
the native HCU byte-cache writer. The original review expected the Plugin
writer name in both arms and failed; the corrected review checks each actual
provider plus the reference's compiled format. Both review scripts and the
failure are retained, with no regenerated responses. Traces cover the final
first-wave request and the concurrent repeat wave; earlier requests are
supported by API responses and runtime policy inspection, not by those traces.
All repeated keys match their own first-wave output.

**Remaining gates:** native leak diagnostics persist in all four arms;
the original15s cleanup also force-kills remaining workers in reference-bf16, flash-bf16, flash-e4m3.
Recorded services/descendants release, but the original lifecycle gate stays
failed. This key-retrieval probe does not replace HE32, resolve strict format
probes, qualify cache reclamation, or clear earlier Qwen3 BF16/E4M3 and gfx938
E5M2 task losses. There is no gfx836 qualification or full-feature/last-good
claim; keep Draft. The separate original HE32 raw/normalized results above
are unchanged. DeepSeek V4 remains last.

Retained evidence: upgrade workspace
`publication/plugin410-long-context-20261118-r145/`, with exact requests,
40 responses, pre/postflight seals, runtime inspections, metrics, profiles,
original cleanup logs, controllers and response/trace review output. The earlier
tokenizer/orchestration failures are retained separately and are not model
results. This follow-up changes documentation and the skill reference only.


## Current merged-Plugin control on actual gfx938 (r146)

The release branch now includes required-entry-point follow-up #464, merged as
`45befb54005d26ea684cccf5568ed63659a3e276`. Its complete tree equals tested
control source `8dad9325966c787e463f7c954b3bbdc44f2a5938`. The new local candidate
`46bd710afdca96b320e7a08926a171d5e3e85d72` combines this MR with that control.
Merging the actual release commit into this MR produces the same complete tree;
this follow-up then changes validation/skill text only. Production is still the
Plugin adaptation of official `unified_attention`; no new Core/Adapter patch.

Actual local devices are **gfx938**, verified in the worker device properties
with UUID/PCI identity. An initial component harness expected gfx936 and refused
before pytest. Its failed preflight is retained. A separate explicitly gfx938
run completes the three affected installed test files: **251/251 pass**, zero
failures/errors/skips, including all64 sliding-window native-writer/forward,
ragged/empty-row, softcap, BF16/FP16/E4M3/E5M2 and changed-input Graph cases.
Package RECORDs match before/after, no package was reinstalled, and owned
processes are absent. This does not qualify gfx936 or gfx836 on this allocation,
or convert the earlier preflight refusal into a test pass.

The BF16 model comparison restores original GPU7/admission0.40/fixedKV1GiB,
Qwen3-8B, default Graph/prefix, batch/token limits and the byte-identical original
HE0–31 requests at concurrency4. Both arms use fresh caches/default tuning;
there is no replay of old tuning choices. Core is `c440c089fe96`, Adapter is
`89fc929c1455`; this is not qualification of a latest Adapter/main wheel or a
new native release build. All36 actual attention layers report the expected
Plugin implementation and BF16 storage. Isolated scoring retains raw and
normalized results separately.

| BF16 arm | Raw | Normalized | Normalized failures | Completion tokens |
| --- | ---: | ---: | --- | ---: |
| merged-Plugin-equivalent FLASH control | 20/32 | 30/32 | /18, /26 | 4198 |
| local current-release + #410 FLASH candidate | 20/32 | 30/32 | /18, /26 | 4199 |

There are no gained/lost raw or normalized passes between these two arms;
31/32 complete emitted token sequences match. Task1 differs at emitted token197,
with lengths202/203, and passes in both. Each arm passes13 arithmetic and3 prefix
probes, records1280 prefix-hit tokens and genuine Graph/kernel traces. There is
no generation truncation. Frontends exit0 within the original15s grace
(9.478s/10.530s); descendants and ports are released, all RECORD/checkpoint seals
match, but nanobind/EngineDeadError diagnostics keep lifecycle unqualified.

The separately completed earlier official-TRITON cohort (r144) remains
raw20/32, normalized31/32, failing only /26. It was not repeated or relabeled as
a concurrent control. The new current Plugin control also loses /18 relative
to that official cohort. This changes attribution of the previous #410-only
claim: this increment has no new pass-set loss in the current BF16 comparison,
but recovery of official accuracy is **not established**. Original failures,
format findings, broader cache-reclamation/architecture gates and the independent
E5M2 history remain recorded; no accuracy threshold or lifecycle gate is waived.
Keep Draft and `last_good=false`.

Evidence: upgrade workspace `evidence/plugin410-released-two-arm-r146/`
(`analysis.json`, `affected-native-gfx938/receipt.json`) and original retained
model receipts under `plugin410-unexecuted-after464-r145/` and
`plugin410-default-after464-r144/`. No DeepSeek V4.1 checkpoint was tested.


### Current-control E5M2 follow-up

The separately completed E5M2 pair preserves the same frozen sources, actual
gfx938 GPU7, original requests/concurrency4, admission0.40/fixedKV1GiB, default
Graph/prefix, fresh default-tuning caches and original15s grace. Worker RPC
verifies36 actual Plugin Flash implementations with public `fp8_e5m2` and byte
cache storage, before/after generation; no installed package or native ABI was
changed during the run.

| E5M2 arm | Raw | Normalized | Normalized failures | Completion tokens |
| --- | ---: | ---: | --- | ---: |
| merged-Plugin-equivalent FLASH control | 21/32 | 30/32 | /19, /26 | 4076 |
| local current-release + #410 FLASH candidate | 21/32 | 30/32 | /19, /26 | 4076 |

All32 emitted token sequences and prompt token IDs match across the pair, with
identical raw/normalized passing-task sets and no truncation. Both pass all13
arithmetic/3 prefix probes, with1280 prefix-hit tokens and actual Graph/native
writer/attention traces. Frontends exit0 in9.429s/10.330s, no owned descendants
remain, ports close, and complete package/checkpoint post-seals match. The logs
still contain native leak/EngineDeadError diagnostics, so lifecycle remains red.

This establishes no new E5M2 pass-set/token loss for this increment against
current Plugin mainline. It does **not** remove the previous KV-dtype /19 loss:
that failure also occurs in the current E5M2 control, whereas the BF16 controls
pass /19. BF16 /18 versus the earlier official cohort, format/lifecycle and
remaining qualification gates are likewise retained. No kernel cause, complete
accuracy recovery, performance or full-upgrade acceptance is claimed. Keep
Draft. `e5m2-analysis.json` and `current-e5m2-pair-inputs-fixed/` retain the
completed evidence. The first E5M2 orchestration attempt failed on missing
helper paths before starting a model; its original failure remains separately
under `current-e5m2-pair/` and was not relabeled as a model result.


## Current-head component CI and fresh baseline (2026-11-19)

PR #410 runtime head `74967a41f9b058d15371941366b1709227e5d26d` completed
[run44503238282](https://github.com/HYGON-AI/vllm-plugin-das/actions/runs/44503238282)
with all eight jobs successful. Artifact ZIP digests, JUnit cases and actual
checkout were checked: source8, portable1382, gfx936 native/Graph401 and gfx938
components8 passed, with no JUnit failures/errors/skips. Logs retain portable46
and native26 deselections. Eleven public-selector CPU ABI records pass without
compute calls or GPU initialization. gfx938 covers a synthetic Channel-FP8 expert
and RoPE components, not a checkpoint or gfx836. The workflow source is6fa728d2;
the resolver targets74967a41 and the immutable runtime image is recorded in the
[structured evidence summary](vllm033-flash-current-20261119.json).

A separate fresh installed environment binds all107 Plugin wheel payload files
to that tested production head, using Core0.33/98a09850 and Adapter0.5.22/8ba62c
whose runtime bytes match main9e2340aa. CPU271 and actual gfx936 native283 pass;
the latter is an exact case-ID subset of CI's401. The initial native run had
33 passes and250 missing-ROCm-device-library compilation failures because its
isolated launcher omitted the existing ROCM_PATH/HIP_PATH. The failure is
retained; restoring those roots makes the unchanged283-case matrix pass without
changing tests, tolerances, provider, source or wheel payloads. Complete package
RECORDs and frozen sources remain unchanged.

The fresh Qwen3-8B no-Plugin official TRITON_ATTN baseline uses TP1/BF16,
block64, context4096, max sequences16, batched tokens1024, fixed1GiB KV, default
FULL_AND_PIECEWISE Graph and enabled prefix caching. Original HumanEval0–31
uses temperature0/seed0/max_tokens2048/thinking-off. This preserved client sends
HE tasks sequentially; its separate arithmetic probes use concurrency4. This
schedule differs from the historical concurrency4 cohorts above and does not
replace them. All32 answers stop normally: raw20/32, existing normalization31/32,
with only task26 failing normalized scoring. The scorer runs in the unchanged
read-only chroot/mount/net/PID/uid65534 isolation. Arithmetic13/13, including
prefix3 and two four-request waves, passes; repeated prefix requests add3840
cache-hit tokens. Worker RPC confirms36 official Triton attention layers and
BF16 storage; the profile records actual Graph and official kernel execution.

The API exits0 after SIGINT in29.013s; the Core's original15-second grace remains
unchanged. No owned process remains and the port is released. Native nanobind
leak diagnostics remain, so full lifecycle is not qualified. Before scoring and
the next arm, the controller's global-resource gate refuses progression: KFD
shows foreign PID-namespace processes, with eight active workers allocating
VRAM across all eight GPUs. Their vpid values are zero in this namespace; they
are not attributed to the completed baseline. The original controller failure
is retained, and existing answers are scored separately without regeneration.

Plugin FLASH BF16/E4M3/E5M2 model arms have **not started** in this fresh cohort.
No fresh model parity, performance, softcap/window checkpoint, gfx836 or full
release claim follows from these component and baseline results. Historical
BF16/task18, E5M2/task19, strict-format and lifecycle failures remain unwaived.
Keep Draft and last_good=false. The fresh local evidence root is
`/data/models/plugin410-e2e-20261119-r722`; remote CI uses its separately recorded
image/dependencies. This documentation follow-up changes no runtime code and
does not relabel old-head CI as a successful full run of a later commit.


## Fresh four-arm results after resource release (2026-11-19)

The previous section is a historical snapshot before device resources became
available. The original controller's resource refusal is preserved. After KFD
was independently empty, a continuation ran only the three unstarted Flash
arms, preserving the original checkpoint seal, requests, client/scorer bytes,
backend/dtypes, default FULL_AND_PIECEWISE Graph, prefix caching and15s Core grace.
No completed baseline or answer was regenerated. All four arms below use the
same recorded Core0.33/98a09850, Adapter0.5.22/8ba62c runtime-equal-main9e, and
vendor cohort; Plugin production is74967a41. Later documentation heads do not
become new runtime or full-CI qualifications.

After E4M3 the unchanged port-bind check refused progression, despite no owned
process or KFD survivor. A separate read13s later found the port bindable. A
final continuation ran only original E5M2 arm3 after checking both ports/KFD;
its post-run port check also refused, with no owned/KFD survivors. Subsequent
TCP-table/bind observations found that port released. Neither transient cause
was established. Both original controller failures/terminals stay unchanged;
existing answers were scored in the unchanged isolated evaluator without
regeneration. A later released port does not retroactively pass either gate.

| Arm | Raw HE0–31 | Existing normalization | Lost normalized tasks vs official | Prefix-hit tokens | Explicit Core force cleanup |
| --- | --- | --- | --- | --- | --- |
| official | 20/32 | 31/32 | none | 3840 | yes |
| candidate-bf16 | 20/32 | 30/32 | HumanEval/18 | 3840 | yes |
| candidate-e4m3 | 20/32 | 31/32 | none | 3840 | yes |
| candidate-e5m2 | 21/32 | 29/32 | HumanEval/15, HumanEval/19 | 3840 | no matching log |

Each arm has32 normal stop responses and13 passing functional arithmetic probes.
Actual worker reports show36 official Triton layers for the no-Plugin baseline
and36 HcuFlashAttentionImpl layers for each candidate. BF16 uses BF16 physical
cache; E4M3/E5M2 use byte physical storage with the explicit requested cache dtype.
Graph/native traces and complete source/RECORD/checkpoint checks are retained.
The structured record preserves actual kernel names, cache resolution, full
extension reports (including unavailable/unrequested capabilities), pass sets,
raw-score failures, stop reasons and per-answer token counts.

Candidate worker RPC identities are bound through saved PID/startticks, KFD
namespace vpid and active queue gpuid to PCI0000:f5:00.0, unique ID0x14bc4a274e18a1,
actual gfx936. Each candidate's nine mapped Core/provider native binaries match
the BF16 bytes; library mapping alone is not execution proof and is paired with
the actual runtime/profiler records. This is not gfx836 qualification.

The BF16 raw passing-task set is identical to official. Existing normalized
scoring losesHumanEval/18 and gains none; full messages differ at/12 and/18.
The candidate's generated regex omits the positive lookahead needed to count
overlapping occurrences. This identifies the wrong generated answer, not a
causal attribution to a numerical kernel or this MR's softcap/window delta.
The two arms select different attention implementations. The checkpoint has
zero softcap and full-window attention, so this model does not exercise the
positive-softcap/window branch added by this MR.

API exit0 and released devices/ports do not qualify normal lifecycle. The
structured record retains each force-cleanup line and native nanobind diagnostic;
the unchanged15s EngineCore deadline is not extended. Accuracy differences,
historical strict-format/softcap/window model failures and gfx836 remain open.
Keep Draft, last_good=false; no performance or full-upgrade claim follows.

The earlier structured summary referenced the mutable convenience file
`models/independent-review.json`. Its exact older derived bytes were not retained
when a new arm was audited; raw generations/scores/logs and their hashes remained
unchanged. That historical hash is now explicitly labeled in the structured
record. Current/future derived audits retain immutable content-addressed snapshots
before a latest-view update. No missing historical snapshot was fabricated.

Immutable final review SHA256: `200d67463edbb487066649918135ee7fb5180f1eb7214fff20ae65aec7cf4367`.
See [the full structured result](vllm033-flash-current-20261119.json) for per-arm
comparisons, raw-input hashes, original failures and evidence paths.


## Current-stack Gemma2 checkpoint and window-boundary comparison (2026-11-19)

This cohort actually exercises positive attention softcap50 and alternating
sliding4096/full layers on the unchanged pretrained Gemma2-2B-IT checkpoint
revision c400b2564e9db9e5a0bc6dfbe07a05da3cd09f52. It is separate from the
preceding full-window, zero-softcap Qwen3 cohort and the older Gemma stack.
Production Plugin remains74967a41; Core0.33/98a09850 and Adapter0.5.22/8ba62c
(runtime-equal-main9e) and the recorded provider cohort are preserved.

Both completed BF16 arms use TP1/BF16 weights, context8192, maxseq16, batch-token1024, block64,
memory-utilization0.7, default V2 Model Runner/FULL_AND_PIECEWISE Graph and prefix
caching. No fixed KV-memory override is applied, matching the original Gemma
command; this differs from Qwen's fixed1GiB cohort. The original HE0–31 client
and isolated scorer are unchanged; only the served model name changes in the
HE request hash. Each completed BF16 service then receives the original six exact-token
completion prompts of4095/4096/4097/6144/6144/4352 tokens and four concurrent
repeats. Existing answers are never regenerated when an original gate stops
progression. Detailed controller and resource failures remain in the record.

| Arm | Raw HE0–31 | Existing normalization | Strict functional probes | Long requests | Key matches ignoring surrounding whitespace | Prefix-hit tokens, HE / long |
| --- | --- | --- | --- | --- | --- | --- |
| official-bf16 | 0/32 | 23/32 | fail (3 recorded) | 10 | 10/10 | 3840 / 42624 |
| flash-bf16 | 0/32 | 23/32 | fail (3 recorded) | 10 | 10/10 | 3840 / 42624 |

Passing-task sets, full HE message differences, raw failures and each long
request's literal text, token strings, sampled logprobs and expected key are
preserved. Key matching after surrounding-whitespace removal is reported
separately from literal text equality; it does not pass strict formatting probes.
Pairwise equality is not used as a substitute for checking the original expected
key. Sampled-logprob deltas are compared only along identical sampled token
strings/histories and do not represent full-vocabulary equivalence.

| Matched pair | Raw lost / gained | Normalized lost / gained | Long equal texts | Long equal token strings | Maximum sampled-logprob delta on equal strings |
| --- | --- | --- | --- | --- | --- |
| official-bf16...flash-bf16 | none / none | none / none | 10/10 | 10/10 | 0.01569509506225586 |

Only the BF16 matched pair completed generation and scoring: official and Flash each raw0/32 and existing-normalization23/32, identical passing-task sets.30/32 full HE messages match. Both differing tasks still fail in both arms: HumanEval/10 constructs an incorrect shortest palindrome; HumanEval/21 uses incorrect rescaling (official omits subtracting the minimum, Flash omits division by the range). These differences are not passed off as formatting-only errors. All32 responses in both arms stop normally.

All10 window-boundary/concurrent long outputs have identical text and sampled token strings between BF16 arms. They match the original expected keys only after surrounding-whitespace removal (10/10), while literal expected-key equality is0/10. Maximum sampled-logprob difference on the matching histories is0.0156950951; this does not measure full-vocabulary equality. HE prefix-hit tokens increase3840 per arm; long-request hits increase42624. Both retain3 original strict-format failures, and Flash retains forced EngineCore cleanup; both report native nanobind leaks. Later port release does not clear those original failures.

The planned official E4M3 service failed before readiness or generation: at worker initialization only30.62/63.98GiB was free versus the unchanged0.7 requirement of44.79GiB. Other-namespace KFD processes acquired the device after an initially idle observation. Zero E4M3 samples were generated, no FP8 accuracy conclusion follows, and the Flash E4M3 arm was not started. Original failed attempt and all source/checkpoint/RECORD checks remain retained. Further device runs must wait for resource availability; the inactive-context admission exception used by this failed attempt is not reused. Neither full lifecycle, model accuracy acceptance, performance, gfx836 nor the full upgrade is qualified.

Each completed BF16 service's worker reports26 actual attention implementations,13 full and13
sliding windows, head256 and softcap50 with the requested BF16 cache dtype. The failed E4M3 startup has no worker RPC or generation evidence.
Actual Graph and kernel_unified_attention traces, physical cache layouts,
complete package RECORD/checkpoint seals and mapped native binaries are retained.
PID/startticks, namespace-aware KFD vpid and queue gpuid bind actual workers to
the recorded physical gfx936 device. This does not qualify gfx836.

Original strict-format probe failures, native nanobind diagnostics and immediate
port-bind failures stay failed. Original TCP rows are preserved; a TIME_WAIT
explanation is applied only to arms whose captured rows establish state06.
Later bare-bind success permits continuation of original unstarted arms and
does not retroactively pass any failed gate. Original15s EngineCore grace is
unchanged; API exit0 does not establish clean native lifecycle. Keep Draft and
last_good=false; no performance or full-upgrade acceptance follows.

Immutable current Gemma review SHA256: `69da52ed059050a4ea136932b3c25084d56ecdb610da44a391ef40ce9f6c31f0`.
The structured record retains per-arm provenance, controller transitions,
original requests, expected-key checks and every raw/normalized task result.
## Official layer-scale attention candidate

The earlier R230 candidate passed scalar layer descales through Core's
`KVQuantMode.FP8_PER_TENSOR`. Its unchanged native ragged-window reference
exposed **24 failures / 96 cases** for scales0.7/0.3 in BF16-query E4M3/E5M2,
both layouts. Core dequantizes FP8 KV in FP32 but then rounds K/V to Q's dtype
before QK/PV. An external diagnostic matches the BF16-rounded-cache reference
while failing the original FP32 reference. Do not relax the original0.016
absolute/relative bounds or treat the earlier portable132 pass as native pass.

The candidate instead represents a scalar layer scale as a zero-stride
`[blocks, page, KV heads]` view and calls the selected Core's official
`KVQuantMode.FP8_PER_TOKEN_HEAD` with those scale-cache views. This mode applies
K scales to FP32 scores before softcap and V scales to probabilities before PV;
FP8 values are exactly representable in the query dtype. Layer scales are the
constant special case of this API. No cache-sized scale allocation, KV copy,
CPU scalar read or Q/output rescale is needed. Typed E4M3/E5M2 storage, native
writers, metadata, Q and output buffers retain their existing owners.
FlashAttention source and binary are neither changed nor rebuilt.

The route is admitted only for an actual scalar or a fully zero-stride
`[batch, KV heads]` layer broadcast. Nonuniform batch/head descales retain the
existing scale adapter: a physical token/head scale cache cannot encode
request-dependent scales for shared blocks. Selection uses tensor metadata.

In R232's external diagnostic only, the original E4M3 and E5M2 window17 failing
tests pass with this official scale-cache mode, including changed-input Graph.
That wrapper is not the installed candidate and does not qualify native/model
acceptance. The production candidate requires its own wheel and full matrix.
The matrix preserves the original references,0.016 bounds, both layouts,
native writer, softcap0/50 and windows1/17/64/4096. It adds FP16-query FP8
coverage and in-place scale changes between Graph replays while retaining the
original fixed-scale BF16 cases. Current model precision, normal shutdown and
performance remain unqualified; historical failures above are not waived.


### Native FP16 E4M3 writer input precision

R233's expanded320-case device matrix passes the original96 cases but exposes
32 failures in the added FP16-query/E4M3,0.7/0.3-scale cases, both layouts,
with fixed or changing scales and softcap0/50. All original references and
0.016 bounds remain unchanged. This is not a passing native candidate.

The selected Core native writer's half specialization stores `x / scale` back
into a half temporary before converting it to FP8; its FP32 specialization
converts directly. R234's external controlled native byte comparison confirms
the mechanism: first-wave half inputs differ from the FP32 quantization
reference at662 K and1058 V elements out of276480 each, while exactly matching
a half-rounded reference. Promoting inputs before the same native writer
matches every FP32-reference byte. Three unchanged window1/17/64 tests,
including softcap50 and changing-input/scale Graph replay, pass in that
diagnostic. This does not change an installed package or qualify production.

The new candidate promotes only FP16 inputs on the E4M3 writer route to FP32,
then calls the unchanged native HCU quantize/scatter. It retains the source
Q/K/V buffers and caller's cache/scales/slot mapping; two temporary FP32 input
buffers scale with the supplied input shape, including padded rows. Other
input/cache dtype routes retain their existing inputs. No native library or
FlashAttention source/build changes. Final wheel/native/model and performance
qualification must include this candidate; the earlier reader-only results
cannot qualify the added writer branch.

### Installed candidate regression and matched controls

Runtime candidate `39eb839235ebb9fed4fa6fcee85d25ac0d51aaab` passes
1426 portable and539 native cases (320 window/scale,32 nonuniform,74 Flash,
113 E4M3), with zero failures/errors/skips and original references/tolerances.
Installed-source, wheel payload,18 RECORDs and checkpoint bytes were sealed.
These results identify that runtime, rather than a fresh CI run on later
documentation-only commits.

Ordinary Qwen3-8B TP1/BF16 weights, default Graph/prefix and fixed HumanEval0-31
requests complete for both KV types, each with strict17/17 probes and1280 prefix
hit-token delta. E4M3 scores raw20/normalized31; E5M2 scores raw21/normalized30.
Both exit0 within the original15s, without forced cleanup or owned survivors,
but both retain nanobind leaks; E5M2 also logs EngineDeadError. Complete
lifecycle therefore fails. This checkpoint does not qualify pretrained
softcap/window behavior.

Matched merged033 control `c0e608de`, with the same proposed Adapter97/Core,
checkpoint, device, commands, environment, requests and fixed dependency
payloads: E4M3 control raw19/normalized30 versus candidate20/31 has no newly
failed task but five changed messages. E5M2 control raw22/normalized31 versus
candidate21/30 has a newly failed HumanEval/19 in both scoring modes, with
messages changed on15/19/31. In19, numeric Enum values become strings, so the
candidate sorts lexicographically. Preserve that loss; one matched pair does
not establish deterministic causality. Ordinary zero-softcap/full-window E5M2
does not require the added softcap/window or FP16/E4M3 branch. An unchanged
baseline fresh-service repeat remains a separate diagnostic, not acceptance.

That unchanged merged-Plugin baseline repeat is now terminal. With the same
physical device, checkpoint, commands, environment, fixed requests and default
Graph/prefix policy, it scores raw21/normalized30 and fails19/26 after
normalization. Baseline distributions and payloads are unchanged from its
original22/31 run. All32 raw and normalized pass flags now match the candidate,
but only27/32 messages match; task19's message is different too. This establishes
that task19 can fail on the unchanged baseline, not that its exact candidate
continuation was reproduced or that outputs are equivalent. The original
observed loss remains recorded; neither numerical causality nor acceptance
follows from this repeat. Both serving arms retain nanobind/EngineDeadError
lifecycle failures despite exit0 within15s and no forced cleanup or survivors.
An independent read-only audit recomputes these counts from the saved score
flags and raw responses, without rerunning generated code. Evidence:
`evidence/plugin410-e5m2-baseline-repeat-r240/` and
`evidence/plugin410-repeat-audit-r245/audit.json`.

A real-device configuration-only probe also admits the available Gemma3
text encoder from an LTX pipeline: original
`Gemma3ForConditionalGeneration`, head256,48 layers (40 sliding/8 full),
window1024, BF16 compute/E5M2 KV, default FULL_AND_PIECEWISE Graph and prefix.
The official `limit_mm_per_prompt={"image": 0}` disables mm-prefix mode and
the real public selector returns `HcuFlashAttentionBackend`. The matching
pipeline tokenizer loads and tokenizes. The earlier device-hidden attempts
are retained as failed: HCU detection correctly declined activation without
an available HIP device, leaving the ordinary ROCm selector. This follow-up
uses a real gfx938 device and initializes HIP; it is not CPU-only evidence.
Installed payloads/18 RECORDs verify before and after. No model weights were
loaded, so this does not qualify pretrained window behavior, accuracy,
softcap or serving performance. Evidence:
`evidence/gemma3-window-config-r245/` and
`evidence/gemma3-window-config-r245-controller/`.

FP16/E4M3 writer component timing on gfx938,8 KV heads/head128 and five warmed
100-call samples per route: corrected Graph replay intervals are about
12.5–13.7us at1–128 unpadded tokens, versus6.8–8.0us for the numerically
incorrect unpromoted diagnostic. At8 active/1024 supplied rows, correction is
about17.3us and adds8MiB of FP32 input payload. The old wrong route is not a
qualified alternative. These GPU-event intervals include submission gaps and
do not establish kernel-only or serving performance. No FA library changes
or tolerance relaxation; keep Draft pending accuracy investigation, real
softcap/window checkpoint, clean lifecycle, serving performance and the
proposed Adapter97 dependency's independent qualification/required approval.

## Real Gemma3 window BF16 pair (2026-11-19)

The original Gemma3 conditional-generation text encoder and bundled tokenizer
from the local LTX pipeline complete an official `TRITON_ATTN` / Plugin
`FLASH_ATTN` pair. Both use the same gfx938 device UUID, checkpoint bytes,
Core0.33 `c440c08`, proposed Adapter97 `6eeea954`, BF16 weights and `auto` KV,
TP1, max length4096, block64, max sequences8/batched tokens1024, official
`limit_mm_per_prompt={"image":0}`, and default Graph/prefix. Plugin runtime
is `39eb839`; later documentation commits do not change its wheel inputs.
The original model configuration, including all48 attention layers, is retained.

An independent audit reads the saved raw responses, scores, worker metadata
and compressed rank0 traces. It checks original source seals and reconciles
the14 checkpoint assets' before/after double-read hashes; it does not rehash
weights or execute generated code again.

| Check | Official Triton | Plugin Flash |
| --- | --- | --- |
| Actual per-layer windows | 40 sliding `(1023,0)`,8 full | Same |
| Literal expected answers, input lengths1023/1024/1025/2049 | 16/16 | 16/16 |
| Concurrent waves1/4/8, repeated real token IDs | Equal | Equal |
| Prefix hit-token delta | 17856 | 17856 |
| Actual `hipGraphLaunch` / correlated GPU kernels | 6 /84 | 6 /72 |
| GPU `unified_attention` kernels in trace | 150 | 126 |
| HumanEval0–31 raw / normalized | 1/32 /32/32 | 1/32 /32/32 |
| Exit within original15s, exit0, no forced cleanup/survivors | Yes | Yes |
| Complete lifecycle | FAIL: nanobind | FAIL: nanobind |

Before/after capture counters remain unchanged while the traces show real
HIP graph launches and correlated GPU work; capture counts alone are not used
as replay evidence. All16 window responses have identical literal texts and
real integer token lists across backends. All32 HumanEval messages, token
lists, finish reasons and raw/normalized pass flags match, with no newly
failed task. The independent `gemma3-pipeline-text` protocol retains temperature0,
seed0,2048 output tokens and concurrency4, without reasoning controls. Most
raw programs include Markdown fences: retain raw1/32 separately from the
isolated format-normalized32/32, without redefining the original accuracy gate.

Each model controller returns1 because nanobind leakage remains, despite
exit0 within15s and no survivors. This is real BF16 window-model evidence;
Gemma3 has no attention softcap, and these runs do not qualify E4M3/E5M2,
clean lifecycle, serving performance or all Plugin features. Separate FP8
cohorts are pending and compare changed KV precision to this reference rather
than claiming a same-precision old-Plugin A/B. FlashAttention is unchanged.
Keep Draft. Evidence: `evidence/gemma3-window-pair-r245/` and immutable
independent audit `evidence/gemma3-window-audit-r248/paired-contract/audit.json`.

## Real Gemma3 window FP8 cohorts completed (2026-11-19)

The E4M3 and E5M2 cohorts pending in the preceding section are now terminal.
Both retain the original Gemma3 checkpoint/tokenizer, BF16 weights, gfx938
device UUID, Core `c440c08`, proposed Adapter97 `6eeea954`, Plugin runtime
`39eb839`, default Graph/prefix, serving options and original15s exit grace.
Only the KV precision changes relative to the FLASH BF16 reference. This is
a quantization comparison, not a same-precision old-Plugin/candidate A/B.

| Check | FLASH BF16 KV | FLASH E4M3 KV | FLASH E5M2 KV |
| --- | --- | --- | --- |
| Literal expected window answers | 16/16 | 16/16 | 16/16 |
| Window real token lists equal to BF16 | Reference | 16/16 | 16/16 |
| Prefix hit-token delta | 17856 | 17856 | 17856 |
| Actual HIP graph launches / correlated GPU kernels | 6 /72 | 6 /84 | 6 /84 |
| GPU unified-attention kernels | 126 | 126 | 126 |
| HumanEval0–31 raw / isolated normalized | 1/32 /32/32 | 4/32 /32/32 | 0/32 /32/32 |
| Newly failed raw task relative to BF16 | Reference | HumanEval/4 | HumanEval/4 |
| Newly failed normalized tasks | Reference | None | None |
| Changed HumanEval messages / real token lists | Reference | 8/32 | 8/32 |
| Exit0 within15s, no forced cleanup/survivors | Yes | Yes | Yes |
| Complete lifecycle | FAIL: nanobind | FAIL: nanobind, EngineDeadError | FAIL: nanobind |

The independent audit verifies source seals, the original18 distribution
attestations, saved checkpoint double-read hashes, normalized launch options,
controlled environment and before/after worker metadata. All48 layers retain
the original40 sliding windows1024 and8 full layers. Boundary inputs remain
1023/1024/1025/2049 tokens with concurrent waves1/4/8; repeated window tokens
are stable, and capture counters remain unchanged during real graph execution.
Profiling is execution evidence, not serving performance acceptance.

E4M3 raw pass flags match BF16 for27/32 tasks; newly passed tasks8/20/28/29
raise the total while task4 is lost. E5M2 raw flags match for31/32 tasks.
All32 normalized pass flags match in both comparisons. E4M3 changes messages
and token lists on tasks4/8/10/15/19/20/28/29; E5M2 changes1/4/7/8/10/15/28/29.
In E4M3 task4, added Markdown fences cause the raw syntax failure, but the
normalized program also adds an empty-input branch: its AST differs from
BF16. Do not call this only a formatting change or output equivalence.
An improved aggregate raw score cannot waive a newly failed individual task.

Each FP8 model controller returns1 because lifecycle diagnostics remain.
Gemma3 has no attention softcap. Preserve the original raw accuracy gate,
per-task losses and output differences; these results do not qualify FP8
accuracy, real softcap checkpoints, clean lifecycle, serving performance or
all Plugin features. No FlashAttention source, binaries or build changes.
Keep Draft pending those gates and the proposed Adapter dependency's approval.

Evidence: `evidence/gemma3-window-fp8-r248/` and immutable independent audit
`evidence/gemma3-window-audit-r249/fp8-complete/audit.json`, SHA256
`61ae8f734751cca35e4ae9a3bc5ceb242b5ecba98243d016dda621c4d6d80d78`.
The audit reuses saved isolated scores without executing generated code or
rehashing the weights again; its executed script is retained beside the result.
