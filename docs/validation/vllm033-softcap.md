# v0.33 HCU Flash attention softcap

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
only the bounded metadata/seal is durably archived. gfx836 remains unqualified.

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
