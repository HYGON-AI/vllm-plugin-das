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
softcap keeps the existing vendor route and optional
prefill gather policy. Enabling prefill gather cannot bypass softcap dispatch.

This is an explicit provider fallback for a correctness boundary, and is not
evidence that the vendor implements softcap. The Core path produces the
public AttentionImpl output only; diagnostic requests for LSE or attention
probabilities are refused. Remove this fallback only after the vendor passes
the same writer/reader, non-unit-scale and changed-input Graph tests for both
prefill and decode, plus applicable model validation.

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
