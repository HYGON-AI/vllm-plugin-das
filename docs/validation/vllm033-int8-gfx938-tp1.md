# Public INT8 MoE on gfx938: scoped TP1 qualification

This increment extends exact architecture admission from gfx936 to gfx938 for
ordinary TP. EP2 remains restricted to qualified gfx936. Publication health checks
the official registry, bound functions/modules and vendor's effective plan
architecture. Compile-factor validation checks the frozen source/native/config
file bytes before ordinary/AOT cache reuse.

Core owns CT policy/loading, quantization, prepare/finalize and reduction. The
Plugin composes existing HCU AITER ASM stages through the merged public INT8
registry. Config/layout misses delegate the complete call to official Triton.

## Source and validation baseline

- Plugin parent: PR373 `3384779276da03ecee783254e92048cacc0f7871`; this gfx938
  increment is independently validated over its EP2 changes. PR372 packaging
  remains a dependency pending merge.
- Core target: `0b7b7bd231d687209ba3144f7006641acde9c14a`, with the independent
  [INT8 output precision fix, MR84](https://developer.sourcefind.cn/codes/OpenDAS/vllm/-/merge_requests/84)
  at `3d845a5e83c3f82bdeb6932bb9b445e0b1696728`, still unmerged at this record.
  That generic Core defect forced BF16 compute from already-INT8 inputs; the
  fix restores floating output precision. Plugin ASM math and gates were kept.
- Adapter0.5.22 public process-health ABI; the tested host additionally contains
  local gfx938 extension of PR66. Its unquantized BF16 hook is not an INT8
  dependency. DTK26.10/Torch2.13/Python3.12; gfx938 physical hardware throughout.
  The original R4 receipts use that paired Adapter's padding policy0. The later
  exact-main counterfactual below establishes independence for its exercised
  public INT8 TP1/default-padding workload; the original receipts stay unchanged.
- All20 Plugin runtime files match source, wheel and installed model environment.
  The Core precision artifact changes one Python file with seven native
  libraries byte-identical. Wheels are ABI-compatible local validation builds,
  not exact-version requirements.

## Components and CPU checks

Public selector/constructor EP-scope regression first passes its gfx936 control
and fails gfx938 refusal (`DID NOT RAISE`); the scoped fix then passes all41
architecture/public CPU cases. The independent source review closes the
gfx938-EP2 over-admission issue and reports no remaining P0–P3 source finding.

Actual host activation, CT raw loader/postprocess, public factory, complete
forward and changed-input Graph: BF16/FP16 × M1/16/128 at E256/H2048/I512/top8,
plus a BF16 small config miss. All seven pass with14 raw loader generations and
42 changed Graph replays. Parameter addresses remain stable across the raw
reloads. Six cases observe ASM; the small case observes ASM0/official fallback1.
The unchanged gates are RMS≥0.001, NMAE/NRMSE≤0.03, maxAbs≤0.001. The formerly
failing FP16/M128 has maxAbs≤0.000030517578125; Graph and eager outputs agree
exactly. This is component raw reloading, not public engine checkpoint reload.

Full project CI source tier:6 pass. Portable tier:385 pass,14 fail,6 marked HCU
cases deselected. The14 failures exactly match the pristine Plugin baseline:
`test_slimquant_ignore_groups.py` needs the unmerged Core PLE API
`Qwen4ExpPLEPerRowEmbeddingMethod`. Optional SlimQuant activation fails for that
same absent API. Full Plugin health/CI green is not claimed.

## Actual model/OpenAI and original HumanEval32

One checkpoint: `/models/Qwen3.5-35B-A3B-W8A8`, total35B/active3B,
25 visible assets/37,603,599,762bytes. Four isolated TP1/MTP0 arms use
TRITON_ATTN, official default FULL_AND_PIECEWISE Graph, max model4096,
max batched1024/max seqs16, prefix caching, greedy seed0/thinking off/max2048.
Original HumanEval0–31 and the existing isolated scorer/normalization are
unchanged. This is one checkpoint, not four unique models.

| Dtype / backend | Raw | Fixed normalized | Raw failed IDs | Completion tokens |
| --- | --- | --- | --- | --- |
| FP16 / Triton |31/32|32/32|24|4791|
| FP16 / AITER |30/32|32/32|13,24|4893|
| BF16 / Triton |31/32|32/32|24|4915|
| BF16 / AITER |31/32|32/32|13|4877|

Each arm has32 HTTP200/stop responses and4203 prompt tokens. Every raw failure
is a code-fence syntax error; no indentation repair or scoring waiver was added.
Each dtype pair has32 equal serialized request bodies. FP16 raw/normalized texts
match23/32 and BF16 matches24/32; ASTs with only docstrings removed match26/32
for each pair. Repeated frozen earlier R3 FP16 scores were raw30/31 versus these
raw31/30; normalized32 holds. No unique cause is assigned to that variation.
Sample score parity does not prove identical arithmetic or general equivalence.

Each actual model has40 expected experts: official TritonExperts or
HcuAiterInt8Experts, with raw INT8 weights and FP32 channel scales. Each AITER
arm has16 capped real startup ASM plans, followed by profiler removal;10
distinct gfx938 code objects are observed in its log. Triton observes no ASM
plans/code objects. These are bounded observations, not all-kernel tracing.
Each arm passes prefix14/14 and concurrent8/8 with14144 effective prefix hits
over26676 queries. Ordinary and AOT cache artifacts are retained per arm.

All25 checkpoint assets are fully posthashed for every arm. All1536 provider
inputs,56 private native copies, Core/host/Plugin installed payloads and
checkpoint identities match frozen manifests; native asset sets remain exact.
The reviewers' terminal audits independently verify small receipts/runtime
bindings and reference those full posthashes, rather than independently rereading
all checkpoint shards again.

## Limits and publication

All API/validator parents exit0; owned groups are empty, ports/cards release.
All four retain a
worker-grace warning and Torch library-finalizer errors (Triton2/AITER6 each).
Both BF16 arms additionally report eight shutdown ERROR lines/EngineDeadError;
the FP16 arms report none. Individual internal worker exit statuses remain
unknown. Lifecycle stays HOLD; no clean-shutdown claim or causal fix is made.

This qualifies the stated gfx938 TP1 component/model scope only. gfx938 EP2,
DP/PCP/SP/EPLB, MTP, padded serving, public engine reload, broader performance and
full148-feature migration acceptance remain open. Required remote device CI is
still blocked before image pull by missing/unauthorized image configuration;
local evidence does not bypass normal review/CI merge gates. Keep PR Draft.

Run-root receipts: `plugin373-gfx938-scoped-validation-r4.json`,
`qwen35-float16-public-int8-pair-review-r2.json` and
`qwen35-bfloat16-public-int8-pair-review-r2.json`. Historical R2/R3 failures and
successes retain their original artifacts. AI assistance was used.


## Later exact Adapter-main counterfactual

This separate experiment retains the same Core precision artifact and Plugin
runtime commit `509212bad08196edda937ee3e8b0d3d4792ecd5c`, provider/native inputs and
checkpoint. A host wheel built from clean Adapter main
`2c06e7430bd6fd9b423457012d1d161b19c07013` has all202 host/startup payloads and
22 paired kernel payloads byte-bound to Git. No PR66 bridge file exists or is
imported. Do not substitute an old same-version wheel: the earlier artifact named
`adapter-pr66-base-r1` differs from main in two files and adds the PR66 bridge.

`VLLM_MOE_SKIP_PADDING` is unset in each child; read-only component/worker
observations record effective Core skip=true. There is no policy0 override.
All seven CT cases/14 raw generations/42 changed-input Graph replays pass at the
original numerical gates, including six ASM cases and one official fallback.
Public CPU41/source CI6 pass; portable385pass/14 exact known PLE API failures
remain. This is not whole-Plugin health or CI acceptance.

The same original HumanEval32 requests, generation limits, scorer and model
settings above are rerun in four isolated main-host server arms:

| Dtype / backend | Raw | Fixed normalized | Raw failed IDs |
| --- | --- | --- | --- |
| FP16 / Triton |30/32|32/32|24,31|
| FP16 / AITER |29/32|32/32|15,24,31|
| BF16 / Triton |31/32|32/32|24|
| BF16 / AITER |30/32|32/32|13,24|

All raw failures are retained syntax-format failures, with no normalization
change or quality waiver. Each arm returns32 HTTP200/stop with40 actual experts,
prefix14/concurrent8, default skip=true and absent PR66 bridge. Each has full25
checkpoint/provider1536/native56/runtime payload postchecks. All32 wire request
bodies match both their backend pair and the corresponding original R4 arm.
FP16 paired texts24/AST27 and BF16 texts22/AST25 match. Different generated
texts/raw scores do not identify a unique host, hardware or arithmetic cause.

API/validator parents exit0 and resources release. All retain worker-grace and
finalizer warnings; BF16 also retains shutdown EngineDeadError. Lifecycle HOLD
and unknown internal child wait statuses are unchanged acceptance limits.
This proves PR66's unquantized hook and generic padding0 setup unnecessary for
these exercised public INT8 TP1/default-padding cases. It does not qualify the
official unquantized AITER backend, every padding/sentinel pattern, gfx938 EP2,
MTP, engine reload or clean lifecycle.

Additional run-root receipts: `adapter-main-int8-counterfactual-artifact-r1.json`,
`int8-adapter-main-env-r1.json`, `int8-adapter-main-ct-all-r1/result.json`,
`qwen35-{float16,bfloat16}-main-int8-pair-review-r1.json`, and
`main-counterfactual-original-protocol-comparison-r1.json`.


## Balanced same-GPU warmed serving measurements

The quality-qualified main2c06/Core3d/Plugin509 stack was measured on the same
physical gfx938 GPU0/PCI0000:49:00.0, serial Triton/AITER/AITER/Triton order.
Each arm uses BF16/TP1/TRITON_ATTN/default Graph and the same four original
prompts with seed0/temperature0/thinking off; only max_tokens64 and ignore_eos=true
change from quality requests. Request logging is disabled identically for both
backends. Every concurrency1/4 cell has two warmup rounds and five measured
rounds per arm: ten measured samples/backend/cell,140 total HTTP200/length
responses, each exactly64 generated tokens. Partial code is not HumanEval-scored.
The original full-length main-host quality receipts remain separate.

Nonstream client-wall generated token/s, combined medians with min–max ranges:

| Concurrency | Triton token/s | AITER token/s | AITER median delta |
| --- | --- | --- | --- |
| 1 | 66.38 (65.92–66.53) | 65.76 (65.72–65.83) | -0.94% |
| 4 | 200.65 (195.50–202.11) | 206.66 (204.33–207.28) | +3.00% |

These are bounded measurements, including a small c1 decrease; they do not
establish general acceleration, TTFT or GPU-kernel latency. Clocks and shared
host CPU are not locked/isolated. The official Triton baseline reports an
untuned default E256/N512/deviceN_A MoE config. No baseline tuning or architecture
spoofing was applied. Prefix caching is enabled, but actual block_size1088 and
short prompts yield0 hits/5096 queries in every arm. Warmup here refers to
kernels/requests, not effective prefix reuse. GPU cache capacity is1458 blocks
for Triton and1457 for AITER at the same0.45 memory-utilization setting.

Actual workers bind40 expected experts, effective skip=true/no PR66 bridge and
no active observer profiler before timing. Sixteen capped AITER startup plans
are observed per AITER arm; this is not exhaustive per-request kernel tracing.
Private copies of compatible quality-run compiler caches are used; original
quality caches are unchanged. Every arm rehashes all25 checkpoint assets,
provider1536/native56 and installed runtime payloads, with API/driver exit0,
owned groups/ports/cards released. Worker/finalizer/EngineDeadError limitations
retain lifecycle HOLD. The earlier single-arm pilot is excluded because its
experimental driver tested the wrong terminal status; no pilot timings enter
this four-arm result. The corrected driver has separate evidence and paths.

Receipts: `int8-main-performance-abba-r2/terminal.json`,
`int8-main-performance-review-r2.json`, and
`int8-performance-driver-terminal-red-green-r2.json`. Reproduce from the frozen
run scripts `run_qwen35_int8_performance_abba_r2.py`,
`run_qwen35_int8_performance_arm_r2.py`, `model_case_int8_performance_r2.py` and
`int8_serving_performance_client_r1.py`; the controller manifest binds the arm,
model-driver and client bytes.
No gfx938 EP2, MTP, reload, clean lifecycle or broad performance acceptance.
