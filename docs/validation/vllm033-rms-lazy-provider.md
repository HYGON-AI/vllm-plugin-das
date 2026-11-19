# Optional LightOp RMS provider loading on vLLM 0.33

The default-OFF RMS extension previously imported `lightop.norm` during every
activation, even when all model execution used official RMSNorm. Keep OOT and
opaque/fake operator registration and the effective-policy compile factor, but
resolve LightOp only for explicit opt-in, a required extension/capability, or an
actual eligible eager custom-op invocation. Compiled calls still delegate to
Core; width/dtype/layout admission, rounding and residual mutation are unchanged.

The resolver reuses Adapter's framework-free `ProcessLocalLifecycle`: concurrent
first use resolves once per PID, a child inheriting incomplete resolution refuses
instead of taking the parent's lock, and import/ABI failures remain latched.
The health callback only reports prior failure; it never initializes or repairs
a provider. The existing synchronous/four-positional-argument ABI check, including
native generated-signature validation, runs before vendor use. Explicit startup
requirements still check ABI before registration while execution switches are OFF.
Those requirements use the published entry-point name `das_hcu`, rather than
the declared spec name `das-hcu`. Four source CPU regressions exercise actual
Adapter activation through the project metadata, covering missing imports and
invalid ABI both before and after optional activation. These regressions do not
extend the recorded wheel/device/model qualification below to a new build.

Publication itself also uses Adapter's process-local lifecycle. A child forked
while the parent is resolving the provider or registering RMS refuses inherited
incomplete activation before taking a lock or importing another provider. This
guard covers both the actual published entrypoint through `activate_extensions`
and the direct provider accessor. A later explicit requirement still preflights
the provider after an earlier optional activation; health remains read-only and
initialization failures stay latched. Regression cases use real metadata and
Adapter activation with controlled CPU provider/registration doubles, preserving
the parent completion and owned-child cleanup checks. Python activation across
fork and native/model teardown remain separate qualification gates.

## Source and component verification

Candidate is based on Plugin `v0.33.0` commit
`a367368977b76f0cbbb91da546f0dec501811692`. The independently built wheel is
`hcu_vllm_das_plugin-0.33.0-py3-none-any.whl`, SHA256
`caa14a67e06cd713535d21f09adf0dacd9f0477ac40843979fa9a6ef1ffd940c`.
All 92 Python payloads match source and installation. All six selected package
RECORDs were checked before and after device execution. This is a recorded local
combination with the Adapter lazy scaled-MM candidate based on #140; Core is the
0.33 build `98a09850d5`, with the existing native providers retained.

- 111 affected CPU cases pass across extension, ordinary/fused/fallback dispatch,
  compile cache and discovery. Eight additional source/packaging contracts pass.
  Initial collection, error-chain and symlink-origin failures remain recorded;
  the corrected cases were rerun without loosening production admission.
- On gfx936, all 30 existing installed-wheel RMS HCU cases pass with original
  tolerances: eager ordinary/fused, fallbacks, compiled/composed residual chains
  and changed-input Graph replay. The trace records 104 `hipGraphLaunch` calls
  and real RMS kernels. General-plugin startup has no LightOp module; enabled
  execution loads it. Both FP16 and BF16 remain covered.
- The paired Adapter candidate also passes its original 410-case scaled-MM
  protocol and 96 Graph launches. Its 200 vendor cases are INT8: FP8 on gfx936
  retains the original official Triton gate. This is separate from FP8 KV cache.

## Model comparison and remaining gates

The same sealed `/model/Qwen3-1.7B` checkpoint ran serially before/after the
combined Adapter/Plugin lazy-loading changes: BF16 weights, TP1, `FLASH_ATTN`,
`fp8_e4m3` KV, block 64, 8192 context, default FULL_AND_PIECEWISE Graph, prefix
caching and the unchanged original HumanEval 0–31 protocol (temperature 0,
seed 0, max_tokens 2048, thinking disabled). Both arms use the same Core and
four unchanged native dependency RECORDs; the reference already includes #140's
atexit-unfreeze change. This compares loading policy, not BF16-KV versus FP8-KV.

Both arms score raw **1/32** and normalized
**28/32** using the unchanged isolated evaluator;
pass sets match: **True**. Message, finish reason and
usage match for **32/32** responses. Each arm
records 3840 prefix-hit tokens and
77/77 actual Graph launches
(reference/candidate); real HCU prefill, paged decode and native E4M3 cache-writer
kernels are present in both traces. All 28 attention layers retain the Plugin
implementation and all physical KV specs retain uint8/E4M3 quant mode.

The reference imports LightOp in the worker before and after generation; the
candidate has none of its root/norm/GEMM modules at either checkpoint. Original
strict output-format probes still fail in both arms
(12/12 failures); these
failures remain separate from HumanEval scoring and have not been reinterpreted.

Reference EngineCore forced termination at the original 15-second deadline:
**True**; candidate: **True**.
Nanobind leak reports: reference **False**, candidate
**False**. Both API processes exit 0 and all device handles
are released; API exit alone is not the cleanup gate. Recorded whole-API stop
times are 31.256/26.001s.
No latency/throughput claim follows from one serial run, and prior Gemma2
softcap/sliding lifecycle failures have not been rerun with this combination.

Source tests, the original component tests, immutable wheel/RECORD admission,
raw responses, scorer inputs/results and shutdown logs are preserved in the
local task evidence under `optional-lightop-loading-20261118-r178`.
Actual device is **gfx936**. gfx836 and full upgrade acceptance remain unqualified.

Reproduce the affected portable contracts with
`pytest tests/vllm033/test_extension.py tests/vllm033/test_rms_norm.py
 tests/vllm033/test_rms_compile_cache.py tests/vllm033/test_activation_discovery.py`
with `VLLM_TARGET_ROOT` naming the actual Core checkout. In the admitted installed
HCU stack, retain the original device suites:
`HCU_VLLM033_RMSNORM_LIVE=1 pytest tests/vllm033/test_rms_norm_hcu.py
 tests/vllm033/test_rms_norm_composed_hcu.py`.


## Current v0.33 integration on gfx938 (R227)

The prospective Plugin merge `d659c2e516e7ef0df1fcefe17c7ae188f5d38a68`
combines PR head `93dcf621` with current v0.33 target `c0e608de` without
conflicts. The RMS runtime and three fork regressions retain their previous bytes.
Adapter is proposed #97 head `6eeea954`; it is not merged main. Core remains
0.33.0+das.c440c08.dtk2610.torch2130.gfx936gfx938.base93. Source-bound wheels
were installed in a separate physical prefix, with 18 complete original
RECORDs and source/wheel/installed payload identities verified before and after.
FlashAttention was neither modified nor rebuilt.

Full unchanged Plugin portable selection: **1360 passed**, zero failures,
errors or skips; 46 HCU cases were deselected by the original portable marker.
Eleven separate public-selector subprocesses pass. Eight affected Adapter
contract files pass **244 cases**, including real metadata discovery/spawn.
All processes reached successful terminal states; this is not log-only evidence.
Native components pass **30 RMS/compiled/Graph** and **43 Adapter native-pair**
cases. A real required RMS owner with the official Worker/FULL Graph manager
passes six changing-input GPU replays across FP16/BF16 and four sticky refusals
before pending sends or replay. Its runner/PP doubles do not qualify real PP.

The separate Qwen3-8B candidate-only cohort uses TP1, BF16 weights,
FLASH_ATTN, `fp8_e4m3`, block64, context8192, maxseq8/batched-tokens1024,
memory0.6, explicit RMS switches, default compilation/Graph and prefix caching.
The original checkpoint is double-read sealed; post-run checkpoint and all
18 RECORD checks pass. Scorer/request/controller sources are sealed as well.

| Original check | Result |
| --- | --- |
| Arithmetic at concurrency1/4/8 | 13/13 |
| Repeated-prefix strict output | 3/3; +1280 prefix-hit tokens |
| Profiled strict output | 1/1 |
| HumanEval0-31 raw | 20/32 |
| Existing format normalization | 31/32 |
| Length-limited responses | 0 |
| Completion tokens | 4159 |
| Owned processes after shutdown | 0 |
| Original15s grace / frontend exit | within grace / exit0, no forced cleanup |
| Clean native lifecycle | FAIL: nanobind leaks and shutdown EngineDeadError |

HumanEval/26 remains an actual semantic failure after normalization: the
completion keeps one copy of repeated values instead of removing every value
occurring more than once. Do not describe it as formatting. This candidate-only
phase did not include a matched current-target control; the completed control
is recorded below. Prior gfx936 scores remain under their old sources.

Before/after worker reports agree: genuine gfx938/BW1100, Adapter Worker,
official MRV2 runner, 36 HCU Flash implementations and 145 RMS OOT instances.
The native trace records the E4M3 uint8 cache writer, Flash prefill, paged decode
and three hipGraphLaunch calls. Default compiled RMS deliberately delegates to
Core IR: the trace contains fused Triton RMS kernels, not a LightOp execution
claim. Eager LightOp is qualified only by its separate component evidence.

Local immutable evidence root:
`evidence/rms-owner-current-combination-r227/`; `cpu-state.json`, native JUnits,
`model-audit.json`, original model logs, requests/responses/scoring, profile,
shutdown and post-seal receipts retain their scopes. These are local evidence,
not a new remote CI qualification or independent approval.

## Matched current-target control on gfx938 (R228)

The separate physical-prefix control uses Plugin target
`c0e608de81532ac808768cd68133412d132b6971` with the same proposed Adapter #97,
Core build, checkpoint, device, requests, protocol, switches and default
compilation/Graph settings as R227. It reached terminal states for all stages:
11 public selectors, **1357 portable**, **244 Adapter**, **30 RMS/compiled/Graph**
and **43 native-pair** cases pass, with no failures, errors or skips in those
selections. The two real required-owner Worker/FULL Graph probes also pass.
The original 46 hardware cases remain deselected by the portable marker.

The read-only pair audit verifies each original installed RECORD against actual
file hashes and sizes. Cross-prefix console-script RECORD hashes can differ
solely because their interpreter shebang changes. For payload comparison only,
the audit normalizes the exact first-line `#!<that-prefix>/bin/python` in those
actual console scripts. It does not rewrite RECORDs, normalize package/native
bytes or excuse additional changes. The sole changed distribution payload is
the intended Plugin. Checkpoint bytes, CLI, controlled environment, physical
device and frozen scoring inputs match.

| Original model check | R227 candidate | R228 target control |
| --- | --- | --- |
| Strict probes / prefix-hit increment | 17/17 / 1280 | 17/17 / 1280 |
| HumanEval0-31 raw | 20/32 | 19/32 |
| Existing format normalization | 31/32 | 30/32 |
| Completion tokens / length-limited responses | 4159 / 0 | 4348 / 0 |
| Original15s grace / frontend exit | within grace / exit0 | within grace / exit0 |
| Forced cleanup / surviving owned processes | none / 0 | none / 0 |
| Clean native lifecycle | FAIL | FAIL |

All 32 requests and finish reasons match; 28/32 messages, choices and usage
records match. Changed tasks are **HumanEval/6, /17, /18 and /25**. The
candidate adds no failed task in this single cohort: it passes /17, whose
control completion omits the final bare `o` in its hand-written ASCII parser;
both fail /26. Tasks /6, /18 and /25 change code form without changing their
pass flags. These observations establish neither 32-output equivalence nor a
causal accuracy improvement. The source of the four changes is not established.

Both serving arms retain nanobind leak diagnostics and a shutdown
`EngineDeadError`. Equal failure presence is evidence that the symptom also
occurs without this fork fix; it neither identifies every leak owner nor
waives the original clean-lifecycle requirement. Both traces show real FP8
cache/attention work and three `hipGraphLaunch` calls. Compiled RMS execution
still delegates to Core IR, so these model results do not qualify LightOp
execution or performance. Neither arm is a last-good upgrade combination.

An additional source-stage compatibility check against actual Adapter main
`8490a6b6072bda681982e95b23cbcae313b1eec6` passes **75 extension cases**,
including real fork regressions. Its imported lifecycle source was checked
explicitly. This is not a full installed-main model A/B or proof that #97 has
merged.

Original control evidence is retained under
`evidence/rms-owner-current-baseline-r228/`, including `paired-audit.json`,
`model-audit.json`, terminal CPU/native results, raw requests/responses/scoring,
traces, original shutdown diagnostics and pre/post provenance seals. Keep
Draft: attribution of changed outputs and clean lifecycle remain open;
performance and the complete upgrade matrix remain unqualified.
