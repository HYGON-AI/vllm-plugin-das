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
