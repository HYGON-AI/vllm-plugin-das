# DSV4.1 Channel-FP8 routing on vLLM 0.33

Status: configuration/routing implemented; CPU source replay passed. No model
or GPU acceptance is claimed.

## Scope and provenance

- Core stays read-only at `368701db06c78f3e8654bb51f44753e5064e262c`
  (official base `4ba787ec9959e566d3b587a796dff07673324da5`).
- Adapter stays read-only at `3257a934e683d31edcf570c7bc455564aa6db00b`.
- Plugin baseline: `aef6f69f26b753de9ac1db118ed437a35253f2cf`.
- Source contract: `23a8d11e` in the v0.32 integration, whose final reviewed
  implementation is present at `1b0b1a14`.
- This increment covers configuration identification and quantization-method
  routing. It does not claim checkpoint loading, GPU kernels, model accuracy,
  PCP, DSpark, W4A8, or Graph execution.

The production approach must use the existing `hcu_adapter.extensions` API and
vLLM's public `register_quantization_config` API under a distinct method name.
It must not replace the official FP8/CT configurations, native methods, or INT8
oracle symbols. The DSV4.1 extension must not depend on RMSNorm, FlashAttention,
LightOp, AITER, or the legacy `vllm_hcu` package for discovery.

## Failure modes recorded before implementation

| ID | Trigger and failure | Verification before implementation |
| --- | --- | --- |
| F01 | Entry-point discovery imports Torch, vLLM or vendor modules, making unrelated features mandatory. | A fresh process blocks framework/vendor imports while loading the factory and its declaration. |
| F02 | Wrong release line or epoch registers a method before rejecting the core. | Reject 0.32/0.34, malformed/missing versions and nonzero epochs before registration; accept 0.33 patch/RC/dev/local variants. |
| F03 | The public quantization name already belongs to another class. | Reject a foreign owner before writing; repeat activation with the same class is idempotent. |
| F04 | Registration fails after publishing part of its state. | A controlled register-then-raise path must propagate an Adapter commit-fatal error, not an optional successful activation. |
| F05 | Merely registering a class does not make ModelConfig select it. | Exercise the actual v0.33 `ModelConfig._verify_quantization` path, not a duplicated selector; report any isolated hardware doubles. |
| F06 | A non-V4.1 model uses the same FP8 schema. | Reject automatic override for other model types/architectures. |
| F07 | Block/tensor/MXFP8/Quark/static-activation or non-channel schemes are incorrectly captured. | Negative schema matrix must preserve native selection. |
| F08 | An explicit incompatible user quantization choice is silently overridden. | Only compatible FP8/custom choices may match; other explicit choices remain native or fail through the official parser. |
| F09 | CPU, CUDA, or ordinary AMD ROCm accidentally gets HCU routing. | Automatic override is limited to the actual HCU platform contract, not a fabricated architecture identity. |
| F10 | Config construction loses native FP8 exclusions, activation settings or packed-name mappings. | Construct through the native v0.33 configuration contract and compare inherited policy; leave the input dictionary unchanged. |
| F11 | BF16 `wo_a` is quantized because the channel scheme targets all Linear layers. | Both ordinary and packed prefixes preserve native exclusion behavior; `.wo_a` selects the native unquantized method. |
| F12 | Native ignored Linear/MoE layers are translated into CT's different ignore syntax. | Check native exclusions before delegation, including exact expert checks and configured matching mode. |
| F13 | A mapped/updated packed-module name is ignored by a cached CT config. | Update the native packed mapping and verify the delegated CT configuration observes it. |
| F14 | New routing also captures embeddings or other unsupported layer types. | Unrelated layers delegate to the native implementation. |
| F15 | The generated CT description has wrong scale grouping or static activations. | Verify channel FP8 weights, dynamic token FP8 activations and the expected float-quantized scheme using the official CT parser. |
| F16 | Registration changes the official FP8/CT classes, their methods, or INT8 oracle. | Compare symbol identities before/after activation; the new entry uses its own name only. |
| F17 | A fake fixture passes while the production module or built wheel differs. | Load the candidate from production source; inspect the built wheel and entry-point metadata; save a reproducible command and output. |
| F18 | CPU configuration checks are reported as kernel or model acceptance. | Keep configuration/package results separate from unrun GPU/model/Graph results. |

## Required implementation boundaries

Extension activation has no model configuration. It may preflight and register
an independent quantization configuration; model/schema selection happens later
through the official configuration parser. Loading-time aliases, runtime
collectives and model callbacks are outside this increment.

The old monkey-patched `DeepseekV4FP8Config.get_quant_method` is a semantic
reference only. The new implementation must own its configuration type and
preserve native exclusion and BF16 `wo_a` semantics without rebinding that
native method. The ordinary and expert routing targets the official CT API;
provider availability and execution remain independent, unverified gates.

The V4.1 family predicate belongs to the official ModelConfig override path;
`from_config` receives a quantization dictionary, not model identity. Direct
low-level construction is not an arbitrary-model admission validator. A raw
checkpoint declaring `quant_method=hcu_dsv41_channel_fp8` is not supported by
the accepted schema and is rejected by `from_config`.

Registration assumes standard serialized plugin startup. Core's public API has
no atomic name-reservation operation; this owner does not provide atomicity
against concurrent external registry writers, or roll back arbitrary Core
registrations. Existing foreign owners are refused at sequential startup and
later lost ownership is observable through health checking.

## Verification record

The inventory and the complete configuration/package tests were written before
production implementation. The pre-implementation run recorded 57 passed,
48 missing-feature setup errors and three failures (the missing entry point,
missing discovery module, and a packaging ownership issue). That record is
retained; it is not a successful feature run.

After implementation, the same cases were rerun. Two fixture mistakes were
corrected without adding cases or weakening expected behavior: ModelConfig's
LazyLoader cached registry attributes across test isolation, and the installed
CT parser represents strategies as strings rather than Enum objects. The
initial root-container wheel build was rejected by Git ownership; the final
run used the repository owner's UID instead of changing global Git config.

The targeted configuration/package/extension/core-contract run passed:
**108 passed, 14 deprecation warnings in 43.26 seconds**. GPU visibility was
empty and `torch.cuda.is_initialized()` was false at the recorded boundary.
The initial full existing `tests/vllm033` portable regression completed with
**312 passed, 66 skipped, 14 warnings in 156.13 seconds**. Final reviewed Python
source passed the same suite with **312 passed, 66 skipped, 14 warnings in
168.70 seconds**. Live-device cases were not executed and do not establish
hardware coverage. Independent review findings about sticky registration
failure and active-publication version drift were fixed. Fault-injected replay
through the actual Adapter extension host confirmed sticky fatal behavior for
partial publication plus version drift, active publication plus version drift,
and interrupted native preparation; CUDA remained uninitialized.

A separate official-selector replay confirmed non-V4.1 native FP8 selection is
preserved, an incompatible explicit custom choice is rejected, and a raw
checkpoint naming the custom method is rejected by `from_config`. Direct
low-level config construction is not a model-identity validator. Concurrent
external registry mutation remains outside the serialized startup guarantee.

The final Plugin wheel has SHA256
`f97f58e8d1ff0f5b6ac673caeecbcb1d3196b5de54d7159d7fe12ed09be870c1`;
all 17 packaged Python files match the final production source bytes.

Reproduce with the selected Core and Adapter as siblings (or set
`VLLM_TARGET_ROOT` and `HCU_ADAPTER_SOURCE`):

```bash
VLLM_TEST_PYTHON=/path/to/python3.12 \
  bash tests/vllm033/run_dsv41_channel_fp8_cpu.sh /path/to/output
```

Use an account owning the checkout. Container users also need a normal matching
username and a writable pytest temporary directory. The script records the
three SHAs, actual ModelConfig source path, installed distribution metadata,
JUnit XML and stdout. It builds the Plugin wheel without installing packages.

**Evidence boundary:** validation ran in an existing container with Python
3.12 and prepared DTK dependencies. Its installed vLLM distribution metadata
remains `0.32.0+rocm723`; Python configuration code was explicitly loaded from
the fixed v0.33 source tree and checked by path. This is CPU source/configuration
replay with hardware/layer-construction doubles, not a verified installed
v0.33 Core/Adapter/Plugin deployment. No model, kernel, graph, collective,
accuracy, performance or gfx938 runtime result is claimed.
