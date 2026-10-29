# vLLM 0.32 Plugin ON/OFF Validation Design

## Intent

Validate the optional DAS optimizations as an independently installable plugin
on top of official vLLM 0.32 and the current `adapter-das` delivery branch. Keep
platform and runtime ownership in Adapter, while proving that Plugin activation,
operator rollback, Graph execution, and model serving remain correct in both ON
and OFF configurations.

## Goals

- Keep Plugin development in the independent `vllm-plugin-das` repository and
  branch `feat/vllm032-plugin-onoff-gates`.
- Base the work on `HYGON-AI/vllm-plugin-das:v0.32.0` at
  `e76220d9c0006cc3c9d886587a4771a31e7d83e8`, synchronized to the personal
  fork, and target that upstream branch for the final MR.
- Pair the Plugin with Adapter commit
  `c99886d41f435e594050d47663e03ea94081fdc6` and the existing HCU vLLM
  `0.32.0+rocm723` wheel.
- Reuse Extension API v1 through `hcu_adapter.extensions`; Adapter remains the
  only owner of `vllm.platform_plugins` and `vllm.general_plugins`.
- Validate the existing LightOp `RMSNorm` and `SiluAndMul` capabilities with the
  master switch and each operator switch independently ON and OFF.
- Compare numerical behavior, Graph capture/replay, model responses, and warmed
  performance. Performance readings are evidence, not a correctness substitute.
- If a defect is found, reproduce it with a failing test before the smallest
  Plugin-only fix and submit that fix as a separate MR.

## Non-goals and anti-goals

- Do not add optimization implementation to `adapter-das`, `sitecustomize.py`,
  or either official vLLM plugin group.
- Do not port additional vLLM 0.28.1 optimizations in this workstream.
- Do not change official vLLM source or claim support beyond the tested HCU
  vLLM 0.32 artifact.
- Do not broaden the Plugin's exact `0.32.0` release gate without live evidence
  from another 0.32 patch wheel.
- Do not claim a performance win from cold compile time or a single un-warmed
  request.

## Approaches

1. **Independent Plugin branch with existing switches and paired A/B validation
   — selected.** Reuse the already migrated extension and its three environment
   switches. Build and test it against the current Adapter branch, changing code
   only for a demonstrated defect.
2. Put ON/OFF policy in Adapter. Rejected because Adapter owns compatibility and
   lifecycle, not optional optimization policy; it would couple two repositories
   and duplicate Plugin behavior.
3. Port more legacy Plugin optimizations while validating the pair. Rejected
   because failures would no longer be attributable to the Extension boundary or
   the two already validated operators.

## Component and switch boundary

```text
official vLLM 0.32
  -> adapter-das (platform/runtime guards + Extension API v1)
  -> vllm-plugin-das (optional LightOp operators)
       VLLM_HCU_USE_CUSTOM_OPS
         +-- VLLM_HCU_USE_CUSTOM_RMS_NORM
         `-- VLLM_HCU_USE_CUSTOM_SILU_AND_MUL
```

The master switch has highest priority. A disabled master or child switch must
execute the supported native vLLM fallback without uninstalling the Plugin. All
switch changes take effect in a fresh process; dynamic in-process unregister is
out of scope.

## Validation matrix

1. Run portable package, Extension lifecycle, version-gate, and ON/OFF fallback
   tests.
2. Build the Plugin wheel and inspect metadata: exactly one
   `hcu_adapter.extensions:das_hcu` entry point and no vLLM platform/general
   entry points.
3. Install vLLM, current Adapter, and Plugin into an isolated environment and
   run the installed verifier. Require the Extension report to be active and OOT
   owners to be `HcuRMSNorm` and `HcuSiluAndMul`.
4. On an idle `gfx936`, compare operator output against the native fallback for:
   master OFF, RMSNorm-only OFF, SiluAndMul-only OFF, and all ON. Include ordinary
   and fused-add RMSNorm, supported dtypes/shapes, and Graph capture/replay with
   changed inputs.
5. Run at least one small dense model in non-eager mode with all ON and master
   OFF. Require FULL and PIECEWISE Graph capture, HTTP 200 probes, deterministic
   semantic answers, and no fatal runtime log.
6. Record warmed operator latency and serving throughput for ON/OFF using the
   same process policy, model, prompts, and sampling configuration. Report
   regressions without hiding them behind correctness success.
7. Stop only owned processes and require VRAM to return to the node baseline.

## Failure and MR policy

- Environment or missing dependency failures are diagnosed and recorded; they
  are not fixed by weakening a gate.
- A Plugin defect gets one minimal failing regression test, a Plugin-only fix,
  portable and live-HCU revalidation, documentation, and a commit that names the
  root cause.
- Adapter changes require separate evidence that the failure belongs to
  platform/runtime compatibility rather than the optional optimization. They do
  not enter this Plugin branch by default.
- A new MR is created only when the branch contains a justified code or contract
  change. Pure validation evidence may be submitted separately without changing
  runtime behavior.

## Acceptance

- Existing portable tests are green against Adapter `c99886d`.
- Installed ownership and Extension activation are correct.
- All ON/OFF combinations select the expected optimized or native path.
- Eager numerical and non-eager Graph replay parity pass on live HCU.
- The selected small-model ON/OFF service checks pass and performance evidence is
  recorded with an environment stamp.
- Any remaining gap, including HumanEval or broader model coverage, is reported
  explicitly rather than inherited from an older Adapter/Plugin pair.
