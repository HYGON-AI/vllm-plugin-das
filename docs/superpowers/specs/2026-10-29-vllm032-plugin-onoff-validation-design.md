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
- Audit the candidate LightOp `RMSNorm` and `SiluAndMul` capabilities against
  the official vLLM 0.32 native/IR paths before deciding whether each operator
  is retained, made opt-in, or removed.
- Validate every retained capability with the master switch and each operator
  switch independently ON and OFF.
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
- Do not use an exact wheel string as a runtime compatibility lock. Admit the
  vLLM 0.32 release line and fail closed on missing required APIs or
  capabilities; keep validation claims scoped to the exact artifact tested.
- Do not claim a performance win from cold compile time or a single un-warmed
  request.

## Approaches

1. **Independent Plugin branch with existing switches and paired A/B validation
   — selected.** Start from the migrated extension and its environment switches.
   Build and test it against the current Adapter branch, retaining only operators
   whose evidence justifies ownership outside upstream vLLM.
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
  -> vllm-plugin-das (optional LightOp RMSNorm)
       VLLM_HCU_USE_CUSTOM_OPS
         `-- VLLM_HCU_USE_CUSTOM_RMS_NORM
```

The master switch has highest priority. A disabled master or child switch must
execute the supported native vLLM fallback without uninstalling the Plugin. All
switch changes take effect in a fresh process; dynamic in-process unregister is
out of scope.

## Frozen source and artifact contract

The immutable identifiers below describe this experiment and its evidence. They
do not become exact-match runtime gates.

| Role | Repository/artifact | Immutable identity |
| --- | --- | --- |
| Plugin upstream base | `HYGON-AI/vllm-plugin-das:v0.32.0` | `e76220d9c0006cc3c9d886587a4771a31e7d83e8` |
| Plugin candidate at audit start | `alexanderbin123/vllm-plugin-das:feat/vllm032-plugin-onoff-gates` | `fdf427fe615d913f9d42c47287c2a1a804eb0021` |
| Adapter candidate | `adapter-das:feat/vllm032-adapter-delivery-gates` | `c99886d41f435e594050d47663e03ea94081fdc6` |
| Official vLLM upstream base | `vllm-project/vllm:v0.32.0` | `5df63a7203cdb9328d824a27b682e77c7b9928be` |
| HCU vLLM build source | `OpenDAS/vllm:v0.32.0` | `7c13676e497262b71a8b107a3186b0f6641dcbf1` (upstream base plus the DTK/gfx936 build patch) |
| Installed HCU vLLM wheel | `vllm-0.32.0+rocm723-cp312-cp312-linux_x86_64.whl` | SHA256 `d7f084b8754288893dd9ba28d24784ec4b4e59350c9bfb5038ec24ac3af9572b` |

The older `de61890d...` rebuild was not installed for this experiment and is
not used as evidence. The current test host also has a global vLLM 0.28.1 installation ahead of a
test venv through inherited `PYTHONPATH`. Every v0.32 command must replace that
variable explicitly and print `vllm.__version__` plus `vllm.__file__`; otherwise
the result is invalid target evidence.

## Upstream-first patch migration ledger

| ID | Legacy/candidate behavior | Official vLLM 0.32 owner | Provisional disposition | Required evidence |
| --- | --- | --- | --- | --- |
| P-032-01 | Adapter Extension API activation | vLLM owns platform/general plugin lifecycle; Adapter owns the HCU extension seam | Keep the single `hcu_adapter.extensions` entry point; no Plugin platform/general entry point | Wheel metadata, parent/child import roots, active extension report |
| P-032-02 | LightOp `SiluAndMul` | `SiluAndMul.forward_native` and the official HIP `_C.silu_and_mul` path | **Remove from Plugin.** Warmed compiled A/B was slower for every measured Qwen dense shape; ship neither capability, switch, module, nor OOT registration | Official class and `forward_native` in both Plugin ON/OFF Graph rows; wheel excludes the module |
| P-032-03 | LightOp ordinary `RMSNorm` | vLLM IR owns native, `vllm_c`, and optional AITER providers | **Retain as explicit opt-in.** LightOp was materially faster for every measured shape; default stays OFF and native remains the complete fallback | Selected forward method, eager and Graph numerics, warmed A/B |
| P-032-04 | LightOp fused-add `RMSNorm` custom op returns mutated inputs | Official/PyTorch mutable custom-op contract | Fix before enabling in Graph: the custom op is side-effect-only and the Python wrapper returns its owned buffers | Schema/no-return test, changed-input compile/Graph replay, alias-warning absence |
| P-032-05 | Master and RMSNorm rollback switches | Plugin policy | Keep, but evaluate in fresh processes and prove OFF reaches the complete official path | ON, child OFF, and master OFF process rows with dispatch evidence |
| P-032-06 | Exact `0.32.0` version comparison | Adapter release-line gate plus Plugin capability checks | Change to vLLM `>=0.32.0,<0.33.0`; preserve fail-closed API/symbol checks | Boundary unit tests and actual 0.32.0 artifact import |
| P-032-07 | Old compatibility lock and service record | No upstream owner | Regenerate for the current three-way SHA pair; old service success is historical only | New wheel hashes, commands, logs, raw results and statuses |

Audit findings on the frozen pair:

- Portable compatibility tests pass (`62 passed, 2 skipped`) when
  `VLLM_TARGET_ROOT` points at the official v0.32.0 source.
- Installed-wheel live eager numerics pass for both operators (`2 passed`) on
  `gfx936` with the current Adapter source and Plugin wheel.
- The default non-eager vLLM configuration resolves to
  `custom_ops=['+sparse_attn_indexer', 'none']`. Although the extension is
  active and OOT classes are selected, both instances bind `forward_native`;
  therefore the old service smoke does not prove either LightOp kernel ran.
- Direct `torch.compile` execution of the fused-add LightOp path emits the
  PyTorch mutable-output alias deprecation warning. This is expected to become
  an error in PyTorch 2.12 and confirms P-032-04 is required.
- After the mutable-op fix, changed-input compiled replay has no alias warning
  and preserves ordinary and fused-add RMSNorm numerics.
- Warmed compiled A/B on token rows 1, 16, and 128 measured official
  `SiluAndMul` at about 99--104 us versus LightOp at about 128--132 us. The
  LightOp candidate was therefore removed. LightOp ordinary RMSNorm measured
  about 150--166 us versus official at about 296--314 us, and fused-add measured
  about 116--129 us versus official at about 351--372 us, so RMSNorm remains an
  opt-in Plugin capability.

## Validation matrix

1. Run portable package, Extension lifecycle, release-line/capability gate, and
   ON/OFF fallback tests.
2. Build the Plugin wheel and inspect metadata: exactly one
   `hcu_adapter.extensions:das_hcu` entry point and no vLLM platform/general
   entry points.
3. Install vLLM, current Adapter, and Plugin into an isolated environment and
   run the installed verifier. Require the Extension report to be active and OOT
   owner to be `HcuRMSNorm`; require the wheel to exclude the candidate Silu
   module and registration.
4. On an idle `gfx936`, record the selected forward method and IR/provider, then
   compare operator output against the native fallback for:
   master OFF, RMSNorm child OFF, and RMSNorm ON. Include ordinary and fused-add
   RMSNorm, supported dtypes/shapes, and Graph capture/replay with changed
   inputs. Require official `SiluAndMul.forward_native` in both ON and OFF rows.
5. Run at least one small dense model in non-eager mode with all ON and master
   OFF in fresh processes. Require evidence that the intended ON/OFF path
   actually executes, FULL and PIECEWISE Graph capture, changed-input replay,
   HTTP 200 probes, deterministic semantic answers, and no fatal runtime log.
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
- Every retained optimization has an upstream-takeover disposition backed by
  dispatch, numerical, Graph, and warmed performance evidence.
- All ON/OFF combinations select the expected optimized or native path.
- Eager numerical and non-eager Graph replay parity pass on live HCU.
- The selected small-model ON/OFF service checks pass and performance evidence is
  recorded with an environment stamp.
- Any remaining gap, including HumanEval or broader model coverage, is reported
  explicitly rather than inherited from an older Adapter/Plugin pair.
