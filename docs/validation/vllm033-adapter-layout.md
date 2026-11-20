# Adapter directory-layout compatibility

Adapter's directory migration moves infrastructure and feature owners away from
the package root. The Plugin's static imports, dynamic seam lookup and installed
alias checks must resolve those real owners together. Adapter's public
`hcu_adapter.extensions` facade and metadata group remain the protocol boundary.

The Plugin uses the presence of `hcu_adapter.infra` to select the declared layout.
Only the exact absence of that namespace selects legacy root modules. A missing
canonical owner or transitive dependency propagates; no old provider is retried.
The resolver returns the original module, preserving mutable state, class identity
and facade alias semantics. It adds no numerical algorithm or version pin.

## Source and artifact binding

- Plugin base: v0.33.0 `c0e608de81532ac808768cd68133412d132b6971`.
- Legacy Adapter: main `8490a6b6072bda681982e95b23cbcae313b1eec6`.
  All 815 Git blobs were verified against the immutable upstream tree.
- Relocated Adapter: proposed #152
  `1629cd664a8b105eb09b7565374e5fbcff56e396`; all 876 Git blobs verified.
- Independent noneditable Plugin and Adapter host-only wheels were installed into
  task prefixes. Production source/wheel/installed payloads and original RECORD
  hashes match. Adapter keeps its independent 0.5.22 version. No new native kernel
  wheel or native dependency pair was built or qualified.

## Completed checks

| Check | Legacy layout | Relocated layout |
| --- | --- | --- |
| Real installed metadata: 12 extension factories | 12 passed | 12 passed |
| RMS lazy import and lifecycle-class identity | Passed | Passed |
| Selected affected portable contracts, 13 files | 402 passed | 402 passed |

Factory discovery ran in independent `-I -S -B` interpreters with only the two
installed prefixes and the packaging dependency. A meta-path guard rejected
Torch, vLLM, SGLang, sgl_kernel, AITER, LightOp and Triton imports. All 12 factories
returned specs. Imported files were bound to installed RECORDs, including exact
declared Adapter aliases and namespace-package locations; no production source
tree was on those discovery processes' import paths.

The portable runs hid `/dev/kfd` and `/dev/dri` in a private mount namespace,
retained the original assertions and initialized no GPU. Core ROCm import used
only a bounded architecture-metadata double, restored before pytest; the tests'
existing operator doubles remain existing test scope. Those results do not prove
GPU kernels, native pairing or model behavior. Each final run seals production
and test source bytes before and after execution.

The pure filesystem resolver and installed-ownership tests pass 52 cases with
34 subtests. They cover both layouts, module identity and mutable state, missing
canonical owners, missing owner/namespace dependencies, and both RECORDs plus
exact object identity for each declared relocated alias.

The source/packaging, CI-helper and upgrade-skill static selection passes 158
cases with 446 subtests. Skill schema validation passes; Torch is not imported
in that process.

Initial relocated-layout portable execution retained four failures: two child
probes still imported root activation names, one missing-module injection targeted
the old owner, and one seam-table assertion expected root names. The tests now
target the actual resolved owner, preserving each original negative assertion.
The final independent runs pass 249 and 153 cases per layout without skips.
Early discovery harness checks also rejected legitimate Adapter-owned aliases
and namespace modules; the corrected audit uses the existing exact alias guard
and validates namespace paths, without a general origin exemption.

## Reproduction

Build and install the Plugin project (`adapter_extension`) and each Adapter
`packaging/hcu-adapter` host-only wheel into separate prefixes with `--no-deps`.
Do not use editable installs. Run `tests/vllm033/test_adapter_layout.py` and
`tests/ci/test_v033_installed_module_ownership.py`, then the affected files listed
below against each installed layout with `HCU_V033_VERIFY_INSTALLED=1`:

```text
test_extension.py                 test_activation_discovery.py
test_fp4_full_admission.py        test_fp4_fp8_admission.py
test_inverse_rope_pcp.py          test_pcp_q_rope.py
test_fp4_pcp_workspace.py         test_v41_indexer_review_fixes.py
test_compilation_owner.py        test_fp4_dspark_rows.py
test_dsv41_channel_fp8.py         test_minimax31_prefill_cpu.py
test_fp4_packer_lowering.py
```

For a separate framework-free discovery check, use an isolated interpreter,
explicitly add only the installed Plugin/Adapter prefixes and packaging path,
and load each entry in the installed distribution's `hcu_adapter.extensions`
group. Check its returned spec, exact metadata origin and all module origins;
block host/provider imports before loading any factory. Do not substitute an
egg-info source entry or a fake entry-point object for installed metadata.

Local source-bound evidence is retained under
`/data/models/plugin-adapter-layout-20261120T032620Z/`: wheel/install bindings,
`installed-discovery-{main,152}.json`, `layout-unit-r3.xml`, and the
`affected-{main,152}-{r2,extra}` logs, JUnits and source seals.

## Remaining acceptance

This is directory/import compatibility only. It does not qualify #152 as a
complete refactor, the full portable suite, activation of all providers, actual
GPU Graph replay, gfx836, model accuracy/HumanEval32, serving performance or
clean Worker shutdown. DeepSeek checkpoint/device adaptation remains last in the
user's order; its existing discovery contract is included here. Keep this MR
Draft until the remaining required combination checks are complete.
