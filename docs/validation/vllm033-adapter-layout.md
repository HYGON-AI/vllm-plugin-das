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
  `bf7309281952df32e2aed0241a8d504651c79c9b`; all 876 Git blobs verified
  against canonical tree `0c0baf5e8336618b09288b916d269e8819d64731`.
  The original affected portable results below used predecessor `1629cd66`;
  they are not relabeled as results for the newer head.
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

The initial affected-file runs hid `/dev/kfd` and `/dev/dri` in a private mount namespace,
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

## Subsequent installed and CI checks

The latest relocated Adapter wheel passes the same real metadata factories and
RMS identity checks (13/13). Its changed SGLang lazy-import guard passes all seven
original cases; the predecessor wheel fails those exact seven cases on the old
root import. This verifies the changed import, not the entire Adapter refactor.

[CI run 44671131998](https://github.com/HYGON-AI/vllm-plugin-das/actions/runs/44671131998)
tests Plugin commit `def84abe9c429f00104a51ca67cdfca39be83adf` and finishes with
all eight jobs successful: source 8, portable 1393, gfx936 components 255 and
gfx938 components 14. Artifact digests, ZIP integrity, JUnit failures/errors/skips
and checkout bindings were checked; portable case IDs exactly match the original
1393-node roster. This run uses the approved runtime Adapter, not proposed #152.

A separate real gfx936 host-startup probe with Adapter main selects the official
`HcuPlatform`, activates the required Flash and Step5 capabilities through real
installed metadata and passes repeated activation/health checks. All 74 loaded
Adapter/Plugin modules resolve to that installed pair. Packages, Core and vendor
trees are mounted read-only, JIT caches are private and copied payload/RECORD
hashes remain unchanged. The worker exits 0 and its owned process group is empty.
This is startup evidence without a model or native numerical acceptance claim.

The initial two complete local portable attempts retain 1390 passes and three
failures each. DeepGEMM cannot obtain a CU count when management nodes are hidden;
two Step5 children use global purelib and import an older Plugin. Repairing the
environment with real venv prefixes, real management queries, empty compute
masks and a Torch-initialization guard makes the unchanged native-import node
and full Step5 file pass 12/12 per layout. No tests or production code changed;
the complete two-layout reruns are tracked separately from this focused result.

Those complete reruns now pass **1393/1393 per layout**, using Adapter main
`8490a6b6` and proposed #152 `bf730928`, respectively. Both match the original
74-file/1393-node roster exactly, with zero failures/errors/skips and the same
46 marker deselections as CI. Production/test source and copied wheel RECORDs
remain unchanged. Each arm records 104 process-origin observations without
global package contamination or Torch GPU initialization. The audit distinguishes
installed modules, existing filesystem fixtures and explicitly loaded
byte-identical source; it does not label every subprocess as installed-only.

## Reproduction

Build and install the Plugin project (`adapter_extension`) and each Adapter
`packaging/hcu-adapter` host-only wheel into separate prefixes with `--no-deps`.
For the complete portable lane, use real venv prefixes whose purelib contains
the candidate, retaining real management queries and empty compute masks. A
`--target` install alone does not change the parent's sysconfig or a child's
discovery paths. Do not use editable installs. Run `tests/vllm033/test_adapter_layout.py` and
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

## Merge scope and remaining runtime validation

Adapter #152 is now merged at `d9bb7fa5d2ed7a46e74b06f5c7c2f9ef172bd702`.
Its old root infrastructure modules are absent, so the released Plugin needs
this compatibility fix when used with the new Adapter. The merge scope is the
directory/import contract, covered by both-layout portable runs, installed
factory/identity checks and the completed CI above.

A final independent build installs the Plugin `e3039e8` and merged Adapter
`d9bb7fa5` host-only wheels into fresh noneditable prefixes. All 12 real metadata
factories plus RMS lifecycle identity pass (13/13) in separate framework-blocked
interpreters. Production source/wheel/installed bytes, original RECORD rows,
installed hashes and exact module/namespace origins are verified. Evidence is in
`/models/zb/vllm033-resume-20261114/evidence/plugin486-ready-merge-20261120/`.
The merged Adapter differs from tested `bf730928` in three files: SGLang FP8
padding, its repro and tier documentation; the final discovery check is new
evidence for the merged wheel, not a relabeling of earlier full portable runs.

Subsequent Qwen3-1.7B controls hold Plugin `e3039e8` fixed and compare actual
paired native wheels for Adapter `bf730928` and old main `8490a6b6`. All 32
messages, finish reasons and completion-token counts match; both retain the
same four normalized HumanEval misses, 13 strict-protocol failures and forced
EngineCore termination after the original grace period. Those failures remain
failures. This comparison does not establish Plugin-main equivalence, qualify
the merged Adapter's full runtime or resolve the shutdown issue.

These outstanding runtime investigations do not extend this import repair into
whole-upgrade acceptance. Activation of all providers, gfx836, full model and
HumanEval32 acceptance, performance and clean shutdown remain separate work.
DeepSeek checkpoint/device adaptation remains excluded from this MR. With its
compatibility checks complete, this MR can leave Draft; historical instructions
to keep it Draft pending the entire runtime matrix are superseded.
