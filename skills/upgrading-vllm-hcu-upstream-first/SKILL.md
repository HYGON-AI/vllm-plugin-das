---
name: upgrading-vllm-hcu-upstream-first
description: Upgrade or rebase the HCU vLLM plugin against an explicit official vLLM target, identify carried patches and API drift, and prepare source-bound validation and review evidence. Use for upgrade work, not ordinary model serving.
---

# Upgrade the HCU plugin upstream first

Use the requested release, tag or commit; resolve it to an immutable SHA.
Read the target branch's README and `adapter_extension/README.md` first.
For v0.33, `adapter_extension` is the production project; repository-root
`vllm_hcu` is a legacy reference and must not be built as this release.

## Choose an owner before carrying a patch

- Core owns models, algorithms, scheduling, cache allocation and official
  execution interfaces. Reuse its supported implementation and registries.
- Adapter owns genuine HCU identity, platform admission and common process
  lifecycle. Do not copy an admission fix into every optional provider.
- Plugin owns optional providers through Adapter extensions. Check defaults,
  explicit enablement, required-owner failures and the existing kill switch.

For a standalone draft config missing from the official registry, first test
public Transformers `AutoConfig.register()` from an Adapter extension before
adding Core registry entries or replacing parsers. Inherit the official config
under a distinct model type, preserve foreign registrations and verify real
`get_config`/save-load paths. See [standalone draft configuration](
references/standalone-draft-config.md); config support does not qualify MTP
execution.

Compare the actual baseline and target implementations before retaining,
rewriting or retiring each patch. Keep a migration row with its owner, reason,
affected call sites and tests. An identical file or matching parameter names
does not establish equivalent behavior or performance.

Inspect the old callback's body before calling it an optimization: a
nonmutating audit callback need not become a replacement for an official
verifier. Verify the registered target implementation and its observable
behavior; direct verifier calls do not qualify full configuration construction
or model execution.

Confirm that tests call the production owner. A class or timeout formula copied
into a test can pass after the production code regresses. Use real ZMQ sockets
for receive behavior and real FutureWrapper constructors for queue ordering and
exception propagation. Bound possible infinite waits and synchronize on explicit
readiness. Where useful, exercise deliberately broken source in a separate
process to establish that the test catches the regression; never install or
commit that broken source. Reuse existing protocol suites and their CI coverage.
Treat composite patches by subfeature: a passing receive-timeout test does not
qualify RPC deadlines, pipeline queue capacity, fork initialization or worker
lifecycle. Preserve unverified subfeatures in the migration ledger.

Before retaining a pre-fork block-size hook, trace worker KV specs back into
EngineCore. A target that derives the parent's cache block size from those specs
before scheduler/hash/KV initialization can already propagate a worker-local
choice. Verify real constructors and returned configurations before copying the
old hook. A public metadata-only custom worker can isolate this negotiation,
but does not qualify attention, device KV allocation, Graph, PP computation or
model lifecycle. Separately check actual worker/manager/page/hash geometry on a
frozen real-model cohort, preserving its original accuracy and exit findings.

Parent block-size feedback alone does not qualify hybrid prefill splitting:
EngineCore can select the smallest group's page while Mamba uses a larger one.
After the real scheduler-config conversion, test unequal attention/Mamba pages
with budgets above and below the Mamba page. Preserve finer-hash tail checkpoints
and shared configuration, and use the actual Mamba spec rather than the scheduler
LCM. Equal-page tests cannot detect this mismatch; device/model gates remain.

Trace worker placements through the target's scheduler-config conversion before
testing an offload connector. A worker's UniformTypeKVCacheSpecs may be unwrapped
there; bypassing that boundary can invent a missing-manager failure. For packed
KV, check shared-allocation accounting and retained layer/offset/stride metadata
through the real builder, conversion, connector and block pool. Keep independent
backings, device copies and model acceptance separate from this CPU coverage.

## Run a static preflight

The companion script reads committed trees without checking out refs,
fetching missing objects, importing repository code or using a device.
It requires Python 3.10+. Git supporting `--no-lazy-fetch` can inspect partial
clones without fetching. Older Git can inspect fully local repositories; it
rejects partial/promisor configuration before reading objects. Prepare missing
history separately; the preflight never retries by fetching objects.

```bash
python -I -B skills/upgrading-vllm-hcu-upstream-first/scripts/upgrade_preflight.py \
  --plugin-root /path/to/plugin --plugin-ref PLUGIN_SHA \
  --fork-root /path/to/vendor-vllm --fork-ref FORK_SHA \
  --upstream-root /path/to/official-vllm --target-ref TARGET_SHA \
  --upstream-base OFFICIAL_BASE_SHA --output /path/outside/repos/new-report
```

The JSON and Markdown report list carried file differences, possible module
moves and statically parsed legacy `TARGET_MODULE` guards. Resolved functions
are compared for parameter kinds/defaults, annotations, decorators, generic
type parameters, async status and body AST. Findings remain pending review.
An unevaluable constant reassignment invalidates its previous binding; an
unresolved `TARGET_MODULE` produces a manual-review warning instead of a
comparison against the old module. A bare annotation preserves its binding.
Control-flow writes, container mutations and lexical shadowing cannot reuse a
stale global constant. Unsupported guard arguments remain explicit unresolved
rows; resolvable symbols still receive contract comparison. Constant starred
sequences are expanded without executing source. Later target rebinding makes
the callable unresolved. Wildcard imports invalidate cached targets and callable
bindings because their exports are unknown without execution.
Inherited `GIT_CONFIG` is cleared so configuration
probes and guarded Git operations inspect the same repository settings.
Uncommitted changes are
reported but excluded from analysis; commit or separately bind their bytes.
Report destinations must also be outside each checkout's private Git directory
and shared storage, including linked worktrees and separate Git directories.
For repositories with configured clean/process filters, working-tree inspection
is reported as unavailable because `git status` can execute those filters.
Committed-tree analysis still runs without invoking them.
Working-tree status excludes submodule recursion so initialized partial-clone
children cannot fetch through an older Git. Committed gitlink changes remain
in the report; uncommitted changes inside submodule worktrees are excluded.
When a regular package and sibling `.py` file share a name, module lookup
prefers the package initializer, matching Python's import precedence.
A regular parent module blocks namespace children; dotted parent components
are checked before resolving a child file.
Dotted callable lookup traverses class scopes; a function-local definition is
not treated as a public attribute. Imported and dynamically assigned attributes
remain unresolved.
Direct classmethod guards distinguish bound access from `.__func__`; aliases
or additional decorators with unresolved binding remain review candidates.
Git paths retain arbitrary filename bytes in escaped JSON/Markdown output.
Python AST parsing retains original blob bytes, honoring source encodings and
UTF-8 BOMs without collapsing distinct non-UTF-8 literals.
Working-tree entry counts treat a rename/copy and its two paths as one record.

This is **not a complete Plugin inventory**. Enumerate Adapter entry points,
OOT registrations, backend/quantization registries, dynamic imports, native
extensions and build changes separately. `plugin_declared_metadata`, when
present, describes the legacy root version file; inspect the production
`adapter_extension` metadata independently. Static matches do not validate
ABI compatibility, HCU execution, correctness, model quality or speed.

For NIXL, derive the package and descriptor ABI from the selected Core target.
Module discovery, distribution-version checks and Ray aliases must select one
native provider. A working older `nixl` UCX transfer does not satisfy a target
requiring `nixl_rocm` and compressed Nx5 descriptors. Prefer the target's pinned
official ROCm build before adding a package shim; verify installed native
dependencies, gapped GPU READ/WRITE and real model P/D with Graph and prefix.

For DeepGEMM, check the selected public platform's support gate, Core's package
discovery and the vendor's actual exports separately. A GPU-hidden CPU probe
cannot establish HCU platform activation; real detection may initialize a device.
The vendor namespace `deepgemm` does not establish the official `deep_gemm` ABI.
Inspect runtime exports as well as source, since wildcard/native imports can add
symbols. Retain scale/layout semantics: do not substitute identity transforms,
enable an unsupported platform globally or install a different host's bridge
merely to pass discovery. Qualify each admitted operator and consumer separately.

## Validate the changed execution path

For interrupted checkpoint downloads and live-worker inspection, read
[checkpoint recovery and runtime probes](references/checkpoint-runtime-probes.md).
Keep the original checkpoint identity and avoid copying live tensor owners.

For v0.33 CI event availability and the trusted exact-head manual PR route, read
[the release CI guide](../../docs/validation/vllm033-ci-coverage.md#explicit-pr-dispatch-on-v0330).
Keep dispatch acknowledgement, the actual PR checkout and aggregate acceptance separate.
For a private Core image candidate, use the guide's
[publisher and credential boundaries](../../docs/validation/vllm033-ci-coverage.md#publish-a-private-core-candidate-image);
inspect embedded architectures before changing a shared CI image.

For a newly published image, prefer a trusted single-run candidate input when
the target workflow provides one; see [candidate CI qualification](../../docs/validation/vllm033-ci-coverage.md#qualify-a-candidate-image-for-one-pr).
Verify the live PR/head, dispatch actor, immutable approved image and concurrency
group before dispatch. Reuse an existing matching run. Keep the shared consumer
and complete source/device/Graph/aggregate gates unchanged; publication or
successful image selection does not establish runtime qualification.

1. Freeze Core/Adapter/Plugin sources and dependency heads. Check capability
   and ABI compatibility instead of introducing an exact-wheel runtime lock.
2. Run affected CPU contracts, then installed native numerics and Graph
   replay with changing inputs, then the relevant real model/topology.
   Keep the requested backend, Graph, prefix, MTP and precision settings.
3. Verify every hashed path from raw wheel RECORD CSV, including recorded
   startup hooks and console scripts. Check parent and actual worker origins;
   an interpreter path or a source-only test is insufficient.
4. Preserve original evaluation requests, data revision, all samples, raw and
   normalized scores, failure sets, finish reasons and token counts. Execute
   generated evaluation code only in the isolated scorer. Report format,
   truncation and numerical failures separately; do not infer acceptance
   thresholds from a previous task or hide a score decrease.
5. On a shared node, observe available devices and other jobs before launch.
   Clean up only recorded owned process identities. Keep numerical success,
   process release and normal teardown as separate results.

For a raw accuracy loss involving output formatting, compare saved reference
continuations through the target's public prompt-logprob API. Verify identical
rendered prompt/token IDs and an unchanged BPE boundary; repeat each identical
request within each arm and record the first greedy-ranking difference, token
probabilities and top-two margins. Probe both original continuation prefixes
where available. Teacher-forced prefill has different execution geometry from
free-running decode: a reproducible ranking change narrows the investigation
but does not identify one native leaf, qualify decode, or waive the raw loss.
Keep serving configuration, package/checkpoint identity and lifecycle results
bound separately; preserve the original evaluation and normalization protocol.

For explicit FP8 KV names, writer bytes, mixed query/cache dtypes and native
Flash prefill/decode boundaries, read [Flash KV formats](references/flash-kv-formats.md).

Graph capture or HTTP health alone does not establish successful generation.
For MTP observations, derive the draft-model field and concrete class from the
selected runner and model registry before launching a full checkpoint. MRV2
uses `speculator.model`; Qwen3.5 MoE resolves to `Qwen3_5MoeMTP`, rather than
the dense `Qwen3_5MTP` base. Bind the worker's class and source to that selected
architecture, and separately verify trained layers, shared embeddings, restored
gate weights and nonzero drafted/accepted tokens. Replay a failed read-only
observer against its saved response before repeating a costly model run;
preserve that failed attempt and its completed cleanup.

When workspace planning moves to construction, build the real layer in CPU
regressions with explicit native leaf doubles and exercise its frozen plan.
Mutate the source configuration afterward and verify reservation and reuse at
the actual workspace boundary; a removed runtime-config import is not a seam
to restore merely to keep an old fixture working. Retain missing-reservation
negative controls. CPU constructor tests do not qualify a model or device.
CPU CLI transport fixtures should use a real temporary local model directory
and explicitly exercise offline resolution, rather than depend on a cached
placeholder repository. For layout-only quantization tests, isolate the
constructor's device-admission leaf while retaining real post-load/reload,
tensor-value, stride and pointer assertions. Match source-origin guards to the
actual Core import path. Keep production admission checks and device tests
unchanged; these fixture doubles do not establish native execution.
For GDN, follow the actual constructor's dispatch selection: an official
CustomOp may replace `_forward_method` after platform dispatch, so overriding
`forward_hip` alone need not execute the provider. An OOT prefill entry does
not imply a public registry for its FLA leaves. Keep the official pipeline
and current prefill/decode configuration semantics; legacy root callbacks or
deprecated switches do not establish coverage in the production wheel.
A formal consumer-alias candidate must preflight all selected Core/provider
ABIs before any publication, retain canonical Core leaves, and reject foreign
owners, partial publication, reentry and inherited incomplete activation.
Freeze effective stage policy and prepared fallbacks before model construction;
include their source identity in official compile factors. Keep Core's official
backend selector authoritative: a selected whole-op/fused route can bypass the
leaf aliases, so require execution evidence rather than counting activation as
coverage. Python guards do not execute inside an already captured device Graph;
serving health must also cover the host boundary before Graph replay.
An installed native test using public startup must call the target Core's
general-plugin loader before checking the ACTIVE Adapter owner. Plain pytest
does not execute serving startup, and host/required-capability environment
variables alone do not activate extensions. Retain the required-owner check;
verify the ordinary CI entry without a harness that silently preloads plugins.
Keep serving-worker policy and completed Python provider submissions separate
from GPU trace events: a Graph replay may expose `hipGraphLaunch` without
individual decode-node kernels. Identify prepared fallbacks per leaf; an
AITER stage policy can still execute BoltOPs chunk-O/recompute. Compare with
a stock arm that verifies every consumer alias still binds canonical Core.
Retain per-task raw losses even when fence-normalized tests all pass; do not
transfer a frozen model result to a head that inherits new target code.
Inspect actual optional-provider signatures even when the public version is
unchanged. BoltOPs chunk-O has an audited trailing `full_chunks=None` variant;
retain masked varlen semantics rather than forcing full chunks. Distinguish
indexed decode NULL slot 0 (`null_state_index=0`) from prefill's already-gathered
dense per-sequence state: row 0 is valid with no index tensor and needs
`null_state_index=-1` in the observed BoltOPs ABI. Test nonzero row-0 initial
state, output and final state rather than transferring the cache sentinel.
Include actual Triton autotune scratch in bounded device probes: the observed HCU benchmark
cache alone needs 256 MiB. A lower allocator cap can pass decode and fail official
prefill before candidate execution. Preserve that resource failure separately,
keep shapes/tuning/precision checks, and reject zero-like outputs with meaningful
absolute tolerances and an independent decode reference.
For compiler layout drift, inspect the resolved Triton build and actual layout
constructor rather than deriving its API from the Torch version. Preserve the
hardware register mode through layout construction and IR round trips; a
constructor-only probe cannot establish native lowering or changed-input Graph
correctness. Separate compiler failures from reference-allocation OOM and
numerical assertions, and retain each original failure.
Probe the MFMA constructor and explicit enum together: a transpose-only API,
an explicit-only API and a dual API can require different calls for the same
register mode. Bind capability choices as kernel constexpr arguments or actual
constexpr globals so JIT compilation and caching include them. Exercise real
native IR for each admitted architecture; Python constructor success misses
native mode rejection. Keep full kernel compilation/target device-library
availability separate from IR checks and device numerical/Graph qualification.
For optional AutoAWQ dense providers, retain the official configuration,
checkpoint loader and complete fallback through the public quantization
registry. Eager and GPU-Graph success of a pybind callable does not establish
Dynamo support: validate a dispatcher custom op with its FakeTensor rule,
fullgraph compilation and changed-input replay separately. Hash the effective
master/child policy, retain batch-invariant fallback, and keep operator tests
separate from model and HumanEval qualification.
Cover every admitted post-TP shape and each packing-tile branch before claiming
the optimized shape set is qualified. Retain recovered allocator OOM warnings
separately from unhandled OOM, numerical failures and performance qualification.

For `torch.library` exit errors, inspect the actual provider's registration:
a Library already owns its namespace, so qualified definition names can leave
duplicate namespaces in cleanup metadata even when dispatch succeeds. Reuse an
applicable upstream fix in that provider rather than suppressing weakref errors
or replacing Torch cleanup in Core/Adapter/Plugin. Cover explicit destruction
and natural child-process exit alongside eager, fake and compile contracts.
Source fixes and diagnostic repackaged wheels remain distinct from a rebuilt
provider release and its device/model qualification.

Resolve an installed provider's recorded commit against its actual source
repository before rebuilding: similarly named mirrors can lack the wheel's
newer ABI. Compare all relevant package files, not only the repaired module;
publish against the maintained source without downgrading algorithms. Label
a local PEP 517 source build separately from a released or CI-produced wheel.
For packaging profiles, native dependencies and source/installed test isolation,
read [native provider validation](references/native-provider-validation.md).
Validate its source/wheel/installed payload and raw RECORD, then rerun affected
CPU, native/Graph and model gates. A new venv can change a registered console
script's interpreter prefix; audit that exact generated script separately
while preserving every payload check and the original RECORD verification.
Inspect real wheels built from recursive checkouts for submodule `.git`
pointers. Exclude Git control metadata in the distribution manifest and verify
the remaining package payload is unchanged. No JIT `.so` in a wheel does not
mean it has no native assets: inventory source-controlled HSA code objects
separately from newly compiled JIT modules and qualify their execution.

Audit installation destinations independently of import isolation. On Python
3.12, `-S` can leave a venv executable using the base `sys.prefix`; injecting
its site-packages into `sys.path` changes package discovery, not pip's write
location. Check `sys.executable`, `sys.prefix` and `sysconfig.get_path("purelib")`
before installation. Use an explicit candidate `--prefix`/`--target`, or a
normal venv interpreter whose prefix has been verified, then verify actual
installed files and RECORD. Check build-backend child interpreters separately.

Chained jobs must wait for completed cleanup, checkpoint post-seal and RECORD
receipts, not just a terminal-looking state string. Write final status only
after the last check, using an atomic snapshot. Preserve consumer failures
and restart only unexecuted work in new output directories.

After a pause or executor restart, recheck the controller's recorded process
identity, descendants, listener and output files. A saved `RESOURCE_WAIT` or
`RUNNING` receipt does not establish that work is still alive. Preserve the
original receipt and record interruption separately; distinguish no service
or samples from an interrupted evaluation before starting a new attempt.

Check the interpreters spawned by structural gates too. A parent launched
with `-I -S` can spawn a normal interpreter whose sitecustomize preloads an
older Adapter, even with an empty PYTHONPATH and HOSTnone. Use a matching
installed owner for that child and retain the first failure; do not regenerate
documentation or relax freshness assertions to hide package-origin drift.

For installed lifecycle tests that launch a bare CLI, prepend the tested
venv bin directory to that arm's PATH, check the selected script and shebang,
and record actual server/worker origins. Preloading the correct parent package
does not select the child CLI. Preserve a wrong-PATH attempt as invalid harness
evidence, clean up its owned identities, and rerun the unchanged original tests
and deadlines in a new output directory.

For pipeline parallelism, count workers and distinct GPU trace ranks as
TP times PP. Hybrid cache projection can preserve globally indexed groups
whose local layer list is empty: allocate only worker-owned layers and check
model-runner setup on stages with no local recurrent layers. Keep global
group indexing intact. Reuse the official planner implementation when it
already handles this case, then validate any remaining stage-local fix with
a serialized hybrid checkpoint and changing prompts/batches. A zero output
head can test startup and communication, but cannot establish model quality.

Re-read device identity and total capacity after an environment change; an old
absolute free-memory threshold can become unreachable and a prior utilization
fraction can become too small for the same weights. Record a new budget and
its changed validation scope before launching.

A shared-node memory preflight is an observation, not a reservation. Preserve
the actual worker admission failure if free memory changes before startup;
do not report a weight-loading failure or a model score for that attempt.
Check when the target measures admission relative to distributed initialization:
a standalone Torch probe omits communicator allocations. With explicit fixed KV,
lowering the admission fraction does not lower that KV allocation or Graph demand.

Matching package entry-point hashes do not establish matching installations.
Compare the complete owned package payloads and native libraries before using
an old score as a control. A quantization method can retain its class while
an Adapter rebinds its execution callable; inspect actual operators/kernels
and topology before attributing an accuracy change to the framework.

For a package-only precision difference, vary one existing provider switch per
fresh process while preserving the full installation and original requests.
Inspect actual worker bindings and probe kernels: changing scaled-MM does not
also isolate mHC. Equal aggregate scores can hide different failed samples and
different generated content. A checkpoint-weight numerical probe with synthetic
activations helps locate a boundary, but does not prove model-level causality.
Keep compile failures and missing process-tracking evidence visible; rerun a
corrected diagnostic separately instead of treating missing results as passes.

Platform API semantics do not establish a kernel's physical lane width. For
warp reductions, check the active compiler target and generated reduction
scope before choosing lane IDs or shared-memory partitions. Preserve the
Adapter's platform identity. Check every executed fusion independently and
keep thread-group branches uniform at that physical width; fixing an unfused
projection does not qualify a fused post/pre or its normalization epilogue.
An exact projection and squared-sum regression
with changed-input Graph replay can expose duplicate lane contributions;
validate the real model separately before claiming its accuracy is restored.

An unchanged serial request has different scheduling and cache history from
the original cohort. Retain both full-cohort and serial results, including
baseline failures. Reproduction on the baseline does not establish numerical
equivalence or erase an earlier candidate failure.

## Prepare a reviewable change

Refresh the target branch, relevant open/merged requests, commits and review
threads before publishing. Compare full hunks and dependencies, including
other release lines when related. Update an existing open request for the
same fix; after it merges, prepare only a still-needed narrow follow-up on
the new target. Old-head test results retain their original scope.
When a new target enables an existing fused kernel, prepare its affected
constructor, kernel and model checks. Preserve both sides of any shared CI
selection file; a clean patch application does not qualify the new runtime.

Include the problem, ownership decision, exact test commands/results,
evaluation scope, remaining dependencies and AI assistance in the PR.
Record unmerged requests with URL, base/head, state, dependency and remaining
gate. Keep local validation, remote CI, review readiness and merge readiness
distinct. Do not mark a whole upgrade complete from a partial feature matrix.

Trace a prepared CI image to the current publisher's workflow and exact
repository, then verify its manifest digest and runtime metadata. A valid
release image can live outside the Plugin's historical image repository;
update an explicit repository allowlist with executable positive and negative
guard cases. Keep image configuration, source checks and actual device jobs
as separate evidence. Registry provenance alone does not qualify the runtime.

If a selected test fails on a missing official Core interface, inspect the
installed Core payload and the publisher's package index before rebuilding
an image. Reusing the same archived wheel preserves the defect. Keep the
original test selected and deliver the required interface through Core.

For images shared by multiple device tiers, inspect embedded code objects in
each GPU extension and record the actual build targets. A Python-only wheel
refresh over native libraries for one architecture cannot qualify another.
After a complete compatible build, rerun installed portable contracts, then
the actual image's selected native/Graph jobs on each supported architecture.
Record a locally assembled image separately from a successfully published
digest and a runtime-qualified image; a pull-only registry token cannot
publish it. Keep private source artifacts in an authorized private channel.

For three-way repair, resolve the live target Git ref: PR metadata `base.sha`
can lag the branch. Preserve independent changes and compare interface owners
when a conflict reveals a newer merged implementation. Check expected head and
live base again before a normal non-forced publication.

Inspect the actual fork job environment before relying on variables/secrets.
Only when the project's trusted policy explicitly permits a recorded baseline,
an authorized published release digest can supply it when variables are unavailable;
otherwise retain required configuration. Preserve overrides and reject mutable/foreign
images. On Actions runner v2.338.2, an empty container credentials object is
valid while present empty username/password fields fail template validation.
Omit unavailable fields for forks; never send secrets to them or relax gates.
After checkout restores its temporary HOME, package metadata may still hit Git
ownership refusal. Trust only the exact checkout path in the ephemeral job,
never `safe.directory=*`, and verify another foreign checkout remains rejected.
A successful container start still requires the actual selected tests to pass.

When PR checks never appear, resolve the workflow's actual event context before
changing labels or pushing again. Current GitHub.com `pull_request_target` uses
the receiving repository's default branch; older GHES versions can use the PR
base. Check the applicable [event documentation](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#pull_request_target),
default ref and workflow file there. Landing it only on another release branch
may leave the PR entry absent. A manual dispatch returning 204 is accepted:
observe its run before retrying. A target-only run does not test a PR head.
Retain trusted authorization, exact-head checkout, credential isolation and
device/aggregate gates when choosing a repair within the authorized branch scope.
Zero check contexts or conflict-free mergeability do not establish CI success.

A workflow can intentionally skip its GPU job for pull_request_target events.
Treat the automatic green summary as structural evidence only. If the project
provides an authorized manual PR test route, inspect its current trusted
workflow and concurrency group before dispatching, then verify the fetched PR
head, merged source and installed wheel identity in the resulting device job.
Do not infer that a draft flag caused the skip or that a manual run qualifies
a newer PR head.

The portable script tests use temporary Git repositories:

```bash
python -I -B -m unittest discover \
  -s skills/upgrading-vllm-hcu-upstream-first/scripts -v
```


Check the actual Core APIs in a prepared image before replacing it with a newer
dated tag. Compare complete source changes and verify package/layer hashes;
rebuilding against the same archived wheel cannot supply a missing interface.
When native, build and dependency inputs are unchanged, exact target Python over
verified published binaries is only a partial candidate. Preserve its provenance
and regenerate RECORD; required native, image and device gates remain separate.
Inspect actual embedded HIP bundle targets in a scratch library copy, not an
architecture substring in a diagnostic path. DTK llvm-objdump --offloading can
extract files beside its input. A gfx938-only library cannot qualify gfx936;
even a compiled dual-target wheel still needs execution on both architectures.
Keep private Core source/wheels in the original private project or an authorized
internal publisher; publishing a candidate does not repin consumers or a release.

When a published runtime lacks a newly merged Adapter capability, deliver the
host wheel and its exact required native-wheel build together. Verify wheel
RECORDs, namespace ownership, build ABI and the host's kernel requirement before
offline installation. Compare the installed payload with the prepared wheels,
verify installed RECORDs and embedded architectures, then rerun the affected
full CI matrix. Publish a unique candidate digest; keep shared consumer changes
and device/model acceptance separate from artifact delivery.
Run metadata-only image inspection with isolated no-site startup, adding the
system interpreter's package directories without executing `.pth` or
`sitecustomize`; a disabled host can still import Torch during bootstrap.

For optional container credentials, distinguish cross-repository PRs from PRs
inside a repository that is itself a fork. Compare head/base repository identity
and event type rather than head.repo.fork. Cover cross-repository and both kinds
of same-repository PR, push and manual events while retaining secret escaping.

When the public Adapter Worker wraps Core execution with a health guard, Plugin
inheritance tests must preserve that Adapter method and reject a Plugin override.
Do not remove the guard to satisfy an outdated direct-Core identity assertion.
Exercise published and proposed compatible Adapter pairs, then the unchanged
portable selection. These tests do not qualify guard behavior, device numerics,
model quality or clean lifecycle. Track exit-time errors even when pytest exits0.


When modular MoE reserves per-expert capacity, retain batched tensors and pass
live-token counts to the target's official masked activation API; GEMM masking
alone leaves activation processing padding. Check both Core and Plugin wrapper
signatures at the actual target. Admission must follow all switches selecting
the scatter/gather provider, whose tile limits may differ from GEMM alignment.
Keep dispatch capacity guards and update positive decode fixtures to fit their
allocated bound, with separate oversized refusals and retained RED evidence.


Trace the complete inherited activation method chain: accepting keyword
arguments does not prove forwarding. Keep that chain real in CPU regressions
and double only its final provider. Verify native batched semantics separately;
a binary may accept the same operator signature yet implement flat counts only.
Retain a failing old-binary result and bind native/Graph successes to the actual
coherent Core build, architecture and submitted source.


## 已有 MR 的目标版本边界

处理已提 MR 时，仍以本次选定的目标版本为交付范围。旧分支请求先作为功能迁移和证据输入；
核对目标版官方实现与当前 owner 后，仅迁移仍缺少的能力。没有明确的旧版维护要求时，
不要为修复历史 CI 新建旧发布线请求，也不要直接改 base 夹带跨版本差异。
较新开发版、较旧 wheel 或组件移植的结果只能定位问题；正式验收须在目标发布线的完整组合上重跑。

静态目标解析只接受无转换、无格式参数的 f-string。`!s`/`!r`/`!a`、
格式宽度/精度、嵌套或空格式参数均须保留为 unresolved 并触发人工复核；
不能丢弃这些 AST 字段后检查另一个模块，再给出静态 unchanged 结论。

For RMSNorm provider migration, inspect both exact-target IR and the actual
compiled serving kernel. Eager native's intermediate low-precision cast may
be elided by default Inductor; do not add an extra multiply solely to match
an eager oracle. Compare official eager, official compiled and the provider
under the same inference/autograd context and compiler configuration, with
changed-input Graph replay. Saved-tensor precision policies can differ with
gradients enabled. Bind generated code to runtime traces; vLLM AOT Inductor
output may be nested under VLLM_CACHE_ROOT/torch_compile_cache rather than
the top-level Inductor cache. Keep FP32 epsilon, residual precision and input
preservation checks separate from intermediate-cast diagnosis. Small constant
inputs near sqrt(epsilon) with an independent FP64 scalar oracle can expose
epsilon narrowing hidden by random unit-scale inputs. Check ordinary and
fused exports separately. A public inference API may default to an in-place
output; provide a separate output for a non-mutating official op and validate
its actual ABI before activation. Do not change compiler settings or force
eager to make the comparison pass. Bind vendor behavior to the tested build;
component repair alone does not prove a HumanEval failure's cause. Retain
per-task pass-set comparisons and exact-release model gates.

Include two successive residual additions without returning an otherwise
unused intermediate residual: default compilation can remove its low-precision
store, whereas an opaque mutating provider or an observation-only output can
force materialization. Single fused calls can miss the resulting composition
difference. Cover the legacy mutating ABI independently from a functional layer.
Measure the actual provider against official compiled IR at relevant shapes;
a transparent decomposition can add launches and intermediate buffers even
when component numerics pass. If official IR covers the required behavior,
consider official ownership of compiled execution and assess eager vendor
dispatch separately. Revalidate the resulting model path before acceptance.

Use the same compilation pipeline for the reference and candidate. A direct
native implementation under ordinary `torch.compile` is not equivalent to an
official IR wrapper compiled without vLLM's functionalization and lowering
passes. Compare genuine Core and OOT layers through the target release's
compiler passes, retain its default arithmetic policy, and record selected IR
providers and generated kernels. Follow the target wrapper's dynamic-dimension
and backend-lifetime contracts; a one-shot backend cannot survive a Dynamo
analysis restart. Disable compile caches only in a diagnostic that needs to
observe passes, and validate normal serving with its intended caches intact.
If independent builds of identical kernel source differ, add a Core-vs-Core
control and inspect selected launch configurations before assigning the cause
to the Plugin. Preserve the failures and full-model gates.

Budget serial device jobs from measured cold execution, including portable
startup, every provider subprocess and the remaining native matrix. Inspect
actual timeout annotations before changing the bounded job budget; preserve
all selection, authorization and aggregate gates. Upload each component's
JUnit alongside the final aggregate so a later cancellation retains completed
evidence. Component logs or XML do not qualify an unfinished aggregate, and
an increased budget is not a passing rerun. Reuse live runs and retain the
original timeout result when validating the corrected workflow.

For provider CI, inspect the final pytest log and result artifact as well as
job status. A session-level version pause can exit successfully without running
any cases; a benchmark driver can then produce no CSV. Record those as no
execution or unavailable performance evidence, retaining the original guard.
Bind real pass/fail/skip counts and timings to the exact submitted head and
dependency matrix. An independently green compliance aggregate, successful
wheel build or selected native subset does not qualify that correctness or
performance job. Diagnose the compatibility limit and rerun its affected
matrix after repair; do not bypass the pause to claim release qualification.

Check each job's actual driver: direct benchmark scripts may execute even
when another job's pytest session is paused. Count JUnit launcher cases,
CSV measurement rows and numerical regression cases separately; selected
benchmark timings do not qualify unrelated kernels or model accuracy.

Derive grouped key/value head counts from the actual checkpoint and the
selected Core's sharding/replication rules. Cover each new per-rank geometry
when changing TP, even if its head ratio matches an already-tested TP1 shape.
Keep prefill, packed decode and speculative varlen/accepted-state Graph
coverage. Add only missing cases and bind their receipts to the actual
installation's package RECORDs; component parity does not qualify TP
communication, model quality or performance, and older counts stay separate.
Interpret shutdown logs against the selected Core source: `mode=abort` may
mean the default zero-timeout request policy rather than an OS abort. Record
actual exit codes, process release and native leak reports independently;
successful exits do not resolve nanobind leaks.
For a suspected frozen-heap leak, compare the public `freeze_gc_heap()` helper
and natural interpreter exit in hidden-device base/candidate processes before
running a model. Keep serving-time freezing intact. Verify shutdown duration,
child-process release and `force killing remaining processes` logs: a parent
exit of 0 cannot establish natural worker cleanup. An import-only fix does not
qualify serving or waive the exit-latency tradeoff of final cyclic collection.

For same-host Mooncake HIP IPC, inspect ROCR visibility separately from each
process's HIP/CUDA compute mask. The tested DTK2610/gfx938 runtime crashes on
the first 64-byte copy with disjoint single-device ROCR masks, also in a direct
HIP program without Torch or Mooncake. A shared ROCR list of the two allocated
peers with separate HIP/CUDA masks passes two changed-input waves in direct HIP
and the original Mooncake buffer API. Verify relative HIP indexes, PCI buses,
full destination bytes and untouched regions. Preserve the failed masks and
record this as a deployment condition for the tested runtime; do not generalize
it to other builds or claim a native source fix. Keep P/D model output, Graph,
accuracy, performance and server lifecycle gates separate; changing SDMA or
transport would be a distinct diagnostic.
