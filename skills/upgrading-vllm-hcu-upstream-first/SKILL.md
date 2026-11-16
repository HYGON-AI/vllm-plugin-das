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

Compare the actual baseline and target implementations before retaining,
rewriting or retiring each patch. Keep a migration row with its owner, reason,
affected call sites and tests. An identical file or matching parameter names
does not establish equivalent behavior or performance.

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
are compared for parameter kinds/defaults, annotations, decorators, async
status and body AST. Findings remain pending review. Uncommitted changes are
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

## Validate the changed execution path

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

Graph capture or HTTP health alone does not establish successful generation.
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
An authorized published release digest can supply an explicit project baseline
when variables are unavailable; preserve overrides and reject mutable/foreign
images. On Actions runner v2.338.2, an empty container credentials object is
valid while present empty username/password fields fail template validation.
Omit unavailable fields for forks; never send secrets to them or relax gates.
After checkout restores its temporary HOME, package metadata may still hit Git
ownership refusal. Trust only the exact checkout path in the ephemeral job,
never `safe.directory=*`, and verify another foreign checkout remains rejected.
A successful container start still requires the actual selected tests to pass.

The portable script tests use temporary Git repositories:

```bash
python -I -B -m unittest discover \
  -s skills/upgrading-vllm-hcu-upstream-first/scripts -v
```
