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
It requires Python 3.10+ and Git supporting `--no-lazy-fetch`.
Prepare the required local history separately; preserve that restriction
when an older Git or partial clone cannot provide the inputs.

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

This is **not a complete Plugin inventory**. Enumerate Adapter entry points,
OOT registrations, backend/quantization registries, dynamic imports, native
extensions and build changes separately. `plugin_declared_metadata`, when
present, describes the legacy root version file; inspect the production
`adapter_extension` metadata independently. Static matches do not validate
ABI compatibility, HCU execution, correctness, model quality or speed.

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

## Prepare a reviewable change

Refresh the target branch, relevant open/merged requests, commits and review
threads before publishing. Compare full hunks and dependencies, including
other release lines when related. Update an existing open request for the
same fix; after it merges, prepare only a still-needed narrow follow-up on
the new target. Old-head test results retain their original scope.

Include the problem, ownership decision, exact test commands/results,
evaluation scope, remaining dependencies and AI assistance in the PR.
Record unmerged requests with URL, base/head, state, dependency and remaining
gate. Keep local validation, remote CI, review readiness and merge readiness
distinct. Do not mark a whole upgrade complete from a partial feature matrix.

The portable script tests use temporary Git repositories:

```bash
python -I -B -m unittest discover \
  -s skills/upgrading-vllm-hcu-upstream-first/scripts -v
```
