# v0.28.1 PR CI on ARC/Kubernetes

The `v0.28.1-dev` PR workflow defaults to the ARC job-container lane. It uses the
runner migration guide's pool names, not combined self-hosted labels. The
existing selector, registration partitions, suite arguments, resource preflight,
environment lock, per-job timeouts and no-skip report policy remain unchanged.

## Select a backend

Repository Actions variable `HCU_CI_RUNNER_MODE_V0281`:

| Value | PR CI backend |
| --- | --- |
| Unset, empty, `k8s` | ARC/Kubernetes |
| `legacy` | Existing self-hosted/Docker PR lane |
| Any other value | Fail the existing `ci-gate` check |

Rollback requires **explicitly setting `legacy`**. Deleting the variable selects
K8s again. Fork `pull_request` runs may not see repository variables; for these
events the K8s default and release-specific image fallback apply. A repository
variable is not a guaranteed fork-event rollback mechanism. Revert the routing
commit if a rollback must cover such events. This migration does not change the
trigger to `pull_request_target`, expose secrets or alter repository settings.

## Scheduling and immutable image

- `hygon-cpu`: image digest resolution and trusted authorization (stdlib Python).
- `hygon-build`: CPU patch contracts and selection in the prepared CI image.
- `bw1000` / `gfx936` and `bw1100` / `gfx938` selected jobs with 1, 2, 4 or 8 cards
  map to `hygon-bw1000-{N}card` / `hygon-bw1100-{N}card` respectively.
- Unsupported or mismatched topologies fail rather than dropping selected tests.
- Jobs marked `isolate_generated_code` remain on the original hardened Docker
  path. A normal ARC job container does not replace its read-only source copy,
  dropped capabilities, no-new-privileges and PID limits. The final gate requires
  these jobs too; this is an explicit security exception, not a skipped test.

`HCU_CI_IMAGE_V0281` retains the existing release-specific fallback
`10.16.1.152:5000/jenkins/model_test_env/vllm:0.28.1rc1-latest`. Before scheduling
any job containers, a CPU job resolves the tag via the internal registry's
manifest API and verifies its SHA256 against `Docker-Content-Digest`. Only the
existing internal registry's `jenkins/model_test_env/vllm` and
`hcu/base_env/vllm` repositories are accepted. Explicit SHA256 references are
passed through; the actual pull and existing environment lock still must pass.
This also avoids the v0.33 migration's prepared-image repository mismatch.

The resolver does not need Docker or install registry tools. It uses the
existing internal HTTP registry endpoint (not an Internet TLS fallback), refuses
redirects and never sends credentials. ARC must have anonymous manifest/pull
access or platform-managed image pull credentials for this registry; if the
manifest endpoint requires authentication, tag resolution fails closed. No
repository secret is passed into untrusted source/static jobs. A resolved digest
provides per-run consistency, not image qualification or transport authenticity.

## Native job execution

Hardware jobs run directly in `container.image`; no nested container is started.
No `container.options`, `container.volumes`, privileged request, host selection,
fixed HCU IDs or thread-budget overrides are added. Device visibility and
`OMP_NUM_THREADS`, `MAX_JOBS`, `CMAKE_BUILD_PARALLEL_LEVEL` remain platform-owned.
The native helper requires the platform-injected `HCU_CI_IN_K8S=1` marker.

The helper sources `/opt/dtk/env.sh`, installs the tested Plugin in the ephemeral
job container, and runs the original dependency/evaluator bootstrap followed by
`run_hcu_ci_job.py`. The same preflight validates the installed v0.28.1 environment
and selected card count. Missing models/datasets remain failures, not skips.
Configured `HCU_CI_MODEL_ROOT` / `HCU_CI_DATASET_ROOT` are paths **inside the job
container**. By default, resources are discovered under
`/public/opendas/DL_DATA/{llm-models,ci_datasets}` and the corresponding `/public4`
paths. No host mount or old `/models` / `/datasets` alias is created.

Logs and JUnit reports stay under a per-job `runner.temp` directory and are
uploaded with the existing partition-specific artifact naming and 30-day
retention, including failed test steps. Pods/workspaces are reclaimed by ARC;
there is no recursive host workspace repair or deletion on the K8s lane.

The image resolver, matrix mapper and native launcher belong to the workflow
revision (`github.workflow_sha`), not the tested revision. The mapper and launcher
are preserved under `runner.temp` before checking out the PR head. This allows
pre-migration PR branches to use the ARC lane without copying the migration into
every branch. Existing selector, preflight, environment lock, plugin source and
tests still come from the exact PR head; the bootstrap does not replace them.
The static job fetches only the workflow, base and head commits (depth 1 each),
retaining both endpoints needed by the existing two-endpoint diff selector.

The trusted-author/`ready-hcu` helper is checked out from the immutable PR base
SHA on a separate CPU runner. Tested source is always the PR head SHA; checkout
credentials are not persisted. The outer `ci-gate` name and fail-closed admission
remain unchanged, including unauthorized hardware selections.

## Scope and acceptance

This change covers **PR CI only**. Nightly, full-enabled, multi-node, image build,
release workflows, and the Cookbook model-smoke change in #401 are not migrated
here. Generated-code sandboxes deliberately retain their original backend.

Before rollout, cluster owners must confirm both architecture pools, their
requested 1/2/4/8-card resources, DTK/vendor mounts, registry access and model/data
paths. In particular, large model jobs must fit the ARC pool's actual memory,
shared-memory and PID budgets; a successful legacy run does not qualify a smaller
ARC allocation. Compare the same immutable image and selected partitions on both
backends, including generated-code isolation and artifact upload on failure.
Local CPU tests and workflow lint are not ARC/HCU or model-quality evidence.
