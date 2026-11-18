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
- At most four selected ARC hardware jobs run concurrently per PR, with
  `fail-fast: false`; this does not change partition selection or card counts.
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

Both static/selection and hardware commands use the preserved native helper,
which restores scheduler-provided thread budgets and device visibility after
sourcing `/opt/dtk/env.sh` (including explicitly empty visibility values). Hardware
steps install the tested Plugin in the ephemeral job container and run the
original dependency/evaluator bootstrap followed by
`run_hcu_ci_job.py`. The same preflight validates the installed v0.28.1 environment
and selected card count. Missing models/datasets remain failures, not skips.
Configured `HCU_CI_MODEL_ROOT` / `HCU_CI_DATASET_ROOT` are paths **inside the job
container**. By default, resources are discovered under
`/public/opendas/DL_DATA/{llm-models,ci_datasets}` and the corresponding `/public4`
paths. No host mount or old `/models` / `/datasets` alias is created.

The writable `/ci_public` mount is required; the launcher fails rather than
creating a missing platform mount on the local disk. Pip temporary downloads
(installation-only `TMPDIR`), `PIP_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR` and
`XDG_CACHE_HOME` use
`/ci_public/hcu-ci/<owner>/<repo>/<run-id>/<attempt>/<job-id>/`. The namespace is
validated and separates partitions, repositories and retries, with reuse only
inside one job/attempt, not a mutable cache from another PR. This reduces local
scratch use; source-tree editable build outputs still live in the checkout.
`HCU_CI_NFS_TMPDIR` is applied only to dependency/plugin installation commands.
Runtime `TMPDIR` is not changed: vLLM/Python may derive Unix socket paths from it,
and a long NFS namespace can exceed Linux socket path limits.
NFS storage is platform-retained (the guide specifies seven days); the workflow
does not recursively delete shared storage. No artifacts are loaded from NFS as
trusted workflow code or authorization data.

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

## Alignment with the supplied ARC guide

- Sections 2 and 3.1: one runner pool label, platform-provided container and
  mounts, no nested Docker, no fixed devices, and injected concurrency budgets.
- Sections 2.1 and 2.2: NFS-backed installer temporary files and caches in both
  build/static and card jobs; runtime socket temporary paths stay unchanged,
  and reports remain on the runner for artifact upload.
- Section 3.2 requires separate build/test jobs when compilation exceeds ten
  minutes. The observed editable build in the initial ARC run took about ninety
  seconds, so the guide's short-build example (`pip install -e .` in the card
  container) still applies. If that duration grows beyond ten minutes, move the
  build to `hygon-build`, explicitly target both selected architectures there,
  and transfer immutable wheels to tests; do not detect devices on a build pool.
- Section 3.3: at most four concurrent card partitions per PR. Original tests,
  assertions, selection rules and the explicit generated-code sandbox exception
  are unchanged.

The supplied guide contains older statements that BW1000 is not yet supported,
while its November 17 update lists all four BW1000 pools. Actual ARC assignment
of the selected BW1000 jobs confirms that the newer pool mapping is applicable.

An `Initialize containers` failure warrants checking resource wait and the pinned
image pull, as in section 5. `137` / `OOMKilled` or ephemeral-storage eviction
requires the actual pod/container reason; exit `130` plus a runner shutdown alone
does not prove either cause (or a human cancellation). If shutdown recurs, retain
the run attempt, runner name and exact timestamp together with runner diagnostics
and the job/runner pod events and controller logs. An `always()` upload cannot
finish after the runner itself has stopped. Successful lint/CPU contracts do not
qualify hardware execution or identify the shutdown trigger.
