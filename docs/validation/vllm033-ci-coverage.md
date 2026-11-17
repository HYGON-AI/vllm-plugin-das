# v0.33 CI coverage candidate

A change to `adapter_extension/src/hcu_vllm_das/environment.py` or
`attention/e5m2_scales.py` previously selected no HCU jobs and did not trigger
fallback. The legacy workflows only target `v0.28.1-dev`.

`hcu-pr-ci-v033.yml` handles pushes and dispatches on `v0.33.0`. Its automatic
PR entry also requires the workflow on the repository's default branch (see
the manual PR route below). It invokes the selector CLI with `--release v0.33.0` and the
independent `hcu-test-map-v033.json`. The default CLI, legacy map, registry and
workflows retain the v0.28.1 behavior. It never sends v0.33 changes to legacy
vllm_hcu jobs or substitutes the v0.28.1 image.

Every mapped code change selects `v033-source-contract` and
`v033-device-gfx936`. Every unmapped non-documentation path selects both and is
reported in `unmapped_paths` with `fallback=true`. Deleted paths are included
in release diff selection. Documentation alone selects no jobs; `--full` and
`--accuracy` expand to the full declared v0.33 scope even for empty/doc diffs.
The CI selector/map/workflow and test tooling themselves select both jobs.
An explicit `--config` is passed to the selected release's parser; the default
map is chosen by release only when no override is provided. Invalid overrides
fail selection rather than silently using the built-in map.

PR hardware execution reuses the existing author/`ready-hcu` authorization
helper from the frozen trusted release checkout on a hosted CPU runner. A trusted
author is authorized automatically; other authors need `ready-hcu` applied
by a trusted actor. Only the `labeled` event that adds `ready-hcu` authorizes,
and it approves the exact head snapshot carried by that run: a later push
leaves hardware locked until a trusted actor removes and re-applies the label.
CPU source contracts run independently. Release pushes/manual runs must target
`refs/heads/v0.33.0`. Image resolution and the self-hosted device job depend
on authorization. The aggregate gate rejects a selected but unauthorized,
skipped, failed or cancelled stage, so a skipped hardware job is not a pass.

The source job runs existing official API AST and package-boundary tests using
the immutable official target `4ba787ec9959e566d3b587a796dff07673324da5`.
It does not certify the whole portable suite, native ABI, or model quality.
Portable collects ordinary contract files and uses `-m 'not hcu'` with devices
hidden. Device uses AST marker discovery without importing pure portable modules,
then `-m hcu` with the live flags. Module/class marks and decorated cases also
cover a file containing
portable format contracts and native numerical cases together, including
SlimQuant's six marked cases in `test_slimquant.py`. Device enables the existing
`HCU_SLIMQUANT_LIVE=1` opt-in. Deselected cases belong to the other tier; executed
cases must still contain no skips or xfails. Script-only selector cases retain
their eleven explicit subprocess invocations outside pytest. The job
installs only the independent `adapter_extension` project, never root
`setup.py`, and explicitly selects the Adapter platform entry point in every
tier so an image's legacy platform entry point cannot be discovered. The
installed tiers pin Core, Adapter and Plugin packages before test imports and
verify loaded module origins against their snapshotted non-editable RECORDs
after collection and at session end. A pure portable module's source-path
fixture is never imported just to select device cases. These are file-ownership
checks, not a replacement for wheel integrity, provider or numerical validation.
The
workflow isolates parent Python discovery and clears inherited `PYTHONPATH`
and user site packages from child environments. Empty
results, failures, errors, skips/xfails and unexpected passes
fail the result gate. JUnit reports preserve each tier separately. The shared
HCU registration assertion recognizes only files actually executed by this
release runner; all other HCU files still require the legacy registry.

Installed-version and source-root discovery use a separate process with the
same interpreter, sanitized environment and working directory as the tests.
The probe queries metadata and the module spec without importing vLLM; both
the release gate and portable AST root use that child's result. A controller's
different `sys.path` or inherited `PYTHONPATH` cannot choose the tested Core.

Set `HARBOR_SITE` to the approved registry host (optionally with a port, without
a scheme or path). Set `HCU_CI_IMAGE_V033` to an immutable
`${HARBOR_SITE}/jenkins/model_test_env/vllm@sha256:<64 lowercase hex digits>` or
`${HARBOR_SITE}/hcu/base_env/vllm@sha256:<64 lowercase hex digits>`
for a prepared DTK image with compatible vLLM 0.33, Adapter, vendor libraries,
pytest, packaging and build tooling. Missing/whitespace values, mutable tags,
malformed digests and other repositories fail before the hardware job starts;
there is no legacy-image fallback or automatic pull to resolve a tag.
On current GitHub.com, `pull_request_target` resolves the workflow from the
receiving repository's default branch. The resolver separately checks out
v0.33.0 and freezes that commit for the authorization helper; PR metadata's
base SHA can lag the branch and lack newly delivered helpers. PR-head
code runs only in read-only hosted jobs or, after a trusted author or an
authorized `ready-hcu` label unlocks hardware, inside the device container.
The device container uses the repository `HARBOR_USERNAME`/`HARBOR_PASSWORD`
secrets for the initial private-image pull. The credentials declaration does
not prove pull success. Missing authorization or pull access remains a failure;
never let a PR head authorize itself or run untrusted PR code in a job that
holds a write-scoped token.
The explicit legacy container mounts `/opt/hyhal` read-only and uses the
existing 32GiB shared-memory policy without privileged mode. The default ARC
backend below delegates mounts, shared memory and devices to the cluster.
The hosted selector/source jobs receive no Harbor secrets or HCU device mounts.
PR-code jobs use read-only contents permissions and every checkout disables
persisted git credentials, including the source job's separate official checkout.
A configured digest is not evidence that the image or hardware tier passed.
The runner must provide the host `/opt/hyhal` runtime, gfx936 devices
and the normal prepared DTK environment. This candidate does not create or
validate such an image, allocate runner resources, or certify models and
other HCU architectures. Image/wheel provenance, model quality, performance
and the remaining release matrix remain separate acceptance gates.

Installed tiers use the existing `packaging` dependency to parse PEP 440 epoch
and release tuples, preserving the runtime activation policy for rc/dev/post/local
builds and explicit epoch zero. The import is lazy so source/selection bootstrap
still works on a stdlib-only interpreter. Script-only public-selector contracts
run as eleven explicit isolated cases; they are not counted as collected pytest
tests. Installed tiers default to `TORCHINDUCTOR_COMPILE_THREADS=1`, while keeping
explicit overrides, torch.compile and changed-input Graph contracts enabled.

## Switch v0.33 device CI between legacy runners and ARC/K8s

The device jobs support the ARC job-container model described in the supplied
runner migration guide and use **ARC/K8s by default**. To override the default,
configure the repository **Actions variable** `HCU_CI_RUNNER_MODE_V033`
(Settings → Secrets and variables → Actions → Variables):

| Value | `v033-device-gfx936` | `v033-device-gfx938` | Container resources |
| --- | --- | --- | --- |
| `legacy` | `[self-hosted, ci, bw1000]` | `[self-hosted, ci, bw1100]` | Existing host devices, read-only `/opt/hyhal`, 32GiB shm |
| unset, empty or `k8s` | `hygon-bw1000-1card` | `hygon-bw1100-1card` | Supplied by the ARC pool; no custom volumes or Docker options |

Explicit non-empty values must be `legacy` or `k8s`. A typo fails image/runner
resolution before publishing any outputs; it never silently falls back or
selects an arbitrary runner. The mode and both container specifications are resolved together, so
changing labels cannot accidentally retain legacy host-device options.

Keep the prepared, digest-pinned `HCU_CI_IMAGE_V033` and approved `HARBOR_SITE`
configuration above. The same image is passed through `container.image` in both
modes. ARC pool names are single strings, not combined with `self-hosted` or
hardware labels. The workflow does not run nested Docker, request privileged
mode, add `container.volumes`, choose a host/node or override visible-device IDs
in K8s mode. Leave the injected `OMP_NUM_THREADS`, `MAX_JOBS` and
`CMAKE_BUILD_PARALLEL_LEVEL` intact. ARC provides standard shared storage and
resource limits; the repository does not configure the cluster or mount it.
`HCU_CI_IN_K8S=1` can be inspected in the job environment for diagnosis, but it
cannot choose `runs-on`: scheduling happens before steps execute.

Before merging or running this workflow revision, the cluster owner must confirm
both single-card pools (including the newly introduced BW1000 pool), prepared
image access and the required DTK runtime are ready. Revisions containing this
default use ARC/K8s without setting a variable; older revisions retain their own
defaults. An existing explicit `legacy` value still selects the old runners.
Compare the two backends with the **same image digest and test tiers**. Roll back
by explicitly setting `HCU_CI_RUNNER_MODE_V033=legacy`; no workflow edit is needed.
Removing or emptying the variable restores **K8s**, not legacy. This change only
updates the workflow default; it does not modify repository variables, cluster
resources or branch protection itself.

Both modes retain job IDs/check names, per-job 45-minute timeouts, concurrency
cancellation, artifact names, source/portable/device suites and the existing
trusted-author/`ready-hcu` authorization. CPU selection/source checks stay on
GitHub-hosted runners without Harbor secrets. Trusted event execution supplies
repository variables and initial device-image pull credentials; automatic PR
events retain the trusted-author/`ready-hcu` policy above. The PR never supplies
the workflow or its authorization helper.
The aggregate `v033-ci-gate` still requires both selected device jobs to pass.
The v0.28.1 selector map and Docker helpers are not used by this release and
are intentionally not migrated here. The already-ARC wheel publishing workflow
is unchanged; this switch covers the v0.33 device CI jobs only.

Regression tests execute the actual image/runner resolver script with fake
registry names and digests, checking unset/empty/explicit K8s, explicit legacy,
invalid inputs, expression wiring and credential isolation. These are CPU configuration contracts, not
proof of ARC scheduling, registry pulls, device isolation, HCU numerical
correctness or model quality; those require a real authorized run.

CPU verification commands:

```bash
python -I -B -m unittest discover -s tests/ci -v
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 HCU_ADAPTER_HOST=none \
  HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 CUDA_VISIBLE_DEVICES=-1 \
  python -I -B -m pytest -q tests/ci tests/patch/test_hcu_ci_selector.py
VLLM_TARGET_ROOT=/path/to/official-v033 \
  python -I -B tools/run_v033_ci.py --tier source --junit v033-source.xml
```

The upgrade audit preserves the original zero-job probe and the first full
regression failure: the old global registration assertion did not recognize
three existing v0.33 HCU-marked files. The candidate accounts for them via the
actual release runner rather than weakening registration for other files.
No GPU job, remote CI, image publication, push or PR is claimed by these local
CPU results.

The integration follow-up incorporates the merged optional SlimQuant format
from MR363. Its generic public Core PLE interface remains an enablement and
deployment dependency; the CI wrapper does not invent that API, override its
capability guard, or count a missing interface/skip as positive device evidence.


## Checked-out package workspace

After checkout, both ephemeral device containers add only the exact
`GITHUB_WORKSPACE` to their effective global Git safe.directory configuration.
Package metadata commands can then inspect the mounted checkout even after the
checkout action restores its temporary HOME. A real-Git regression first reproduces
the ownership refusal, checks the workspace command succeeds and verifies that
an unrelated repository remains untrusted. Wildcard trust is never used.

Trusted workflow execution owns variable and registry credential availability.
The configured immutable image and registry host remain required, with no recorded-image
fallback. Authorization, head checkout, image allowlist, ARC defaults and the full
portable/operator/Graph gate stay under the release workflow. CPU/source checks
do not qualify the prepared image, requested gfx836 hardware or model accuracy.

## Explicit PR dispatch on v0.33.0

Current GitHub.com resolves `pull_request_target` from the default branch.
Here that branch is `v0.28.1-dev`, where this workflow is absent; merely landing
the workflow on v0.33.0 does not establish automatic PR runs. No default-branch
setting or older release workflow is changed by this route.

Dispatch on **v0.33.0** with both `pr_number` and the full current
`expected_head` SHA, or leave both empty to test the target branch:

```bash
PR_HEAD=$(gh pr view 421 --repo HYGON-AI/vllm-plugin-das --json headRefOid --jq .headRefOid)
gh workflow run hcu-pr-ci-v033.yml --repo HYGON-AI/vllm-plugin-das \
  --ref v0.33.0 -f pr_number=421 -f expected_head="$PR_HEAD"
```

The trusted release resolver runs before submitted code. It requires an open
PR against this repository's v0.33.0, compares its live head with the supplied
SHA, and checks the dispatch actor's current permission using the existing
trusted-permission set. The explicit dispatch authorizes that exact head;
a persistent `ready-hcu` label grants no manual authority. A malformed pair,
stale head, wrong repository/release, closed PR or untrusted actor fails before
checkout of submitted code. Automatic PR events retain their separate label
policy. API/permission errors fail closed.

All four submitted-code jobs check out the resolved SHA, with read-only
permissions and no persisted checkout credentials. Manual PR selection runs
the full declared matrix. Concurrency includes the PR number so independent
manual PRs do not cancel one another through a shared release-ref group.
The existing image, device, no-skip/xfail and aggregate requirements remain.

A 204 dispatch response means accepted; observe that run before retrying.
Record the trusted workflow revision, resolver output, actual checkout and
installed payload in its logs. GitHub associates a manual run with its workflow
ref, so a base-SHA check context alone does not qualify the submitted PR. This
route supplies explicit PR evidence; it does not create automatic PR statuses,
repair prepared dependencies or establish model/release acceptance.
