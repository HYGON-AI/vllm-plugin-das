# v0.28.1-dev CI Baseline Design

## Objective

Make `v0.28.1-dev` run its own pull-request, static, nightly, and image
validation gates without silently using the `v0.25.1` runtime contract. Repair
the current portable test baseline and make the Docker dependency install
repeatable enough that a transient public mirror metadata gap does not block
the release image.

All changes ship in one dedicated merge request based directly on
`v0.28.1-dev`. No model-runtime optimization or model-serving behavior is in
scope.

## Release isolation

The checked-out branch remains the source of truth for its environment lock.
On `v0.28.1-dev`, `.github/workflows/configs/hcu-runner-environment.json` must
describe the v0.28.1 release line on the DTK 26.04 release line. Patch, release
candidate, nightly, and local build suffixes within those lines remain valid;
other vLLM or DTK release lines fail preflight.

HCU workflows must select a version-scoped repository variable,
`HCU_CI_IMAGE_V0281`. The generic `HCU_CI_IMAGE` variable may remain available
to legacy workflows, but a v0.28.1 gate must never fall back to the hard-coded
`vllm:0.25.1-latest` image. A missing v0.28.1 image is a configuration error
and must fail closed with a useful diagnostic.

The reusable selected-test workflow continues to accept an immutable image as
an explicit input. Its image-resolution job pulls the configured tag and
passes a digest to hardware jobs, preserving the existing immutable execution
boundary.

## Workflow coverage

The v0.28.1 branch enables these gates for pull requests or pushes targeting
`v0.28.1-dev`:

- HCU PR static selection and authorized hardware execution;
- the shared quality gate;
- patch-coverage validation;
- nightly/manual HCU execution using a v0.28.1 default ref and image.

Workflow-contract tests exercise rendered workflow behavior: target branch,
selected version-scoped image, explicit environment-lock propagation, and
failure when the required image is absent. They do not merely grep for one
source line.

## Portable test baseline

The current branch has seven actionable failures in the portable patch suite.
They are compatibility-baseline drift introduced by recent v0.28.1 changes,
not seven independent runtime defects:

1. Register the Kimi K2.5 HumanEval test in the HCU registry with its existing
   model/resource contract.
2. Add a direct contract reference for the GLM-5.3 PCP KV-cache patch so the
   patch-coverage audit can prove ownership.
3. Update the explicit platform dispatcher inventory for the new GLM-5.3 PCP
   callback and its `vllm.v1.core.kv_cache_coordinator` target.
4. Update the corresponding registration and callback counts.
5. Characterize the v0.28.1 modules imported by the upstream platform path,
   while continuing to assert that HCU business adapters are lazy.
6. Characterize the additional v0.28.1 worker modules imported by upstream,
   with the same HCU-laziness guarantee.
7. Make the model-runner lifecycle test double reproduce the upstream
   constructor contract by retaining `vllm_config`.

The HCU bootstrap test that fails solely because a locally built `hcu_ops.so`
is unavailable remains an HCU-environment test. It is not weakened or counted
as a portable regression.

## Docker dependency reliability

The Docker image currently overrides the configured internal index with an
Aliyun-only Ray install. An unbounded Ray dependency selected pandas metadata
that the mirror advertised but could not serve, producing HTTP 404.

The image build will use the already configured internal devpi hierarchy as
the primary package source for Ray and its extras. Versions whose compatibility
matters to the branch will be constrained in a dedicated requirements file,
rather than resolving every transitive dependency from the newest public
metadata on every build. The Dockerfile will install from that file without an
Aliyun-only `--index-url` override.

A dependency-resolution check will validate the requirements independently of
the full DTK image build. The real image workflow remains the integration proof
because the private base image and DTK archive are runner-only resources.

## Error handling

- Missing `HCU_CI_IMAGE_V0281`: fail before starting hardware jobs.
- Environment-lock mismatch: retain the existing preflight error and report
  the expected and installed distributions.
- Unsupported workflow ref: fail rather than selecting a different release
  line.
- Dependency resolution failure: fail at the dedicated dependency layer with
  the unresolved package visible in the build log.

## Verification

Verification is layered:

1. New workflow and registry contract tests are observed failing before edits.
2. Targeted tests pass after each minimal fix.
3. `tools/check_patch_test_coverage.py`, HCU registration verification,
   environment-lock parsing, workflow YAML parsing, and Python compilation
   pass.
4. The portable patch suite is run in full; HCU-only failures are reported
   separately and never hidden.
5. Docker dependency resolution is tested locally where network policy allows,
   then the MR image workflow provides the authoritative private-runner result.

## Delivery

The work is committed on `fix/v0281-ci-baseline`, reviewed locally, pushed,
and opened as one MR targeting `v0.28.1-dev`. Branch-protection settings are
repository administration and are reported as a follow-up rather than mutated
by this code MR.
