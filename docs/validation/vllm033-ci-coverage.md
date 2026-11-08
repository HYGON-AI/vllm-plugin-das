# v0.33 CI coverage candidate

A change to `adapter_extension/src/hcu_vllm_das/environment.py` or
`attention/e5m2_scales.py` previously selected no HCU jobs and did not trigger
fallback. The legacy workflows only target `v0.28.1-dev`.

`hcu-pr-ci-v033.yml` handles PRs and pushes to `v0.33.0` and dispatches on that
release. It invokes the existing selector CLI with `--release v0.33.0` and the
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
helper from the PR's trusted base commit on a hosted CPU runner. A trusted
author is authorized automatically; other authors need `ready-hcu` applied
by a trusted actor. Label additions/removals trigger a new check. CPU source
contracts run independently. Release pushes/manual runs must target
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
`${HARBOR_SITE}/jenkins/model_test_env/vllm@sha256:<64 lowercase hex digits>`
for a prepared DTK image with compatible vLLM 0.33, Adapter, vendor libraries,
pytest, packaging and build tooling. Missing/whitespace values, mutable tags,
malformed digests and other repositories fail before the hardware job starts;
there is no legacy-image fallback or automatic pull to resolve a tag.
The job container uses the existing `HARBOR_USERNAME`/`HARBOR_PASSWORD` secrets
when the event permits them, for the initial private-image pull. Fork PRs
normally cannot read repository secrets: a private-image pull therefore still
requires reviewed pre-existing runner registry authentication or credentials
available to that event. The credentials declaration does not prove pull success.
Missing authorization or pull access remains a failure; do not switch to
`pull_request_target`, expose secrets to PR code, or let a PR head authorize itself.
The container mounts `/opt/hyhal` read-only and uses the existing
`scripts/ci/hcu/hcu_ci_start_container.sh` default 32GiB shared-memory policy
without privileged mode.
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
