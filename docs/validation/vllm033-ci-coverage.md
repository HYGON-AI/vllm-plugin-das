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

The source job runs existing official API AST and package-boundary tests using
the immutable official target `4ba787ec9959e566d3b587a796dff07673324da5`.
It does not certify the whole portable suite, native ABI, or model quality.
The prepared-image job runs all non-`*_hcu.py` files in `tests/vllm033` with
devices hidden, then all `*_hcu.py` files with the actual live flags. It
installs only the independent `adapter_extension` project, never root
`setup.py`, and explicitly selects the Adapter platform entry point in every
tier so an image's legacy platform entry point cannot be discovered. The
workflow isolates parent Python discovery and clears inherited `PYTHONPATH`
and user site packages from child environments. Empty
results, failures, errors, skips/xfails and unexpected passes
fail the result gate. JUnit reports preserve each tier separately. The shared
HCU registration assertion recognizes only files actually executed by this
release runner; all other HCU files still require the legacy registry.

Set `HCU_CI_IMAGE_V033` to a prepared DTK image with compatible vLLM 0.33,
Adapter, vendor libraries, pytest, packaging and build tooling. The workflow
fails explicitly when it is absent. A configured image name is not evidence
that the image or hardware tier passed. The runner must provide gfx936 devices
and the normal prepared DTK environment. This candidate does not create or
validate such an image, allocate runner resources, or certify models and
other HCU architectures. Image/wheel provenance, model quality, performance
and the remaining release matrix remain separate acceptance gates.

CPU verification commands:

```bash
python -B -m unittest discover -s tests/ci -v
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 HCU_ADAPTER_HOST=none \
  HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 CUDA_VISIBLE_DEVICES=-1 \
  python -B -m pytest -q tests/ci tests/patch/test_hcu_ci_selector.py
VLLM_TARGET_ROOT=/path/to/official-v033 \
  python -B tools/run_v033_ci.py --tier source --junit v033-source.xml
```

The upgrade audit preserves the original zero-job probe and the first full
regression failure: the old global registration assertion did not recognize
three existing v0.33 HCU-marked files. The candidate accounts for them via the
actual release runner rather than weakening registration for other files.
No GPU job, remote CI, image publication, push or PR is claimed by these local
CPU results.
