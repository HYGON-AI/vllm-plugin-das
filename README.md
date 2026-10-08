<h3 align="center">
vLLM HCU Plugin
</h3>

[English](README.md) | [简体中文](README.zh-CN.md)

---

`vllm-plugin-das` provides HCU platform, model, and operator integration for
[vLLM](https://github.com/vllm-project/vllm). Use the repository branch that
matches your HCU/OpenDAS vLLM deployment.

## Version Compatibility

Package versions and integration revisions are maintained in
[vllm_hcu/version.py](vllm_hcu/version.py), with wheel version construction in
[setup.py](setup.py). Consult those files on the branch you are installing;
this README intentionally does not duplicate their changing version numbers.

The [compatibility gate](vllm_hcu/compatibility.py) requires the installed vLLM
package to have the same PEP 440 epoch and release tuple as
`__vllm_target_version__`. Pre-release, development, post-release, and local
build suffixes may differ. Missing packages, malformed versions, and different
release lines are rejected before patch registration.

The recorded revisions identify the integration baseline, not an exact wheel
pin. Passing the version gate alone does not validate the PyTorch/DTK/operator
ABI or guarantee compatibility with an arbitrary upstream vLLM wheel. Use a
matching HCU/OpenDAS runtime stack.

## Upstream, License, and Third-Party Notices

This repository is licensed under the Apache License 2.0. See [LICENSE](LICENSE)
and [NOTICE](NOTICE). Some files are adapted from vLLM and other Apache-2.0
compatible third-party sources; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for source and modification
notices. Modified by Hygon Information Technology Co., Ltd.

Historical source attributions in those notices are not the runtime dependency
requirement for this branch; use the version contract above.

## Install

Build and run on a Linux HCU system with the matching DTK toolchain, PyTorch,
and HCU vLLM base package already installed. Use a Python version supported by
[setup.py](setup.py), which imports PyTorch's extension build utilities and
does not provision runtime dependencies for you.

[docker/Dockerfile](docker/Dockerfile) describes the container build flow.
Operator packages such as AITER, FlashMLA, DeepGEMM, LightOp, and DeepEP
must match the chosen model/backend and runtime ABI; a generic PyPI stack is
not a substitute for those HCU builds.

The [CI environment lock](.github/workflows/configs/hcu-runner-environment.json)
checks DTK using `match: release_line` against its configured major/minor
release. Numeric patch releases and `-`/`+` build suffixes within that line are
accepted; other release lines and malformed versions are rejected. Preflight
reports the full installed version.
Custom locks without `rocm.match` retain exact matching. Other dependency
checks remain independent, and this CI policy does not prove binary ABI
compatibility for every build in the series.

From the root of the selected branch, build and install the plugin in that
prepared environment:

```bash
python3 setup.py bdist_wheel
python3 -m pip install --no-deps dist/vllm_hcu-*.whl
```

Use a fresh checkout/dist directory, or select one specific wheel if multiple
builds are present. `--no-deps` preserves the prepared runtime; it does not
install or verify missing dependencies.

### Optional Build Configuration

These are environment preparation examples, not commands required on every
installation. Install build tools only if your environment is missing them;
set the DTK path when an explicit location is needed, and limit parallel
compilation according to available resources:

```bash
# Only if the build tools are not already provided.
python3 -m pip install ninja wheel setuptools
# Only if the DTK path needs to be set explicitly; adjust to your installation.
export ROCM_PATH=/opt/dtk
# Optional build parallelism limit; this value is an example.
export MAX_JOBS=16
```

`MAX_JOBS` defaults to the host CPU count. `ROCM_PATH` is also used to discover
DTK metadata for the wheel version when it is set.

`ADD_GIT_VERSION=1` is the default and includes the detected Git revision in
the wheel's local version. Set `ADD_GIT_VERSION=0` to omit that revision; the
`+das` suffix and any detected DTK metadata remain. `MAX_JOBS` defaults to the
host CPU count when unset.

No post-install source-patching step is required. Installation and plugin startup
do not rewrite files in the vLLM package and do not create source-tree symlinks.
`setup.py` computes the wheel version from read-only Git/environment metadata;
it no longer rewrites the tracked `vllm_hcu/version.py` or changes Git's global
`safe.directory` configuration. At runtime, `vllm_hcu.version` obtains the full
installed build version from distribution metadata and falls back to the source
release series only when no distribution is installed. The extension build can
still copy its compiled `.so` into this checkout, so use a clean checkout when
build-artifact changes matter.
The historical `vllm-hcu-apply-patches` command remains a read-only compatibility
check; new automation should use:

```bash
vllm-hcu-doctor
```

For metadata/source diagnostics without arming platform patches, use
`vllm-hcu-doctor --no-arm --json`. The default doctor also checks process-local
platform patch activation; neither mode proves model accuracy or GPU execution.

## Runtime Settings

This branch's [HCU worker](vllm_hcu/v1/worker.py) constructs
`HcuGPUModelRunnerV2` and rejects the legacy runner. Use Model Runner V2 when
launching a server, as the model validation profiles do:

```bash
export VLLM_USE_V2_MODEL_RUNNER=1
```

The following settings are not interchangeable:

| Setting | Current behavior |
| --- | --- |
| `VLLM_USE_V2_MODEL_RUNNER` | Must resolve to V2; the HCU worker has no legacy-runner fallback. |
| `VLLM_HCU_USE_CUSTOM_OPS` | Enabled by default; `0` disables optional HCU optimized paths governed by the master switch, not the entire plugin or all native dependencies. |

See [environment settings](vllm_hcu/platforms/envs.py) for HCU-specific
switches. Select attention/MoE backends, KV-cache format, speculative decoding,
and parallel topology from a matching model profile; these options are not
validated in every combination.

### MHA/GQA With FlashAttention

For MHA/GQA models using `--attention-backend FLASH_ATTN`, set
`VLLM_KV_CACHE_LAYOUT=HND` when all of the model's selected backends support
that layout, as in the matching [model profiles](tests/models/):

```bash
VLLM_KV_CACHE_LAYOUT=HND vllm serve /path/to/model \
  --attention-backend FLASH_ATTN
```

This is a recommended layout for the supported FlashAttention routes, not a
requirement for every model or backend. Hybrid models can select additional
attention backends that do not support HND even with `FLASH_ATTN` specified.
In those cases, leave `VLLM_KV_CACHE_LAYOUT` unset and let vLLM resolve a common
supported layout. For MLA and other backends, follow the model profile rather
than applying HND globally; see the [validation records](docs/validation/) for
model-specific constraints.

## Runtime Integration

vLLM discovers the HCU platform, model registry, and operator registry through
standard plugin entry points. The plugin installs exact, process-local import
callbacks and applies patches in two explicit phases:

- `apply_platform_patches()` prepares platform fixes and framework integration.
- `apply_worker_patches(vllm_config)` prepares Worker-only model, operator, and
  communication integration.

HCU-only feature settings are stored in
`vllm_config.additional_config["hcu"]`; vLLM configuration classes are not
modified. `patch_report()` reports the process role, target symbols, patch
status, failure details, and feature activation state.

All three plugin entry points and both patch-application phases share one
fail-closed compatibility gate using the source-defined release-line contract.
Incompatible installations are rejected before patch registration. The
corresponding doctor check is named `vllm_compatible`.

For `FLASHMLA_SPARSE`, the platform maps public `--kv-cache-dtype fp8_e4m3`
to the internal `fp8_ds_mla` format. This packed sparse-MLA layout is
backend-specific, not a generic cache format shared by all attention backends.

## Models and Validation

Model registration or the presence of a YAML profile is not an accuracy pass.
Use the exact checkpoint, quantization, hardware, backend, and topology in the
profile and compare against the recorded evidence:

- [Model configurations](tests/models/): checkpoint-specific TP/DP/EP, MTP,
  attention/MoE backend, and KV-cache settings.
- [Model configuration guide](tests/models/README.md): profile selection and
  additional Kimi, DeepSeek-V4 DSpark, GLM PCP, and multimodal test routes.
- [Validation records](docs/validation/): frozen environments, hardware,
  per-profile scores, diagnostic failures, and validation limits.

Recorded results do not establish accuracy for another quantization, parallel
topology, graph policy, or a newer build.
Diagnostic profiles must not be counted as accepted accuracy runs, and the
recorded results are not a fresh test of every subsequent branch commit.

HumanEval executes model-generated code. Run it in a properly isolated
evaluation environment without credentials or access to sensitive host data;
follow the isolation requirements in the validation record. An environment
flag by itself does not create a sandbox.

## Architecture

```text
vllm_hcu/
├── __init__.py                  # three vLLM plugin entry points
├── compatibility.py             # shared source-defined compatibility gate
├── doctor.py                    # read-only installation diagnostics
├── patch/
│   ├── __init__.py              # public platform/Worker patch lifecycle API
│   ├── import_coordinator.py    # exact lazy callbacks and module replacement
│   ├── module_exchange.py       # canonical vLLM -> HCU module inventory
│   ├── runtime_state.py         # process role, idempotence, failure latch, report
│   ├── runtime_callbacks.py     # small symbol/method compatibility callbacks
│   ├── tokenizer_callbacks.py   # tokenizer compatibility callbacks
│   ├── config.py                # HcuFeatureConfig sidecar contract
│   ├── platform/
│   │   ├── __init__.py          # ordered process-wide dispatcher
│   │   ├── core_fix/            # config, env, parser and registry adapters
│   │   └── framework_opt/       # executor and KV/PD integration
│   └── worker/
│       ├── __init__.py          # ordered Worker dispatcher and feature gates
│       ├── core_fix/            # model-specific compatibility adapters
│       ├── op_opt/              # attention, quantization, GEMM and MoE adapters
│       └── framework_opt/       # collectives, DBO, MTP and context adapters
├── runtime_compat/              # small HCU-owned replacement implementations
│   ├── base_linear_parameter.py
│   ├── scaled_mm.py
│   └── weight_loading.py
├── model_executor/
│   └── layers/
│       ├── linear.py            # HCU linear/custom-op implementation
│       ├── fused_moe/           # MoE, DeepEP and DeepGEMM runtime
│       └── quantization/        # compressed-tensor and SlimQuant runtime
├── models/                      # DeepSeek, HY and GLM model implementations
├── ops/                         # HCU custom operators and fallbacks
├── platforms/
│   ├── hcu.py                   # public HCUPlatform implementation
│   └── envs.py                  # HCU environment settings
└── v1/
    ├── worker.py                # Worker patch boundary and device lifecycle
    ├── hcu_model_runner_v2.py    # active Model Runner V2 implementation
    ├── hcu_model_runner.py       # retained legacy code, not a Worker fallback
    ├── attention/               # HCU attention backends, metadata and ops
    ├── executor/                # multiprocessing executor implementation
    └── spec_decode/             # speculative decoding runtime
```

The `patch/` tree owns registration, ordering, target validation, and small
adapters only. Scheduling remains owned by upstream vLLM. Substantial Mooncake,
attention, MoE, communicator, and executor behavior stays in the corresponding
HCU-owned implementation module. Consumers import canonical `vllm.*` module
names when an entry exists in `module_exchange.py`; the import coordinator
resolves those names to the HCU implementation before the upstream module is
loaded.

Platform patches are armed during plugin discovery. Worker patches are bound to
the deserialized `vllm_config` in `HcuGPUWorker.__init__`, before the parent
Worker imports model runners and custom operators. Required incompatibilities
are fail-closed and retained in the process-local patch registry; they do not
fall back to source rewriting.

## Development Guardrails

Run the source-boundary and patch-coverage audits from the repository root:

```bash
python3 tools/check_production_boundary.py --json
python3 tools/check_patch_test_coverage.py --json
```

With the test dependencies and matching HCU vLLM source available, run the
portable contract suite:

```bash
python3 -m pip install -r requirements-test.txt
python3 tools/run_patch_tests.py --suite contract --vllm-source /path/to/vllm-root
```

`/path/to/vllm-root` must contain the matching `vllm/` package. See
[tests/README.md](tests/README.md) for suite selection. Portable contract checks
do not replace HCU kernel, distributed, or model accuracy tests.

`tools/check_production_boundary.py` verifies that migration-only metadata,
versioned private markers, and version-specific runtime module names do not
enter `vllm_hcu/`. Internal runtime markers use the version-neutral
`_vllm_hcu_*` prefix. Restart all Python/vLLM processes after installing a new
wheel so stale module identities and custom-op registrations cannot survive an
upgrade.

---
