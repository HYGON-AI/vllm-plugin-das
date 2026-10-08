<h3 align="center">
vLLM HCU Plugin
</h3>

---

`vllm-plugin-das` provides HCU platform, model, and operator integration for
[vLLM](https://github.com/vllm-project/vllm). This README describes the
`v0.28.1-dev` branch and its HCU/OpenDAS vLLM `0.28.1` runtime contract.

## Version Compatibility

The source of truth is [vllm_hcu/version.py](vllm_hcu/version.py), with wheel
version construction in [setup.py](setup.py).

| Item | Current source value |
| --- | --- |
| Plugin base version | `0.28.1rc1.dev491` |
| Recorded HCU vLLM target build | `0.28.1rc1.dev491+g462fdb097.das.462fdb0.dtk2604` |
| Recorded upstream vLLM revision | `58ad1f3b8973b23943107b51230d594050b42ec3` |
| Recorded OpenDAS vLLM revision | `462fdb097c66b487ef4826e8009431c10fe88fb8` |

The [compatibility gate](vllm_hcu/compatibility.py) requires the same PEP 440
epoch and release tuple as the target: `0.28.1`. Pre-release, development,
post-release, and local build suffixes are accepted, including `0.28.1`,
`0.28.1rc1.dev491+das.vendor`, and `0.28.1+dtk2604.torch2110`. This is **not** a
blanket `0.28.x` check: `0.25.1`, `0.28.0`, and `0.28.2` are rejected, as are
missing or malformed versions.

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
and HCU vLLM base package already installed. `setup.py` requires Python 3.10 or
newer and imports PyTorch's extension build utilities; it does not provision
the runtime dependencies for you.

The checked-in [gfx938 validation record](docs/validation/v0281-gfx938-provenance.md)
uses Python 3.10.12, DTK 26.04.1, a DTK 26.04 PyTorch 2.11.0 build, and an HCU
vLLM `0.28.1` wheel. It also records exact wheel versions for reproducing that
campaign. [docker/Dockerfile](docker/Dockerfile) describes the container build
flow. Operator packages such as AITER, FlashMLA, DeepGEMM, LightOp, and DeepEP
must match the chosen model/backend and runtime ABI; a generic PyPI stack is
not a substitute for those HCU builds.

The [CI environment lock](.github/workflows/configs/hcu-runner-environment.json)
checks DTK using `match: release_line` with `version: 26.04`. This accepts
`26.04`, numeric patch releases such as `26.04.1`, and `-`/`+` build suffixes
such as `26.04.1-72cu-0911`; it does not accept other release lines or arbitrary
strings starting with `26.04`. Preflight reports the full installed version.
Custom locks without `rocm.match` retain exact matching. Other dependency
checks remain independent, and this CI policy does not prove binary ABI
compatibility for every build in the series.

In that prepared environment, build and install the plugin:

```bash
git clone --branch v0.28.1-dev https://github.com/HYGON-AI/vllm-plugin-das.git
cd vllm-plugin-das
python3 -m pip install ninja wheel setuptools
export ROCM_PATH=/opt/dtk
export MAX_JOBS=16
python3 setup.py bdist_wheel
python3 -m pip install --no-deps dist/vllm_hcu-*.whl
```

Adjust `ROCM_PATH` to the DTK installation and `MAX_JOBS` to available build
resources. Use a fresh checkout/dist directory, or select one specific wheel
if multiple builds are present. `--no-deps` preserves the prepared runtime; it
does not install or verify missing dependencies.

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

Leave `VLLM_PLUGINS` unset for normal use so vLLM loads all three HCU entry
points.  If a plugin allow-list is required, include every HCU entry point:

```bash
export VLLM_PLUGINS=hcu,hcu_model,hcu_ops
```

Setting only `VLLM_PLUGINS=hcu` loads the platform plugin but excludes the HCU
model and operator general plugins.

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
| `VLLM_PLUGINS` | Normally unset; an explicit HCU allow-list needs `hcu,hcu_model,hcu_ops`. |
| `VLLM_USE_V2_MODEL_RUNNER` | Must resolve to V2; the HCU worker has no legacy-runner fallback. |
| `VLLM_HCU_USE_CUSTOM_OPS` | Enabled by default; `0` disables optional HCU optimized paths governed by the master switch, not the entire plugin or all native dependencies. |
| `VLLM_HCU_GLM53_GATE_UP_DEEPGEMM` | Explicit opt-in with `1` for GLM5Next shared-expert gate/up projection on gfx938; also requires the custom-op master and a matching DeepGEMM installation. Not enabled by default. |

See [environment settings](vllm_hcu/platforms/envs.py) and the
[GLM5Next adapter](vllm_hcu/patch/worker/core_fix/patch_glm5next_channel_fp8.py)
for the corresponding gates. Select attention/MoE backends, KV-cache format,
speculative decoding, and parallel topology from a matching model profile;
these options are not validated in every combination.

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
fail-closed compatibility gate using the `0.28.1` release-line contract above.
Incompatible installations are rejected before patch registration. The
corresponding doctor check is named `vllm_compatible`.

For `FLASHMLA_SPARSE`, the platform maps public `--kv-cache-dtype fp8_e4m3`
to the internal `fp8_ds_mla` format. This packed sparse-MLA layout is
backend-specific, not a generic cache format shared by all attention backends.

## Models and Validation

Model registration or the presence of a YAML profile is not an accuracy pass.
Use the exact checkpoint, quantization, hardware, backend, and topology in the
profile and compare against the recorded evidence:

- [gfx938 model matrix](tests/models/v0281_gfx938_humaneval16.yaml): profiles for
  DeepSeek, GLM (including GLM-5.3 Channel-FP8), HY, Qwen, and MiniMax, with
  profile-specific TP/DP/EP, MTP, and KV-cache settings.
- [Model configuration guide](tests/models/README.md): profile selection and
  additional Kimi, DeepSeek-V4 DSpark, GLM PCP, and multimodal test routes.
- [gfx938 validation provenance](docs/validation/v0281-gfx938-provenance.md):
  frozen wheels, hardware, per-profile scores, known diagnostic failures,
  and limits of the recorded validation campaign.

For example, `glm53_channel_fp8_tp8` specifies TP8, `FLASHMLA_SPARSE`, AITER
MoE, MTP3, and E4M3 KV cache. Its recorded HumanEval results do not establish
accuracy for another quantization, DP topology, graph policy, or a newer build.
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
├── compatibility.py             # shared vLLM 0.28.1 release-line gate
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

See [Runtime patch architecture](docs/runtime_patch_architecture_v0251.md) for
lifecycle, ownership, and module-replacement background. That document was
written for the earlier `v0.25.1` migration; its version-specific details are
historical, not the current branch's compatibility or validation contract.

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

`/path/to/vllm-root` must contain the matching `vllm/` package. Pass this
explicitly: the test runner retains legacy `VLLM_V0251_SOURCE_ROOT` and
`vllm_0251` fallback names; those names do not change the runtime version
requirement. See [tests/README.md](tests/README.md) for suite selection. Portable
contract checks do not replace HCU kernel, distributed, or model accuracy tests.

`tools/check_production_boundary.py` verifies that migration-only metadata,
versioned private markers, and version-specific runtime module names do not
enter `vllm_hcu/`. Internal runtime markers use the version-neutral
`_vllm_hcu_*` prefix. Restart all Python/vLLM processes after installing a new
wheel so stale module identities and custom-op registrations cannot survive an
upgrade.

---
