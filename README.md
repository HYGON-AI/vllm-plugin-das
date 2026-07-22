<h3 align="center">
vLLM HCU Plugin
</h3>

---

## Install

vLLM-HCU v0.25 is a runtime plugin for the DCU vLLM v0.25 base package. Install
the matching vLLM package first, then install this repository:

```bash
python3 setup.py install
```

To build a wheel instead:

```bash
python3 setup.py bdist_wheel
```

Set `ADD_GIT_VERSION=1` when the wheel version should include the Git revision.

No post-install source-patching step is required. Installation and plugin startup
do not rewrite files in the vLLM package and do not create source-tree symlinks.
`setup.py` computes the wheel version from read-only Git/environment metadata;
it no longer rewrites the tracked `vllm_hcu/version.py` or changes Git's global
`safe.directory` configuration. At runtime, `vllm_hcu.version` obtains the full
installed build version from distribution metadata and falls back to the source
release series only when no distribution is installed. The extension build can
still copy its compiled `.so` into this checkout, so use a clean checkout when
build-artifact changes matter.
The historical `vllm-hcu-apply-patches` command is retained for one release as a
read-only compatibility check; new automation should use:

```bash
vllm-hcu-doctor
```

Leave `VLLM_PLUGINS` unset for normal use so vLLM loads all three HCU entry
points.  If a plugin allow-list is required, include every HCU entry point:

```bash
export VLLM_PLUGINS=hcu,hcu_model,hcu_ops
```

Setting only `VLLM_PLUGINS=hcu` loads the platform plugin but excludes the HCU
model and operator general plugins.

## Runtime integration

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
fail-closed compatibility gate. This branch accepts installed vLLM `0.25.x`
(including local builds such as `0.25.0+das...`) and rejects missing, malformed,
or other release series before any patch is registered. The corresponding
doctor check is named `vllm_compatible`.

## Architecture

```text
vllm_hcu/
├── __init__.py                  # three vLLM plugin entry points
├── compatibility.py             # shared vLLM 0.25.x compatibility gate
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
│   │   └── framework_opt/       # scheduler, executor and KV/PD adapters
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
    ├── worker.py                 # Worker patch boundary and device lifecycle
    ├── hcu_model_runner.py       # model execution and draft-token channel
    ├── attention/                # HCU attention backends, metadata and ops
    ├── core/                     # KV-cache and Scheduler implementations
    ├── executor/                 # multiprocessing executor implementation
    └── spec_decode/              # speculative decoding runtime
```

The `patch/` tree owns registration, ordering, target validation, and small
adapters only. Substantial Scheduler, Mooncake, attention, MoE, communicator,
and executor behavior stays in the corresponding HCU-owned implementation
module. Consumers import canonical `vllm.*` module names when an entry exists in
`module_exchange.py`; the import coordinator resolves those names to the HCU
implementation before the upstream module is loaded.

Platform patches are armed during plugin discovery. Worker patches are bound to
the deserialized `vllm_config` in `HcuGPUWorker.__init__`, before the parent
Worker imports model runners and custom operators. Required incompatibilities
are fail-closed and retained in the process-local patch registry; they do not
fall back to source rewriting.

See [Runtime patch architecture](docs/runtime_patch_architecture_v025.md) for
the lifecycle, ownership rules, module-replacement contract, and extension
guidelines.

## Development guardrails

`tools/check_production_boundary.py` verifies that migration-only metadata,
versioned private markers, and version-specific runtime module names do not
enter `vllm_hcu/`. Internal runtime markers use the version-neutral
`_vllm_hcu_*` prefix. Restart all Python/vLLM processes after installing a new
wheel so stale module identities and custom-op registrations cannot survive an
upgrade.

---
