# Step5 runtime integration

Step5 uses the existing vLLM-HCU exact-import patch coordinator. Installation
and runtime do not modify the installed vLLM package.

## Ownership

- `vllm_hcu/models/step5_text.py`, `step5_vl.py`, `step5_mtp.py`: model implementations.
- `vllm_hcu/models/step5_config.py`: checkpoint configuration classes.
- `vllm_hcu/model_executor/layers/step4_dsa.py` and `step4_dsa_kernels.py`: DSA backend and kernels.
- `vllm_hcu/models/step5_swa_fused.py` and `vllm_hcu/ops/step5`: SWA, router, cached Triton driver launcher and FP8 kernels.
- `vllm_hcu/v1/step5_*.py`: Step5 metadata, execution and multi-module MTP implementations.
- `vllm_hcu/distributed/eplb*_runtime.py`: mixed-precision expert movement.

`patch/module_exchange.py` lists eight exact canonical Step5 aliases. They are
armed atomically with the existing cold replacements before importing model
or custom-op modules. Canonical and plugin imports resolve to the same objects.

## Registration order

Platform callbacks first install the compatible DeepGEMM exports before cold
MoE imports bind those functions during configuration deserialization. Later
Worker registration reuses the same idempotent adapters. Platform callbacks
then register configuration exports, model types, MTP overrides, compile
splitting operations, compile prefix hashing and EP graph capture.
EP capture is registered before consumers copy the public distributed export.
The Worker dispatcher registers metadata, block-table mirrors, MTP loading and
sampling, DP synchronization, MoE combine, token FP8, logits and EPLB callbacks.

MoE combine hook state is owned by the plugin and can be registered before
Worker dispatch. Step5 execution and EPLB helpers resolve the canonical or owned
implementation rather than retaining exports copied before callback dispatch.

Substantial implementations remain in owned modules. Small adapters validate
exact target names and signatures, verify reviewed source fingerprints where
shared methods are replaced, and record target identities for idempotency.
Incompatible or stale targets fail startup instead of silently losing behavior.
No source rewriting or runtime patch-file application is used.

Step5 runner and metadata paths are selected by model type. Other model families
retain their native execution and MTP paths. Existing HCU feature sidecars and
operator switches remain the configuration boundary; upstream config and
attention metadata dataclasses are not extended in place.

Quantization tools, server launchers, performance clients and evaluation run
artifacts remain outside the source package. The package contains runtime
implementations, compatibility adapters, their tests and integration guidance.

## Validation

Use the supported vLLM 0.28.1 base and the plugin on `PYTHONPATH`:

```bash
python -m pytest -q tests/runtime_patch/test_step5_runtime.py
python tools/check_production_boundary.py
python tools/check_patch_test_coverage.py
```

The focused runtime tests cover adapter contracts and registration.

For the full `tests/patch` suite, set `VLLM_V0251_SOURCE_ROOT` to the directory
containing the reviewed vLLM 0.28.1 package. This is the existing test harness's
legacy variable name; it does not require a separate vLLM 0.25.1 checkout.

The adapters accept the reviewed clean base and, where needed, the previous
source-patched API. Updating vLLM requires reviewing the compatibility contracts.
A clean-base import/configuration test verifies independence from historical
source patches; CPU tests do not establish GPU correctness or performance.
GPU startup and inference should be verified separately when resources allow.
