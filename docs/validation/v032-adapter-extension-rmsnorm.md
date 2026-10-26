# vLLM 0.32 DAS extension: LightOp RMSNorm validation

Date: 2026-10-26 (UTC)

## Scope and provenance

This record covers `hcu-vllm-das-plugin` `0.2.0.dev0`, which adds ordinary and
fused-add LightOp `RMSNorm` to the previously validated `SiluAndMul` capability.
Both capabilities are loaded through `hcu-vllm-adapter`; the plugin owns no
vLLM platform or general-plugin entry point.

| Component | Version or revision |
| --- | --- |
| Official vLLM source contract | `v0.32.0`, commit `5df63a7203cdb9328d824a27b682e77c7b9928be` |
| Installed vLLM wheel | `0.32.0+rocm723` |
| Adapter source | `45525d0056e788fd189172d53756c4b53a151e8a` |
| Adapter wheel | `0.1.0.dev0` |
| DAS reference baseline | `v0.28.1-dev`, commit `cd412a0d53b9b517fbc3dcdf96121dc3daa1096c` |
| DAS extension wheel | `0.2.0.dev0` |
| Device | HCU `gfx938` (device 0 selected for focused tests) |

The tested wheel files and SHA256 values are:

```text
0c9e3512dccbe5a4018218c710fb46608b4b6776cb5ef21b52d7f73e0cdfcd5d  hcu_vllm_adapter-0.1.0.dev0-py3-none-any.whl
5510cc193997579455b61d6d6e841bbfa6d91ff2df281678dd8fa820d3315dc7  hcu_vllm_das_plugin-0.2.0.dev0-py3-none-any.whl
da7bda7ce51871b064f957076cfae43b40fee7226711432e88dcf2a5c0c91640  vllm-0.32.0+rocm723-cp312-cp312-linux_x86_64.whl
```

Artifacts are staged under
`/models/artifacts/vllm032-das-rmsnorm-RejxtR`; final wheels are in its
`wheels-final` directory. This path identifies the validation host's evidence
directory; the hashes, not the path, identify the tested wheels.

## Installed-state attestation

The wheels were installed with `--no-deps` into a fresh virtual environment
created with `--system-site-packages` to reuse the prepared HCU runtime. Python
was invoked with `-E` so the host's older system vLLM could not precede the
environment's wheel.

```bash
VLLM_PLUGINS=hcu_adapter \
HSA_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 \
/models/artifacts/vllm032-das-rmsnorm-RejxtR/venv-final/bin/python -E \
  adapter_extension/tools/verify_installed.py
```

Result: exit code 0. Schema 2 of the attestation selected vLLM
`0.32.0+rocm723`, adapter `0.1.0.dev0`, and plugin `0.2.0.dev0`. Extension
`das_hcu` was active with capabilities `lightop-rms-norm` and
`lightop-silu-and-mul`. The exact OOT owners were:

```text
RMSNorm:   hcu_vllm_das.rms_norm.HcuRMSNorm
SiluAndMul: hcu_vllm_das.silu_and_mul.HcuSiluAndMul
```

The installed `RECORD` content hashes reported by the verifier were:

```text
vllm:                f4f09c43f5826ad3f47f49e1b3d235e31878e542cbe0fa5a5e32387dc8e574f7
hcu-vllm-adapter:    ab7c62eb0254e19c0601d58495dfac10f192682b3593705e9d5f8b94a6c1367e
hcu-vllm-das-plugin: cf466c8a54b1a61d9abc3358c0edd0eb507078b313a92457fc1b731a8d7194f0
```

## Live HCU numerical checks

```bash
HCU_VLLM032_RMSNORM_LIVE=1 VLLM_PLUGINS=hcu_adapter \
HSA_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 \
/models/artifacts/vllm032-das-rmsnorm-RejxtR/venv-final/bin/python -E -m pytest \
  tests/vllm032/test_rms_norm_hcu.py -q -rs
```

Result: `1 passed` on `gfx938`. For BF16 and FP16, the test checked ordinary
RMSNorm and fused-add RMSNorm against independently calculated float32
references. It also checked both rollback switches and the narrow-width,
weightless, and variance-size-override fallback cases. The optimized numerical
tolerance was `rtol=4e-2, atol=4e-2`; the fused residual tolerance was
`rtol=2e-2, atol=2e-2`.

The previously validated `SiluAndMul` live test was rerun against this exact
wheel and also completed with `1 passed`.

## Release gate and retained reference failures

The installed-wheel vLLM 0.32 suite completed with `59 passed, 2 skipped`.
Those two skips are the opt-in RMSNorm and SiluAndMul live tests documented
above; each passed separately against the final candidate wheel. The installed
attestation also exited 0.

The retained repository-root vLLM 0.28 suite is not a gate for this vLLM 0.32
wheel. Its previously recorded failures remain unchanged: the legacy root
native module `vllm_hcu.hcu_ops` was not built, and a later gfx938 FP8 indexer
test aborted the process. Those known aborting legacy kernels were not rerun
for this revision and are not represented as passing.

## Deployment boundary

RMSNorm optimization is eligible only for BF16 or FP16 contiguous, same-dtype
inputs, weights, and optional residuals; equal input/residual shapes; width
equal to the configured hidden size and at least 64; a real weight; no
variance-size override; and non-batch-invariant mode. Every unsupported case
routes to the vLLM native implementation.

This record does not validate `GemmaRMSNorm`, `RMSNormGated`, quantized fusion,
model accuracy, serving, concurrency, soak behavior, performance, attention,
MoE, or `gfx92a`. On `gfx92a`, production deployments must set
`VLLM_HCU_USE_CUSTOM_RMS_NORM=0` until a separate live-device record exists.
