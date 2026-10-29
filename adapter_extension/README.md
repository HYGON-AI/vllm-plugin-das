# HCU vLLM DAS optimization plugin

`hcu-vllm-das-plugin` is the optimization-only extension for official vLLM
0.32 and `adapter-das`. The adapter owns the HCU platform and runtime
lifecycle; this wheel contributes optional, independently removable operator
optimizations through the adapter extension API.

Version `0.2.0.dev0` contains one retained, default-OFF `gfx936` candidate:

- LightOp ordinary and fused-add `RMSNorm`.

vLLM's native `SiluAndMul` remains the selected implementation. Warmed
compiled A/B measurements on `gfx936` did not justify replacing the upstream
operator, so the plugin neither registers nor ships a `SiluAndMul` override.

No attention, MoE, model, topology, or performance support is implied by these
operator-level capabilities.

## Compatibility

- vLLM release line: `>=0.32.0,<0.33.0`; prerelease, postrelease,
  development, patch, and local version suffixes are accepted when the
  required operator APIs and LightOp capabilities are present.
- Adapter distribution: `sgl-kernel>=0.1.0`, extension API v1 from
  `hcu_adapter.extensions`.
- Runtime: a prepared HCU PyTorch/DTK/LightOp stack. LightOp is intentionally
  not fetched by pip because production deployments must use an ABI-matched
  local build.
- The plugin does not own `vllm.platform_plugins` or `vllm.general_plugins`.

The repository root remains the vLLM 0.28.1 development reference. Build only
this directory for vLLM 0.32:

```bash
python3 -m pip wheel --no-deps --wheel-dir ./dist ./adapter_extension
```

## Production installation

Start with a fresh virtual environment or image containing the matching HCU
PyTorch, DTK, and LightOp builds. Install audited local wheels without dependency
resolution, in this order:

```bash
python3 -m pip install --no-deps /wheels/vllm-0.32.0+rocm*.whl
python3 -m pip install --no-deps /wheels/sgl_kernel-*.whl
python3 -m pip install --no-deps /wheels/hcu_vllm_das_plugin-*.whl
```

If the shell exports `PYTHONPATH`, clear it or invoke Python with `-E` during
verification so an older system vLLM cannot precede the environment's wheel:

```bash
VLLM_PLUGINS=hcu_adapter python3 -E adapter_extension/tools/verify_installed.py
```

The verifier fails closed unless all three distributions have non-empty wheel
records, their imported modules are owned by those records, the adapter is the
sole expected vLLM entry-point owner, the DAS extension is active with its
declared capability, and vLLM's OOT registry selects this exact class:

```text
hcu_vllm_das.rms_norm.HcuRMSNorm
```

## RMSNorm eligibility

The LightOp RMSNorm path is selected only when all of the following hold:

- the master and RMSNorm-specific switches are enabled;
- vLLM batch-invariant mode is disabled;
- the operation uses a real weight and no variance-size override;
- the input dtype is BF16 or FP16;
- input width equals the layer hidden size and is at least 64;
- input, weight, and optional residual are contiguous;
- input, weight, and optional residual use the same dtype; and
- an optional residual has the same shape as the input.

All other cases use vLLM's native RMSNorm implementation. `GemmaRMSNorm`,
`RMSNormGated`, quantized fusion, model patches, and performance claims are out
of scope for this release.

## Opt in and rollback

The optimized path defaults to OFF. Enable it explicitly; the master switch
remains the highest-priority rollback and returns RMSNorm to vLLM's complete
implementation without uninstalling the plugin:

```bash
# Explicitly enable the validated RMSNorm path.
export VLLM_HCU_USE_CUSTOM_OPS=1
export VLLM_HCU_USE_CUSTOM_RMS_NORM=1

# Roll back all optional DAS custom operators.
export VLLM_HCU_USE_CUSTOM_OPS=0

# Or roll back only the LightOp RMSNorm path.
export VLLM_HCU_USE_CUSTOM_RMS_NORM=0
```

The numerical evidence for this release is limited to `gfx936`. On another HCU
architecture, keep RMSNorm disabled until a separate live-device validation
record exists:

```bash
export VLLM_HCU_USE_CUSTOM_RMS_NORM=0
```

For complete removal, uninstall `hcu-vllm-das-plugin`; the adapter and vLLM
remain installed and retain platform ownership.

The [`SiluAndMul`](../docs/validation/v032-adapter-extension-silu.md) and
[`RMSNorm`](../docs/validation/v032-adapter-extension-rmsnorm.md) records are
historical candidate evidence. The current three-way identities, upstream
takeover decision, and pending final-wheel gates are recorded in the
[`vLLM 0.32 upstream-first validation`](../docs/validation/vllm032-upstream-first-operator-ab.md).
