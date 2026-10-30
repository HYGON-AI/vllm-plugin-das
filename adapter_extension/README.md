# HCU vLLM DAS optimization plugin

`hcu-vllm-das-plugin` is the optimization-only extension for official vLLM
0.32 and `adapter-das`. The adapter owns the HCU platform and runtime
lifecycle; this wheel contributes optional, independently removable operator
optimizations through the adapter extension API.

Version `0.2.0.dev0` contains two independently selected capabilities:

- a default-OFF LightOp ordinary and fused-add `RMSNorm` candidate; and
- an explicit AITER INT8 W8A8 MoE route selected by `--moe-backend aiter`.

It also provides an explicit HCU FlashAttention backend selected by
`--attention-backend FLASH_ATTN`. The backend keeps vLLM's official metadata,
scheduler, allocation, cache writer, and forward orchestration. It changes only
the provider boundaries needed by the HCU vendor varlen ABI: dense HND K/V page
views, FA2 keyword compatibility, and a 64-token kernel page.

vLLM's native `SiluAndMul` remains the selected implementation. Warmed
compiled A/B measurements on `gfx936` did not justify replacing the upstream
operator, so the plugin neither registers nor ships a `SiluAndMul` override.

No model, topology, FP8-KV, or broad performance support is implied by these
operator-level capabilities.

## Compatibility

- vLLM release line: `>=0.32.0,<0.33.0`; prerelease, postrelease,
  development, patch, and local version suffixes are accepted when the
  required operator APIs and LightOp capabilities are present. FLASH_ATTN also
  requires the OpenDAS vLLM cache-view and kernel-call extension seams.
- Adapter distribution: `sgl-kernel>=0.1.0`, extension API v1 from
  `hcu_adapter.extensions`.
- Runtime: a prepared HCU PyTorch/DTK/LightOp/AITER stack. LightOp and AITER
  are intentionally not fetched by pip because production deployments must
  use ABI-matched local builds.
- The plugin does not own `vllm.platform_plugins` or `vllm.general_plugins`.

The repository root remains the vLLM 0.28.1 development reference. Build only
this directory for vLLM 0.32:

```bash
python3 -m pip wheel --no-deps --wheel-dir ./dist ./adapter_extension
```

## Production installation

Start with a fresh virtual environment or image containing the matching HCU
PyTorch, DTK, LightOp, and AITER builds. Install audited local wheels without
dependency resolution, in this order:

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
sole expected vLLM entry-point owner, the DAS extension is active with both
declared capabilities, vLLM's OOT registry selects this exact class, and
explicit `aiter` INT8 selection resolves to the Plugin-owned expert:

```text
hcu_vllm_das.rms_norm.HcuRMSNorm
hcu_vllm_das.moe.int8_oracle.HcuAiterInt8Experts
```

## AITER INT8 W8A8 MoE

The Plugin reuses vLLM 0.32's official channel-weight/token-activation W8A8
quant configuration and kernel construction. It adds only the explicit AITER
selector and translates the resulting tensors to the public `aiter.moe` API;
the vendor wheel continues to own kernels and tuned configurations.

Select the route through vLLM's existing CLI option:

```bash
vllm serve /model/Qwen3.5-35B-A3B-W8A8 --moe-backend aiter
```

Each unseen shape logs whether it found and executed a native AITER config. A
shape with no compatible config, or a config requiring an unsupported shuffle
layout, is logged as an official vLLM Triton INT8 fallback and is not counted
as an AITER hit. Once a compatible AITER config is selected, a vendor execution
error is raised rather than silently changing providers.

Leaving `--moe-backend` at its default preserves vLLM's upstream backend
priority. Uninstalling this wheel removes the added selector without changing
the Adapter's platform/runtime ownership.

## HCU FlashAttention varlen backend

The initial supported path is BF16/FP16 dense MHA/GQA with LBHNC (HND) cache
layout and 64-token pages. The public vendor ABI is
`flash_attn.flash_attn_varlen_func`; no classic, custom, or alternate public
attention entry point is registered.

Run with Graph enabled (the vLLM default):

```bash
vllm serve /model/Qwen2-0.5B-Instruct \
  --attention-backend FLASH_ATTN \
  --block-size 64
```

The Plugin reinterprets each allocated page as dense `[K/V, H, N, D]`
storage. This is required because the vendor kernel does not consume vLLM's
default non-dense views produced by splitting `[..., 2D]`. The official
stride-aware HIP cache writer writes through a zero-copy NHD alias of those
dense views. The Plugin does not copy the full KV cache on decode. The vendor's
regular prefill path may gather paged KV internally; rectangular short prefill
uses that path by passing a conservative `max_seqlen_q` upper bound, avoiding
the vendor wrapper's incompatible MTP tile selection.

FA4-only `dynamic_causal`, `mask_mod`, and `aux_tensors` are accepted only
when inactive (`None`). An active feature fails closed instead of silently
changing its mask. FP8 KV, DCP, hybrid caches, and layouts other than LBHNC
remain unvalidated and are not claimed by this release.

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
# Explicitly enable the retained RMSNorm candidate.
export VLLM_HCU_USE_CUSTOM_OPS=1
export VLLM_HCU_USE_CUSTOM_RMS_NORM=1

# Roll back all optional DAS custom operators.
export VLLM_HCU_USE_CUSTOM_OPS=0

# Or roll back only the LightOp RMSNorm path.
export VLLM_HCU_USE_CUSTOM_RMS_NORM=0

# Roll back FlashAttention without changing the Plugin installation.
vllm serve /model/Qwen2-0.5B-Instruct --attention-backend TRITON_ATTN
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
takeover decision, and base final-wheel gates are recorded in the
[`vLLM 0.32 upstream-first validation`](../docs/validation/vllm032-upstream-first-operator-ab.md).
The explicit AITER W8A8 route, public vendor-kernel evidence, Graph replay,
model smoke, and isolated HumanEval64 result are recorded in the
[`vLLM 0.32 AITER INT8 MoE validation`](../docs/validation/vllm032-aiter-int8-moe.md).
