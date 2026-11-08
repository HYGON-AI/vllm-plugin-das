# HCU vLLM DAS optimization plugin

Independent Flash public-registry publication and health contract:
[v0.33 provider ownership](../docs/validation/vllm033-upgrade.md#ownership-and-changes). Its
activation dependencies are independent of the optional RMS provider.

`hcu-vllm-das-plugin` is the optimization-only extension for official vLLM
0.33 and `adapter-das`. The adapter owns the HCU platform and runtime
lifecycle; this wheel contributes optional, independently removable operator
optimizations through the adapter extension API.

Version `0.3.0.dev0` targets the `v0.33.0` branch. Migration status and current
evidence are recorded in [v0.33 validation](../docs/validation/vllm033-upgrade.md);
the v0.32 documents below are historical evidence and do not certify v0.33.
The operator extension currently advertises only the default-OFF LightOp
ordinary and fused-add `RMSNorm` candidate. INT8 model selection, weight
conversion, quantization and kernel construction remain official vLLM owners.

It also provides an explicit HCU FlashAttention backend selected by
`--attention-backend FLASH_ATTN`. The backend keeps vLLM's official metadata,
scheduler and allocation. Its public `AttentionImpl` owns the HCU provider call
and dense HND/NHD cache views, routes BF16 writes to the native vLLM writer,
and retains a Plugin E5M2 writer and descale adapter. The kernel page is 64
tokens. It neither copies nor replaces the official Flash forward method and
does not require private HCU hooks in the Core.

vLLM's native `SiluAndMul` remains the selected implementation. Warmed
compiled A/B measurements on `gfx936` did not justify replacing the upstream
operator, so the plugin neither registers nor ships a `SiluAndMul` override.

No model, topology, FP8-KV, or broad performance support is implied by these
operator-level capabilities.

## Compatibility

- vLLM release line: `>=0.33.0,<0.34.0`; prerelease, postrelease,
  development, patch, and local version suffixes are accepted when the
  required operator APIs and LightOp capabilities are present. FLASH_ATTN uses
  the official public backend registry and AttentionImpl contract.
- Adapter distribution: `sgl-kernel>=0.1.0`, extension API v1 from
  `hcu_adapter.extensions`.
- Runtime: a prepared HCU PyTorch/DTK/LightOp/AITER stack. LightOp and AITER
  are intentionally not fetched by pip because production deployments must
  use ABI-matched local builds.
- The plugin does not own `vllm.platform_plugins` or `vllm.general_plugins`.

The repository root remains the vLLM 0.28.1 development reference. Build only
this directory for vLLM 0.33:

```bash
python3 -m pip wheel --no-deps --wheel-dir ./dist ./adapter_extension
```

## Production installation

Start with a fresh virtual environment or image containing the matching HCU
PyTorch, DTK, LightOp, and AITER builds. Install audited local wheels without
dependency resolution, in this order:

```bash
python3 -m pip install --no-deps /wheels/vllm-0.33.0+*.whl
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
sole expected vLLM entry-point owner, the RMS extension is active, and
vLLM's public OOT registry selects `hcu_vllm_das.rms_norm.HcuRMSNorm`.
Schema4 additionally verifies that the seven official INT8 oracle symbols
remain identical across activation, the module is owned by the Core wheel,
and explicit Triton selection resolves to an official expert.

## INT8 ownership and unavailable AITER model integration

Use official `--moe-backend auto` or `--moe-backend triton` for INT8 models.
The retired Plugin installer extended/rebound the official oracle enum and
functions; that private integration conflicts with the upstream-first skill
and has been removed. No `aiter-int8-moe` capability is advertised. Calling
its old `install()` entry point explicitly rejects before importing or
modifying Core. An Adapter request requiring that capability must fail.

The official v0.33 INT8 oracle provides Triton/Humming/CPU and has no public
expert registry for adding AITER. The general public quantization-config
registry is not equivalent: a future distinct quantization method would need
its own loader/lifecycle/full-dataflow evidence. It must not overwrite the
official `compressed-tensors` entry or rebind oracle globals.

The standalone INT8 provider, ASM composition and unregistered expert factory
are retained for operator experiments. Their prior native/Graph and full-model
results are frozen historical evidence; they do not make AITER selectable in
the corrected candidate. Preserving official ownership does not certify that
unavailable capability. See [ownership correction](../docs/validation/vllm033-official-ownership.md).

## Experimental SlimQuant checkpoint format

The isolated candidate adds the distinct `slimquant_w4a8` format through the
public quantization registry. Request capability `hcu-slimquant-w4a8` explicitly.
It requires the public per-row PLE method in the associated Core candidate;
activation fails before publication when that API is absent.

Packed expert bytes decode high nibble first to signed INT4 values. The decoded
INT8 weights and expert scales converted from the serialized high-nibble units
(times16) enter the official CT expert loader,
backend selection and kernels. Dense INT8 layers also use official CT methods.
Declared n-gram tables use the public per-row INT8 PLE method with FP32 scales.
This consumes more weight memory than a packed native W4A8 kernel. It neither
registers AITER as an official INT8 backend nor replaces `compressed-tensors`.
The current parser requires the three explicit mixed-precision groups used by
the Qwen3.8 checkpoint. Verify loader numerics, changed-input Graph and model
accuracy for the selected Core/provider/checkpoint combination before deployment.

## HCU FlashAttention varlen backend

The supported path is BF16/FP16 dense MHA/GQA with head size 64, 128, 192, or
256, LBHNC (HND) or LBNHC (NHD) cache layout, and 64-token pages. The public
vendor ABI is
`flash_attn.flash_attn_varlen_func`; no classic, custom, or alternate public
attention entry point is registered.

Run with Graph enabled (the vLLM default):

```bash
vllm serve /model/Qwen2-0.5B-Instruct \
  --attention-backend FLASH_ATTN \
  --block-size 64
```

The Plugin keeps LBHNC first in its preference list to preserve the existing
default. Select either layout through vLLM's native setting when a deployment
needs an explicit layout:

```bash
# LBHNC storage, vendor layout="bhsd"
VLLM_KV_CACHE_LAYOUT=HND vllm serve /model/Qwen2-0.5B-Instruct \
  --attention-backend FLASH_ATTN --block-size 64

# LBNHC storage, vendor layout="bshd"
VLLM_KV_CACHE_LAYOUT=NHD vllm serve /model/Qwen2-0.5B-Instruct \
  --attention-backend FLASH_ATTN --block-size 64
```

The Plugin reads vLLM's resolved cache layout rather than inferring it from
tensor geometry. It reinterprets each allocated page as dense
`[K/V, H, N, D]` or `[K/V, N, H, D]` storage. This is required because the
vendor kernel does not consume vLLM's non-dense views produced by splitting
`[..., 2D]`. The official stride-aware HIP cache writer always receives its
NHD contract through a zero-copy alias. The Plugin does not copy the full KV
cache on decode. The vendor's regular prefill path may gather paged KV
internally; rectangular short prefill uses that path by passing a conservative
`max_seqlen_q` upper bound, avoiding the vendor wrapper's incompatible MTP tile
selection.

An additional prefill composition is available as an explicit candidate:
`VLLM_HCU_FLASH_PREFILL_GATHER=1`. It reuses vLLM's existing C++
`cp_gather_cache` per head, then the public contiguous vendor varlen call.
It applies to BF16/FP16 paged queries with more than one query token and
static batch-times-maximum-context capacity at most 262144 tokens. It retains
device cumulative lengths, actual HND/NHD strides and caller output ownership.
Single-token decode, FP8 KV and larger capacities use their existing paths.
No new gather kernel or Core/Adapter source change is introduced.

The default is OFF; unset the variable or set it to `0` and restart to roll
back. This Flash-specific switch is independent of the RMSNorm switches.
Its effective boolean is registered through vLLM's official environment
compile factors before backend publication, covering ordinary and AOT caches.
Conflicting factor ownership is rejected. Scoped component, installed-device
and full Qwen3.5-4B quality/performance results are recorded in
[the bounded-prefill validation](../docs/validation/vllm033-bounded-prefill.md).
Other model, topology and architecture gates remain separate.

FA4-only `dynamic_causal`, `mask_mod`, and `aux_tensors` are accepted only
when inactive (`None`). An active feature fails closed instead of silently
changing its mask. The bounded-prefill path does not implement FP8 KV or DCP;
FP8 retains the existing path. Hybrid-cache validation is limited to the
recorded models and recipes. Layouts other than LBHNC and LBNHC are not claimed.

## RMSNorm eligibility

Provider admission checks the actual four-positional-argument call ABI before
compile-factor or operator publication. It accepts normal Python signatures and,
only for uninspectable native builtins, a compatible concrete declaration from
generated binding documentation, including overloads and optional additions.
Declarations are parsed without evaluating annotations or defaults. Missing or
incompatible documentation, uninspectable Python callables, and asynchronous or
generator providers are rejected. This extends the existing activation preflight;
its provider checks still run even when the RMS execution flags are OFF.
It neither invokes a provider nor proves kernel, Graph or model correctness.

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

# Roll back the optional RMSNorm custom operators.
export VLLM_HCU_USE_CUSTOM_OPS=0

# Or roll back only the LightOp RMSNorm path.
export VLLM_HCU_USE_CUSTOM_RMS_NORM=0

# Roll back FlashAttention without changing the Plugin installation.
vllm serve /model/Qwen2-0.5B-Instruct --attention-backend TRITON_ATTN
```

Device acceptance for v0.33 is tracked separately in the
[current validation record](../docs/validation/vllm033-upgrade.md).
Historical gfx936 results do not certify this release or another architecture.
Keep RMSNorm disabled until the intended stack has a live-device validation
record:

```bash
export VLLM_HCU_USE_CUSTOM_RMS_NORM=0
```

For complete removal, uninstall `hcu-vllm-das-plugin`; the adapter and vLLM
remain installed and retain platform ownership.

## Historical v0.32 evidence

The [`SiluAndMul`](../docs/validation/v032-adapter-extension-silu.md) and
[`RMSNorm`](../docs/validation/v032-adapter-extension-rmsnorm.md) records are
historical candidate evidence. The v0.32 three-way identities, upstream
takeover decision, and base final-wheel gates were recorded in the
[`vLLM 0.32 upstream-first validation`](../docs/validation/vllm032-upstream-first-operator-ab.md).
The explicit AITER W8A8 route, public vendor-kernel evidence, Graph replay,
model smoke, and isolated HumanEval64 result are recorded in the
[`vLLM 0.32 AITER INT8 MoE validation`](../docs/validation/vllm032-aiter-int8-moe.md).
