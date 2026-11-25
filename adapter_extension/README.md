# HCU vLLM DAS optimization plugin

Independent Flash public-registry publication and health contract:
[v0.33 provider ownership](../docs/validation/vllm033-upgrade.md#ownership-and-changes). Its
activation dependencies are independent of the optional RMS provider.

`hcu-vllm-das-plugin` is the optimization-only extension for official vLLM
0.33 and `adapter-das`. The adapter owns the HCU platform and runtime
lifecycle; this wheel contributes optional, independently removable operator
optimizations through the adapter extension API.

Version `0.33.0` targets the `v0.33.0` branch. Migration status and current
evidence are recorded in [v0.33 validation](../docs/validation/vllm033-upgrade.md);
the v0.32 documents below are historical evidence and do not certify v0.33.
Separate extensions provide default-OFF RMSNorm, HCU FlashAttention, SlimQuant
and the explicit INT8 ASM candidate below. Core retains CT scheme selection,
weight loading, generic quantization and kernel construction.

It also provides an explicit HCU FlashAttention backend selected by
`--attention-backend FLASH_ATTN`. The backend keeps vLLM's official metadata,
scheduler and allocation. Its public `AttentionImpl` owns the HCU provider call
and dense HND/NHD cache views, routes BF16 and explicit E4M3 writes to the native
vLLM writer, and retains a Plugin E5M2 writer and shared descale adapter. E4M3
prefill uses Core's byte-preserving gather, explicit OCP E4M3 conversion to the
query dtype, and vendor varlen attention; decode without LSE retains vendor
paged attention, while explicit decode-LSE requests use the bounded composition.
See the [E4M3 implementation and validation](../docs/validation/vllm033-flash-e4m3.md).
The kernel page is 64
tokens. It neither copies nor replaces the official Flash forward method and
does not require private HCU hooks in the Core.

## V4.1 indexer cache default

With the paired Adapter's V4.1 default policy, `indexer_kv_dtype=auto` resolves
to `mxfp4` during HCU platform configuration when the deployment meets FP4
admission and the V4.1 Indexer and E2M1 packer extensions are active and
healthy. Deployments outside FP4 admission retain Core's FP8 default. The
effective dtype is set before Core model construction, cache allocation and
configuration hashing. Explicit
`--attention-config '{"indexer_kv_dtype":"fp8"}'` selects FP8, and explicit
`mxfp4` retains the existing FP4 admission. Other model families retain their
Core defaults.

`HCU_VLLM_FP4_INDEXER_IMPL=stock`, disabling the Indexer or packer through the
existing kill switches, a stock packer, or absent paired providers preserves
the existing FP8 resolution of `auto` in a fresh process. Earlier compatible
Adapter versions retain their original auto policy. General-plugin activation
defers an unresolved V4.1 auto configuration until the platform resolves it.
The existing FP4 deployment admission and provider checks still apply.

The LightOp FP4 decode path masks logits beyond each device sequence length
before candidate selection and top-k, including during CUDA Graph replay.

## V4.1 W4A8 masked MoE scheduling hint

The small-batch masked GEMM hint defaults on for eligible DeepEP low-latency
`gfx938` 64-CU W4A8 paths. It uses the local token extent for both masked GEMMs
and clamp/quantization. Workspace capacity and device-side per-expert row
counts retain their roles. Set `HCU_DSV41_W4A8_MOE_EXPECTED_M=0` before startup
to use the capacity hint for all configurations.

## E2M1 packer lowering (default triton, `stock` opts out)

`das_hcu_fp4_packer` rebinds Core's `_fp32x2_to_fp4x2` E2M1 packer at startup.
The audited upstream body emits PTX `cvt.rn.satfinite.e2m1x2.f32` with the
`=r,f,f` constraint, which DTK clang cannot lower (029 `BLOCKED_TOOLCHAIN`).
The replacement is Plugin-owned arithmetic: `min(|x|, 6)` saturation, seven
thresholds, tie-to-even on the magnitude code, and the IEEE sign bit (so `-0`
stays `0x8`). The first argument keeps the low nibble.

Activation requires real HCU `gfx938`, vLLM `0.33.*`, an exact AST match with
the audited definition plus identical aliases in six namespaces, an unhashed
and uncompiled caller set, no foreign compile factor, and no existing runtime
record. The paired Adapter registry defaults `HCU_VLLM_FP4_PACKER_IMPL` to
`triton`, so the candidate is selected unless a deployment opts out. Explicit
`stock` (`HCU_VLLM_FP4_PACKER_IMPL=stock`) or `HCU_DISABLE_OPS=vllm_fp4_packer`
leaves every original object in place; a
policy change needs a fresh process because JIT and Graph caches may already
refer to the published kernel. A failure after publication is commit-fatal and
is never presented as a rollback.

Upstreaming position: the arithmetic body is a workaround for a toolchain gap,
so the intended end state is deleting it once DTK lowers the PTX instruction or
Core ships a lowering it can compile. This module therefore declares no
upstream-format contribution; it carries no cache format, scale recipe or
roofline claim of its own.

Evidence level: **spec-consistent, hardware-unverified**. CPU/mock contracts
only (83 tests): explicit E2M1 code literals, FP32 tie neighbours, saturation
including `2**28`/`FLT_MAX`, NaN/Inf policy, nibble order, the audit gate, and
policy/rollback refusals. The Triton kernel was neither compiled nor executed;
native compilation, kernel hit, Graph replay, accuracy and performance remain
separate gates.

vLLM's native `SiluAndMul` remains the selected implementation. Warmed
compiled A/B measurements on `gfx936` did not justify replacing the upstream
operator, so the plugin neither registers nor ships a `SiluAndMul` override.

No model, topology, FP8-KV, or broad performance support is implied by these
operator-level capabilities.

## HY4 frontend migration

`das_hcu_hy4_frontend` provides bounded native ByteLevel prompt priming with
`VLLM_HCU_ENABLE_BYTELEVEL_DETOKENIZER_TAIL=1` (default off). Core keeps its
detokenization and stop handling. Adapter owns admission and process health;
the Plugin does not register a platform/general entry point. Source revisions,
ownership, CPU evidence and the remaining W4A8/PCP/R16 gates are recorded in
[the HY4 migration ledger](../docs/validation/hy4-v033-migration.md).

The separate `das_hcu_hy4_ct_w4a8` capability supplies the distinct
`hcu_hy4_ct_w4a8` configuration through the paired Adapter's CT registration
transaction. It reuses the existing packed W4A8 method and Core loader/runner;
other model types and quantization formats keep their original selection.
Require `HCU_ADAPTER_VLLM_REQUIRED_CAPABILITIES=hy4-ct-w4a8` for HY4 validation.
This is a source candidate: offline-R16 model binding, native ABI, checkpoint
accuracy and performance remain open in the ledger.

The owned HY4 CT/static backbone can opt into native iHC launch tuning with
`VLLM_HCU_USE_CUSTOM_OPS=1 HCU_VLLM_HY4_IHC_LAUNCH=1`. It uses the original
Core kernels with stage-1/stage-2 warps 4/4 on gfx938, BF16, hidden size 6144,
HC multiplier 4, MRV2 and compilation mode 0. Other configurations retain the
native launch. MTP is unchanged. This defaults off; unset the child or disable
the master and restart to roll back. Reduction ordering can change slightly;
component tests do not replace checkpoint accuracy or serving measurements.

With the paired Adapter iHC bridge, text-only eager PCP16 can additionally set
`HCU_VLLM_HY4_IHC_VENDOR=1`. Only contiguous BF16 inputs with at least 378 rows
use BoltOps pre/post plus Core RMS; smaller inputs and D retain the Core chain.
Topology is captured on owned backbone instances during construction. Native
decoder bodies, live checkpoint parameters and BF16 rounding boundaries remain.
This option defaults off and requires whole-model accuracy/performance validation.

The owned HY4 Mooncake worker reuses equal block-ID groupings only within one
native builder call. `HCU_DISABLE_OPS=hy4_mooncake_grouping_reuse` restores the
unmodified builder. Transfer regions, byte offsets and descriptor validation
remain native.

HY4 W4A8 PCP16 prefills can opt into per-token INT8 DeepEP HT transport with
`VLLM_HCU_USE_CUSTOM_OPS=1 HCU_VLLM_HY4_HT_INT8_DISPATCH=1` (default off).
This quantizes source rows before dispatch; Core retains receive metadata,
communication handles and finalize. Inputs below 756 source tokens per rank
keep BF16 transport. Admission requires gfx938, BF16, eager MRV2, synchronous
scheduling, TP1/PP1/DP1/DCP1, HT16, and no DBO, ubatching, LoRA or online EPLB.
The child or master can be disabled before restarting to retain the existing
prepare path. CPU contracts and 16-rank transport/scale equivalence are covered;
whole-model accuracy and serving performance are separate validation gates.

The same HY4 PCP16 HT scope can opt into the legacy LightOp expert output
gather with `HCU_VLLM_HY4_EXPERT_GATHER=1` (default off). Adapter owns the
provider bridge; the owned expert instance retains the native permutation,
GEMMs, activation and finalize. Disabling the child or custom-ops master
restores native gather. No process-wide Core binding is changed.

## Standalone Step5 MTP configuration

`das_hcu_step5_config` declares the `step5-standalone-mtp-config` capability.
Adapter activates it before vLLM constructs model configuration. It registers
only the separate `step5_mtp` model type through Transformers' public
`AutoConfig.register()` API. Ordinary `step5` and `step5_vl` configs, their
validation and the in-checkpoint MTP conversion stay owned by vLLM.

The Plugin class inherits official `Step5Config`, defaults standalone drafts
to one prediction layer and the `Step5MTP` architecture, and preserves explicit
checkpoint values. Both ordinary `hf` and `auto` config formats work with
`trust_remote_code=False`; no custom config-format flag or Core registry patch
is needed. Discovery imports no framework or native provider. A pre-existing
`step5_mtp` owner is refused without overwriting it. Failed publication or
subsequent ownership drift is fatal for that process; health checks do not
repair or silently republish the mapping. Spawned processes register their own
mapping, and a fork must explicitly activate its inherited exact mapping.

To require configuration support at startup, use the Adapter's existing
`HCU_ADAPTER_VLLM_REQUIRED_CAPABILITIES=step5-standalone-mtp-config` requirement.
This is a **configuration capability**. It does not establish Step5 checkpoint
loading, tokenizer, sparse attention, Graph, drafted/accepted tokens, model
accuracy or performance. See the [configuration validation boundary](
../docs/validation/vllm033-step5-config.md).

## HYV4 Flash configuration metadata

`das_hcu_hyv4_flash_config` provides `hyv4-flash-config`. It registers the two
distinct `hy_v4_flash` and `hy_v4_flash_vl` config types through public
`AutoConfig.register()` during Adapter extension activation. The text config
inherits official Core `HYV4Config`; ordinary HYV4 configs and model registry
entries retain their official owners. Flash-specific IndexPool, SWA, Engram
and vision fields use the existing candidate's validation and serialization.

Both names are checked for foreign ownership before publication. A partial
commit or later ownership drift latches a fatal failure; health checks never
repair the mapping. Discovery imports no framework. Forked children must
activate their inherited mapping, and spawned children discover the installed
entry point. The capability can be required using the existing Adapter
`HCU_ADAPTER_VLLM_REQUIRED_CAPABILITIES=hyv4-flash-config` setting.

The optional `das_hcu_hyv4_flash_model` entry point delivers the active Flash
model implementation. Select it with
`HCU_ADAPTER_VLLM_REQUIRED_CAPABILITIES=hyv4-flash-model` on pinned vLLM 0.33.0.
Unselected discovery imports no model and claims no model capability. This
model owns the `HYV4FlashForCausalLM`/`HYV4FlashVLForConditionalGeneration` names;
the separate optimized HY4 static-EPLB architecture and its entry points remain
available. Flash quantized/latent parallel admission remains narrower than
that static path and must not be treated as EPLB qualification.

Activation health certifies registry/config/DFly ownership only. Checkpoint,
DFly generation, HiSparse admission, changed-input Graph replay and measured
performance need their own installed-artifact validation. See the
[model integration and runtime gates](../docs/hy4_flash.md). In particular,
HiSparse plus DFly or an unqualified Graph path remains unsupported.

## DSV4.1 HCU decode defaults

Eligible DSpark draft LL receive quantization, INT8 direct GEMM, and Shared
Expert INT8 fusion default on. Set `HCU_DSV41_DRAFT_LL_LIVE_QUANT=0`,
`HCU_DSV41_WQ_B_INT8_DIRECT=0`, or `HCU_DSV41_SHARED_EXPERT_INT8_FUSION=0`
respectively before startup to restore the corresponding fallback. The Adapter
uses LightOp in eligible INT8 and FP8 graph capture by default. Set
`HCU_LIGHTOP_PER_TOKEN_INT8_GRAPH=0` or `HCU_LIGHTOP_PER_TOKEN_FP8_GRAPH=0`
to restore upstream quantization for that capture path.

On the validated gfx938 vLLM 0.33 build (`g7364da`),
`HCU_VLLM_WO_A_TOKEN_MAJOR` defaults to enabled. It applies to eligible
BF16 eager decode batches of 2–32 rows. The owned MXFP4 attention path
also enables `VLLM_HCU_DSV41_INT8_WQ_B` by default, subject to the existing
FULL capture, quantization and tensor layout checks. Set either variable to
`0` before starting the server to disable its optimization.

## DSV4.1 Channel-FP8 configuration routing

The independent `das_hcu_dsv41` Adapter extension advertises only
`dsv41-channel-fp8-routing` and registers the distinct public quantization name
`hcu_dsv41_channel_fp8`. It does not replace the official `fp8`,
`deepseek_v4_fp8`, or `compressed-tensors` entries or their methods, and does
not depend on the RMS/Flash extensions or vendor optimizations for discovery.

Automatic selection requires the HCU platform, a V4.1 model configuration,
`quant_method=fp8` (or `deepseek_v4_fp8`), `is_per_channel=true`, no weight block
size, dynamic activations, and a compatible user quantization choice. Other
models and schemes preserve their official selection. The configuration reuses
the official V4.1 FP8 policy, checks native exclusions before delegating to the
official CT channel-weight/token-activation scheme, and leaves BF16 `wo_a`
unquantized. At normal serialized startup, an existing foreign owner of the
new name is rejected; partial registration failure refuses further activation
in that process. The Core registry has no atomic reservation API: this extension
does not guarantee inter-owner atomicity against concurrent external writes or
roll back arbitrary Core registrations. Its health check detects a later lost
owner without repairing it.

This increment is configuration/routing infrastructure. It does **not** certify
checkpoint or target/draft scale loading, HCU kernel support, model inference,
W4A8, PCP, DSpark, Graph, accuracy or performance. Those are separate migration
gates. Requiring `dsv41-channel-fp8-routing` at startup proves activation only,
not those execution properties. See the [routing contract and evidence](../docs/validation/dsv41-channel-fp8-routing.md).

For the selected Channel-FP8 W8A8 Linear scheme, a Plugin-owned CT LinearMethod
returns correctly shaped empty outputs before local quantization/GEMM. It
retains CT weight creation/finalization and native nonempty execution, registers
through the public weight-loader-v2 API, and leaves the surrounding Linear
forward and collectives intact. Native `QuantizedActivation` is reused.
The pinned v0.33 V41 target already resolves direct/inverse scale spellings;
small CPU source replay verifies ordinary and fused projection loading without
adding parameter aliases or changing that loader. Hardware admission,
quantization and GEMM are doubled in that replay. DSpark draft scale aliases,
actual checkpoint loading, distributed execution and GPU correctness remain
unverified. See [target loading and empty-row validation](../docs/validation/dsv41-channel-fp8-loading.md).

## PCP Channel-FP8 providers

`das_hcu_v41_pcp` remains the sole owner of the V4.1 target model registration;
`das_hcu_dsv41` independently owns Channel-FP8 quantization admission. Select
PCP with `VLLM_HCU_DSV41_PCP_ATTENTION=1` and `HCU_VLLM_PCP_IMPL=native`.
The enabled PCP owner also advertises the `dsv41-pcp-megamoe` capability from
#399; requiring that capability does not enable PCP by itself.

PCP attention and FP8 reference scoring stay adapter-owned. Native mHC class
names retain the adapter's public OOT dispatch, including its provider/shape
fallbacks; the PCP model does not install another local mHC implementation.
The additional `dsv41.pcp_attention` and `models.dsv41_pcp` import paths are
compatibility aliases to these same classes, not independent implementations.

For Channel-FP8 TP1/DP1/PCP8/EP8 with DeepEP HT, explicit `deep_gemm` selects
the scoped PCP DeepGEMM method; explicit `deep_gemm_mega_moe` selects MegaMoE.
Vendor target-shape, eager, no-replay and other provider admission checks remain
mandatory. Existing decode HT/LL and DSpark AG-RS methods remain selected for
their original configurations. CT W4A8 uses its independent quantization route.

`VLLM_HCU_DSV41_PCP_PROFILE_WORKER=1` (default) selects rank-local memory
profiling only for that resolved Channel-FP8 PCP8 topology and only when Core's
worker choice was automatic. Engram staging and vendor buffers share the same
local token bound. Disabling automatic selection or using a different explicit
worker retains global capacity. Explicitly naming this PCP worker still applies
its topology/quantization predicate to both profiling and capacity reservation;
configurations outside that predicate retain global capacity. Changing the
policy after PCP publication requires a fresh process. The scheduler's global
budget is not reduced.

CPU contracts cover registration, dispatch, local capacity, empty ranks and
loader preservation. They do not certify multi-rank execution, vendor kernel
availability, model accuracy or performance. PCP8 hardware acceptance remains
required after merging these providers.

## Optional PCP inverse RoPE

Select the HCU provider in an `HCU_CONFIG` file with the PCP model owner enabled.

```yaml
version: 1
ops:
  vllm_pcp: native
  vllm_inverse_rope: hcu
```

Inverse RoPE defaults to `stock`. `HCU_DISABLE_OPS=vllm_inverse_rope` selects
Core inverse RoPE in a fresh process. The choice enters the PCP compile factor.
Source or policy drift after publication requires a process restart.

The provider applies `hcu_vllm_das::deepseek_v4_inv_rope` to the unrotated
64-dimension BF16 output tail on gfx938. PyTorch builds the packaged HIP sources
in its user build cache before model construction. Explicit HCU selection
requires HIP PyTorch, matching hipcc and a C++ build toolchain. A missing selected
provider fails during preparation. Ineligible tensor shapes use Core inverse RoPE.
Prefill retains Core's compressed KV and SWA gather and dequantization.

## DeepSeek V4.1 static EPLB candidate

The owned PCP model accepts `additional_config.hcu_static_eplb` for a fixed
version-2 expert map or offline calibration. The FP4 constructor composes
static placement with the existing PCP attention, Engram and decoder SWA
replay gate. The scoped DeepGEMM/DeepEP HT method admits the configured
physical expert count. Global runtime EPLB remains disabled.
Configuration, migration ownership and validation scope are recorded in
[static EPLB](../docs/validation/v41-static-eplb.md).

## Compatibility

- vLLM release line: `>=0.33.0,<0.34.0`; prerelease, postrelease,
  development, patch, and local version suffixes are accepted when the
  required operator APIs and LightOp capabilities are present. FLASH_ATTN uses
  the official public backend registry and AttentionImpl contract.
- Adapter host distribution: `hcu-adapter>=0.5.22`, extension API v1 with
  process health checks from `hcu_adapter.extensions`.
- Runtime: a prepared HCU PyTorch/DTK/LightOp/AITER stack. LightOp and AITER
  are intentionally not fetched by pip because production deployments must
  use ABI-matched local builds.
- The plugin does not own `vllm.platform_plugins` or `vllm.general_plugins`.

The repository-root legacy build is disabled on this branch. Build only
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
python3 -m pip install --no-deps /wheels/sgl_kernel-*.whl /wheels/hcu_adapter-*.whl
python3 -m pip install --no-deps /wheels/hcu_vllm_das_plugin-*.whl
```

Current Adapter releases split the pure `sgl-kernel` package from the
`hcu-adapter` startup/bridge package. Install both audited wheels; the Adapter
metadata owns their kernel version pairing. When migrating an existing legacy
combined `sgl-kernel` installation, uninstall that old wheel before installing
the pair, as required by Adapter upstream, or use a fresh environment. Do not
install the host over a legacy combined RECORD: both can claim `hcu_adapter/`
and a later uninstall can remove the other distribution's files. The verifier
also accepts an existing genuine legacy combined owner for ABI-compatible
manual validation, but rejects duplicate host owners and pure-kernel-only
installs. No framework or model implementation is moved into this plugin.

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

## Shared channel-FP8 AITER MoE interface

`hcu_vllm_das.moe.aiter_fp8` exposes `Problem`, `get_config`,
`prepare_weights`, and `run`. Callers supply actual expert/input/output shapes,
top-k, compute dtype and activation parameters. `get_config` calls the installed
public `aiter.moe.get_aiter_moe_config` first, including the `use_shuffle` preference and the
requested solution type. It records `vendor_query` or `qualified_profile` rather
than treating a missing tuning row as a hit. An optional `QualifiedProfile`
must match the exact static problem, dtype, architecture, layout and token bound
and identify its independent native evidence. The common interface contains no
HY4 shapes, router scaling or expert-count constants.

```python
from hcu_vllm_das.moe import aiter_fp8

problem = aiter_fp8.Problem(M, E, N1, N2, K, topk, x.dtype,
    activation="silu", gated=True, gemm1_limit=limit,
    architecture=actual_architecture)
selected = aiter_fp8.get_config(problem, use_shuffle=True,
    profiles=consumer_qualified_profiles, solution_type="asm")
# Once, after loading and before capture. Keep only the prepared parameters.
prepared = aiter_fp8.prepare_weights(w1, w2, s1, s2, selected)
y = aiter_fp8.run(x, topk_weights, topk_ids, prepared, selected,
    routed_scaling_factor=1.0)
```

Returned `need_shuffle` / `need_shuffle_scale` flags determine preparation. Packing uses public `aiter_moe_shfl_weight` and, only if required by the returned
config, its scale counterpart. Prepared weights retain a layout signature and
own no duplicate canonical buffers. Later token-count configs must be layout
compatible; different packed stage IDs refuse instead of repacking during
capture. Config mutation and replaced/modified prepared buffers refuse. Treat
inference weights as immutable: they may have no Torch version counter. A reload
needs canonical buffers, a new prepared owner and recaptured graphs. This API
does not qualify every model, geometry, dtype or activation supported by AITER;
each consumer retains native and full-model gates.

HYV4 Flash opts in through `--moe-backend aiter` for serialized channel-FP8
routed experts. Dense/DFly and auto/triton retain their official Core owners.
`--hf-overrides '{"flash_aiter_moe_use_shuffle": false}'` selects the raw control;
the explicit AITER route otherwise prepares shuffled weights. The layout policy
is frozen at construction; the resolved HF flag participates in Core ModelConfig hash factors. Current admitted
expert geometries are genuine gfx938 ordinary TP4/TP8, E512/K1536/N384 or192,
BF16 input, top16 and clamp10. The qualified ASM11007+21000/BM32 profile is used
only after a vendor lookup miss; no global architecture or FP8 oracle is changed.
Core keeps CT loading, routing, prepare/finalize and TP reduction. Its router
already applies3.998, so the vendor receives factor1.0. Startup logs distinguish
the selected backend, config provenance and raw/shuffled layout. Component
qualification is not full-model accuracy or performance acceptance.

## Explicit INT8 ASM candidate

Request `HCU_ADAPTER_VLLM_REQUIRED_CAPABILITIES=hcu-aiter-int8-moe` and
`--moe-backend aiter`. This requires the associated Core public
`register_int8_moe_backend` API; stock v0.33 without that API refuses activation.
The retired `int8_oracle.install()` remains unavailable and no oracle functions,
already-imported aliases, CT classes or builtin enum members are replaced.
Builtin auto/triton behavior stays unchanged.

The architecture candidate admits exact gfx936 and gfx938 identities, with
raw symmetric per-channel INT8 expert weights,
dynamic per-token INT8 activations, BF16/FP16 inputs, gated SiLU and ordinary TP.
The gfx938 component/model gates are separate from the existing gfx936 results.
Static TP-only EP2 is admitted on homogeneous gfx936 or gfx938 worker groups,
with BF16 inputs, global256/local128 experts, hidden2048/intermediate512/top8.
Architecture-specific validation remains separate; the gfx936 records below
do not certify gfx938.
On gfx936, the public factory/CT loader/runner component has passed a controlled two-rank
fixture, including shared output, reload and changed-input Graph replay. The real
Qwen3.5 INT8 checkpoint also completes original HumanEval32 with both AITER and
official Triton scoring raw30/normalized32 under default Graph. See the
[component record](../docs/validation/vllm033-int8-ep2-component.md) and
[full-checkpoint evidence and limits](../docs/validation/vllm033-int8-ep2-he32.md).
DP/PCP/sequence parallelism, EPLB, elastic topology, unsupported EP geometries,
bias, LoRA, padded layouts, router-weight-on-input and non-neutral activation
parameters refuse explicitly. `passthrough` refuses because this expert delegates
cross-rank reduction to the official runner; it does not reduce internally.

Canonical expert maps retain actual local-index permutations. An actual local
vendor profile takes priority. After a local E128 lookup miss, the EP candidate
can reuse only the stage-qualified raw E256 profiles for M1/2/3/4/8/16/17/32/64/
128/256/512/1024, with the exact first-stage ID, second-stage ID and sorting
block size checked together. Compatibility selection is recorded explicitly.
Unknown profiles/token counts
use complete official Triton fallback before native launch. This is not an E128
CSV tuning or a broad native/performance claim.
The additional shapes are a new isolated candidate: their public two-rank
runner/reload, model and performance gates remain pending. The component and
model records below belong to the earlier M1/17/128 policy at runtime7e088e9b.
See the [native profile evidence and remaining gates](../docs/validation/vllm033-int8-ep2-native-batches.md).
Core retains CT scheme selection, parameter metadata, weight loading, generic
quant config, prepare/finalize and reductions. Preprocessed weights retain the
official method's default refusal.

Plugin composes native ASM GEMMs with Core quantization/activation semantics.
A shape/config miss uses the official whole-call Triton fallback, including its
own quantization and scratch allocation. Native launch failures propagate.
The extension publishes a content fingerprint for normal/AOT compile factors,
binding its source, loaded native modules, ASM objects and channel config before
model construction; later file drift refuses cache reuse. The vendor's resolved
plan architecture must match the detected HCU before native acquisition. The
published architecture, provider getter and module identity remain bound to
process health and the compile factor; later provider drift refuses selection.
Activation must precede
any vendor ASM configuration consumer in the process; an already imported vendor
ASM module (including one inherited by fork) refuses and requires a fresh spawned
process. This avoids binding stale vendor plan caches to new config bytes.
Activation occurs exclusively at Adapter startup before model consumers. No exact wheel string
is required for compatible builds.

Public-path validation on gfx936 covers ten installed CT factory/expert/full
forward cases and thirty changed-input Graph replays against official Triton.
Single-card Qwen3.5-35B-A3B-Channel-INT8-w8a8, TP1/MTP0, BF16 activations and
TRITON_ATTN passes original HumanEval tasks0–31: AITER raw32/32, Triton raw31/32;
both normalize to32/32 (Triton task24 contains a code fence). The actual forty
CT INT8 experts and default FULL_AND_PIECEWISE Graph dispatch were verified.
ASM config selections and native code-object loads corroborate execution;
these are not an all-kernel trace. Both arms release the GPU but retain a worker
grace-period SIGTERM warning, so clean lifecycle remains unaccepted.

The EP2 model pair retains internal worker-grace SIGTERM and EngineCore forced
termination logs despite API exit0 and released resources. Clean lifecycle,
warmed EP2 performance, full-model checkpoint reload and remote device CI remain
pending. Earlier TP1 warmed-serving evidence belongs to its older frozen Core
combination. Core public registry: [MR83](https://developer.sourcefind.cn/codes/OpenDAS/vllm/-/merge_requests/83).
Historical private integration results do not certify this public path. The
general RMS installation verifier checks official INT8 symbol ownership but
does not validate this capability.

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
When `mixed_precision` is absent, the parser preserves the historical format:
applicable Linear layers use per-channel INT8, routed experts use packed INT4,
and `ignore` patterns remain authoritative. Ordinary embedding/lm-head and
model-owned router construction keep their official unquantized behavior.
An explicit `mixed_precision` map requires the packed-expert declaration.
Dense INT8 and n-gram INT8 groups are optional; omitting either retains its
unquantized weights. Every present group needs a complete module list and
matching format. A malformed declaration never falls back to historical
defaults. The legacy format does not infer quantized PLE tables. Verify loader
numerics, changed-input Graph and model accuracy
for the selected Core/provider/checkpoint combination before deployment.

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
generator providers are rejected. Explicit RMS opt-in, a required `das-hcu` extension, or a required
`lightop-rms-norm` capability performs these checks before registration.
With RMS disabled and no explicit requirement, registration retains the official
fallback and compile factor without importing LightOp. If later enabled, the
first eligible eager custom-op call resolves and validates the provider once
per process; ABI/import failures are latched rather than silently falling back.
The read-only extension health checker reports prior provider failures without
loading or repairing a provider. Direct custom-op calls also require admission.
See [optional-provider loading](../docs/validation/vllm033-rms-lazy-provider.md).
Admission neither invokes a provider nor proves kernel, Graph or model correctness.

Compiled calls delegate to Core's complete RMSNorm IR, including its compiler
precision policy and fusion across consecutive residual additions. The optional
LightOp provider is retained for eligible eager execution; this capability does
not promise a vendor normalization kernel in compiled serving. Direct ordinary
and mutating-fused custom-op ABIs remain available and independently tested.

The eager LightOp RMSNorm path is selected only when all of the following hold:

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

The optional eager ordinary and fused RMSNorm paths retain eager native IR's rounding
order: normalize in FP32, cast to the input/weight dtype, then multiply by the
weight. LightOp computes the unweighted normalization and PyTorch applies the
weight after that cast. The residual still stores the rounded FP32 sum. This
uses an extra dtype-preserving multiply; measure its serving cost independently
of accuracy. A final-cast-only FP32 formula is insufficient as a regression
oracle, especially before INT8 activation quantization.
See the scoped [rounding validation](../docs/validation/vllm033-rms-rounding.md)
for the regression, model results, and component cost.

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

## Optional HY4 LL tuning

```bash
export VLLM_HCU_USE_CUSTOM_OPS=1
export HCU_VLLM_HY4_LL_TUNING=1
```

This defaults to off and requires Adapter's
`hcu_adapter.features.ct_w4a8.masked_activation` provider. It is restricted to
text HY4 W4A8 on gfx938: BF16, DP16/EP16, TP/PP/PCP/DCP1, DeepEP LL,
H6144/I2048, 256 experts with top-k8, a 32-token per-rank capacity, positive
SwiGLU clamp, and FULL Graph with compilation mode 0. Other configurations
retain the original experts and providers.

The two masked GEMMs reuse Core's CPU DP estimate with a 128-token scheduling
floor; missing DP metadata retains the full-capacity hint. Device counts and
capacity remain unchanged. The Adapter uses 64 persistent slots per expert and
the v0.25 FP32 SwiGLU/INT8 quantization math. This differs from the current
LightOp vendor path's rounding and requires model accuracy validation.
Component timings do not establish serving throughput or TTFT improvements.

Restart with `HCU_VLLM_HY4_LL_TUNING=0` to disable this policy, or
`VLLM_HCU_USE_CUSTOM_OPS=0` for the master rollback. P-side HT/static EPLB is
unaffected.

## Optional HY4 fused routing

```bash
export VLLM_HCU_USE_CUSTOM_OPS=1
export HCU_VLLM_HY4_FUSED_ROUTING=1
```

This defaults to off. The owned HY4 backbone and MTP routers reuse Core's
`fused_topk_bias` for sigmoid routing with one expert group, renormalization,
256 experts and top-k8. The original `GroupedTopKRouter` class, routing method,
parameters, callbacks and `select_experts` remain unchanged.

Admission requires text HY4 BF16 H6144 on gfx938, MRV2, compilation mode 0,
DP16/EP16 with TP/PP/PCP/DCP1 and DeepEP LL. LoRA, online EPLB, elastic EP,
DBO, multiple ubatches, batch-invariant mode and AITER MoE retain native routing.
Unsupported tensor metadata also retains native routing. P-side HT/static EPLB
is unaffected.

Core consumes the device padding mask: padded expert IDs become -1 while
their normalized/scaled weights remain unchanged. Single-device operator and
changing-input Graph checks passed; actual 16-rank LL dispatch, experts,
combine and whole-model validation remain outstanding. Component speedups
are not serving throughput or TTFT measurements.

Restart with `HCU_VLLM_HY4_FUSED_ROUTING=0` to disable this policy, or
`VLLM_HCU_USE_CUSTOM_OPS=0` for the master rollback. Changing a captured policy
requires restarting the service.

## Adapter directory layouts

Adapter's root-module and `infra`/`features` layouts use the same Plugin entry
points. Plugin imports resolve the actual Adapter owner through `adapter_api`;
module identity and shared state remain owned by Adapter. A missing `infra`
namespace selects the legacy layout. Missing owners or dependencies in the new
layout are errors and never retry legacy providers. See the
[layout compatibility validation](../docs/validation/vllm033-adapter-layout.md).

## Historical v0.32 evidence

The [`SiluAndMul`](../docs/validation/v032-adapter-extension-silu.md) and
[`RMSNorm`](../docs/validation/v032-adapter-extension-rmsnorm.md) records are
historical candidate evidence. The v0.32 three-way identities, upstream
takeover decision, and base final-wheel gates were recorded in the
[`vLLM 0.32 upstream-first validation`](../docs/validation/vllm032-upstream-first-operator-ab.md).
The explicit AITER W8A8 route, public vendor-kernel evidence, Graph replay,
model smoke, and isolated HumanEval64 result are recorded in the
[`vLLM 0.32 AITER INT8 MoE validation`](../docs/validation/vllm032-aiter-int8-moe.md).
