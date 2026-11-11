# HCU mHC HIP compatibility on vLLM 0.33

## Scope and anchors

Start Plugin `479e8ab`, immutable Core `368701db06`, Adapter `3257a934`.
Reference `75275f87` and historical archive documents
`mhc_full_integration.md:24-37,53-66`, `mhc_post_v3_validation.md:15-38`.
Historical GPU evidence is not this increment's device acceptance.

Core `custom_op.py:107-126,register_oot` provides public instance substitution.
Use four owned subclasses of MHCPreOp, MHCPreDelayedOp, MHCPostOp and
MHCFusedPostPreOp. Never rebind native methods, rocm_aiter_ops methods or global
HAS_TILELANG_MHC/HAS_AITER_MHC. HIP dispatch bypasses TileLang; native Torch
references supply fallback. Other platforms retain their native paths. Do not
replace the separate direct Torch DSpark head call in this increment.

The native delayed ABI `layers/mhc.py:384-402,516-558` returns residual plus
post/comb/input/next-pre and carries previous-sublayer pre_mix. It handles
model-entry narrow x and applies norm once. Reuse native Torch fallbacks with
those exact semantics. Installed AITER pre has `use_tf32`, not n_splits/norm;
project and big_fuse use output buffers, so stage with split-K=1, pad GEMM width
32 exposing 24 columns, and recover next-pre from the same projection/statistics.
HIP provider failures propagate. Missing optional providers select native before
execution. Post width5120 remains native; ordinary/delayed Pre5120 is eligible
under the historical BF16/HC4/contiguous/FP32-coefficients contract. No fused
AITER or TileLang provider is enabled. Existing custom-op master remains default
OFF; dedicated MHC/PRE/POST default OFF too. Freeze/check policy and include it
in compilation environment factors to avoid stale graph/AOT selection.

## Failure modes fixed before implementation

| ID | Failure | Replay observation |
|---|---|---|
| M01 | Registry bypasses real native constructors. | Instantiate the four native classes through actual CustomOp API; check subclass and original symbols. |
| M02 | Native fallback still calls TileLang/AITER unexpectedly. | Force native flags True and block all native vendor/TL calls; use actual Torch reference. |
| M03 | Delayed path uses current rather than previous mix. | Compare five outputs against native delayed Torch over 200 seeds with pre_mix, narrow x and folded post. |
| M04 | RMSNorm is omitted or applied twice. | Compare normed ordinary/delayed/fused outputs and norm_weight/eps forwarding. |
| M05 | Provider pre tuple order differs or statistics are wrong. | CPU AITER ABI doubles implement exact pre/project/big_fuse/post layouts; reference comparison. |
| M06 | Post5120 or unsupported shapes enter vendor kernels. | Routing matrix HC/dtype/width/empty/noncontiguous incl5120; no vendor call on fallback. |
| M07 | Zero tokens launch kernels or fail ambiguous reshape. | Explicit empty tensors with exact shapes for four operators. |
| M08 | Non-HCU platform gets owned publication. | CPU/native platform activation leaves OOT registry untouched. |
| M09 | Partial/foreign publication or lost owner appears healthy. | Public preflight/idempotence, sticky fatal and unchanged Core identities. |
| M10 | Master/children or env/AOT policy changes silently. | Read actual switch values; register frozen policy compile factor, reject active-policy drift; no fused/TL. |
| M11 | Provider errors are swallowed after writes. | Inject provider error, require original error propagation with no second dispatch. |
| M12 | Provider import/signature drift executes unreviewed ABI. | Exact inspectable signature bind before publication, missing optional capability native fallback. |
| M13 | CPU doubles treated as numerical hardware acceptance. | Record source hashes/SHAs/metadata, CUDA false and exact doubled boundaries. |

## Implementation and routing

The independent `das_hcu_mhc` entry advertises `hcu-mhc-hip-routing`, not device
or model acceptance. Four owned OOT subclasses are published on HCU through
`CustomOp.register_oot`. Their constructors check the frozen policy/ownership
for V41 and delegate complete native construction, including delayed Pre's
nested Post. Non-V41 operators delegate the native HIP method unchanged; no
Core symbols or global capability values are assigned. The native DSpark head
has a direct Torch Post call and is unchanged.

| V41 route | Width5120 | Other historically eligible widths | Empty / unsupported |
|---|---|---|---|
| Ordinary Pre | optional AITER HIP | optional AITER HIP | explicit empty / native Torch |
| Delayed Pre | optional staged HIP; previous Post native at5120 | optional staged HIP | explicit empty / native delayed Torch |
| Post | native Torch | optional AITER HIP v3 | explicit empty / native Torch |
| Combined Post+Pre | native Post+Pre | separate admitted Post then Pre | explicit empty / native components |

Eligibility is HC4, contiguous same-device GPU, BF16 Pre, BF16/FP16 Post,
positive1024-aligned hidden width, FP32 coefficients. Staged Pre uses split-K1
and the same padded projection/statistics to recover next_pre. Previous pre_mix
collapses the current residual; narrow x normalization is preserved. Norm runs
once outside the opaque provider op. Provider exceptions are not swallowed.
Disabled/absent providers select explicit native Torch, not Core forward_hip
(which could choose TileLang). TileLang/fused requests are rejected.

Provider imports and native operator preparation are inside a fatal/fork-aware
boundary. All four registry owners are preflighted; partial publication/later
owner loss is sticky fatal. Policy drift is checked on V41 construction or an
explicit health call; forward does not read environment variables. Compilation
factor `VLLM_HCU_DAS_MHC_POLICY` reflects the frozen effective policy. Changing
switches requires a fresh process; no inference-time hot switching is promised.

All master/child flags default OFF. In this state V41 gets the native Torch
compatibility path and no AITER import is required. To opt into eligible HIP
providers in a newly started process:

```bash
export VLLM_HCU_USE_CUSTOM_OPS=1
export VLLM_HCU_USE_AITER_MHC=1
export VLLM_HCU_USE_AITER_MHC_PRE=1
export VLLM_HCU_USE_AITER_MHC_POST=1
export VLLM_HCU_USE_AITER_TILELANG_MHC=0
export VLLM_HCU_USE_AITER_MHC_FUSED=0
```

Activation logs raw master/child values and actual provider availability.
`AITER_MHC_POST_KERNEL` defaults to v3 only when the Post provider is requested;
an explicit caller value is preserved. Historical evidence does not validate a
different explicitly chosen Post kernel.

## Results

The pre-implementation fixed replay recorded **0/7 passed** for missing modules.
The production replay passed **7/7** after correcting two fixture errors without
new cases or changed thresholds: explicitly import the native Torch reference
submodule (the package's `torch` attribute names PyTorch itself), and read empty
next_pre from index4 of the native five-result delayed ABI. The V41 model scope
was supplied as an explicit construction fixture after review narrowed ownership.
Early failed/intermediate logs remain alongside final artifacts.

The replay uses real native CustomOp construction and dispatch, native Torch
references and Plugin custom-op registration. Two differential groups exercise
200 deterministic seeds each, ordinary/delayed/Post/combined, previous pre_mix,
folded Post, narrow x, optional norm, empty tensors and the5120 routing matrix.
FP32 comparison uses atol=rtol=2e-5; BF16 uses atol=.002,rtol=.01. They are
reference/double differences, not a measured AITER GPU tolerance/result.

AITER functions are CPU doubles with the actual inspected signatures; hardware
eligibility and RMSNorm are also doubles. No real AITER kernel, norm kernel,
compiler/Graph, full model, GPU numerics, distributed execution or gfx938 runtime
has been accepted. Installed vendor Python signatures and ordinary Pre return
ordering were read without importing/executing the provider. Actual import/ABI
availability and performance remain separate runtime gates. The bw47 existing
container was reused; its observed metadata is `vllm0.33.0+rocm72`. The selected
Core source path and revisions are recorded, not inferred from metadata.
CUDA remained uninitialized. The `_C_AVX2` warning is retained as an ABI limit.

Reproduce with existing Python3.12 and Core/Adapter sibling sources:

```bash
VLLM_TEST_PYTHON=/path/to/python3.12 \
  bash tests/vllm033/run_mhc_hip_cpu.sh /path/to/output
```

Evidence root: `tests/vllm033/.cache/mhc-hip/`; final JSON/log in `final/`,
JUnit/full-suite output in `regression-final.xml`/`regression-final.log`.
The final reviewed portable suite completed **312 passed, 66 skipped,
14 warnings in188.66 seconds**. The66 skipped live-device cases are not
hardware passes. AST, shell syntax and `git diff --check` passed.
Source SHA256s are saved in the replay report. Original pytest cases/assertions
are unchanged except the pre-implementation entry-point metadata expectation.
The replay script and runner match a broad repository ignore pattern `*_hip*`;
they are deliberately included as final deliverables, not left as ignored files.

Review findings about import-fatal timing and actual constructor health checking
were corrected before commit. A subsequent review hypothesis that redirected
CustomOp construction bypasses the subclass initializer is contradicted by
both the actual native-class construction replay and a minimal CPython subtype
construction example (checked mixin then native initializer). A direct diagnostic
using the existing fixture also printed both actual owned types with
`_hcu_mhc_owned=True` and successfully executed folded delayed Pre; see
`construction-review.log` under the evidence root. No production
change was made for that hypothesis. These new error boundaries and non-V41 delegation
have source review evidence; the fixed CPU replay verifies math/registration but
is not presented as exhaustive concurrency or hardware coverage. Default policy
is intentionally different from old0.29/0.32 default-ON AITER policy. No README,
Core, Adapter or old vllm_hcu package files were modified; separate build/egg-info
artifacts already present in Adapter are untouched. Only local commits are made.
