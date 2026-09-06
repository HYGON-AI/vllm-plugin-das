# SGLang operator adaptation validation for vLLM 0.25.1

## Scope

This change compares `HYGON-AI/sglang-das` at
`0457572a14edd55e05220726ff4a8af8b1931771` with the vLLM HCU plugin based on
the plugin branch baseline at `6925ca37`. The corresponding vLLM source is
`/models/zb/vllm_025/vllm` at `7b108ad1`. It adapts only operators with an exact vLLM
0.25.1 ownership seam, validates the installed LightOp/AITER ABI, and keeps
all HCU routes subordinate to `VLLM_HCU_USE_CUSTOM_OPS`.

The test host used BW1100 (`gfx938:sramecc+:xnack-`), PyTorch 2.11.0, installed
vLLM `0.25.1+das185.dtk2604.torch2110.2608171710.g7b108a`, LightOp
`0.6.0+das.dtk2604`, and AITER
`0.1.5+das185.dtk2604.torch2110.2608180853.g40a705`. Raw benchmark JSON is
written under `/tmp`; it is intentionally not committed.

## Disposition

| Candidate | Result | Production action |
|---|---|---|
| LightOp `moe_fused_gate_sqrtsoftplus` | Accepted | Added the non-hash DeepSeek-V4 router route with master/leaf gates and official hash fallback. |
| LightOp W16A16 Marlin MoE | Accepted | Added a TP-only, exact-shape unquantized oracle backend with one-time packed weights. |
| LightOp `ds_cat` mode 0 | Accepted | Replaced the exact FlashMLA decode concatenation seam, with `torch.cat` fallback. |
| LightOp `layer_norm_fwd_1pass_opt` | Accepted | Added a strict Qwen3.5/3.6 gated RMSNorm custom-op route, bound only at the exact Qwen module, with master/leaf gates and canonical Triton fallback. |
| LightOp `topk_softmax` | Dependency-rejected | Installed output-index ABI has no executable dtype; no switch or route was added. |
| AITER fused recurrent packed decode | Performance-rejected | Index 0 violates vLLM's state ABI; the required safe remap makes all shapes slower. |
| AITER LayerNorm2D | Accuracy-rejected | Installed mixed BF16-input/FP32-affine contract produces invalid output. |
| AITER MXFP4 prequant GEMMs | Dependency-rejected | BW200B configs are absent and explicit MI350 configs cannot lower DotScaleOp on gfx938. |
| AITER `silu_and_mul` | Performance-rejected | Kept the existing categorized LightOp implementation. |
| AITER `tgemm` | Performance-rejected | Kept the existing vLLM linear implementation. |
| AITER `batched_gemm_bf16` | Performance-rejected | Kept the DeepSeek-V4 WO_A einsum; the callable replacement is substantially slower after mandatory layout conversions. |
| EP `ep_build_m_indices` | Already covered | Existing LightOp `ep_scatter` produces indices and inverse permutation in one launch. |
| Fused RMS plus dynamic quant | Already covered | Existing public categorized LightOp route has the same contract and avoids private LMSlim APIs. |
| Generic RMS/RoPE/KV-store fusions | No vLLM seam | Not adapted; SGLang token-pool mutation is not ABI-compatible with vLLM paged KV cache ownership. |
| Newer AITER fused Qwen/gating symbols | No compatible seam / dependency unavailable | The fused QK/RMS/RoPE/cache implementation is present only in a categorized module but does not match vLLM paged-cache ownership; absent symbols get no dormant switch. |

The complete symbol-by-symbol audit and Triton alternatives are in
`docs/operator_adaptation_audit_v0251.md`.

## Controls and fallbacks

| Route | Leaf switch | Default | Fallback |
|---|---|---:|---|
| DeepSeek-V4 sqrt-softplus gate | `VLLM_HCU_USE_LIGHTOP_SQRTSOFTPLUS_GATE` | on | Official vLLM router for disabled, unavailable, unsupported, or hash-routing inputs. |
| W16A16 Marlin MoE | `VLLM_HCU_USE_LIGHTOP_W16A16_MOE` | off | AITER/Triton selection before any layout conversion. Kernel failures after packing fail closed. |
| FlashMLA decode concat | `VLLM_HCU_USE_LIGHTOP_MLA_DECODE_CAT` | on | `torch.cat` for disabled, unavailable, or ineligible layouts; the legacy `VLLM_USE_OPT_CAT=0` remains an opt-out. |
| Qwen gated RMSNorm | `VLLM_HCU_USE_LIGHTOP_QWEN_RMSNORM_GATED` | on | vLLM Triton for disabled, missing categorized LightOp export, or any non-HCU/non-BF16/non-contiguous/non-Qwen-width/group/activation/unmeasured-row input. |

The W16 backend is restricted to BF16
`E=256,K=2048,N=512,top-k=8,max_num_tokens<=16`, no EP/EPLB/DP/PCP/SP, no
expert bias, and auto backend selection. Its packed layout cannot safely be
sent to AITER or Triton, so all rejection happens before weight replacement.

## Operator accuracy and performance

| Route | Accuracy | Stable performance result |
|---|---|---|
| sqrt-softplus gate | 75/75 live-HCU cases | 40/40 shapes exceeded 5%; speedup 6.83% min, 22.61% median, 44.92% max versus official vLLM. |
| W16A16 Marlin MoE | 6/6 live-HCU cases, including the exact Qwen shape | For M=1/2/4/8/16: 31.08%--68.22% faster than Triton and 6.28%--71.35% faster than AITER. M>16 is rejected by the oracle. |
| FlashMLA decode concat | 4/4 live-HCU cases, including production strides | 33/33 shapes exceeded 5%; speedup 8.84% min, 47.03% median, 77.46% max versus `torch.cat`. |
| Qwen gated RMSNorm | 40/40 width-128/256 screened shapes; 14/14 live routed cases plus FP32-weight and missing-export Triton fallback cases | Every screened shape exceeded 5%; width-128 stable median speedup 74.62%--81.06%, width-256 sequential observation 77.45%--85.39%. Exact Qwen3.6-35B M=512 improved 80.43%. |
| AITER SiLU screening | Accuracy passed | Rejected: 40.58%--96.17% slower than current LightOp and 5.20%--26.18% slower than vLLM native. |
| AITER `tgemm` screening | 36/36 exact comparisons | Rejected: 0/36 shapes exceeded 5%; speedup -27.51% min, -0.12% median, 0.46% max. |
| AITER WO_A batched BF16 GEMM | 6/6 live-HCU cases against FP32 (`rtol=0.02,atol=0.05`) | Rejected: 0/28 TP/decode shapes exceeded 5%; complete candidate latency was 2.04x--5.24x the existing einsum. Screening used `rtol=0.02,atol=0.5`, with maximum absolute FP32 difference 0.9863. |

Raw reports:

- `/tmp/vllm-hcu-sqrtsoftplus-benchmark.json`
- `/tmp/vllm-hcu-w16a16-benchmark.json`
- `/tmp/vllm-hcu-mla-decode-cat-benchmark.json`
- `/tmp/vllm-hcu-aiter-silu-benchmark.json`
- `/tmp/vllm-hcu-aiter-tgemm-benchmark.json`
- `/tmp/vllm-hcu-aiter-batched-gemm-bf16.json`
- `/tmp/vllm-hcu-norm-candidates-hcu5.json`
- `/tmp/lightop-layer-norm-fwd-qwen-exact-hcu4.json`
- `/tmp/lightop-layer-norm-fwd-qwen-token-heads-hcu4.json`
- `/tmp/lightop-layer-norm-fwd-qwen-width256-hcu3.json`
- `/tmp/vllm-hcu-lightop-topk-softmax-screening.json`
- `/tmp/fused_recurrent_aiter_hcu4_0457572.json`
- `/tmp/vllm-hcu-aiter-mxfp4-hcu7.json`

## Model-level acceptance

The committed profiles run feature-off and feature-on servers separately,
require exactly 32 HumanEval predictions and 32 reviews, require feature-on
Pass@1 not to regress, and inspect only the fresh log bytes from each server
invocation. Positive-coverage feature-on profiles must contain their exact
LightOp route marker and feature-off must not contain it; intentional fallback
controls require the marker to remain absent in both profiles. Output TPS is
recorded as an observation;
operator acceptance is based on the isolated multi-shape benchmarks above so
shared-host load and generation-length variance cannot create a false gate.

Local results:

- DeepSeek-V4-Flash Channel-INT8 TP4 completed both profiles. Feature-off and
  feature-on each produced and reviewed exactly 32 samples with Pass@1 1.0;
  observed output throughput was 7.12 and 7.20 tok/s respectively. Only the
  fresh feature-on log contains `Using LightOp sqrt-softplus MoE routing.`
- Qwen3.6-35B-A3B TP1 completed both profiles with 32 predictions/reviews and
  Pass@1 1.0 after the exact captured-class fix. The final observations were
  14.93 tok/s off and 16.86 tok/s on. Only the fresh feature-on invocation
  contains both `Using LightOp W16A16 Marlin MoE backend.` and
  `Using LightOp Qwen gated RMSNorm.`; the paired pytest acceptance completed
  successfully in 992.45 seconds.
- Qwen3.6-27B TP1 completed both profiles with 32 predictions/reviews and
  Pass@1 0.875. The latest pre-binding observations were 13.23 tok/s off and
  19.20 tok/s on. As above, these runs establish model accuracy but not the
  post-fix gated-route marker.
- Qwen3.5-35B-A3B-W8A8 TP1 completed both profiles with 32
  predictions/reviews and Pass@1 1.0; observed throughput was 7.14 and 11.05
  tok/s. Its checkpoint stores `linear_attn.norm.weight` as FP32, so both runs
  correctly retained the canonical Triton implementation under the strict
  BF16 contract. This is an intentional fallback control.

When the HCU was released, the final Qwen3.6-35B-A3B run supplemented the
14/14 live-HCU class/registered-op accuracy matrix, live FP32-weight and
missing-export fallback cases, the 40-shape accuracy/performance matrix, and
the worker cold-import binding test with real-model route evidence. The
fixed-size Torch fallback used when the pinned vLLM MoE-align ABI is absent
also completed a live-HCU graph capture and replay test.

## Additional `/models` coverage inventory

The local checkpoint scan found one additional positive-coverage model and
several useful negative controls:

- `/models/DeepSeek-V4-Flash-0731-Channel-FP8-w8a8` has the same
  `DeepseekV4ForCausalLM`, 256-expert sqrt-softplus router and `512+64` MLA
  geometry as the validated INT8 checkpoint. It can exercise the accepted
  router and decode-concat routes, and its `G=8,R=1024,K=4096` WO_A dimensions
  are included in the rejected batched-GEMM operator matrix.
- `/models/Qwen3.5-35B-A3B-W8A8` has the Qwen `E=256,K=2048,N=512,top-k=8`
  expert geometry but quantized weights and an FP32 gated-norm weight. It is a
  fallback/non-activation control for both strict BF16 routes.
- Qwen3.6-27B supplies positive BF16 width-128 gated-normalization geometry,
  while the two HY-V4 checkpoints have `K=6144,N=2048`, Kimi-K2.6 MLA uses
  `128+64`, and Qwen3-VL-8B/Gemma4 are different architectures.

The user allowed operator-level accuracy when model coverage is incomplete;
the exact positive operator shapes are covered by the live-HCU matrices for
additional checkpoints that were not rerun after the final logging fix.

## Reproduction

```bash
HIP_VISIBLE_DEVICES=4 python3 -m pytest -s -q \
  tests/integration/server/test_evalscope_operator_adaptation_humaneval.py::test_qwen36_35b_a3b_operator_adaptation_humaneval32

HIP_VISIBLE_DEVICES=4,5,6,7 python3 -m pytest -s -q \
  tests/integration/server/test_evalscope_operator_adaptation_humaneval.py::test_deepseek_v4_int8_operator_adaptation_humaneval32
```

The model and YAML paths can be overridden with the environment variables
documented in `tests/integration/server/README.md`.
