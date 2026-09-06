# SGLang operator adaptation validation for vLLM 0.25.1

## Scope

This change compares `HYGON-AI/sglang-das` at
`0457572a14edd55e05220726ff4a8af8b1931771` with the vLLM HCU plugin based on
`origin/v0.25.1` at `88f8a07`. It adapts only operators with an exact vLLM
0.25.1 ownership seam, validates the installed LightOp/AITER ABI, and keeps
all HCU routes subordinate to `VLLM_HCU_USE_CUSTOM_OPS`.

The test host used BW1100 (`gfx938:sramecc+:xnack-`), PyTorch 2.11.0, vLLM
0.25.1, LightOp 0.6.0, and AITER
`0.1.5+das185.dtk2604.torch2110.2608180853.g40a705`. Raw benchmark JSON is
written under `/tmp`; it is intentionally not committed.

## Disposition

| Candidate | Result | Production action |
|---|---|---|
| LightOp `moe_fused_gate_sqrtsoftplus` | Accepted | Added the non-hash DeepSeek-V4 router route with master/leaf gates and official hash fallback. |
| LightOp W16A16 Marlin MoE | Accepted | Added a TP-only, exact-shape unquantized oracle backend with one-time packed weights. |
| LightOp `ds_cat` mode 0 | Accepted | Replaced the exact FlashMLA decode concatenation seam, with `torch.cat` fallback. |
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

## Model-level acceptance

The committed profiles run feature-off and feature-on servers separately,
require exactly 32 HumanEval predictions and 32 reviews, require feature-on
Pass@1 not to regress, and inspect only the fresh log bytes from each server
invocation. Feature-on must contain its exact LightOp route marker and
feature-off must not contain it. Output TPS is recorded as an observation;
operator acceptance is based on the isolated multi-shape benchmarks above so
shared-host load and generation-length variance cannot create a false gate.

Local results:

- Qwen3.6-35B-A3B TP1 feature-off: HumanEval 32/32, Pass@1 1.0, observed
  11.82 output tok/s.
- Qwen3.6-35B-A3B TP1 feature-on: the fresh log contains
  `Using LightOp W16A16 Marlin MoE backend.`, HumanEval 32/32, Pass@1 1.0,
  observed 5.71 output tok/s. The host had concurrent accelerator load and an
  earlier unchanged Triton rerun varied from 11.82 to 7.96 tok/s, so these
  non-isolated end-to-end observations are not used to override the stable
  per-operator result.
- DeepSeek-V4-Flash Channel-INT8 TP4 loaded the 274.13 GiB checkpoint and
  reached a healthy server. HumanEval was stopped during its first prediction
  because concurrent load held all four selected devices busy and generation
  fluctuated around 0.1--0.8 tok/s. No DeepSeek model-level accuracy claim is
  made from that incomplete run; acceptance relies on the 75/75 router
  accuracy matrix, as allowed when a local model run cannot provide clean
  coverage.

## Additional `/models` coverage inventory

The local checkpoint scan found one additional positive-coverage model and
several useful negative controls:

- `/models/DeepSeek-V4-Flash-0731-Channel-FP8-w8a8` has the same
  `DeepseekV4ForCausalLM`, 256-expert sqrt-softplus router and `512+64` MLA
  geometry as the validated INT8 checkpoint. It can exercise the accepted
  router and decode-concat routes, and its `G=8,R=1024,K=4096` WO_A dimensions
  are included in the rejected batched-GEMM operator matrix.
- `/models/Qwen3.5-35B-A3B-W8A8` has the Qwen `E=256,K=2048,N=512,top-k=8`
  expert geometry but quantized weights, so it is a fallback/non-activation
  control rather than positive coverage for the unquantized W16A16 backend.
- The two HY-V4 checkpoints have `K=6144,N=2048`; Kimi-K2.6 MLA uses
  `128+64`; Qwen3.6-27B, Qwen3-VL-8B and Gemma4 are dense or different
  architectures. They do not satisfy the accepted routes' exact eligibility
  contracts and are therefore useful only for proving non-activation.

No extra large-model run is claimed from this inventory. The user allowed
operator-level accuracy when model coverage is incomplete, and the exact
positive operator shapes are already covered by the live-HCU matrices.

## Reproduction

```bash
HIP_VISIBLE_DEVICES=4 python3 -m pytest -s -q \
  tests/integration/server/test_evalscope_operator_adaptation_humaneval.py::test_qwen36_35b_a3b_operator_adaptation_humaneval32

HIP_VISIBLE_DEVICES=4,5,6,7 python3 -m pytest -s -q \
  tests/integration/server/test_evalscope_operator_adaptation_humaneval.py::test_deepseek_v4_int8_operator_adaptation_humaneval32
```

The model and YAML paths can be overridden with the environment variables
documented in `tests/integration/server/README.md`.
