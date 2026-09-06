# SGLang operator adaptation validation for vLLM 0.25.1

## Scope

This change compares `HYGON-AI/sglang-das` at
`0457572a14edd55e05220726ff4a8af8b1931771` with the vLLM HCU plugin based on
`origin/v0.25.1` at `88f8a07`. It adapts only operators with an exact vLLM
0.25.1 ownership seam, validates the installed LightOp/AITER ABI, and keeps
all HCU routes subordinate to `VLLM_HCU_USE_CUSTOM_OPS`.

The test host used BW1100 (`gfx938:sramecc+:xnack-`), PyTorch 2.11.0, vLLM
0.25.1, LightOp 0.6.0, and the installed HCU AITER package. Raw benchmark JSON
is written under `/tmp`; it is intentionally not committed.

## Disposition

| Candidate | Result | Production action |
|---|---|---|
| LightOp `moe_fused_gate_sqrtsoftplus` | Accepted | Added the non-hash DeepSeek-V4 router route with master/leaf gates and official hash fallback. |
| LightOp W16A16 Marlin MoE | Accepted | Added a TP-only, exact-shape unquantized oracle backend with one-time packed weights. |
| LightOp `ds_cat` mode 0 | Accepted | Replaced the exact FlashMLA decode concatenation seam, with `torch.cat` fallback. |
| AITER `silu_and_mul` | Performance-rejected | Kept the existing categorized LightOp implementation. |
| AITER `tgemm` | Performance-rejected | Kept the existing vLLM linear implementation. |
| EP `ep_build_m_indices` | Already covered | Existing LightOp `ep_scatter` produces indices and inverse permutation in one launch. |
| Fused RMS plus dynamic quant | Already covered | Existing public categorized LightOp route has the same contract and avoids private LMSlim APIs. |
| Generic RMS/RoPE/KV-store fusions | No vLLM seam | Not adapted; SGLang token-pool mutation is not ABI-compatible with vLLM paged KV cache ownership. |
| Newer AITER fused Qwen/gating symbols | Dependency unavailable | No dormant switch was added for symbols absent from the installed package. |

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

Raw reports:

- `/tmp/vllm-hcu-sqrtsoftplus-benchmark.json`
- `/tmp/vllm-hcu-w16a16-benchmark.json`
- `/tmp/vllm-hcu-mla-decode-cat-benchmark.json`
- `/tmp/vllm-hcu-aiter-silu-benchmark.json`
- `/tmp/vllm-hcu-aiter-tgemm-benchmark.json`

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

## Reproduction

```bash
HIP_VISIBLE_DEVICES=4 python3 -m pytest -s -q \
  tests/integration/server/test_evalscope_operator_adaptation_humaneval.py::test_qwen36_35b_a3b_operator_adaptation_humaneval32

HIP_VISIBLE_DEVICES=4,5,6,7 python3 -m pytest -s -q \
  tests/integration/server/test_evalscope_operator_adaptation_humaneval.py::test_deepseek_v4_int8_operator_adaptation_humaneval32
```

The model and YAML paths can be overridden with the environment variables
documented in `tests/integration/server/README.md`.
