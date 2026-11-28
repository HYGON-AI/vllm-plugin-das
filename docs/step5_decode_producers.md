# Step5 decode producer profile

This branch packages the highest measured end-to-end decode path from the
2026-11-04 comparison: Eager graph compilation with DeepEP low-latency transport,
channel-wise FP8 DeepGEMM experts and fused decode producers. The optimization
is opt-in; existing model forward paths remain the default.

## Code path

`Step5Model` carries the previous FFN output with its FP32 residual into the next
layer's input producer. That producer combines the residual, applies Gemma
RMSNorm, rounds the activation to BF16, and computes per-token FP8 values/scales
in one kernel. A second producer combines the attention output and residual
before the post-attention norm. The final FFN/residual addition and the model's
final BF16 narrowing remain at the model boundary.

The selected profile requires 92 layers, hidden size 4096, a BF16 compute dtype,
FP32 residual connections, the FP8 dense path, all model layers on the same
pipeline stage, and compilation backend `eager`. Fused kernels support at most
64 rows. Larger service-prefill batches retain the original library norm and
token-quantization fallback. The MTP decoder layers keep their existing forward
path. Dynamic small-M dispatch remains inside opaque operators so that tracing
a larger prefill batch does not remove the later decode kernels.

The residual sum is FP32. FP8 quantization operates on the rounded BF16 value,
using the dense path's original scale floor `1e-12` and rounded FP32 `/448`
constant. Component comparisons observed at most one BF16 step of normalized
output difference against the library reduction; the fused reduction is not
claimed bitwise identical for every possible input.

## Enabling the measured route

Use a matching HCU vLLM 0.28.1 / DTK 26.10 / gfx938 environment. The measured
DeepGEMM HCU build was based on source commit
`7da8cd9db6859d69e8a460cd0da2d9d6ffd65633`, with its gfx938 masked entry points
rebuilt as code-object V5. DeepEP must provide the layered-opt low-latency ABI
and per-device HCA mapping used by the plugin. A generic package with a similar
version string is not equivalent to those validated native extensions.

The producer-specific opt-in and its prerequisite compute flags are:

```bash
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_HCU_USE_CUSTOM_OPS=1
export VLLM_HCU_STEP5_FP8_DENSE=1
export VLLM_HCU_STEP5_FP8_BF16_MOE=1
export VLLM_HCU_USE_CUSTOM_GEMMA_RMS_NORM=1
export VLLM_HCU_TRITON_TOKEN_FP8_QUANT=1
export VLLM_HCU_STEP5_SWA_FUSED_QK=1
export VLLM_HCU_STEP5_FUSED_GATE=1
export VLLM_HCU_STEP5_QKV_INDEXER=1
export VLLM_HCU_STEP5_DSA_FUSED_QK=1
export VLLM_HCU_STEP5_DECODE_PRODUCERS=1
```

The measured serving configuration was TP4 / DP4 / EP16 across two eight-card
nodes, `--all2all-backend deepep_low_latency --moe-backend deep_gemm`, maximum
model length 70000, maximum batch tokens 512, prefill threshold 32, maximum
sequences 64, FP8 E4M3 KV, block size 64 and no prefix caching. Graph compilation:

```json
{"mode":3,"backend":"eager","use_inductor_graph_partition":false,"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["none","+swiglustep_and_mul","+GemmaRMSNorm"],"compile_sizes":[16,64]}
```

Here `eager` selects the FX compiler backend; CUDA Graph replay remains enabled.
Do not add `--enforce-eager` when reproducing this profile.

The profile used `VLLM_HCU_DEEPEP_AUTO_SHARED_WEIGHTS=1`,
`DEEPEP_ENABLE_LL_LAYERED_OPT=1`, `VLLM_DISABLE_SHARED_EXPERTS_STREAM=1`,
`HCU_STEP5_ROUTER_SHARED_MAX_M=64`, `HCU_STEP5_CONTEXT_GRAPH_REGIONS=8736`,
`VLLM_STEP4_DSA_GRAPH_REGIONS=8736`, `GPU_MAX_HW_QUEUES=16` and
`NCCL_MAX_NCHANNELS=8`. GPU/NIC ordering and the expert-placement asset must be
configured for the deployment; they are not embedded in this change.

## Recorded validation

| Measurement | Result |
| --- | --- |
| Common decode window, C64 / I65536 / O4096 | 2821.9351 output tokens/s |
| Output tokens in the common window | 256310 over 90.827744476 s |
| Common-window coverage | 97.8129% of eligible decode tokens |
| Preemptions during the benchmark | 0 |
| Minimum observed physical free memory | 14361 MiB |
| Fixed100 GSM8K, standard MTP3 verification | 97/100, no request errors |

The throughput run used MTP3 synthetic acceptance rates
`[0.838992, 0.598322, 0.362686]`, nominal available capacity 96 GiB per GPU and
GPU memory utilization 0.5. Synthetic acceptance is a benchmark control, not
standard serving verification. The separate quality run used standard MTP3,
temperature 1.0, top-p 0.95, seed 42 and an output limit of 32768; all 100
responses finished with `stop`.

GPU qualification also covered 432 producer cases on each of two GPUs,
FP32-residual bit equality, dynamic 512-to-64-to-16 dispatch and changed-input
graph replay. FP8 values/scales matched the original quantization formula
applied to the candidate's rounded BF16 activation. This is distinct from
claiming identical normalized activations across norm implementations. The
numbers above are recorded measurements of this specific environment, not a
claim of 3000 TPS.

This publication excludes the lower-throughput Inductor variant, the G2-only
grid candidate, numeric-pivot TopK, alternate Native chunks and private
profiling/controller helpers. Those experiments are retained locally.
