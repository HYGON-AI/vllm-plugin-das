# HY4 SlimQuant W4A8 PCP

The SlimQuant W4A8 MoE method now has a fixed DeepEP high-throughput route
for `HYV4ForCausalLM`, PCP > 1, DP = 1 and expert parallelism enabled.
Use `--all2all-backend deepep_high_throughput --moe-backend deep_gemm`
(or MoE backend `auto`). Keep the existing HY4 PCP attention/runner setup.
The original TP1/PCP8 launch does not need a DP argument or `deepep_auto`.
Existing PCP topology restrictions still apply.

This route uses the existing HT prepare/finalize communicator and contiguous
HIPC W4A8 experts. Weights are packed in place using the existing fixed-HT
weight owner, including reload handling. Dynamic per-token INT8 scales,
expert maps and router weights retain the modular kernel contract. HY4's
positive `swiglu_limit` is passed to the clamp-and-quantize activation.
The DP+EP `deepep_auto` route and pure TP AITER/Triton routes are unchanged.

The server's HCU libraries must export:

- `deepgemm.pack_w4a8_moe_hipc_weight`
- `deepgemm.m_grouped_w4a8_gemm_nt_contiguous_hipc`
- `lightop.fuse_silu_mul_clamp_quant` for a positive SwiGLU limit,
  otherwise `lightop.activation.fuse_silu_mul_quant`

Fixed HT does not require the LL masked GEMM, N32 view or LL activation op.
Unsupported quantization metadata still fails before kernel construction.

## Optional HYV4 INT8 activation reuse

HYV4 Channel-INT4 checkpoints use dynamic INT8 activations for MLA and expert
projections. Enable the reviewed activation-quantization reuse path with:

```bash
export VLLM_HCU_USE_CUSTOM_OPS=1
export VLLM_HCU_USE_FUSED_RMS_QUANT=1
export VLLM_HCU_USE_FUSED_SILU_MUL_QUANT=1
```

One quantized input is shared by q-a and kv-a projections, the normalized
Q-LoRA input is quantized once for q-b and the sparse indexer, and pure-TP MoE
paths can share one input pair with routed and shared experts. DeepEP keeps its
routed dispatch in BF16; supported overlap modes reuse the pair only for the
shared experts. Dense feed-forward layers also fuse SiLU/multiply with the
down-projection quantization. The optimization activates only for INT8
consumers; launches without the switches retain their existing path.

## Verification status

Regression coverage includes HT route selection/rejection, required operator
checks, quant-method load/apply routing, fixed HT factory construction without
an LL interface, and SwiGLU clamp propagation. Existing tests cover fixed HT
in-place packing/reload and contiguous W4A8 GEMMs with expert maps/scales.

Final validation ran 2,574 portable contract tests and 372 focused HYV4,
DeepEP, and quantized-GEMM tests. An HCU DP8/EP8/MTP3 DeepEP low-latency
service using FP8 E4M3 KV cache and FULL_AND_PIECEWISE graphs completed all 64
HumanEval cases at pass@1 = 1.0. The controlled input-1/output-1024 benchmark
also confirmed that shared-only reuse does not increase steady per-token
latency; full-run output throughput remains sensitive to a small number of DP
tail requests.

Run in the matching vLLM/HCU environment after installing the updated plugin:

```bash
python -m pytest -q tests/runtime_patch/test_quant_gemm_aiter.py \
  tests/runtime_patch/test_moe_deepep.py \
  tests/runtime_patch/test_deep_gemm_utils.py -k 'w4a8'
```

Then run the original W4A8 PCP8 service command. The selected route logs
`Using HYV4 PCP DeepEP HT W4A8 MoE kernel.` Check short generation, long
prefill, repeated prompts with prefix-cache hits and concurrent short/long
requests. Compare deterministic outputs against a working W4A8 deployment
of the same checkpoint; successful startup alone does not establish numerical
correctness.
