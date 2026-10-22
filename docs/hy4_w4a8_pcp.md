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

## Verification status

Regression coverage includes HT route selection/rejection, required operator
checks, quant-method load/apply routing, fixed HT factory construction without
an LL interface, and SwiGLU clamp propagation. Existing tests cover fixed HT
in-place packing/reload and contiguous W4A8 GEMMs with expert maps/scales.

The development host has no available Python interpreter or HCU runtime;
these tests and the HY4 W4A8 PCP8 service have not been run for this change.
This implementation is not a new hardware validation result for the older
`hy4_v0251_clean_validation.md` report.

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
