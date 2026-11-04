# BF16 DeepEP auto with DeepGEMM (work in progress)

Qwen3.5-122B-A10B is unquantized BF16. The original unquantized MoE oracle
rejects `--moe-backend deep_gemm`; the existing HCU auto experts accept only
channel FP8/INT8. This implementation adds an explicit BF16 selection for
`--all2all-backend deepep_auto --moe-backend deep_gemm`.

The HT path calls `deepgemm.m_grouped_bf16_gemm_nt_contiguous` with canonical
weights. The LL path calls `deepgemm.m_grouped_bf16_gemm_nt_masked` with HIPC
packed weights and MODE 2000. GEMMs operate on BF16 weights/activations.
BF16 scatter uses the generic Triton copy kernel because LightOp's fused
quantized scatter accepts only byte activations. Completely padded HT tiles
use a valid expert index because this BF16 scheduler does not skip -1 tiles.
The inverse permutation excludes these padded results from the output.

Supported configurations currently require BF16, SiLU gate/up, dimensions
that are multiples of 256, DeepEP auto, no bias/LoRA, and output router
weighting. Unsupported explicit requests fail during backend selection.

LL needs a separate packed weight copy, created during weight loading and
included in memory profiling. For Qwen3.5-122B-A10B with DP4/EP4, the extra
48-layer MoE weight storage is approximately 54 GiB per rank. This memory
cost and the original 180000 context configuration require model validation.
Weight updates refresh the packed copy while retaining its storage address.

## Validation on 2026-11-04

- 69 runtime tests passed across `test_deepep_bf16_moe.py`,
  `test_lightop_w16a16_moe.py`, and `test_deepep_exact_ht_layout.py`.
- Real BF16 HT and LL GEMM probes (E=2, N=512, K=512) matched the BF16 PyTorch
  reference exactly on valid rows.
- Two full expert pipeline cases passed (E=4, hidden=512, intermediate=256,
  tokens=1/17, top-k=2), exercising HT -> LL -> HT, invalid routes, an empty
  expert, router weighting, and output/workspace aliasing.
- The hidden=3072/intermediate=1024 cases could not run: GPU allocation failed
  with no available memory while physical GPUs 4-7 were externally occupied
  at approximately 142757 MiB each. Those failures are not accuracy passes.
- Distributed DeepEP auto dispatch/combine, full model startup, graphs, BF16
  DeepGEMM HumanEval32, and AITER comparison remain unverified.

A previous, separate eager Triton + DeepEP high-throughput run achieved
HumanEval pass@1 32/32. It does **not** validate this DeepGEMM implementation
or the original AITER + auto configuration.

Reproduce the numerical cases on a free HCU:

```bash
HIP_VISIBLE_DEVICES=4 PYTHONPATH="$PWD" pytest -q tests/accuracy/test_deepep_bf16_moe_accuracy.py
```

Use the repository source and matching built extension when testing a server.
The validation should retain explicit backend logging and verify both HT and LL
selection before collecting model evaluation results.
