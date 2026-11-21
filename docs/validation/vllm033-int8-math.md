# v0.33 INT8 math diagnosis and ASM follow-up

Production fix: `529855a5`, on existing [PR330](https://github.com/HYGON-AI/vllm-plugin-das/pull/330),
target `v0.33.0`. The [candidate JSON](../../validation/v0.33.0/int8-math-gfx936-candidate.json)
retains the distinct wheel, raw-evidence hashes, before/after tests and limitations.
This follows the [original review](vllm033-review.md); its former ASM numerical
failure is preserved as a historical observation. AI assistance was used.

## Numerical diagnosis

The original sixteen EP cases used global512/local256 experts, BF16 activations,
INT8 per-channel weights, dynamic per-token activations, linear/round-robin maps,
both logical ranks, and M16/M128. Every M128 case failed the third changed-input
Graph replay's original max-absolute-error limit of0.001. Separate same-input
Graph/eager results were already bit-identical, so Graph storage did not explain
that vendor/reference discrepancy.

Two stage semantics differ in this installed DTK26.10 vendor stack:

- Official per-token INT8 uses `round`; vendor ASM and vendor Triton share a
  `nearbyint` quantizer and a different scale-reciprocal association. An original
  failing input contains exact normalized0.5, quantized to1 by official and0 by
  vendor. Quantizer outputs and scales were compared directly on identical inputs.
- Official BF16 SwiGLU rounds SiLU to BF16 before multiplying by up. The vendor
  kernel rounds only after the product. Independent FP64 single-/double-round
  references distinguish them. In one shared-input example, official matches the
  double-round reference exactly; vendor differs at68,237 values. The vendor
  single-round result may be closer to the unrounded mathematical expression;
  that does not waive the requested upstream compatibility gate.

External diagnostic replacements compare all four combinations on the same eight
M128 cases and four input waves. They do not edit installed files or ship a
production alias replacement:

| Quantization / activation | Strict passes | Worst absolute error |
| --- | --- | --- |
| original vendor / original vendor | 24/32 | 0.001709 |
| official / original vendor | 22/32 | 0.001709 |
| original vendor / official | 24/32 | 0.001907 |
| official / official | 32/32 | 0.000244 |

A stage capture additionally compares the first GEMM to FP64 on selected routes.
With shared official quantization, vendor Triton's first GEMM equals official in
the inspected case; ASM has three differing intermediate values, up to0.003906,
and two sampled differing routes favor official against FP64. These are bounded
stage observations, not a proof about every model or vendor kernel. Matching both
quantization and activation is sufficient for the exercised strict output gate.

## Production implementation and verification

The Plugin composes public AITER `moe_sorting_fwd` and two `asm_fmoe_a8` launches
with official per-token quantization, the official activation native op and
official reduction. Vendor configuration still selects both ASM solutions and
the sorting block. Remote routes are zeroed before activation/reduction. No Core
runtime or Adapter change, global rebinding, copied generic runner or vendor-file
edit is introduced. The framework-owned output destination is still copied by
the existing expert wrapper.

Missing stage APIs, padded layouts, unsupported scales/dtypes/activation/storage
and the native32-bit workspace bound fall back before native launch. The fallback
retains the original canonical expert map and unchanged weights/scales. Failures
after a native launch propagate. CPU tests cover these boundaries, including a
large metadata-only workspace, while actual device tests verify kernel execution.

- Four new M128 installed regressions fail against the previous wheel, at the
  original strict limit. The failed log is retained.
- Full portable suite: **253 passed,66 skipped**. This includes the focused43
  CPU stage/fallback/provider/oracle tests. Native tests remain skipped there.
- Rebuilt installed wheel: **11 passed**,14 retained deprecation warnings. Three
  non-EP shapes M1/16/128 and eight EP cases pass. All24 changed-input EP Graph
  replays compare to a fresh official reference and same-input eager output;
  addresses remain fixed. Each case observes16 Python native-GEMM launches
  across eager/warmup/capture/checks, with both real ASM stages executed.
  Maximum absolute error over the EP eager/replay checks is **0.000244**.
- All14 packaged Python files match source, wheel and both isolated installation
  environments. The previous full hybrid profile's Flash file remains identical.

The fixtures are synthetic fixed-seed operator weights, not an INT8 model
checkpoint. Thresholds remain min-reference-RMS0.001, NMAE/NRMSE0.03 and maximum
absolute error0.001. No acceptance tolerance or input range was relaxed.

## Performance and remaining gates

Single-device warmed Graph microbenchmark, three samples of100 replays per arm,
same weights/input and BF16 output; values are medians in microseconds:

| M | Local EP map | Official Triton | Plugin ASM | ASM / official latency |
| --- | --- | --- | --- | --- |
| 1 | none | 52.77 | 59.20 | 1.122 |
| 1 | round-robin rank0 | 53.27 | 83.06 | 1.559 |
| 16 | none | 456.25 | 409.05 | 0.897 |
| 16 | round-robin rank0 | 250.84 | 270.45 | 1.078 |
| 128 | none | 965.34 | 872.08 | 0.903 |
| 128 | round-robin rank0 | 944.57 | 868.43 | 0.919 |

Results are mixed. Explicit opt-in remains; there is no universal or model
speedup claim. This is not distributed EP, complete W8A8 model, arbitrary
padded/shuffled layouts, other architectures, FP16 numerical or full topology
acceptance. Vendor Triton/CK/MoE-C retain their previous pipeline; the diagnostic
counterfactual does not certify those production paths. Previous vendor-Triton
strict failures remain open. Existing E5M2 and Qwen3.5 BF16 HumanEval quality gates
also remain open, so **PR330 stays draft and unmerged; last_good=false**.

Initial tool/environment failures are preserved separately: one uncontrolled
legacy-platform diagnostic was repeated with explicit Adapter selection and a
platform assertion; missing source/import/target-root settings and the unavailable
`build` frontend were corrected before final tests and the normal PEP517 wheel.
All owned probes/tests/benchmarks are terminal, prior owned ports remain closed,
and all eight cards return to about2MiB. GPU5/6 and unrelated services were not
modified; no device resets.
