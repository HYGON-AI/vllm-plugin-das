# Channel-FP8 Q normalization producer

The owned `Fp4Attention` can produce one native `QuantizedActivation` for its
Q projection and optional indexer Q projection. Both linears retain the native
quantization scheme, checkpoint weights, scale layout, GEMM and output dtype.
The producer delegates to Adapter's BF16-boundary norm/quantization kernel.
Core attention flow, parallel fork/join, KV handling and other global symbols
are unchanged.

Admission requires TP/PCP/DCP=1, the exact owned attention/indexer classes,
matching native dynamic-token E4M3 quantizers and their public input capability.
Backends that own query normalization/RoPE, MXFP8, AITER block scales, INT8,
unknown providers and consumer drift retain the native producer. The backend
normalization contract is checked before either linear is rebound. Main and
indexer capabilities are inspected before either is
updated. BF16 Q is rounded before FP8 conversion; graph/compile quantizer
selection preserves the existing LightOp/upstream choice. No new launch flags
or environment variables are introduced.

Validation before full-model testing:

- 35 exact native norm/quantizer cases in the Adapter study.
- 18 actual Channel-FP8 Linear/projection consumer cases with checkpoint
  weights, empty rows, rows 1 through 257, with/without indexer and three
  changing Graph replays. Every output matches BF16 bytes exactly.
- 10 capability/fallback tests plus 33 existing DSpark Q/KV split tests pass.
- Adapter's 20 maintained kernel tests and L0 pass.

Full-model HE164, all-eight-rank trace and unprofiled serving must be measured
on these exact commits before claiming a serving performance improvement.
The shared artifact directory is `run/dsv4.1/033/fp8_optimization_20261123`.
