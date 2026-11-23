# DSV4.1 W4A8 live-row INT8 activation

The earlier FP8 activation experiment's 2048+256 column decomposition is
reused for the W4A8 routed expert width2304. Quantization is rewritten to
preserve the native clamped SiLU's two BF16 rounding boundaries, INT8 RNE
codes and FP32 token-scale bits. Contiguous execution skips expert-alignment
padding using native row IDs. Masked execution uses current device counts;
expected_m only controls scheduling and never truncates active rows.

Admission requires the verified LightOp source and binary, gfx938 with64CUs,
H5120, gate/up width4608, clamp10 and either16 contiguous experts or16/48
masked experts with fixed256-per-rank receive capacity on8ranks. Unknown
contracts retain the native operator. The DSpark AG/RS backend, target
communication, capacity, request limits and graph buckets are unchanged.

GPU operator evidence on nmz18 (2026-11-19):

- 75 cases passed byte equality for active INT8 codes and FP32 scales,
  input immutability, zero/tiny/saturated/random inputs, all finite BF16 gate
  bit patterns, changing graph counts through2048 and actual packed W4A8
  down-projection consumption.
- Contiguous16-expert median147us ->19.49/21.06/29.15us for8/32/64
  live rows per expert-alignment segment.
- Masked16-expert10.72/25.25/44.51us ->6.54/14.02/24.32us.
- Masked48-expert21.63/61.56/111.18us ->12.93/35.62/65.22us.

Each latency is a CUDA-event median over80 graph replays with10 launches per
replay. These are operator measurements, not an end-to-end performance claim.
Evidence: `run/dsv4.1/033/w4a8_reverify_20261119/nmz18/moe_reuse_probes_20261119_v3/clamped_int8.json`.

The first contiguous experiment redundantly visited later rows and failed its
latency gate; commit2435071 fixed the launch stride before the above rerun.
The published baseline remains Plugin526b626 + Adapter8a1c05c, full HumanEval
159/164. Serving HE32/full164 and paired performance are required for this new
integration before promotion. FULL_AND_PIECEWISE and the accepted scheduling
changes are retained.
