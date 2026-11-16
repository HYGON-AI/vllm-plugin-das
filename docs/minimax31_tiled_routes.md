# MiniMax tiled local-route pack/restore

The opt-in HCU_MINIMAX_HIPC_PACK_TILED=1 path tiles32 rows by256 columns in both pack and restore. It retains the existing DeepGEMM HIPC weight format, sort,128-row expert padding,32768-row GEMM chunks, native OAI and route-weight timing. Four-step upper-bound lookup handles duplicate expert starts from zero counts. Pointer products use64-bit indices. Restore keeps sequential FP32 top-k accumulation followed by one BF16 store. The existing row-wise pack remains the default fallback.

This is runtime activation movement. It does not repack weights, transpose scales or mutate received hidden states. Derive a real sidecar for changed runtime source guards. Reuse a producer only when the entire weight identity is naturally equal; never rewrite a key.

The optional HCU_MINIMAX_ROUTE_COUNTS_ANNOTATION=1 adds CPU record_function labels containing the16 host counts already returned by DeepEP. It introduces no GPU tensor readback or file IO, but is diagnostic instrumentation and may affect timing. It was disabled in all formal unprofiled samples.

Validation on2026-11-13:
- Synthetic and two real checkpoint expert seeds through both actual HIPC GEMMs/OAI were bit-exact against the old route path and met the unchanged independent FP64 gates.
- Counts changed across rows129–136 and32768 with one compiled variant per new kernel; the360000×6144 address case crossed2^31 and passed with two variants for the large allocation.
- A single controlled M32768/H6144 kernel test measured pack9.732→3.133ms and restore3.299→0.969ms. It excluded sort/GEMM/model overhead.
- Full60-layer strict IPC from the retained8 owners succeeded, startup profile33.83s,12/12 functional cases passed. All24 target output tokens exactly matched the untiled combination.
- Five matched unprofiled target samples had median4.420266s, range4.387660–4.457780s. Each matched suffix improved; paired median difference was−0.531460s versus untiled4.921495s.
- The subsequent8rank diagnostic trace measured4.442054s versus untiled4.136546s and historical SGLang2.475923s. It reduced rank0 pack/restore393.485→112.848ms, but notify wait and idle gaps increased. Extra count annotations differed between captures, so GPUspan improvement is not established and the remaining competition target is unmet.

The unchanged combination still differs from frozen baseline on3/24 random target tokens. Same-weight MSA-on/off controls restore two differences but leave other changes; they are not exact ties and are not waived. See the task report sol_joint_gap_20261113.md for exact margins, route statistics, teacher comparisons and source artifacts.

GPU resources were explicitly released after the diagnostic trace. Integration copied the two tested runtime files byte-for-byte from isolated_plugin_sol_hipc_tiled_stats_v1, without changing permanent weight packing or W8 math. No GPU work was run for integration.
