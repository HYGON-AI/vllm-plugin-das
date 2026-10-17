# AITER CK GDN prefill validation

## Scope and environment

This validation covers the fused `aiter.ck_gdn_prefill` route added at
`ChunkGatedDeltaRule.forward_native`. It is controlled by the existing
`VLLM_HCU_USE_CUSTOM_OPS` master switch and the existing `hcu-aiter` prefill
provider selection. Explicit `triton`, explicit `hcu-boltops`, and a disabled
master switch retain the original path.

Operator measurements were taken on a BW HCU with PyTorch 2.11.0,
vLLM `0.28.1+dtk2604.torch2110.2609171627.g77acaf`, AITER
`0.1.7+dtk2604.torch2110.2610161749.g722998`, and BoltOps
`0.1.0+dtk2604.torch2110.2610131506.g1f36f1`.

## Supported contract

- BF16/FP16 packed activations with FP32 gates, beta, and state.
- Key and value dimensions equal to 128.
- Local value/key head ratios 1, 2, and 4. Ratio 3 uses the original path.
- Fixed single-sequence graph capture omits `cu_seqlens`; replay after in-place
  input changes is exact to eager CK.
- Packed variable-length eager execution is supported. Packed variable-length
  graph capture falls back because AITER 0.1.7 reads its metadata on the host.
- AITER 0.1.7 cannot marshal its empty final-state result when
  `output_final_state=False`; the integration requests the state and discards it
  so the public vLLM result remains `(output, None)`.
- Workspace comes from vLLM's active `WorkspaceManager` lane. A locked manager
  that is too small raises before launch rather than allocating during capture.

## Native correctness and graph validation

The native suite compares the patched public call against official FLA for
lengths 1, 16, 63, 64, 65, 127, 128, 256, 512, 1024, 2048, and 4096; ratios
1/2/4; fixed and packed-varlen inputs; zero and nonzero initial state; both
final-state modes; ratio-3 fallback; and graph replay. Acceptance is finite
output, max absolute error at most 0.01, and NRMSE at most 0.01.

```bash
HIP_VISIBLE_DEVICES=0 VLLM_HCU_USE_CUSTOM_OPS=1 \
  pytest -q tests/kernels/test_aiter_ck_gdn_prefill.py -m hcu
```

The repository has no registered `native` or `graph` markers, so graph-only
coverage is selected by name:

```bash
HIP_VISIBLE_DEVICES=0 VLLM_HCU_USE_CUSTOM_OPS=1 \
  pytest -q tests/kernels/test_aiter_ck_gdn_prefill.py -m hcu -k graph
```

## Operator performance

The command below generated
`/data/models/aiter_ck_gdn_operator_results.json`. Each table value is the
median of the per-seed P50 values for seeds 7, 17, and 29; each seed used three
timed repetitions after warmup. Cold time is recorded separately in the JSON.

```bash
HIP_VISIBLE_DEVICES=0 VLLM_HCU_USE_CUSTOM_OPS=1 \
  python tests/kernels/benchmark_aiter_ck_gdn_prefill.py \
  --lengths 1,64,65,128,512,1024,2048,4096 --repetitions 3 \
  --json-out /data/models/aiter_ck_gdn_operator_results.json
```

| Tokens | Official P50 ms | Mixed P50 ms | CK P50 ms | Mixed / CK | Workspace bytes |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.911 | 0.630 | 0.208 | 3.03x | 1,367,808 |
| 64 | 0.920 | 0.624 | 0.235 | 2.65x | 1,821,184 |
| 65 | 0.915 | 0.623 | 0.240 | 2.60x | 2,090,752 |
| 128 | 0.918 | 0.625 | 0.246 | 2.55x | 2,544,128 |
| 512 | 0.917 | 0.622 | 0.252 | 2.47x | 6,881,792 |
| 1024 | 0.912 | 0.630 | 0.288 | 2.19x | 12,665,344 |
| 2048 | 0.903 | 0.687 | 0.363 | 1.89x | 24,232,704 |
| 4096 | 1.006 | 0.793 | 0.487 | 1.63x | 47,367,424 |

All providers produced finite output. Across the matrix, CK output NRMSE versus
official FLA was at most 0.00467. These are isolated eager operator numbers;
the TP4 `FULL_AND_PIECEWISE` model comparison is recorded below after the
isolated-wheel validation is complete.

## Qwen3.5-35B TP4 model validation

Pending isolated baseline/candidate wheel runs against
`/model/Qwen3.5-35B-A3B`.
