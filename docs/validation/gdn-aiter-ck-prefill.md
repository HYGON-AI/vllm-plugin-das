# AITER CK GDN prefill validation

## Scope and environment

This validation covers the fused `aiter.ck_gdn_prefill` route evaluated at
`ChunkGatedDeltaRule.forward_native`. Production routing is fail-closed because
the audited AITER 0.1.7 kernel produces non-finite results for legal Qwen GDN
inputs. The existing mixed AITER/BoltOps path remains active.

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
- Production model traffic does not enter CK prefill while the AITER 0.1.7
  strong-decay failure remains. Native CK tests explicitly bypass this safety
  gate to characterize the optional dependency.

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

All providers produced finite output for this original synthetic matrix, whose
log-decay values were limited to approximately -0.01 through -0.03. Across the
matrix, CK output NRMSE versus official FLA was at most 0.00467. The later
model failure showed that this input distribution was not sufficient for a
production-safety decision.

## Strong-decay and Qwen3.5-35B TP4 validation

Direct AITER probes with legal FP32 log-decay inputs found:

| Constant `g` | CK output finite | CK final state finite |
|---:|:---:|:---:|
| -1 | yes | yes |
| -2 | no | no |
| -5 | no | no |
| -10 | no | no |

Both `use_exp2=False` and the equivalent base-2 representation with
`use_exp2=True` fail. The result was reproduced with builds
`0.1.7+dtk2604.torch2110.2610161749.g722998` and
`0.1.7+dtk2604.torch2110.2610171925.ga76f3d`.

The isolated TP4 candidate at commit `8f9f4f25` completed target and speculator
`FULL_AND_PIECEWISE` capture. Its pre-reserved 8192-token workspace also
eliminated the earlier 71.37 MiB locked-workspace failure: correctness,
performance, 8192-token, and four-way concurrent requests all returned HTTP
200. However, model semantics failed. Unrelated prompts repeatedly generated
`duct Register Register ...`; the baseline produced prompt-dependent output.
The candidate also had zero initial MTP draft acceptance and substantially
worse TPOT. The same collapsed output reproduced with `--enforce-eager`, which
rules out graph capture and workspace sharing as the semantic root cause.

Artifacts are under:

- Baseline: `/data/models/validation/gdn-aiter-ck-prefill/68f20511`
- Candidate: `/data/models/validation/gdn-aiter-ck-prefill/8f9f4f25`

Consequently, operator speedups are informational only and CK fused prefill is
not enabled for production model traffic until AITER fixes the non-finite
strong-decay behavior and the complete TP4 validation passes again.

## Fail-closed candidate verification

Commit `b5d64fbe` was built as an isolated wheel with SHA256
`7e2efb80a565ced9fd9dd75171be5b442930b3581131a0a50cad0ab649704f92`.
The same TP4/EP4, MTP3, FP8 KV, 8192-token, `FULL_AND_PIECEWISE` configuration
logged the strong-decay safety reason and retained the existing mixed prefill
path. It completed all correctness, prefix-cache, sequential performance,
8192-token, and four-way concurrent requests with HTTP 200. Natural-language
sanity prompts produced prompt-dependent baseline output (`Paris`, `2`, and
`Green`) rather than the fused-CK collapse.

Median TTFT was 89.54 ms at 64 tokens, 199.87 ms at 1024, 398.50 ms at 4096,
and 426.50 ms at 8192. These are within normal run-to-run variation of the
baseline medians (91.42, 204.91, 400.38, and 426.85 ms respectively). Raw
artifacts are under
`/data/models/validation/gdn-aiter-ck-prefill/b5d64fbe`.
