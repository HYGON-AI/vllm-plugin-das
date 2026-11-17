# DSV4.1 Channel-FP8 MoE with DeepGEMM HT/LL

This migration owns only the DSV4.1 Channel-FP8 quantization configuration's RoutedExperts method and its new expert/runtime helpers. Core vLLM, the external Adapter, the legacy `vllm_hcu` tree, the oracle/factory/global registries, Graph, Engram, PCP model files, and shared CT W4A8 files are outside the change.

## Failure contract, fixed before production implementation

1. The native vLLM DeepGEMM and BatchedDeepGEMM expert classes accept only block128 activation/weight keys and assert a block shape. Channel weight + token activation (`kFp8StaticChannelSym`, `kFp8DynamicTokenSym`) must select an owned expert class; opening a backend gate alone is insufficient.
2. Route only the exact DSV4.1 channel FP8 schema: FP8 weights, static per-output-channel scales, dynamic per-token FP8 activations, no weight block shape. Preserve `.wo_a`, ignored layers, non-MoE linear routing, and native non-channel schemes.
3. Preserve public CT construction and `RoutedExperts` parameter loaders. The owned method may reuse native weight creation and mapper/loader callbacks, then replace only the modular expert owner. It must not reconstruct Core constructor signatures or replace the CT factory.
4. HCU DeepGEMM HT uses contiguous packed FP8 `[E,N,K]` operands and channel scale tensors; LL uses masked packed FP8 operands and batched `[E,T,K]` activations with per-token scales and `expert_num_tokens`. Physical packed dimensions and logical unpadded N/K must be retained separately. A signature or dtype alone is not layout evidence.
5. Weight packing is one-time and layout-tagged. Repeated process with the same layout is idempotent only when weight/scale shape, stride, dtype, device, logical dimensions and backend layout agree. A conflicting layout or replaced storage fails closed. Unpacked or non-contiguous weights do not reach DeepGEMM.
6. HT route requires DeepEP high-throughput semantics and 256-token expert alignment; LL route requires DeepEP low-latency semantics, batched expert buffers, a valid `expert_num_tokens` vector, and masked GEMM ABI. Missing metadata, invalid expert maps, overlap between HT/LL requests, or an unknown ABI fails explicitly.
7. Activation preparation must validate actual request rows against padded workspace rows, preserve ragged/prefill distinctions, clamp only through the selected native-compatible SiLU fused contract, and reject unsupported overlap. Empty ranks return through the outer modular collectives without invoking a GEMM provider.
8. Nonempty execution must call the owned modular expert implementation. Provider, quantization and EP/GEMM doubles in CPU replay are explicit test doubles; replay is not GPU, DeepEP, multi-card, graph, numerical accuracy, or performance evidence.
9. Auto/explicit backend selection is validated after construction. Explicit `deep_gemm` must select owned HT or LL according to the actual parallel configuration; a mismatch fails. Unsupported backends do not silently fall back to channel DeepGEMM.
10. Registration and activation are idempotent and health-checked. A lost public owner or changed native method signature fails closed. No source mutation outside task-2 files and no pytest additions are permitted.

## Fixed CPU replay scope

The replay imports the production plugin files from the current source tree and constructs a real owned Channel-FP8 config, a native/public `RoutedExperts` method path, and its modular expert owner. It uses real FP8 parameter creation and loader callbacks where the installed CPU source supports them. Hardware platform admission, FP8 quantization, DeepEP exchange, packing kernels and grouped GEMM are doubles with call records.

The preimplementation matrix planned 200 seeds, rectangular and square logical N/K, direct and merged expert shards, empty ranks, one token, padding and 256-row HT boundaries, ragged LL token counts, invalid metadata, non-contiguous/replaced storage, wrong scale layouts, repeated processing, explicit backend selection and rejected schemas. This is the planned failure matrix, not a claim that all those cases have executed. Actual coverage is listed below.

The replay does not import existing scripts that execute at module import, does not add or extend pytest tests, and writes only to `tests/vllm033/.cache/channel-fp8-moe/`.

## Evidence

All evidence is under `tests/vllm033/.cache/channel-fp8-moe/`.

- `restart-fixture/report.json`: 1/9; the real chain exposed the scale-layout flag mismatch. The owned quant config now explicitly specifies `is_scale_swizzled=False`.
- `restart-layout/report.json`: 6/9; all HT forwards passed. The LL CPU fixture was missing Core's platform-conditional `DEEPEP_QUANT_BLOCK_SHAPE` import. The replay now supplies that exact native constant alongside the native HT/LL classes; the factory function is not replaced.
- `restart-transport/report.json`: **9/9**, exit 0, `cuda_initialized=false`. One case runs 200 physical-layout/witness seeds across rank-6/rank-7 tensors; eight cases exercise real public construction, native CT parameter creation/loaders, owned method, `mk.FusedMoEKernel`, owned HT/LL receivers and experts, native finalize and `RoutedExperts.forward_modular`. HT rows are 0/1/255/256/257; LL rows are 0/1/16. Nonempty paths record two GEMMs and one activation. Empty HT retains dispatch/combine without GEMM; fixed-capacity LL retains masked GEMMs and combine even when the local input is empty, because another rank may supply work. The blanket empty-rank rule in planned contract item 7 therefore applies to empty received HT tensors, not to LL local request count.
- Maximum absolute error against the independent token/expert reference is **2.384185791015625e-7**. The provisional draft's absolute tolerance of 0.02 could admit zero results at these scales; it was tightened before the first green chain run to `rtol=0.02, atol=1e-7`, with nonzero reference amplitude and a normalized maximum-error bound below 0.02. This is a CPU source differential, not an FP8 checkpoint or GPU accuracy result.
- The replay fixture's routing import and HCU LL positional signature were corrected. EP size is two with a real native canonical expert map; transport is still a one-process double. No pytest files were added or modified.
- Independent read-only review found the scale-layout flag issue and a conditional HT double application of routing weights. Both were corrected: the quant config explicitly chooses continuous scales, and HT gather uses unit weights when native prepare has already weighted the input. No additional concrete issue was reported for the ordinary BF16/gated-SiLU path. The conditional preweighted-input fix has static review evidence; the current forward matrix uses the ordinary output-weighted mode.
- Existing complete `tests/vllm033` regression: **311 passed, 1 failed, 66 skipped, 14 warnings in 166.87 s**, exit 1. The failure is `test_matching_layers_delegate_to_official_ct_channel_scheme[experts]`: its `RoutedExperts.__new__` fixture omits checkpoint projection names and expects every expert method to delegate to the native CT factory. That prior expectation conflicts with the newly owned Channel-FP8 method. It was not bypassed, deleted, weakened, or rewritten after implementation. `regression.log`, `regression.xml` and `run_regression.sh` preserve the failure and reproduction command.
- Five changed/new Python files passed AST parsing; the runner passed Bash syntax and tracked changes passed `git diff --check`.

## Implementation and routing

The existing public quantization owner remains `hcu_dsv41_channel_fp8`, registered by the existing DSV4.1 activation entry. Exact channel/token FP8 expert schemes select `HcuDSV41ChannelFP8MoEMethod` before the native block-only backend oracle. Native CT parameter creation and public loaders remain inherited. The method composes an owned prepare and owned experts directly with `mk.FusedMoEKernel`; no AITER backend alias, global factory/oracle rebinding, new registry entry, or global EP group replacement is used.

| Route | Activation and layout | GEMM/reduction |
| --- | --- | --- |
| DeepEP HT | Native BF16 dispatch, owned per-token receiver quant, fixed 256-row expert alignment using native scatter with `block_size=K`; FP32 `[M,1]` token scales | HCU `deepgemm` contiguous pack/GEMM, LightOp SiLU/FP8 quant, second GEMM, native gather and HT combine |
| DeepEP LL | Owned five-positional-argument HCU `quant_type=2` dispatch; native handle slots and finalize; FP32 contiguous `[E,T]` token scales | HCU masked pack/GEMM, LightOp EP activation/quant, second GEMM, native LL combine |

Provider rank-6 `[E,K/64,N/16,4,16,16]` and rank-7 `[E,K/64,N/32,2,4,16,16]` shapes are checked against independent logical dimensions. Packing runs once, with storage/offset/shape/stride/device/dtype witnesses for both weights and scales. Changed storage, inconsistent physical layout, missing ABI, unsupported backend or custom alpha/beta fails closed. EPLB is explicitly unsupported.

## Known limits and delivery state

- This local feature commit records the implementation and passing CPU source E2E with the existing portable-suite failure disclosed. It is a development snapshot, not a clean full-regression, GPU, Graph, or model-accuracy acceptance statement. No push or remote PR update is part of this change.
- Core and external Adapter tracked files remain unchanged. The existing container has installed 0.32 distribution metadata while source imports point at the selected 0.33 Core/Adapter/new plugin; this is not installed-wheel acceptance.
- Hardware admission, EP transport/events, packing, quantization, scatter/gather, DeepGEMM and LightOp kernels are CPU doubles. Real HCU provider packing, GPU kernels, eight-rank communication, PCP8/DP8 full models, checkpoint accuracy, Graph capture/replay and performance remain unverified. D-side Graph was outside this implementation.
- The current executed matrix does not establish every preimplementation planned refusal or mixed/merged-checkpoint case, ragged multi-rank LL counts, clamp numerics, custom routing-input weighting, 200-seed full forwards, LoRA, RL reloading, or other deployment combinations. The 200-seed result is physical-layout/witness coverage only.

Repeat the source E2E in the existing container with:

```bash
VLLM_TEST_PYTHON=/usr/local/bin/python3.12 \
  bash tests/vllm033/run_channel_fp8_moe_cpu.sh \
  "$PWD/tests/vllm033/.cache/channel-fp8-moe/reproduce"
```

`source-manifest.json` and `migration.patch` capture the candidate; the requested archive directory contains a report and copies of the final source-replay and portable-regression outputs.
