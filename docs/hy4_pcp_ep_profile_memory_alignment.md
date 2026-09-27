# Hy4 PCP + EP profile-memory alignment

## Scope and provenance

This change adapts the remaining PCP+EP profile-memory behavior from
`origin/v0.25.1@6ea7b12` commits `3f30e7c` and `2b7de5f` to the current MRV2
owners on `origin/v0.28.1-dev@0be6e56`. It does not port the v0.25.1 V1
communicator monkey patch or claim PCP CUDA Graph support.

The post-review implementation tested on hardware is commit `e97b794`. Its
isolated wheel is
`vllm_hcu-0.28.1rc1.dev491+das.e97b794.dtk2604-cp310-cp310-linux_x86_64.whl`,
SHA-256 `03e46f804fe99ade98492802cec365557a870f86e72be187fb9131648926e3d2`.

## Changes

- Fixed DeepEP high-throughput construction derives `num_nvl_bytes` from
  `VLLM_DEEPEP_BUFFER_SIZE_MB`; the auto manager's existing maximum-size rule
  remains unchanged.
- `HcuGPUModelRunnerV2.profile_run()` uses a conservative PCP rank-local
  bound that accounts for partitioned prefill, replicated decode, per-request
  layout headroom, and MTP decode width. It restores the configured value in
  `finally`; PCP1 delegates without mutation.

For global token budget `B`, PCP width `P`, maximum requests `N`, and decode
query width `Q`, the temporary profile budget is:

```text
partitioned = floor(ceil(B / P) * 9 / 8)
replicated  = min(B, N * max(Q, 2))
profile     = min(B, max(1, partitioned + replicated))
```

The partitioned term retains the 12.5% imbalance allowance. The replicated
term covers decode tokens that remain on every PCP rank and two possible
`DualChunkSwap` segments per request. The global cap is safe because dispatch
cannot exceed the admitted global token budget.

## Review regression and tests

The first implementation divided the complete budget by PCP width. That was
unsafe because `HcuPCPManager.get_num_tokens_for_dispatch()` partitions only
prefill; decode remains replicated. Tests first reproduced the reviewer's
three counterexamples with the real dispatch method:

| Configuration / admitted batch | Old profile | Actual rank | Fixed profile |
| --- | ---: | ---: | ---: |
| `B=4096,N=256,Q=3`; 255 decode requests plus one 3331-token prefill | 1152 | 1599 | 1920 |
| `B=4096,N=512,Q=3`; 512 decode requests | 1152 | 1536 | 2688 |
| `B=512,N=256,Q=1`; 256 decode requests | 144 | 256 | 512 |

The third case also asserts that `_dummy_run()` can create all 256 sampling
requests rather than being truncated to 144. PCP1 behavior and restoration
after an exception are covered separately.

The complete relevant selection passed **158 tests** with 14 existing
warnings:

```text
tests/runtime_patch/test_glm52_pcp_runner.py
tests/runtime_patch/test_worker_framework_opt.py
tests/runtime_patch/test_moe_deepep.py
```

A repository-wide bounded post-review run is intentionally reported
separately because the repository baseline is not green. With the frozen
v0.25.1 source root it stopped at the configured fifth failure with 727 passed
and 64 skipped. The failures were the known W4A8 shared-storage numerical
case, two clean-process tests polluted by that old-source-root environment,
the LightOp public-category audit, and the patch-module coverage audit; none
exercised either changed production file. The two clean-process cases passed
2/2 when rerun without the legacy source-root variable.

## Eight-card post-review service command

```bash
env -u VLLM_PLUGINS -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  -u http_proxy -u https_proxy -u all_proxy \
  VLLM_USE_V2_MODEL_RUNNER=1 VLLM_USE_NN=1 \
  VLLM_DEEPEP_BUFFER_SIZE_MB=256 \
  VLLM_PP_LAYER_PARTITION=41,37 \
  HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
  PYTHONNOUSERSITE=1 \
  VLLM_CACHE_ROOT=<isolated-cache> \
  PYTHONPATH=<isolated-plugin-wheel> \
  /usr/bin/python -m vllm.entrypoints.cli.main serve \
  /models/Hy4-preview-Channel-FP8-w8a8-v2 \
  --served-model-name hy4-pcp-ep-mtp2-review-fix \
  --host 127.0.0.1 --port 8015 --trust-remote-code \
  --tensor-parallel-size 1 --pipeline-parallel-size 2 \
  --prefill-context-parallel-size 4 --enable-expert-parallel \
  --moe-backend deep_gemm --all2all-backend deepep_high_throughput \
  --attention-backend FLASHMLA_SPARSE --kv-cache-dtype fp8_e4m3 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
  --enforce-eager --enable-prefix-caching --max-model-len 4096 \
  --block-size 64 --max-num-seqs 256 --max-num-batched-tokens 4096 \
  --default-chat-template-kwargs '{"reasoning_effort":"no_think"}' \
  --seed 0
```

The service constructed eight `HcuGPUModelRunnerV2` workers for
PP2/TP1/PCP4/EP4/MTP2, selected `DeepEPHTAll2AllManager` and the contiguous
DeepGEMM HT path, used FP8 E4M3 KV, and resolved CUDA Graph mode to `NONE` as
expected for this eager PCP topology. The fixed profile used 1920 tokens;
peak activation was 3.8 GiB on both stages, with 14.13 GiB PP0 and 18.91 GiB
PP1 available KV cache.

Post-review gates:

- HumanEval/0-7: **8/8**.
- 256-client decode stress: 256/256 HTTP 200, all `stop`, 0 preemptions,
  109.19 completion tok/s, 4.605 s p95 latency.
- Mixed high-concurrency gate: 255 active decode requests plus a delayed
  3000-token prefill request; 256/256 HTTP 200, all `stop`. The long prefill
  completed in 13.274 s and the service remained healthy.
- No Traceback, OOM, timeout, reallocation, or NCCL-timeout marker appeared.
  All eight cards returned to zero reported utilization and memory after
  shutdown.

## Historical low-concurrency A/B

Before the review fix, the same topology with `--max-num-seqs 16` showed PP0
peak activation 1.93 GiB versus 2.38 GiB on the unoptimized baseline, and PP0
available KV cache 17.75 GiB versus 15.67 GiB. Both variants passed
HumanEval/0-7 at 8/8 and 320/320 varied-length requests. This remains useful
evidence for the DeepEP/profile-memory direction, but it is not presented as
post-review-wheel A/B evidence and the small throughput delta is not a speedup
claim.
