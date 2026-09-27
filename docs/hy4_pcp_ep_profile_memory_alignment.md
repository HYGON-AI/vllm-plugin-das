# Hy4 PCP + EP profile-memory alignment

## Scope and provenance

This change adapts the remaining PCP+EP profile-memory behavior from
`origin/v0.25.1@6ea7b12` commits `3f30e7c` and `2b7de5f` to the current MRV2
owners on `origin/v0.28.1-dev@0be6e56`. It does not port the v0.25.1 V1
communicator monkey patch or claim PCP CUDA Graph support.

The second-review implementation tested on hardware is commit `376f2a5`. Its
isolated wheel is
`vllm_hcu-0.28.1rc1.dev491+das.376f2a5.dtk2604-cp310-cp310-linux_x86_64.whl`,
SHA-256 `d1e440c93b03db8981cad64d37f1f78470a0c5f73446a6a155a56ee93fc95146`.

## Changes

- Fixed DeepEP high-throughput construction derives `num_nvl_bytes` from
  `VLLM_DEEPEP_BUFFER_SIZE_MB`; the auto manager's existing maximum-size rule
  remains unchanged.
- Without a loaded speculator, `HcuGPUModelRunnerV2.profile_run()` uses a
  conservative PCP rank-local bound that accounts for partitioned prefill,
  replicated decode, and per-request layout headroom. It restores the
  configured value in `finally`; PCP1 delegates without mutation.
- With a loaded speculator, it preserves the original global token budget.
  Upstream profiles the target model and `speculator.propose()` from the same
  dummy batch, while runtime MTP restores the global PCP batch before the
  draft prefill. Shrinking that shared batch would under-profile MTP.

For global token budget `B`, PCP width `P`, maximum requests `N`, and decode
query width `Q`, the target-only temporary profile budget is:

```text
partitioned = floor(ceil(B / P) * 9 / 8)
replicated  = min(B, N * max(Q, 2))
profile     = min(B, max(1, partitioned + replicated))
```

The partitioned term retains the 12.5% imbalance allowance. The replicated
term covers decode tokens that remain on every PCP rank and two possible
`DualChunkSwap` segments per request. The global cap is safe because dispatch
cannot exceed the admitted global token budget. This formula is not applied
when a speculator is loaded; that path profiles with `B`.

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

The second review identified another shared-batch boundary: upstream
`profile_run()` also invokes MTP `propose()`, and the first MTP `_prefill()`
sees the complete global PCP batch during real sampling. A test executes the
installed upstream `AutoRegressiveSpeculator.propose()` body and records the
actual `_prefill()` input. For PCP4/MTP2 with `B=4096,N=16`, the previous
candidate profiled 1200 MTP tokens; the corrected path profiles all 4096.
The same guard covers the `N=256` case that previously profiled only 1920.

The complete relevant selection passed **159 tests** with 14 existing
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
  --served-model-name hy4-pcp-ep-mtp2-mtp-profile-fix \
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
expected for this eager PCP topology. The MTP-bearing PP1 stage retained the
global 4096-token profile: peak activation was 3.8 GiB on PP0 and 4.13 GiB on
PP1, with 14.13 GiB PP0 and 16.98 GiB PP1 available KV cache. The prior
under-profiled wheel reported only 3.8 GiB PP1 activation and 18.91 GiB PP1
KV, so the corrected profile materially exercises the missing MTP budget.

Post-review gates:

- HumanEval/0-7: **8/8**.
- Mixed high-KV gate: 255 unique prompts, each 1445 API prompt tokens with up
  to 128 output tokens, plus a delayed 4032-token API prefill. All 256
  requests returned HTTP 200 with `stop`; the long prefill completed in
  58.128 s. The run admitted 237 concurrent requests, reached 68.2% GPU KV
  usage with 0% prefix-cache hits, generated 22,608 draft tokens, accepted
  22,529, and recorded zero preemptions.
- No Traceback, OOM, timeout, reallocation, or NCCL-timeout marker appeared.
  All eight cards returned to zero reported utilization and memory after
  shutdown.

## PCP8/EP8/MTP3 FP8 validation

The exact TP1/PCP8/PP1 topology requires Hy4's existing PCP K-sharded
`linear_gate`. Without it, the model replicated every BF16 elementwise gate:
each rank loaded 129.27 GiB, and profiling left -6.39 GiB KV at the default
0.9 memory utilization (-0.78 GiB even at 0.959). A 0.98 launch failed the
initial free-memory contract, so tuning `gpu_memory_utilization` cannot create
a valid window on these 143.98 GiB devices.

Set `VLLM_HCU_ENABLE_LINEAR_GATE_PCP_SHARD=1`. The same isolated `35d188f`
wheel then loaded 116.30 GiB per rank, profiled 4.13 GiB peak activation, and
created a 65,600-token FP8 E4M3 KV cache (3.2-3.47 GiB per rank) at the default
0.9 utilization. The PCP8 config gate now rejects a missing opt-in before
weight loading instead of allowing this late negative-KV failure.

```bash
env -u VLLM_PLUGINS -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  -u http_proxy -u https_proxy -u all_proxy \
  VLLM_USE_V2_MODEL_RUNNER=1 VLLM_USE_NN=1 \
  VLLM_DEEPEP_BUFFER_SIZE_MB=256 \
  VLLM_HCU_ENABLE_LINEAR_GATE_PCP_SHARD=1 \
  HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
  PYTHONNOUSERSITE=1 \
  VLLM_CACHE_ROOT=<isolated-cache> \
  PYTHONPATH=<isolated-plugin-wheel> \
  /usr/bin/python -m vllm.entrypoints.cli.main serve \
  /models/Hy4-preview-Channel-FP8-w8a8-v2 \
  --served-model-name hy4-pcp8-ep8-mtp3-fp8 \
  --host 127.0.0.1 --port 8017 --trust-remote-code \
  --tensor-parallel-size 1 --prefill-context-parallel-size 8 \
  --enable-expert-parallel \
  --moe-backend deep_gemm --all2all-backend deepep_high_throughput \
  --attention-backend FLASHMLA_SPARSE --kv-cache-dtype fp8_e4m3 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --enforce-eager --enable-prefix-caching --max-model-len 4096 \
  --block-size 64 --max-num-seqs 256 --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.9 \
  --default-chat-template-kwargs '{"reasoning_effort":"no_think"}' \
  --seed 0
```

Runtime evidence selected eight `HcuGPUModelRunnerV2` workers, PCP ranks 0-7,
EP ranks 0-7, `DeepEPHTAll2AllManager`, the contiguous DeepGEMM HT expert
path, `FLASHMLA_SPARSE`, FP8 E4M3 KV, and MTP3. CUDA Graph was intentionally
disabled by the validated eager PCP contract.

- Concurrent HumanEval/0-7 passed **8/8**. All requests stopped normally.
- MTP drafted 849 and accepted 804 tokens during the smoke/accuracy run, with
  per-position accepted totals 279/270/255.
- A mixed gate kept eight 256-token decodes active while adding a delayed
  3,533-token prefill. All nine requests returned HTTP 200; the prefill
  returned `LONG_PREFILL_OK`, the service remained healthy, and observed
  windows reached 389.3 prompt tok/s and 114.5 generation tok/s.
- After both gates, cumulative MTP acceptance was 2,346/2,415 (97.1%), with
  zero request errors or aborts. These rates are workload observations, not a
  general performance guarantee.

The isolated hardware-test wheel was
`vllm_hcu-0.28.1rc1.dev491+das.35d188f.dtk2604-cp310-cp310-linux_x86_64.whl`,
SHA-256 `73e8e907d050f2664ade09d8ae79d9c3e92d328183b2d62d21faee92650fe2e4`.

## Historical low-concurrency A/B

Before the review fix, the same topology with `--max-num-seqs 16` showed PP0
peak activation 1.93 GiB versus 2.38 GiB on the unoptimized baseline, and PP0
available KV cache 17.75 GiB versus 15.67 GiB. Both variants passed
HumanEval/0-7 at 8/8 and 320/320 varied-length requests. This remains useful
evidence for the DeepEP/profile-memory direction, but it is not presented as
post-review-wheel A/B evidence and the small throughput delta is not a speedup
claim.
